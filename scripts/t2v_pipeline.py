# https://github.com/modelscope/modelscope/tree/master/modelscope/pipelines/multi_modal Apache 2.0
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import datetime
import json
import os
import tempfile
from os import path as osp
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.cuda.amp as amp
from einops import rearrange
import cv2
from .t2v_model import UNetSD, AutoencoderKL, FrozenOpenCLIPEmbedder, GaussianDiffusion, beta_schedule


__all__ = ['TextToVideoSynthesis']

try:
    import gc
    import torch

    def torch_gc():
        """Performs garbage collection for both Python and PyTorch CUDA tensors.

        This function collects Python garbage and clears the PyTorch CUDA cache
        and IPC (Inter-Process Communication) resources.
        """
        gc.collect()  # Collect Python garbage
        torch.cuda.empty_cache()  # Clear PyTorch CUDA cache
        torch.cuda.ipc_collect()  # Clear PyTorch CUDA IPC resources

except:

    def torch_gc():
        """Dummy function when torch is not available.

        This function does nothing and serves as a placeholder when torch is
        not available, allowing the rest of the code to run without errors.
        """
        pass

class TextToVideoSynthesis():
    r"""
    task for text to video synthesis.

    Attributes:
        sd_model: denosing model using in this task.
        diffusion: diffusion model for DDIM.
        autoencoder: decode the latent representation into visual space with VQGAN.
        clip_encoder: encode the text into text embedding.
    """

    def __init__(self, model_dir):
        r"""
        Args:
            model_dir (`str` or `os.PathLike`)
                Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co
                      or modelscope.cn. Valid model ids can be located at the root-level, like `bert-base-uncased`,
                      or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
        """
        super().__init__()
        self.model_dir = model_dir
        self.device = torch.device('cpu')
        # Load the configuration from a file
        with open(model_dir+'/t2v/configuration.json', 'r') as f:
            config_dict = json.load(f)

        # Convert the dictionary to a namespace object
        self.config = SimpleNamespace(**config_dict)
        print("config",self.config)


        cfg = self.config.model["model_cfg"]
        cfg['temporal_attention'] = True if cfg[
            'temporal_attention'] == 'True' else False

        # Initialize unet
        self.sd_model = UNetSD(
            in_dim=cfg['unet_in_dim'],
            dim=cfg['unet_dim'],
            y_dim=cfg['unet_y_dim'],
            context_dim=cfg['unet_context_dim'],
            out_dim=cfg['unet_out_dim'],
            dim_mult=cfg['unet_dim_mult'],
            num_heads=cfg['unet_num_heads'],
            head_dim=cfg['unet_head_dim'],
            num_res_blocks=cfg['unet_res_blocks'],
            attn_scales=cfg['unet_attn_scales'],
            dropout=cfg['unet_dropout'],
            temporal_attention=cfg['temporal_attention'])
        self.sd_model.load_state_dict(
            torch.load(
                osp.join(self.model_dir, self.config.model["model_args"]["ckpt_unet"])),
            strict=True)
        self.sd_model.eval()
        self.sd_model.half()

        # Initialize diffusion
        betas = beta_schedule(
            'linear_sd',
            cfg['num_timesteps'],
            init_beta=0.00085,
            last_beta=0.0120)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            mean_type=cfg['mean_type'],
            var_type=cfg['var_type'],
            loss_type=cfg['loss_type'],
            rescale_timesteps=False)

        # Initialize autoencoder
        ddconfig = {
            'double_z': True,
            'z_channels': 4,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0
        }
        self.autoencoder = AutoencoderKL(
            ddconfig, 4,
            osp.join(self.model_dir, self.config.model["model_args"]["ckpt_autoencoder"]))
        self.autoencoder.to('cpu')
        self.autoencoder.eval()

        # Initialize Open clip
        self.clip_encoder = FrozenOpenCLIPEmbedder(
            version=osp.join(self.model_dir,
                             self.config.model["model_args"]["ckpt_clip"]),
            layer='penultimate')

        self.clip_encoder.model.to('cpu')

        self.clip_encoder.to("cpu")

    #@torch.compile()
    def infer(self, prompt, n_prompt, steps, frames, scale, width=256, height=256, eta=0.0, cpu_vae=False, latents=None):
        r"""
        The entry function of text to image synthesis task.
        1. Using diffusion model to generate the video's latent representation.
        2. Using vqgan model (autoencoder) to decode the video's latent representation to visual space.

        Args:
            input (`Dict[Str, Any]`):
                The input of the task
        Returns:
            A generated video (as pytorch tensor).
        """
        print(self.sd_model.use_fps_condition)
        self.sd_model.use_fps_condition = False
        self.device = torch.device('cuda')
        self.clip_encoder.to(self.device)
        y, zero_y = self.preprocess(prompt, n_prompt)
        self.clip_encoder.to("cpu")
        #self.clip_encoder = None
        #del self.clip_encoder
        torch_gc()

        context = torch.cat([zero_y, y], dim=0).to(self.device)
        # synthesis

        with torch.no_grad():
            num_sample = 1
            max_frames = frames
            latent_h, latent_w = height // 8, width // 8
            self.sd_model.to(self.device)
            if latents == None:
                latents = torch.randn(num_sample, 4, max_frames, latent_h,
                                          latent_w).to(
                                              self.device)
            else:
                latents.to(self.device)
            with amp.autocast(enabled=True):
                self.sd_model.to(self.device)
                x0 = self.diffusion.ddim_sample_loop(
                    noise=latents,  # shape: b c f h w
                    model=self.sd_model,
                    model_kwargs=[{
                        'y':
                        context[1].unsqueeze(0).repeat(num_sample, 1, 1)
                    }, {
                        'y':
                        context[0].unsqueeze(0).repeat(num_sample, 1, 1)
                    }],
                    guide_scale=scale,
                    ddim_timesteps=steps,
                    eta=eta)
                self.last_tensor = x0
                self.last_tensor.cpu()
                self.sd_model.to("cpu")
                torch_gc()
                scale_factor = 0.18215
                bs_vd = x0.shape[0]
                if cpu_vae == True:
                    vd = x0.cpu()
                    x0 = None
                    del x0
                    vd = rearrange(vd, 'b c f h w -> (b f) c h w')
                    print("CREATING FRAME")
                    print(vd.shape)
                    #self.autoencoder.to(self.device)
                    vd.float()
                    # Split the tensor into chunks along the first dimension
                    chunk_size = 1
                    chunks = vd.chunk(vd.size(0) // chunk_size)
                    # Apply the autoencoder to each chunk
                    output_chunks = []
                    self.autoencoder.to("cpu")
                    print("STARTING VAE ON CPU")
                    x = 0
                    for chunk in chunks:
                        ch = chunk.cpu().float()
                        ch = 1. / scale_factor * ch
                        ch = rearrange(ch, 'b c f h w -> (b f) c h w')
                        #print(ch)
                        chunk = None
                        del chunk
                        output_chunk = self.autoencoder.decode(ch)
                        output_chunk.cpu()
                        output_chunks.append(output_chunk)
                        x += 1
                else:
                    chunk_size = 1
                    chunks = x0.chunk(x0.size(0) // chunk_size)
                    x0.cpu()
                    del x0
                    print("CREATING FRAME")
                    #print(x0.shape)
                    self.autoencoder.to(self.device)
                    # Split the tensor into chunks along the first dimension
                    # Apply the autoencoder to each chunk
                    output_chunks = []
                    print(f"STARTING VAE ON GPU {len(chunks)}")
                    torch_gc()
                    x = 0
                    for chunk in chunks:
                        chunk = 1. / scale_factor * chunk

                        chunk = rearrange(chunk, 'b c f h w -> (b f) c h w')
                        output_chunk = self.autoencoder.decode(chunk)
                        cpu_chunk = output_chunk.cpu()
                        del output_chunk
                        output_chunks.append(cpu_chunk)
                        x += 1
                print("FINISHED VAE ON CPU")
                torch_gc()
                # Concatenate the output chunks back into a single tensor
                vd_out = torch.cat(output_chunks, dim=0)
                #video_data = self.autoencoder.decode(video_data)
                print(vd_out.shape)
                vd_out = rearrange(
                    vd_out, '(b f) c h w -> b c f h w', b=bs_vd)
        vd_out = vd_out.type(torch.float32).cpu()

        video_path = self.postprocess_video(vd_out)
        self.clip_encoder.to("cpu")
        self.sd_model.to("cpu")
        self.autoencoder.to("cpu")
        self.autoencoder.encoder.to("cpu")
        self.autoencoder.decoder.to("cpu")

        #self.autoencoder = None
        #del self.autoencoder
        del vd_out
        del context
        del latents
        x0 = None
        del x0
        video_data = None
        del video_data
        torch_gc()
        last_tensor = self.last_tensor
        return video_path, last_tensor
    def cleanup(self):
        pass
    def preprocess(self, prompt, n_prompt, offload=True):
        self.clip_encoder.to(self.device)
        text_emb = self.clip_encoder(prompt)
        text_emb_zero = self.clip_encoder(n_prompt)
        if offload:
            self.clip_encoder.to('cpu')
        return text_emb.type(torch.float16), text_emb_zero.type(torch.float16)

    def postprocess_video(self, video_data):
        video = tensor2vid(video_data)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')
        filename = f"output/mp4s/{timestamp}.mp4"

        output_video_path = filename
        if output_video_path is None:
            output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name

        print(output_video_path)

        """fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w, c = video[0].shape
        video_writer = cv2.VideoWriter(
            output_video_path, fourcc, fps=8, frameSize=(w, h))"""
        return_samples = []
        for i in range(len(video)):
            img = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
            #video_writer.write(img)
            return_samples.append(img)
        del video
        del video_data
        return return_samples
    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Run the forward pass for a model.

        Returns:
            Dict[str, Any]: output from the model forward pass
        """
        pass


def tensor2vid(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(
        mean, device=video.device).reshape(1, -1, 1, 1, 1)  # ncfhw
    std = torch.tensor(
        std, device=video.device).reshape(1, -1, 1, 1, 1)  # ncfhw
    video = video.mul_(std).add_(mean)  # unnormalize back to [0,1]
    del mean
    del std
    video.clamp_(0, 1)
    images = rearrange(video, 'i c f h w -> f h (i w) c')
    images = images.unbind(dim=0)
    images = [(image.numpy() * 255).astype('uint8')
              for image in images]  # f h w c
    return images


