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
import random
import torch.cuda.amp as amp
from einops import rearrange
import cv2
from scripts.t2v_model import UNetSD, AutoencoderKL, GaussianDiffusion, beta_schedule
from modules import devices
from modules import prompt_parser
import math

__all__ = ['TextToVideoSynthesis']

from scripts.t2v_model import torch_gc
from scripts.clip_hardcode import FrozenOpenCLIPEmbedder

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
        with open(model_dir+'/configuration.json', 'r') as f:
            config_dict = json.load(f)

        # Convert the dictionary to a namespace object
        self.config = SimpleNamespace(**config_dict)
        print("config", self.config)

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
                osp.join(self.model_dir, self.config.model["model_args"]["ckpt_unet"]),
                map_location='cpu' if devices.has_mps() else None, # default to cpu when macos, else default behaviour
            ),
            strict=True,
        )
        self.sd_model.eval()
        if not devices.has_mps():
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
                             device='cpu',
            layer='penultimate')

        self.clip_encoder.model.to('cpu')
        self.clip_encoder.to("cpu")
        self.noise_gen = torch.Generator(device='cpu')

    def compute_latents(self, vd_out, cpu_vae='GPU (half precision)', device=torch.device('cuda')):
        self.device = device
        with torch.no_grad():
            bs_vd, c, max_frames, height, width = vd_out.shape
            scale_factor = 0.18215
            vd_out_scaled = vd_out

            if 'CPU' in cpu_vae:
                print("STARTING VAE ON CPU")
                self.autoencoder.to("cpu")
                vd_out_scaled = vd_out_scaled.cpu()
            else:
                print("STARTING VAE ON GPU")
                self.autoencoder.to(self.device)
                if 'half precision' in cpu_vae:
                    self.autoencoder.half()
                    print("VAE HALVED")
                    vd_out_scaled = vd_out_scaled.half()

            vd_out_scaled = rearrange(
                vd_out_scaled, 'b c f h w -> (b f) c h w')

            # Split the tensor into chunks along the first dimension
            chunk_size = 1
            chunks = vd_out_scaled.chunk(vd_out_scaled.size(0) // chunk_size)

            latents_chunks = []
            for chunk in chunks:
                if 'CPU' in cpu_vae:
                    ch = chunk.cpu().float()
                else:
                    ch = chunk.to(self.device).float()
                    if 'half precision' in cpu_vae:
                        ch = ch.half()

                latents_chunk = self.autoencoder.encode(ch)
                latents_chunk = torch.tensor(
                    latents_chunk.mean).cpu() * scale_factor
                # latents_chunks.append(latents_chunk.cpu())
                latents_chunks.append(latents_chunk)

            # Concatenate the latents chunks back into a single tensor
            latents = torch.cat(latents_chunks, dim=0)
            latents = rearrange(latents, '(b f) c h w -> b c f h w', b=bs_vd)

        out = latents.type(torch.float32).cpu()
        return out

    # @torch.compile()
    def infer(self, prompt, n_prompt, steps, frames, seed, scale, width=256, height=256, eta=0.0, cpu_vae='GPU (half precision)', device=torch.device('cpu'), latents=None, strength=0, percentile=0):
        r"""
        The entry function of text to image synthesis task.
        1. Using diffusion model to generate the video's latent representation.
        2. Using vqgan model (autoencoder) to decode the video's latent representation to visual space.

        Args:
            prompt (str, optional): A string describing the scene to generate. Defaults to "A bunny in the forest".
            n_prompt (Optional[str], optional): An additional prompt for generating the scene. Defaults to "".
            steps (int, optional): The number of steps to run the diffusion model. Defaults to 50.
            frames (int, optional): The number of frames in the generated video. Defaults to 15.
            scale (float, optional): The scaling factor for the generated video. Defaults to 12.5.
            width (int, optional): The width of the generated video. Defaults to 256.
            height (int, optional): The height of the generated video. Defaults to 256.
            eta (float, optional): A hyperparameter related to the diffusion model's noise schedule. Defaults to 0.0.
            cpu_vae (bool, optional): If True, the VQGAN model will run on the CPU. Defaults to 'GPU (half precision)'.
            latents (Optional[Tensor], optional): An optional latent tensor to use as input for the VQGAN model. Defaults to None.
            strength (Optional[float], optional): A hyperparameter to control the strength of the generated video when using input latent. Defaults to None.

        Returns:
            A generated video (as list of np.arrays).
        """

        seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
        self.device = device
        self.clip_encoder.to(self.device)
        self.clip_encoder.device = self.device
        c, uc = self.preprocess(prompt, n_prompt, steps)
        self.clip_encoder.to("cpu")
        torch_gc()

        # synthesis
        percentile = None if percentile == 0.0 else percentile
        with torch.no_grad():
            num_sample = 1
            max_frames = frames
            latent_h, latent_w = height // 8, width // 8
            self.sd_model.to(self.device)
            if latents == None:
                self.noise_gen.manual_seed(seed)
                latents = torch.randn(num_sample, 4, max_frames, latent_h,
                                          latent_w, generator=self.noise_gen).to(
                                              self.device)
            else:
                latents.to(self.device)

            print("latents", latents.shape, torch.mean(
                latents), torch.std(latents))

            with amp.autocast(enabled=True):
                self.sd_model.to(self.device)
                x0 = self.diffusion.ddim_sample_loop(
                    noise=latents,  # shape: b c f h w
                    model=self.sd_model,
                    c=c,
                    uc=uc,
                    num_sample=1,
                    guide_scale=scale,
                    ddim_timesteps=steps,
                    eta=eta,
                    percentile=percentile,
                    skip_steps=int(math.floor(steps*max(0, min(1 - strength, 1)))),
                )

                self.last_tensor = x0
                self.last_tensor.cpu()
                self.sd_model.to("cpu")
                torch_gc()
                scale_factor = 0.18215
                bs_vd = x0.shape[0]
                if 'CPU' in cpu_vae:
                    x0 = x0.cpu()
                    print("DECODING FRAMES")
                    print(x0.shape)
                    # self.autoencoder.to(self.device)
                    x0.float()
                    # Split the tensor into chunks along the first dimension
                    chunk_size = 1
                    chunks = torch.chunk(x0, chunks=max_frames, dim=2)
                    # Apply the autoencoder to each chunk
                    output_chunks = []
                    self.autoencoder.to("cpu")
                    print("STARTING VAE ON CPU")
                    x = 0
                    for chunk in chunks:
                        ch = chunk.cpu().float()
                        ch = 1. / scale_factor * ch
                        ch = rearrange(ch, 'b c f h w -> (b f) c h w')
                        # print(ch)
                        chunk = None
                        del chunk
                        output_chunk = self.autoencoder.decode(ch)
                        output_chunk.cpu()
                        output_chunks.append(output_chunk)
                        x += 1
                else:
                    chunk_size = 1
                    chunks = torch.chunk(x0, chunks=max_frames, dim=2)
                    x0 = x0.cpu()
                    del x0

                    print(
                        f"STARTING VAE ON GPU. {len(chunks)} CHUNKS TO PROCESS")
                    self.autoencoder.to(self.device)
                    if 'half precision' in cpu_vae:
                        self.autoencoder.half()
                        print(f"VAE HALVED")
                    print("DECODING FRAMES")

                    # Split the tensor into chunks along the first dimension
                    # Apply the autoencoder to each chunk
                    output_chunks = []
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
                print("VAE FINISHED")
                torch_gc()
                # Concatenate the output chunks back into a single tensor
                vd_out = torch.cat(output_chunks, dim=0)
                # video_data = self.autoencoder.decode(video_data)
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

        # self.autoencoder = None
        # del self.autoencoder
        del vd_out
        del latents
        x0 = None
        del x0
        video_data = None
        del video_data
        torch_gc()
        last_tensor = self.last_tensor
        return video_path, last_tensor, create_infotext(prompt, n_prompt, steps, frames, seed, scale, width, height, eta, strength)

    def cleanup(self):
        pass

    def preprocess(self, prompt, n_prompt, steps, offload=True):
        cached_uc = [None, None]
        cached_c = [None, None]

        def get_conds_with_caching(function, model, required_prompts, steps, cache):
            if cache[0] is not None and (required_prompts, steps) == cache[0]:
                return cache[1]

            with devices.autocast():
                cache[1] = function(model, required_prompts, steps)

            cache[0] = (required_prompts, steps)
            return cache[1]

        self.clip_encoder.to(self.device) 
        self.clip_encoder.device = self.device       
        uc = get_conds_with_caching(prompt_parser.get_learned_conditioning, self.clip_encoder, [n_prompt], steps, cached_uc)
        c = get_conds_with_caching(prompt_parser.get_learned_conditioning, self.clip_encoder, [prompt], steps, cached_c)
        if offload:
            self.clip_encoder.to('cpu')
        return c, uc

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
            # video_writer.write(img)
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

from modules import generation_parameters_copypaste

def create_infotext(prompt, n_prompt, steps, frames, seed, scale, width=256, height=256, eta=0.0, strength=0, comments=None):

    generation_params = {
        "Frames": frames,
        "Steps": steps,
        "Sampler": 'DDIM', # only DDIM is currently supported
        "CFG scale": scale,
        #"Image CFG scale": getattr(p, 'image_cfg_scale', None),
        "Seed": seed,
        #"Face restoration": (opts.face_restoration_model if p.restore_faces else None),
        "Size": f"{width}x{height}",
        #"Model hash": getattr(p, 'sd_model_hash', None if not opts.add_model_hash_to_info or not shared.sd_model.sd_model_hash else shared.sd_model.sd_model_hash),
        #"Model": (None if not opts.add_model_name_to_info or not shared.sd_model.sd_checkpoint_info.model_name else shared.sd_model.sd_checkpoint_info.model_name.replace(',', '').replace(':', '')),
        #"Variation seed": (None if p.subseed_strength == 0 else all_subseeds[index]),
        #"Variation seed strength": (None if p.subseed_strength == 0 else p.subseed_strength),
        #"Seed resize from": (None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": strength,
        "Eta": eta,
        #"Conditional mask weight": getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) if p.is_using_inpainting_conditioning else None,
        #"Clip skip": None if clip_skip <= 1 else clip_skip,
        #"ENSD": None if opts.eta_noise_seed_delta == 0 else opts.eta_noise_seed_delta,
    }

    #generation_params.update(p.extra_generation_params)

    generation_params_text = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in generation_params.items() if v is not None])

    negative_prompt_text = "\nNegative prompt: " + n_prompt if len(n_prompt) > 0 else ""

    return f"{prompt}{negative_prompt_text}\n{generation_params_text}".strip()
