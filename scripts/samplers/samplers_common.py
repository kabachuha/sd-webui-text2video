import torch
from samplers.ddim.sampler import DDIMSampler
from samplers.ddim.gaussian_sampler import GaussianDiffusion
from samplers.uni_pc.sampler import UniPCSampler
from tqdm import tqdm
from modules.shared import state
from modules.sd_samplers_common import InterruptedException

def get_height_width(h, w, divisor):
    return h // divisor, w // divisor

def get_tensor_shape(batch_size, channels, frames, h, w, latents=None):
    if latents is None:
        return (batch_size, channels, frames, h, w)
    return latents.shape

def inpaint_masking(xt, step, steps, mask, add_noise_cb, noise_cb_args):
    if mask is not None and step < steps - 1:

        #convert mask to 0,1 valued based on step
        v = (steps - step - 1) / steps
        binary_mask = torch.where(mask <= v, torch.zeros_like(mask), torch.ones_like(mask))

        noise_to_add = add_noise_cb(**noise_cb_args)
        to_inpaint = noise_to_add
        xt = to_inpaint * (1 - binary_mask) + xt * binary_mask

class SamplerStepCallback(object):
    def __init__(self, sampler_name: str, total_steps: int):
        self.sampler_name = sampler_name
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_bar = tqdm(desc=self.progress_msg(sampler_name, total_steps), total=total_steps)

    def progress_msg(self, name, total_steps=None):
        total_steps = total_steps if total_steps is not None else self.total_steps
        return f"Sampling Using {name} for {total_steps} steps."

    def set_webui_step(self, step):
        state.sampling_step = step

    def is_finished(self, step):
        if step >= self.total_steps:
            self.progress_bar.close()
            self.current_step = 0

    def interrupt(self):
        return state.interrupted or state.skipped

    def cancel(self):
        raise InterruptedException

    def update(self, step):
        self.set_webui_step(step)

        if self.interrupt():
            self.cancel()

        self.progress_bar.set_description(self.progress_msg(self.sampler_name))
        self.progress_bar.update(1)

        self.is_finished(step)  

    def __call__(self,*args, **kwargs):
        self.current_step += 1
        step = self.current_step

        self.update(step)

class SamplerBase(object):
    def __init__(self, name: str, Sampler, frame_inpaint_support=False):
        self.name = name
        self.Sampler = Sampler
        self.frame_inpaint_support = frame_inpaint_support

    def register_buffers_to_model(self, sd_model, betas, device):
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        setattr(sd_model, 'device', device)
        setattr(sd_model, 'betas', betas)
        setattr(sd_model, 'alphas_cumprod', self.alphas_cumprod)

    def init_sampler(self, sd_model, betas, device, **kwargs):
        self.register_buffers_to_model(sd_model, betas, device)
        return self.Sampler(sd_model, betas=betas, **kwargs)
        
available_samplers = [
    SamplerBase("DDIM_Gaussian", GaussianDiffusion, True),
    SamplerBase("DDIM", DDIMSampler), 
    SamplerBase("UniPC", UniPCSampler),
]   

class Txt2VideoSampler(object):
    def __init__(self, sd_model, device, betas=None, sampler_name="UniPC"):
        self.sd_model = sd_model
        self.device = device
        self.noise_gen = torch.Generator(device='cpu')
        self.sampler_name = sampler_name
        self.betas = betas
        self.sampler = self.get_sampler(sampler_name, betas=self.betas)
    
    def get_noise(self, num_sample, channels, frames, height, width, latents=None, seed=1):
        if latents is not None:
            latents.to(self.device)

            print(f"Using input latents. Shape: {latents.shape}, Mean: {torch.mean(latents)}, Std: {torch.std(latents)}")
        else:
            print("Sampling random noise.")

        num_sample = 1
        max_frames = frames

        latent_h, latent_w = get_height_width(height, width, 8)
        shape = get_tensor_shape(num_sample, channels, max_frames, latent_h, latent_w, latents)

        self.noise_gen.manual_seed(seed)
        noise = torch.randn(shape, generator=self.noise_gen).to(self.device)
        
        return latents, noise, shape

    def encode_latent(self, latent, noise, strength, steps):
        encoded_latent = None
        denoise_steps = None

        if hasattr(self.sampler, 'unipc_encode'):
            encoded_latent = self.sampler.unipc_encode(latent, self.device, strength, steps, noise=noise)

        if hasattr(self.sampler, 'stochastic_encode'):
            denoise_steps = int(strength * steps)
            timestep = torch.tensor([denoise_steps] * int(latent.shape[0])).to(self.device)
            self.sampler.make_schedule(steps)
            encoded_latent = self.sampler.stochastic_encode(latent, timestep, noise=noise).to(dtype=latent.dtype)
            self.sampler.sample = self.sampler.decode
        
        if hasattr(self.sampler, 'add_noise'):
            denoise_steps = int(strength * steps)
            timestep = self.sampler.get_time_steps(denoise_steps, latent.shape[0])
            encoded_latent = self.sampler.add_noise(latent, noise, timestep[0].cpu())

        if encoded_latent is None:
            assert "Could not find the appropriate function to encode the input latents"
        
        return encoded_latent, denoise_steps
            
    def get_sampler(self, sampler_name: str, betas=None, return_sampler=True):
        betas = betas if betas is not None else self.betas

        for Sampler in available_samplers:
            if sampler_name == Sampler.name:
                sampler = Sampler.init_sampler(self.sd_model, betas=betas, device=self.device)

                if Sampler.frame_inpaint_support:
                    setattr(sampler, 'inpaint_masking', inpaint_masking)

                if return_sampler:
                    return sampler
                else:
                    self.sampler = sampler
                    return

        raise ValueError(f"Sample {sampler_name} does not exist.")
        
    def sample_loop(
        self, 
        steps, 
        strength, 
        conditioning, 
        unconditional_conditioning,
        batch_size,
        latents=None,
        shape=None,
        noise=None,
        is_vid2vid=False,
        guidance_scale=1,
        eta=0,
        mask=None,
        sampler_name="DDIM"
    ):
        denoise_steps = None
        # Assume that we are adding noise to existing latents (Image, Video, etc.)
        if latents is not None and is_vid2vid:
            latents, denoise_steps = self.encode_latent(latents, noise, strength, steps)
        
        # Create a callback that handles counting each step
        sampler_callback = SamplerStepCallback(sampler_name, steps)

        # Predict the noise sample
        x0 = self.sampler.sample(
            S=steps,
            conditioning=conditioning,
            strength=strength,
            unconditional_conditioning=unconditional_conditioning,
            batch_size=batch_size,
            x_T=latents if latents is not None else noise,
            x_latent=latents,
            t_start=denoise_steps,
            unconditional_guidance_scale=guidance_scale,
            shape=shape,
            callback=sampler_callback,
            cond=conditioning,
            eta=eta,
            mask=mask
        )

        return x0