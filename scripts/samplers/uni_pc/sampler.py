"""SAMPLING ONLY."""

import torch

from .uni_pc import NoiseScheduleVP, model_wrapper, UniPC

class UniPCSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def unipc_encode(self, latent, device, strength, steps, noise=None):
        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)
        uni_pc = UniPC(None, ns, predict_x0=True, thresholding=False, variant='bh1')
        t_0 = 1. / ns.total_N

        timesteps = uni_pc.get_time_steps("time_uniform", strength, t_0, steps, device)
        timesteps = timesteps[0].expand((latent.shape[0]))

        noisy_latent = uni_pc.unipc_encode(latent, timesteps, noise=noise)
        return noisy_latent

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               strength=None,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):

        # sampling
        B, C, F, H, W = shape
        size = (B, C, F, H, W)

        if x_T is None:
            img = torch.randn(size, device=self.model.device)
        else:
            img = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)
        model_fn = model_wrapper(
            lambda x, t, c: self.model(x, t, c),
            ns,
            model_type="noise",
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )
        
        uni_pc = UniPC(model_fn, ns, predict_x0=True, thresholding=False, variant='bh1')
        x = uni_pc.sample(
            img, 
            steps=S, 
            t_start=strength,
            skip_type="time_uniform", 
            method="multistep", 
            order=3, 
            lower_order_final=True, 
            initial_corrector=True,
            callback=callback
        )

        return x.to(self.model.device)