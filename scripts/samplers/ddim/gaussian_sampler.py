import torch
from modelscope.t2v_model import _i
from t2v_helpers.general_utils import reconstruct_conds

class GaussianDiffusion(object):
    r""" Diffusion Model for DDIM.
    "Denoising diffusion implicit models." by Song, Jiaming, Chenlin Meng, and Stefano Ermon.
    See https://arxiv.org/abs/2010.02502
    """

    def __init__(self,
                model,
                betas,
                mean_type='eps',
                var_type='learned_range',
                loss_type='mse',
                epsilon=1e-12,
                rescale_timesteps=False,
                **kwargs):

        # check input
        self.check_input_vars(betas, mean_type, var_type, loss_type)

        self.model = model
        self.betas = betas
        self.num_timesteps = len(betas)
        self.mean_type = mean_type
        self.var_type = var_type
        self.loss_type = loss_type
        self.epsilon = epsilon
        self.rescale_timesteps = rescale_timesteps

        # alphas
        alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([alphas.new_ones([1]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:],alphas.new_zeros([1])])

        # q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def check_input_vars(self, betas, mean_type, var_type, loss_type):
        mean_types = ['x0', 'x_{t-1}', 'eps']
        var_types = ['learned', 'learned_range', 'fixed_large', 'fixed_small']
        loss_types = ['mse', 'rescaled_mse', 'kl', 'rescaled_kl', 'l1', 'rescaled_l1','charbonnier']

        if not isinstance(betas, torch.DoubleTensor):
                betas = torch.tensor(betas, dtype=torch.float64)

        assert min(betas) > 0 and max(betas) <= 1
        assert mean_type in mean_types
        assert var_type in var_types
        assert loss_type in loss_types
        
    def validate_model_kwargs(self, model_kwargs):
        """
        Use the original implementation of passing model kwargs to the model.
        eg:  model_kwargs=[{'y':c_i}, {'y':uc_i,}]
        """
        if len(model_kwargs) > 0:
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2

    def get_time_steps(self, ddim_timesteps, batch_size=1, step=None):
        b = batch_size

        # Get thhe full timestep range
        arange_steps = (1 + torch.arange(0, self.num_timesteps, ddim_timesteps))
        steps = arange_steps.clamp(0, self.num_timesteps - 1)
        timesteps = steps.flip(0).to(self.model.device)

        if step is not None:
            # Get the current timestep during a sample loop
            timesteps = torch.full((b, ), timesteps[step], dtype=torch.long, device=self.model.device)

        return timesteps

    def add_noise(self, xt, noise, t):
        noisy_sample = self.sqrt_alphas_cumprod[t.cpu()].to(self.model.device) * \
            xt + noise * self.sqrt_one_minus_alphas_cumprod[t.cpu()].to(self.model.device)

        return noisy_sample

    def get_dim(self, y_out):
        is_fixed = self.var_type.startswith('fixed')
        return y_out.size(1) if is_fixed else y_out.size(1) // 2

    def fixed_small_variance(self, xt, t):
        var = _i(self.posterior_variance, t, xt)
        log_var = _i(self.posterior_log_variance_clipped, t, xt)

        return var, log_var

    def mean_x0(self, xt, t, x_out):
        x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - _i(
                self.sqrt_recipm1_alphas_cumprod, t, xt) * x_out
        mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)

        return x0, mu

    def restrict_range_x0(self, percentile, x0, clamp=False):
        if not clamp:
            assert percentile > 0 and percentile <= 1  # e.g., 0.995
            s = torch.quantile(x0.flatten(1).abs(), percentile,dim=1)
            s.clamp_(1.0).view(-1, 1, 1, 1)

            x0 = torch.min(s, torch.max(-s, x0)) / s
        else:
            x0 = x0.clamp(-clamp, clamp)

        return x0

    def is_unconditional(self, guide_scale):
        return guide_scale is None or guide_scale == 1

    def do_classifier_guidance(self, y_out, u_out, guidance_scale):
        """
        y_out: Condition
        u_out: Unconditional
        """
        dim = self.get_dim(y_out)
        a = u_out[:, :dim]
        b = guidance_scale * (y_out[:, :dim] - u_out[:, :dim])
        c = y_out[:, dim:]
        out = torch.cat([a + b, c], dim=1)

        return out
        
    def p_mean_variance(self,
                        xt,
                        t,
                        model_kwargs={},
                        clamp=None,
                        percentile=None,
                        guide_scale=None,
                        conditioning=None,
                        unconditional_conditioning=None,
                        only_x0=True,
                        **kwargs):
        r"""Distribution of p(x_{t-1} | x_t)."""

        # predict distribution
        if self.is_unconditional(guide_scale):
            out = self.model(xt, self._scale_timesteps(t), conditioning)
        else:
            # classifier-free guidance
            if model_kwargs != {}:
                self.validate_model_kwargs(model_kwargs)
                conditioning = model_kwargs[0]
                unconditional_conditioning = model_kwargs[1]

            y_out = self.model(xt, self._scale_timesteps(t), conditioning)
            u_out = self.model(xt, self._scale_timesteps(t), unconditional_conditioning)

            out = self.do_classifier_guidance(y_out, u_out, guide_scale)

        # compute variance
        if self.var_type == 'fixed_small':
            var, log_var = self.fixed_small_variance(xt, t)

        # compute mean and x0
        if self.mean_type == 'eps':
            x0, mu = self.mean_x0(xt, t, out)

        # restrict the range of x0
        if percentile is not None:
           x0 = self.restrict_range_x0(percentile, x0)
        elif clamp is not None:
            x0 = self.restrict_range_x0(percentile, x0, clamp=True)

        if only_x0:
            return x0
        else:
            return mu, var, log_var, x0

    def q_posterior_mean_variance(self, x0, xt, t):
        r"""Distribution of q(x_{t-1} | x_t, x_0).
        """
        mu = _i(self.posterior_mean_coef1, t, xt) * x0 + _i(
            self.posterior_mean_coef2, t, xt) * xt
        var = _i(self.posterior_variance, t, xt)
        log_var = _i(self.posterior_log_variance_clipped, t, xt)
        return mu, var, log_var

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * 1000.0 / self.num_timesteps
        return t

    def get_eps(self, xt, x0, t, alpha, condition_fn, model_kwargs={}):
        # x0 -> eps
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / _i(
                self.sqrt_recipm1_alphas_cumprod, t, xt)
        
        if condition_fn is not None:
            eps = eps - (1 - alpha).sqrt() * condition_fn(
                    xt, self._scale_timesteps(t), **model_kwargs)
            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - _i(
                    self.sqrt_recipm1_alphas_cumprod, t, xt) * eps

        return eps, x0

    @torch.no_grad()
    def sample(self,
                    x_T=None,
                    S=5,
                    shape=None,
                    conditioning=None,
                    unconditional_conditioning=None,
                    model_kwargs={},
                    clamp=None,
                    percentile=None,
                    condition_fn=None,
                    unconditional_guidance_scale=None,
                    eta=0.0,
                    callback=None,
                    mask=None,
                    **kwargs):
        r"""Sample from p(x_{t-1} | x_t) using DDIM.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        
        # Shape must exist to sample
        if shape is None and x_T is None:
            assert "Shape must exists to sample from noise"

        # Assign variables for sampling
        steps = S
        stride = self.num_timesteps // steps
        guide_scale = unconditional_guidance_scale
        original_latents = None

        if x_T is None:
            xt = torch.randn(shape, device=self.model.device)
        else:
            xt = x_T.clone()
            original_latents = xt
        
        timesteps = self.get_time_steps(stride, xt.shape[0])

        for step in range(0, steps):
            c, uc = reconstruct_conds(conditioning, unconditional_conditioning, step)
            t = self.get_time_steps(stride, xt.shape[0], step=step)

            # predict distribution of p(x_{t-1} | x_t)
            x0 = self.p_mean_variance(
                xt, 
                t,
                 model_kwargs, 
                 clamp, 
                 percentile, 
                 guide_scale, 
                 conditioning=c,
                 unconditional_conditioning=uc,
                 **kwargs
                )

            alphas = _i(self.alphas_cumprod, t, xt)
            alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)

            eps, x0 = self.get_eps(xt, x0, t, alphas, condition_fn)

            a = (1 - alphas_prev) / (1 - alphas)
            b = (1 - alphas / alphas_prev)
            sigmas = eta * torch.sqrt(a * b)

            # random sample
            noise = torch.randn_like(xt)
            direction = torch.sqrt(1 - alphas_prev - sigmas**2) * eps
            mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
            xt_1 = torch.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
            xt = xt_1
    
            if hasattr(self, 'inpaint_masking') and mask is not None:
                add_noise_args = {
                    "xt":xt, 
                    "noise": torch.randn_like(xt), 
                    "t": timesteps[(step - 1) + 1]
                }
                self.inpaint_masking(xt, step, steps, mask, self.add_noise, add_noise_args)

            if callback is not None:
                callback(step)

        return xt
        


        