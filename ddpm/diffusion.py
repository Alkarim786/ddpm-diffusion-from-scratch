"""Forward process q(x_t | x_0), the epsilon-prediction training loss, and the DDPM / DDIM samplers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .schedule import Schedule


class Diffusion:
    def __init__(self, model: torch.nn.Module, schedule: Schedule):
        self.model, self.sch = model, schedule

    # ---- forward process --------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps   (closed form, any t in one shot)."""
        noise = torch.randn_like(x0) if noise is None else noise
        e = self.sch.extract
        return e(self.sch.sqrt_ac, t, x0.shape) * x0 + e(self.sch.sqrt_one_minus_ac, t, x0.shape) * noise

    # ---- training ------------------------------------------------------------
    def loss(self, x0: torch.Tensor, y: torch.Tensor | None = None, p_uncond: float = 0.1) -> torch.Tensor:
        """Simple DDPM objective  E ||eps - eps_theta(x_t, t)||^2  (Ho et al. 2020, L_simple).

        With labels, each is replaced by the null class with probability `p_uncond` so a single
        network learns both conditional and unconditional scores (classifier-free guidance).
        """
        b = x0.shape[0]
        t = torch.randint(0, self.sch.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        if y is not None and p_uncond > 0:
            drop = torch.rand(b, device=x0.device) < p_uncond
            y = torch.where(drop, torch.full_like(y, self.model.num_classes), y)
        pred = self.model(self.q_sample(x0, t, noise), t, y)
        return F.mse_loss(pred, noise)

    # ---- reverse process -------------------------------------------------------
    @torch.no_grad()
    def _eps(self, x, t_batch, y, guidance):
        """Predicted noise, optionally with classifier-free guidance:
        eps = (1 + w) eps(x, y) - w eps(x, null)."""
        if y is None or guidance == 0:
            return self.model(x, t_batch, y)
        both = self.model(torch.cat([x, x]), torch.cat([t_batch, t_batch]), torch.cat([y, torch.full_like(y, self.model.num_classes)]))
        cond, uncond = both.chunk(2)
        return (1 + guidance) * cond - guidance * uncond

    def _x0_from_eps(self, x, t_batch, eps, clip=True):
        e = self.sch.extract
        x0 = e(self.sch.sqrt_recip_ac, t_batch, x.shape) * x - e(self.sch.sqrt_recipm1_ac, t_batch, x.shape) * eps
        return x0.clamp(-1, 1) if clip else x0

    @torch.no_grad()
    def ddpm_sample(
        self, shape, y=None, guidance: float = 0.0, generator: torch.Generator | None = None, device="cpu",
        return_trajectory_every: int = 0, clip: bool = True,
    ):
        """Ancestral sampling: x_{t-1} ~ N(mu_theta(x_t, t), sigma_t^2 I), t = T-1 ... 0."""
        x = torch.randn(shape, generator=generator, device=device)
        traj = [x.clone()] if return_trajectory_every else None
        e = self.sch.extract
        for i in reversed(range(self.sch.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = self._eps(x, t, y, guidance)
            x0 = self._x0_from_eps(x, t, eps, clip)
            mean = e(self.sch.posterior_coef1, t, x.shape) * x0 + e(self.sch.posterior_coef2, t, x.shape) * x
            if i > 0:
                z = torch.randn(shape, generator=generator, device=device)
                x = mean + e(self.sch.posterior_variance, t, x.shape).sqrt() * z
            else:
                x = mean
            if traj is not None and (i % return_trajectory_every == 0):
                traj.append(x.clone())
        return (x, traj) if traj is not None else x

    @torch.no_grad()
    def ddim_sample(
        self, shape, steps: int = 50, eta: float = 0.0, y=None, guidance: float = 0.0,
        generator: torch.Generator | None = None, device="cpu", clip: bool = True,
    ):
        """DDIM (Song et al. 2021) on a sub-sequence of `steps` timesteps.

        eta = 0 gives a deterministic map from the initial noise to the image; eta = 1 recovers
        DDPM-like stochasticity.
        """
        ts = torch.linspace(self.sch.T - 1, 0, steps).round().long().tolist()
        ac = self.sch.alphas_cumprod
        x = torch.randn(shape, generator=generator, device=device)
        for k, i in enumerate(ts):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = self._eps(x, t, y, guidance)
            x0 = self._x0_from_eps(x, t, eps, clip)
            if clip:  # keep eps consistent with the clipped x0
                eps = (x - ac[i].sqrt() * x0) / (1 - ac[i]).sqrt()
            a_t = ac[i]
            a_prev = ac[ts[k + 1]] if k + 1 < len(ts) else torch.tensor(1.0)
            sigma = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).sqrt()
            dir_xt = (1 - a_prev - sigma**2).clamp(min=0).sqrt() * eps
            x = a_prev.sqrt() * x0 + dir_xt
            if eta > 0 and k + 1 < len(ts):
                x = x + sigma * torch.randn(shape, generator=generator, device=device)
        return x
