"""Noise schedules and the derived constants used by the forward and reverse processes."""

from __future__ import annotations

import math

import torch


def linear_betas(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Ho et al. 2020, defined for T=1000; rescaled so other T keep the same total noise."""
    scale = 1000 / T
    return torch.linspace(beta_start * scale, beta_end * scale, T, dtype=torch.float64)


def cosine_betas(T: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Nichol & Dhariwal 2021: alpha_bar follows a squared-cosine curve, which destroys
    information more gradually than the linear schedule (important for small images)."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=max_beta)


class Schedule:
    """Holds beta_t, alpha_t, alpha_bar_t and the posterior q(x_{t-1} | x_t, x_0) coefficients.

    Time is 0-indexed: index t corresponds to the paper's step t+1.
    """

    def __init__(self, T: int = 1000, kind: str = "cosine"):
        if kind == "linear":
            betas = linear_betas(T)
        elif kind == "cosine":
            betas = cosine_betas(T)
        else:
            raise ValueError(f"unknown schedule '{kind}'")
        self.T, self.kind = T, kind
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, 0)
        ac_prev = torch.cat([torch.ones(1, dtype=torch.float64), ac[:-1]])
        post_var = betas * (1 - ac_prev) / (1 - ac)
        f = lambda a: a.to(torch.float32)
        self.betas, self.alphas, self.alphas_cumprod, self.alphas_cumprod_prev = map(f, (betas, alphas, ac, ac_prev))
        self.sqrt_ac = f(ac.sqrt())
        self.sqrt_one_minus_ac = f((1 - ac).sqrt())
        self.sqrt_recip_ac = f((1 / ac).sqrt())
        self.sqrt_recipm1_ac = f((1 / ac - 1).sqrt())
        self.posterior_variance = f(post_var)
        # Clipped log variance: the posterior variance is 0 at t=0.
        self.posterior_log_variance = f(torch.log(post_var.clamp(min=1e-20)))
        self.posterior_coef1 = f(betas * ac_prev.sqrt() / (1 - ac))  # multiplies x0
        self.posterior_coef2 = f((1 - ac_prev) * alphas.sqrt() / (1 - ac))  # multiplies x_t

    @staticmethod
    def extract(a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """Gather a[t] and reshape to broadcast against a tensor of `shape`."""
        out = a.to(t.device)[t]
        return out.reshape(-1, *([1] * (len(shape) - 1)))
