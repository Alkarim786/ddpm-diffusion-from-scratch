import numpy as np
import pytest
import torch

from ddpm.diffusion import Diffusion
from ddpm.ema import EMA
from ddpm.metrics import frechet_distance
from ddpm.schedule import Schedule, cosine_betas, linear_betas
from ddpm.unet import UNet, timestep_embedding

torch.set_num_threads(2)


# ---- schedules --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_schedule_properties(kind):
    s = Schedule(1000, kind)
    ac = s.alphas_cumprod
    assert (ac > 0).all() and (ac < 1).all()
    assert (ac[1:] < ac[:-1]).all()  # strictly decreasing: information is only ever destroyed
    assert ac[-1] < 1e-3  # essentially pure noise at t = T
    assert (s.betas > 0).all() and (s.betas < 1).all()
    assert s.posterior_variance[0] == 0 and (s.posterior_variance[1:] > 0).all()
    assert (s.posterior_variance <= s.betas + 1e-9).all()  # posterior variance <= beta (tilde beta <= beta)


def test_cosine_schedule_is_gentler_early_than_linear():
    lin, cos = Schedule(1000, "linear"), Schedule(1000, "cosine")
    # at 20% of the way through, cosine has destroyed less of the signal
    assert cos.alphas_cumprod[200] > lin.alphas_cumprod[200]


def test_linear_betas_rescale_with_T():
    assert torch.isclose(linear_betas(500).sum(), linear_betas(1000).sum(), rtol=1e-3)
    assert cosine_betas(10).max() <= 0.999


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError):
        Schedule(10, "nope")


# ---- forward process -----------------------------------------------------------------------
def test_q_sample_statistics_match_closed_form():
    s = Schedule(1000, "cosine")
    d = Diffusion(torch.nn.Identity(), s)
    x0 = torch.full((20000, 1, 1, 1), 0.7)
    for t_val in (10, 500, 999):
        t = torch.full((20000,), t_val)
        xt = d.q_sample(x0, t)
        ac = s.alphas_cumprod[t_val]
        assert abs(xt.mean().item() - ac.sqrt().item() * 0.7) < 0.03
        assert abs(xt.var().item() - (1 - ac).item()) < 0.03


def test_q_sample_at_last_step_is_close_to_standard_normal():
    s = Schedule(1000, "cosine")
    d = Diffusion(torch.nn.Identity(), s)
    xt = d.q_sample(torch.ones(50000, 1, 1, 1), torch.full((50000,), 999))
    assert abs(xt.mean()) < 0.03 and abs(xt.std() - 1) < 0.03


# ---- network ------------------------------------------------------------------------------------
@pytest.mark.parametrize("size", [16, 28, 32])
@pytest.mark.parametrize("classes", [None, 10])
def test_unet_preserves_shape(size, classes):
    m = UNet(base=8, image_size=size, num_classes=classes)
    x = torch.randn(3, 1, size, size)
    y = torch.randint(0, 10, (3,)) if classes else None
    assert m(x, torch.randint(0, 1000, (3,)), y).shape == x.shape


def test_unet_starts_as_zero_output_and_class_conditioning_changes_output_after_perturbation():
    m = UNet(base=8, num_classes=10)
    x, t = torch.randn(2, 1, 28, 28), torch.tensor([5, 50])
    assert m(x, t, torch.tensor([1, 2])).abs().max() == 0  # zero-init output conv
    for p in m.parameters():  # zero-init residual convs also block conditioning at init: randomise everything
        torch.nn.init.normal_(p, std=0.05)
    a, b = m(x, t, torch.tensor([1, 1])), m(x, t, torch.tensor([7, 7]))
    assert not torch.allclose(a, b)  # labels actually reach the network


def test_timestep_embedding_shape_and_uniqueness():
    e = timestep_embedding(torch.arange(100), 32)
    assert e.shape == (100, 32)
    assert torch.cdist(e, e).fill_diagonal_(1).min() > 1e-3  # distinct timesteps -> distinct embeddings
    assert timestep_embedding(torch.arange(4), 33).shape == (4, 33)


# ---- training signal ---------------------------------------------------------------------------------
def test_model_can_overfit_a_single_image():
    torch.manual_seed(0)
    m = UNet(base=16, num_res=1, image_size=16)
    d = Diffusion(m, Schedule(100, "cosine"))
    x0 = torch.randn(1, 1, 16, 16).clamp(-1, 1).repeat(32, 1, 1, 1)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    first = np.mean([d.loss(x0).item() for _ in range(5)])
    for _ in range(150):
        opt.zero_grad()
        d.loss(x0).backward()
        opt.step()
    last = np.mean([d.loss(x0).item() for _ in range(5)])
    assert last < 0.6 * first, (first, last)


def test_cfg_label_dropout_uses_null_class():
    seen = []

    class Spy(torch.nn.Module):
        num_classes = 10

        def forward(self, x, t, y=None):
            seen.append(y.clone())
            return torch.zeros_like(x)

    d = Diffusion(Spy(), Schedule(10, "linear"))
    d.loss(torch.zeros(4000, 1, 2, 2), torch.zeros(4000, dtype=torch.long), p_uncond=0.25)
    frac_null = (seen[0] == 10).float().mean().item()
    assert 0.2 < frac_null < 0.3


# ---- samplers: correctness against an oracle noise predictor ---------------------------------------------
class Oracle(torch.nn.Module):
    """A perfect eps-predictor for a data distribution that is the single point x0."""

    num_classes = None

    def __init__(self, x0, schedule):
        super().__init__()
        self.x0, self.s = x0, schedule

    def forward(self, x, t, y=None):
        ac = self.s.alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return (x - ac.sqrt() * self.x0) / (1 - ac).sqrt()


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_ddpm_sampler_recovers_x0_with_perfect_model(kind):
    s = Schedule(200, kind)
    x0 = torch.linspace(-0.8, 0.8, 16).reshape(1, 1, 4, 4)
    d = Diffusion(Oracle(x0, s), s)
    out = d.ddpm_sample((2, 1, 4, 4), generator=torch.Generator().manual_seed(0))
    assert (out - x0).abs().max() < 0.05


def test_ddim_sampler_recovers_x0_with_perfect_model_even_with_few_steps():
    s = Schedule(1000, "cosine")
    x0 = torch.linspace(-0.8, 0.8, 16).reshape(1, 1, 4, 4)
    d = Diffusion(Oracle(x0, s), s)
    for steps in (5, 50):
        out = d.ddim_sample((2, 1, 4, 4), steps, 0.0, generator=torch.Generator().manual_seed(0))
        assert (out - x0).abs().max() < 0.05, steps


def test_sampling_is_reproducible_with_a_seed_and_ddim_eta0_is_deterministic():
    m = UNet(base=8, num_classes=10)
    torch.nn.init.normal_(m.conv_out.weight, std=0.05)
    d = Diffusion(m.eval(), Schedule(50, "cosine"))
    y = torch.tensor([1, 2])
    g = lambda: torch.Generator().manual_seed(123)
    assert torch.equal(d.ddpm_sample((2, 1, 28, 28), y, 1.0, g()), d.ddpm_sample((2, 1, 28, 28), y, 1.0, g()))
    a = d.ddim_sample((2, 1, 28, 28), 10, 0.0, y, 1.0, g())
    assert torch.equal(a, d.ddim_sample((2, 1, 28, 28), 10, 0.0, y, 1.0, g()))
    assert not torch.equal(a, d.ddim_sample((2, 1, 28, 28), 10, 0.0, y, 1.0, torch.Generator().manual_seed(124)))
    # eta > 0 injects fresh noise each step: same initial noise, different generator state -> differs from eta = 0
    assert not torch.equal(a, d.ddim_sample((2, 1, 28, 28), 10, 1.0, y, 1.0, g()))


def test_classifier_free_guidance_formula():
    class Fake(torch.nn.Module):
        num_classes = 10

        def forward(self, x, t, y=None):
            # conditional branch returns 3, unconditional (null class) returns 1
            return torch.where((y == 10)[:, None, None, None], torch.ones_like(x), 3 * torch.ones_like(x))

    d = Diffusion(Fake(), Schedule(10, "linear"))
    x, t, y = torch.zeros(2, 1, 2, 2), torch.zeros(2, dtype=torch.long), torch.tensor([4, 5])
    assert torch.allclose(d._eps(x, t, y, 0.0), torch.full_like(x, 3.0))  # w = 0: plain conditional
    assert torch.allclose(d._eps(x, t, y, 2.0), torch.full_like(x, 3.0 + 2.0 * (3.0 - 1.0)))  # (1+w)c - w u


# ---- EMA / metrics ---------------------------------------------------------------------------------------------
def test_ema_tracks_weights():
    m = torch.nn.Linear(2, 2)
    ema = EMA(m, 0.9)
    with torch.no_grad():
        m.weight.fill_(1.0)
    for _ in range(300):
        ema.update(m)
    assert torch.allclose(ema.model.weight, torch.ones(2, 2), atol=1e-3)
    assert not any(p.requires_grad for p in ema.model.parameters())


def test_frechet_distance_properties():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2000, 5))
    b = rng.normal(loc=1.0, size=(2000, 5))
    mu, cov = a.mean(0), np.cov(a, rowvar=False)
    assert frechet_distance(mu, cov, mu, cov) < 1e-6
    d = frechet_distance(mu, cov, b.mean(0), np.cov(b, rowvar=False))
    assert 4.0 < d < 6.5  # shifting all 5 dims by 1 gives ~5
