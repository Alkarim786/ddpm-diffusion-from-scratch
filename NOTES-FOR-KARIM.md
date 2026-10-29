# Notes for Karim: ddpm-diffusion-from-scratch (Project 22)

## Resume bullets (measured on my Ryzen 5 7520U, CPU only)

- Implemented a **DDPM/DDIM diffusion model from scratch in PyTorch** (cosine/linear schedules, time- and class-conditioned U-Net with attention, EMA, ancestral and DDIM samplers, **classifier-free guidance**); trained a 0.28M-parameter model on MNIST in about an hour on a laptop CPU, loss 0.74 to 0.042.
- Cut sampling cost **21x (252.6 s to 11.8 s for 16 images)** with DDIM-50 vs 1000-step DDPM, and showed guidance lifts label agreement from **80% to 98-100%**, with a measured fidelity/diversity trade-off (FID-proxy 20.5 at w=1 vs 48.6 at w=4).
- Validated the samplers with **oracle-predictor tests** (a perfect noise model must reproduce x0), plus schedule/forward-process statistics, overfit and determinism tests; **24 tests**, CI on GitHub Actions.

## Likely interview questions and pointers

1. **Why predict the noise instead of the image?** Unit-variance target at every t; x0 = (x_t - sqrt(1-abar) eps)/sqrt(abar). `diffusion.py::_x0_from_eps`. Connect to the ELBO: L_simple is a re-weighted variational bound; also equals denoising score matching.
2. **Forward vs reverse process; why the closed form matters.** `q_sample` jumps to any t in one step (sum of Gaussians), so training samples random t cheaply. Reverse step uses the Gaussian posterior q(x_{t-1}|x_t,x0): `schedule.py` posterior coefficients.
3. **DDPM vs DDIM.** DDIM is a non-Markovian sampler with the same training objective; eta=0 gives a deterministic mapping and allows skipping steps (`ddim_sample`). I measured 21x speed-up and no quality loss on this small setup (single seed, so don't over-claim).
4. **How does classifier-free guidance work and what does it trade?** Train with label dropout; sample with eps = (1+w)·cond − w·uncond. Higher w => more class-faithful, less diverse (my sweep: agreement 80%→100%, FID-proxy best at w=1).
5. **Why cosine schedule, why EMA, why GroupNorm?** Cosine: gradual information loss on small images (test). EMA: measured 30.3 vs 38.5 FID-proxy. GroupNorm: batch-size independent, works with small CPU batches.

## Understand this before interviews

- **`schedule.py`**: everything derives from betas: alpha = 1-beta, abar = cumprod(alpha). Posterior variance = beta·(1-abar_{t-1})/(1-abar_t); it is 0 at t=0, so the last step adds no noise.
- **`diffusion.py::ddpm_sample`**: x0 prediction (clipped) → posterior mean = c1·x0 + c2·x_t → add sqrt(posterior_var)·z except at t=0. **`ddim_sample`** re-derives eps from the *clipped* x0 for consistency, then x_prev = sqrt(abar_prev)·x0 + sqrt(1-abar_prev-σ²)·eps (+ σz when eta>0).
- **`unet.py`**: time embedding is sinusoidal → MLP, class embedding has an extra "null" index (num_classes) used for unconditional passes; both are added and injected in every ResBlock. Output conv and each block's last conv are zero-initialised.
- **Oracle test idea** (`tests/test_ddpm.py`): for data = a single point x0 the exact noise predictor is known in closed form; if the sampler reproduces x0 the sampler maths is right, independent of training. Great story for "how do you test generative code?".
- **Metric honesty**: FID-proxy uses my own CNN's features, so it is only meaningful within this repo (floor 17.2 for real-vs-real, ceiling 794 for noise). At n=128 it is biased upward (floor 46.1).
- **Things that went wrong, worth telling**: the laptop slept during training/eval and corrupted wall-clock logs (training meta said 50,358 s, a DDPM timing said 30,732 s), so I computed training time from the loss log excluding the gap and re-measured sampler timing in a clean run; a test I wrote assumed class conditioning worked at init, but zero-initialised blocks make it inert until weights move (fixed the test, kept the design).
- **Limits to state yourself**: tiny model, MNIST only, speckled samples, no NLL, one seed.
