"""Generate figures and measure sample quality / sampling speed from a trained checkpoint.

    python -m ddpm.evaluate --ckpt runs/mnist/ckpt.pt --out docs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from .data import load
from .diffusion import Diffusion
from .metrics import fid_proxy, sample_report, train_classifier
from .schedule import Schedule
from .train import build_model


def to01(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) / 2)


def load_model(ckpt_path: str, which: str = "ema"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"])
    model.load_state_dict(ck[which])
    return model.eval(), ck


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/mnist/ckpt.pt")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--n-fid", type=int, default=1000)
    ap.add_argument("--n-ddpm", type=int, default=128, help="samples for the (slow) 1000-step DDPM comparison")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    model, ck = load_model(args.ckpt, "ema")
    raw, _ = load_model(args.ckpt, "model")
    cfg = ck["cfg"]
    sch = Schedule(cfg["T"], cfg["schedule"])
    diff, diff_raw = Diffusion(model, sch), Diffusion(raw, sch)
    train_ds, test_ds = load(cfg["dataset"], args.data_root, True), load(cfg["dataset"], args.data_root, False)
    res: dict = {"checkpoint_step": ck["step"]}

    # ---- classifier used by the metrics ------------------------------------------
    t0 = time.perf_counter()
    clf, clf_acc = train_classifier(train_ds, test_ds, epochs=2)
    res["classifier"] = {"test_accuracy": round(clf_acc, 4), "train_seconds": round(time.perf_counter() - t0, 1)}
    print("classifier", res["classifier"], flush=True)

    real_test = torch.stack([test_ds[i][0] for i in range(args.n_fid)])
    real_train = torch.stack([train_ds[i][0] for i in range(args.n_fid)])
    res["fid_proxy_reference"] = {
        "real_train_vs_real_test (floor)": round(fid_proxy(clf, real_train, real_test), 3),
        "gaussian_noise_vs_real_test (ceiling)": round(fid_proxy(clf, torch.randn_like(real_test).clamp(-1, 1), real_test), 3),
    }

    # ---- figures ---------------------------------------------------------------------
    x0 = test_ds[0][0][None]
    ts = [0, 50, 100, 200, 350, 500, 700, 850, 999]
    strip = torch.cat([diff.q_sample(x0, torch.tensor([t]), torch.randn_like(x0)) for t in ts])
    save_image(to01(strip), out / "forward_noising.png", nrow=len(ts), padding=2, pad_value=1.0)

    labels = torch.arange(10).repeat_interleave(10)
    g = torch.Generator().manual_seed(1)
    grid, ddim_grid_s = timed(lambda: diff.ddim_sample((100, 1, 28, 28), 50, 0.0, labels, 2.0, g))
    save_image(to01(grid), out / "samples_ddim50_cfg2.png", nrow=10, padding=2, pad_value=1.0)
    res["grid_100_samples_ddim50_seconds"] = round(ddim_grid_s, 1)

    g = torch.Generator().manual_seed(2)
    traj_labels = torch.arange(8)
    (_, traj), _ = timed(lambda: diff.ddpm_sample((8, 1, 28, 28), traj_labels, 2.0, g, return_trajectory_every=125))
    # one row per sample, columns = noisy -> clean (every 125 steps, ending at t=0)
    rows = torch.stack(traj, dim=1).flatten(0, 1)
    save_image(to01(rows), out / "reverse_denoising_ddpm.png", nrow=len(traj), padding=2, pad_value=1.0)

    # ---- guidance sweep (DDIM 50, conditional) -------------------------------------------
    res["guidance_sweep_ddim50"] = {}
    lab = torch.arange(10).repeat(args.n_fid // 10)
    for w in (0.0, 1.0, 2.0, 4.0):
        g = torch.Generator().manual_seed(3)
        s = torch.cat([diff.ddim_sample((250, 1, 28, 28), 50, 0.0, lab[i:i + 250], w, g)
                       for i in range(0, len(lab), 250)])
        rep = sample_report(clf, s, lab)
        rep["fid_proxy"] = round(fid_proxy(clf, s, real_test), 3)
        res["guidance_sweep_ddim50"][str(w)] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()}
        print("guidance", w, res["guidance_sweep_ddim50"][str(w)], flush=True)

    # ---- EMA vs raw weights --------------------------------------------------------------
    g = torch.Generator().manual_seed(3)
    s_raw = torch.cat([diff_raw.ddim_sample((250, 1, 28, 28), 50, 0.0, lab[i:i + 250], 2.0, g)
                       for i in range(0, len(lab), 250)])
    res["ema_vs_raw_fid_proxy_ddim50_w2"] = {
        "ema": res["guidance_sweep_ddim50"]["2.0"]["fid_proxy"], "raw": round(fid_proxy(clf, s_raw, real_test), 3)}

    # ---- sampler comparison: DDPM-1000 vs DDIM-k (same classifier, same labels) ------------
    n = args.n_ddpm
    lab_n, real_n = lab[:n], real_test[:n]
    cmp = {}
    g = torch.Generator().manual_seed(4)
    s, secs = timed(lambda: diff.ddpm_sample((n, 1, 28, 28), lab_n, 2.0, g))
    cmp["ddpm_1000"] = {"seconds": round(secs, 1), "ms_per_image": round(1000 * secs / n, 1),
                        "fid_proxy": round(fid_proxy(clf, s, real_n), 3), **{k: v for k, v in sample_report(clf, s, lab_n).items() if k != "class_histogram"}}
    for steps in (50, 20, 10):
        g = torch.Generator().manual_seed(4)
        s, secs = timed(lambda st=steps, gen=g: diff.ddim_sample((n, 1, 28, 28), st, 0.0, lab_n, 2.0, gen))
        cmp[f"ddim_{steps}"] = {"seconds": round(secs, 1), "ms_per_image": round(1000 * secs / n, 1),
                                "fid_proxy": round(fid_proxy(clf, s, real_n), 3), **{k: v for k, v in sample_report(clf, s, lab_n).items() if k != "class_histogram"}}
    res["sampler_comparison_w2"] = {"n_samples": n, "note": "fid_proxy on only n samples is biased upward vs the n=1000 numbers", **cmp}
    res["fid_proxy_floor_at_same_n"] = round(fid_proxy(clf, real_train[:n], real_n), 3)

    # ---- determinism of DDIM eta=0 -----------------------------------------------------------
    a = diff.ddim_sample((4, 1, 28, 28), 20, 0.0, labels[:4], 2.0, torch.Generator().manual_seed(9))
    b = diff.ddim_sample((4, 1, 28, 28), 20, 0.0, labels[:4], 2.0, torch.Generator().manual_seed(9))
    res["ddim_eta0_deterministic"] = bool(torch.equal(a, b))

    (out / "eval.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
