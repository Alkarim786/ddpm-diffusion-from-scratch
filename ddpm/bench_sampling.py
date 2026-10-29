"""Clean sampler timing (no metrics, nothing else running):  python -m ddpm.bench_sampling"""

import argparse
import json
import time

import torch

from .diffusion import Diffusion
from .evaluate import load_model
from .schedule import Schedule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/mnist/ckpt.pt")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="docs/sampling_timing.json")
    args = ap.parse_args()
    model, ck = load_model(args.ckpt, "ema")
    d = Diffusion(model, Schedule(ck["cfg"]["T"], ck["cfg"]["schedule"]))
    y = torch.arange(args.n) % 10
    res = {"n_images": args.n, "guidance": 2.0, "threads": torch.get_num_threads()}
    for name, fn in {
        "ddpm_1000": lambda: d.ddpm_sample((args.n, 1, 28, 28), y, 2.0, torch.Generator().manual_seed(0)),
        "ddim_50": lambda: d.ddim_sample((args.n, 1, 28, 28), 50, 0.0, y, 2.0, torch.Generator().manual_seed(0)),
        "ddim_20": lambda: d.ddim_sample((args.n, 1, 28, 28), 20, 0.0, y, 2.0, torch.Generator().manual_seed(0)),
        "ddim_10": lambda: d.ddim_sample((args.n, 1, 28, 28), 10, 0.0, y, 2.0, torch.Generator().manual_seed(0)),
    }.items():
        fn()  # warm-up
        t0 = time.perf_counter()
        fn()
        secs = time.perf_counter() - t0
        res[name] = {"seconds": round(secs, 2), "ms_per_image": round(1000 * secs / args.n, 1)}
        print(name, res[name], flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
