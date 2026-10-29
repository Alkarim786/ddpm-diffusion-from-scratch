"""Train a class-conditional DDPM on MNIST.

    python -m ddpm.train --steps 5000 --out runs/mnist

CPU defaults are deliberately tiny. On a GPU, raise --base (64-128), --steps (50k+) and
--batch (128-256); see the README for the scale-up recipe.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from .data import load, loader
from .diffusion import Diffusion
from .ema import EMA
from .schedule import Schedule
from .unet import UNet


def build_model(cfg: dict) -> UNet:
    return UNet(
        in_ch=1, base=cfg["base"], mults=tuple(cfg["mults"]), num_res=cfg["num_res"],
        image_size=28, num_classes=10,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--mults", type=int, nargs="+", default=[1, 2, 2])
    ap.add_argument("--num-res", type=int, default=1)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/mnist")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {k: getattr(args, k.replace("-", "_")) for k in ("dataset", "base", "mults", "T", "schedule")}
    cfg["num_res"] = args.num_res
    model = build_model(cfg)
    diffusion = Diffusion(model, Schedule(args.T, args.schedule))
    ema = EMA(model, args.ema)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.3f}M  threads: {torch.get_num_threads()}", flush=True)

    ds = load(args.dataset, args.data_root)
    step, t0 = 0, time.perf_counter()
    csv_f = open(out / "loss.csv", "w", newline="")  # noqa: SIM115 (closed at process exit)
    writer = csv.writer(csv_f)
    writer.writerow(["step", "loss", "elapsed_s"])
    running = []
    epoch = 0
    while step < args.steps:
        epoch += 1
        for x, y in loader(ds, args.batch, seed=args.seed + epoch):
            opt.zero_grad()
            loss = diffusion.loss(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            step += 1
            running.append(loss.item())
            if step % 50 == 0:
                avg = sum(running) / len(running)
                running.clear()
                el = time.perf_counter() - t0
                writer.writerow([step, f"{avg:.5f}", f"{el:.1f}"])
                csv_f.flush()
                print(f"step {step:5d}  epoch {epoch}  loss {avg:.4f}  {el:7.1f}s", flush=True)
            if step % 500 == 0 or step >= args.steps:
                torch.save({"cfg": cfg, "model": model.state_dict(), "ema": ema.model.state_dict(), "step": step},
                           out / "ckpt.pt")
            if step >= args.steps:
                break
    total = time.perf_counter() - t0
    meta = {"params": n_params, "steps": step, "batch": args.batch, "epochs_seen": step * args.batch / len(ds),
            "train_seconds": round(total, 1), "sec_per_step": round(total / step, 3), "cfg": cfg, "lr": args.lr,
            "ema_decay": args.ema, "seed": args.seed, "threads": torch.get_num_threads()}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print("done", meta, flush=True)


if __name__ == "__main__":
    main()
