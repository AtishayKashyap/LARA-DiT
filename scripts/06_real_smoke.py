#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data.dataset import build_dataloader
from src.models.dit import build_dit
from src.models.vae_wrapper import VAEWrapper
from src.models.teachers import TierProjectionHeads
from src.training.diffusion import GaussianDiffusion
from src.training.losses import LossComputer


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        default="all",
        choices=["all", "baseline", "repa", "lara_dit", "ablation"],
    )

    ap.add_argument(
        "--data-root",
        default="/workspace/data/imagenet100/raw",
    )

    ap.add_argument(
        "--cache-dir",
        default="/workspace/cache",
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    args = ap.parse_args()

    modes = (
        ["baseline", "repa", "lara_dit", "ablation"]
        if args.mode == "all"
        else [args.mode]
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-data smoke test.")

    device = torch.device("cuda")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    for mode in modes:
        print(f"\n=== {mode.upper()} ===", flush=True)

        cfg = Config(mode=mode)

        cfg.data.data_root = args.data_root
        cfg.data.cache_dir = args.cache_dir
        cfg.train.batch_size = args.batch_size

        loader = build_dataloader(cfg, "train")

        images, labels, targets = next(iter(loader))

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        targets = {
            k: v.to(device, non_blocking=True)
            for k, v in targets.items()
        }

        vae = VAEWrapper(
            cfg.model.vae_model_id,
            device,
            torch.bfloat16,
        )

        model = build_dit(cfg.model).to(device)

        diffusion = GaussianDiffusion()

        loss_computer = LossComputer(
            cfg.tiers,
            cfg.train.total_steps,
        )

        heads = (
            TierProjectionHeads(cfg.tiers).to(device)
            if cfg.uses_alignment
            else None
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            z = vae.encode_from_imagenet(images)

            t = diffusion.sample_timesteps(
                len(z),
                device,
            )

            xt, noise = diffusion.add_noise(z, t)

            if cfg.uses_alignment:
                out, hidden = model(
                    xt,
                    t,
                    labels,
                    return_hidden=True,
                )
            else:
                out = model(
                    xt,
                    t,
                    labels,
                    return_hidden=False,
                )
                hidden = None

            if out.shape != noise.shape:
                raise RuntimeError(
                    "Diffusion prediction shape mismatch: "
                    f"model={tuple(out.shape)}, "
                    f"noise={tuple(noise.shape)}"
                )

            total, logs = loss_computer(
                out,
                noise,
                0,
                hidden,
                heads,
                targets if targets else None,
            )

        if not torch.isfinite(total):
            raise RuntimeError(
                f"{mode}: non-finite loss: {total.item()}"
            )

        total.backward()

        print("input :", tuple(xt.shape), flush=True)
        print("output:", tuple(out.shape), flush=True)
        print("noise :", tuple(noise.shape), flush=True)
        print("loss  :", logs, flush=True)

        del model
        del vae
        del heads
        del loss_computer

        torch.cuda.empty_cache()

        print(f"{mode}: PASS", flush=True)

    print("\nREAL DATA SMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
