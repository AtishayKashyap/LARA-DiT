#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.datasets import ImageFolder

from cleanfid import fid as cleanfid_fid


def inception_score(images, device, splits=10, batch=32):
    model = inception_v3(
        weights=Inception_V3_Weights.DEFAULT,
        aux_logits=True,
    ).to(device).eval()

    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )

    probs = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(images), batch),
            desc="Inception Score",
        ):
            x = images[i:i + batch].to(device)
            x = F.interpolate(
                x,
                size=(299, 299),
                mode="bilinear",
                align_corners=False,
            )
            x = normalize(x)
            logits = model(x)
            probs.append(F.softmax(logits, dim=1).cpu())

    p = torch.cat(probs, dim=0)

    assert len(p) % splits == 0, (
        f"{len(p)} samples cannot be evenly divided into {splits} splits"
    )

    n = len(p) // splits
    scores = []

    for i in range(splits):
        q = p[i * n:(i + 1) * n]
        py = q.mean(0, keepdim=True)
        kl = (
            q
            * (
                q.add(1e-8).log()
                - py.add(1e-8).log()
            )
        ).sum(1).mean()

        scores.append(torch.exp(kl).item())

    scores = torch.tensor(scores)

    return (
        float(scores.mean()),
        float(scores.std(unbiased=False)),
    )


def hf_mask(x, percentile=75):
    gray = (
        0.299 * x[:, 0]
        + 0.587 * x[:, 1]
        + 0.114 * x[:, 2]
    ).unsqueeze(1)

    kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)

    lap = F.conv2d(
        gray,
        kernel,
        padding=1,
    ).abs()

    threshold = torch.quantile(
        lap.flatten(1),
        percentile / 100,
        dim=1,
    )

    return lap >= threshold[:, None, None, None]


def load_generated_images(generated_dir):
    paths = sorted(generated_dir.glob("*.png"))

    print(f"Found {len(paths)} generated images")

    images = []

    for p in tqdm(paths, desc="Loading generated images"):
        images.append(
            transforms.ToTensor()(
                Image.open(p).convert("RGB")
            )
        )

    return torch.stack(images)


def compute_lpips(
    tensors,
    data_root,
    device,
    batch_size=32,
):
    import lpips

    net = lpips.LPIPS(net="alex").to(device).eval()

    ds = ImageFolder(Path(data_root) / "val")

    by = {i: [] for i in range(100)}

    for p, y in ds.samples:
        by[y].append(p)

    vals = []

    with torch.no_grad():
        for s in tqdm(
            range(0, len(tensors), batch_size),
            desc="LPIPS",
        ):
            ids = list(
                range(
                    s,
                    min(s + batch_size, len(tensors)),
                )
            )

            real = []
            fake = []

            for i in ids:
                p = by[i % 100][
                    (i // 100) % len(by[i % 100])
                ]

                real.append(
                    transforms.ToTensor()(
                        Image.open(p)
                        .convert("RGB")
                        .resize((256, 256))
                    )
                )

                fake.append(tensors[i])

            r = torch.stack(real).to(device) * 2 - 1
            f = torch.stack(fake).to(device) * 2 - 1

            mask = hf_mask(
                (r + 1) / 2,
                75,
            ).float()

            vals.append(
                net(
                    f * mask,
                    r * mask,
                ).mean().item()
            )

    return float(sum(vals) / len(vals))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--generated-dir", required=True)
    ap.add_argument(
        "--data-root",
        default="/workspace/data/imagenet100",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)

    args = ap.parse_args()

    device = torch.device("cuda")

    generated_dir = Path(args.generated_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Verify generated samples
    # ------------------------------------------------------------

    paths = sorted(generated_dir.glob("*.png"))

    if len(paths) != 10000:
        raise RuntimeError(
            f"Expected exactly 10000 generated images, "
            f"found {len(paths)}"
        )

    print("Verified 10,000 generated images.")

    # ------------------------------------------------------------
    # FID
    # ------------------------------------------------------------

    print("\n===== FID =====")

    t0 = time.time()

    fid_value = cleanfid_fid.compute_fid(
        str(generated_dir),
        str(Path(args.data_root) / "val"),
        device=device,
        verbose=True,
    )

    print(f"FID: {fid_value:.6f}")
    print(f"FID seconds: {time.time() - t0:.2f}")

    torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Load generated images for IS + LPIPS
    # ------------------------------------------------------------

    tensors = load_generated_images(generated_dir)

    # ------------------------------------------------------------
    # Inception Score
    # ------------------------------------------------------------

    print("\n===== INCEPTION SCORE =====")

    is_mean, is_std = inception_score(
        tensors,
        device,
        splits=10,
        batch=args.batch_size,
    )

    print(f"IS mean: {is_mean:.6f}")
    print(f"IS std : {is_std:.6f}")

    del tensors
    torch.cuda.empty_cache()

    # Reload for LPIPS to keep GPU memory clean.
    tensors = load_generated_images(generated_dir)

    # ------------------------------------------------------------
    # LPIPS
    # ------------------------------------------------------------

    print("\n===== LPIPS =====")

    lpips_value = compute_lpips(
        tensors,
        args.data_root,
        device,
        batch_size=args.batch_size,
    )

    print(f"HF LPIPS: {lpips_value:.6f}")

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    result = {
        "samples": 10000,
        "fid": float(fid_value),
        "is_mean": float(is_mean),
        "is_std": float(is_std),
        "hf_lpips_label_matched": float(lpips_value),
        "hf_lpips_definition": (
            "AlexNet LPIPS on top-25% Laplacian pixels, "
            "label-matched validation references"
        ),
        "inception_batch_size": args.batch_size,
        "lpips_batch_size": args.batch_size,
    }

    # Carry sampling provenance (seed, checkpoint hash, CFG, steps...) written by 07_evaluate.py.
    manifest_path = generated_dir.parent / "manifest.json"
    if manifest_path.exists():
        result = {**json.loads(manifest_path.read_text()), **result}
    else:
        print(f"WARNING: no {manifest_path}; sampling seed/settings unrecorded.")

    result_path = out / "metrics.json"
    result_path.write_text(
        json.dumps(result, indent=2)
    )

    print("\n===== FINAL METRICS =====")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
