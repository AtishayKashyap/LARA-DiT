# LARA-DiT revamp — GPU runbook (budget: under 60 GPU-hours)

Run phases in order. Each phase has a **gate**: check it before spending more time.
All commands run from this folder: `bash run_revamp.sh <phase>`.

## What this fixes

| Weakness in the current paper | Fix in this run | Phase |
|---|---|---|
| Sampling noise unseeded and unmatched; settings not recorded | Seeded generator for all sampler randomness, identical noise across checkpoints, full manifest | reeval, final |
| FID: 5k references, raw val folder, 1,000-step sampler | Adds a 50k-image train reference (ADM-style crop); 250 respaced steps (the DiT/ADM convention); keeps `val5k_raw` for continuity with the old numbers | setup, reeval |
| IS only; can't tell fidelity from diversity | Adds precision/recall (Kynkäänniemi et al. 2019) | all evals |
| HF-LPIPS used unpaired references | Dropped from the new evaluator | — |
| "REPA-style" was a global, linear-head variant that underperformed | Faithful REPA: patch tokens, 3-layer MLP projector, block 3 (REPA's layer 4 for B-size models), λ = 0.5 | r2 |
| LARA-DiT also global/linear | Patch-level LARA with REPA projectors | r3 |
| Teacher placement untested (Park et al. predicts the opposite order) | Reversed assignment MAE / CLIP / DINOv2 | r4 |
| "Constraint strength" hypothesis untested | Per-tier gradient norm and cosine with the denoising gradient, logged every 1,000 steps (`grad/*`) | all training |
| Are pooled CLIP/MAE targets uninformative? | Pairwise cosine, effective rank, centroid accuracy of every cache | setup |
| No convergence evidence | FID at 4 checkpoints per new run | evalckpt |
| Training was probably VAE-bound (1.76 it/s) | Cached VAE posteriors: no encoder or JPEG decode per step | setup, bench |
| Class list unknown | `fid_ref/reference.json` records the class folders | setup |

**Not fixable at this budget:** multiple *training* seeds. Report sampling-seed variation honestly and keep the single-training-run caveat.

## 0. Before the GPU window (do now)

1. Copy to the GPU machine:
   - this `Lara_Dit/` folder
   - the four original checkpoints, as `/workspace/old_checkpoints/{baseline,repa,lara_dit,ablation}_final.pt`
   - ImageNet-100 at `/workspace/data/imagenet100/{train,val}/<class>/...`
2. Check that `sha256sum` of the checkpoints matches `LARA_DiT_BACKUP/SHA256SUMS.txt`.
3. Check free disk space. You need about **85 GB**:
   - token caches: 3 × 12.8 GB
   - latents: 2.1 GB
   - FID-reference scratch: ~6 GB
   - new checkpoints: ~2 GB × 5 per run × 4 runs. To save space, delete intermediate checkpoints after `evalckpt`.
4. Check that the packages are installed: `python -c "import torch, diffusers, timm, open_clip, cleanfid, pandas"`.

## 1. Setup and smoke tests (~2–3 h, estimate)

```bash
bash run_revamp.sh setup        # tests, VAE latents, 3 token caches, FID references, teacher diagnostics
bash run_revamp.sh smoke-eval   # 200 samples from the old baseline: checks the whole evaluator
```

**Gate 1:**
- `pytest` passes.
- The smoke-eval `metrics.json` has `fid_train50k`, `fid_val5k`, `fid_val5k_raw`, `precision_train`, `recall_train` and `is_mean`. FID from 200 samples will be very high; that's expected.
- `reference.json` lists 100 classes.
- The CLIP token cache is 64×768. If open_clip refuses `output_tokens`, stop and fix `CLIPTeacher._forward_tokens` (the only version-sensitive line).

## 2. Benchmark (~0.5 h)

```bash
bash run_revamp.sh bench        # writes evaluation/benchmark.json with projected hours
```

**Gate 2: choose `STEPS` from the measurements.** Let

- `B_left` = GPU-hours remaining
- `E` = `projected_hours.eval_10k_samples`
- `T100` = `projected_hours.train_100k_steps`

Reserve:
- `6E` for reeval
- `8E` for final evaluation (4 runs × 2 seeds)
- `4 × 4 × E/2` for convergence evaluation (4 runs × 4 checkpoints × 5k samples)
- 15% for failures and restarts

Then `STEPS = 100000 × (B_left_after_reserves / 4) / T100`. Round down to a multiple of 20,000; the minimum useful value is 40,000.

Gate 2 also confirms that `baseline_latents` is faster than `baseline_images`. If it isn't, the loader isn't the bottleneck; use the `baseline_images` speed for planning.

> If the budget only covers 3 runs at ≥40K steps, drop **r4**. If it only covers 2, run **r1 + r2** (baseline vs faithful REPA) first: that answers the most important reviewer question.

## 3. Re-evaluate the original checkpoints (6 × E)

```bash
bash run_revamp.sh reeval
```

This yields seed-controlled, recorded numbers for all four original models, plus a second sampling seed for baseline and LARA-DiT. These replace Tables 6–7 in the paper.

## 4. New training runs (priority order, one at a time)

```bash
export STEPS=<from gate 2>
bash run_revamp.sh train r1_baseline
bash run_revamp.sh train r2_repa_patch
bash run_revamp.sh train r3_lara_patch
bash run_revamp.sh train r4_lara_reversed      # only if budget allows
```

**Gate 3:** after the first 2,000 steps of each run, check its log:
- `perf/it_per_s` is close to the benchmark
- losses are finite
- for alignment runs, `grad/cos_*` is being logged

If a run dies, resume it with `--resume <last checkpoint>` (append to the same command).

Notes:
- The decay schedule is relative to `STEPS`, so shorter runs still end at λ = 0.068 / 0.112 / 0.184.
- All new runs share seed 42, cached latents and identical loaders. They are therefore matched with each other, but **not** with the original 200K-step runs. Never mix the two in one table.

## 5. Convergence and final evaluation

```bash
for r in r1_baseline r2_repa_patch r3_lara_patch r4_lara_reversed; do bash run_revamp.sh evalckpt $r; done
bash run_revamp.sh final
bash run_revamp.sh collect      # evaluation/results_all.csv + results_summary.csv
```

## 6. Bring back for the paper

Copy back these files:
- `evaluation/results_*.csv`
- every `metrics.json` and `manifest.json`
- `benchmark.json` and `teacher_diagnostics.json`
- `outputs_v2/*/logs/` (TensorBoard)
- the first 100 PNGs of each final evaluation (for qualitative grids)

Checkpoints are optional, but keep their SHA-256 values.

## What results would mean

| Outcome | What the paper can then say |
|---|---|
| r2 < r1 in FID | REPA reproduces in this setting; the old "REPA-style hurt" result was due to global alignment. |
| r3 < r2 | Depth-stratified multi-teacher alignment beats single-teacher REPA under matched conditions: the main claim, as a single-seed result. |
| r3 ≈ r2 or r3 > r2 | An honest negative or neutral result. It's still publishable as an analysis paper, especially with the gradient and teacher diagnostics. |
| r4 < r3 | Teacher *placement* matters and Park et al.'s ordering is better. That's a finding either way. |
| `grad/cos_deep` more negative for DINOv2 at depth than for MAE | Supports the constraint-strength explanation of the original ablation. |
