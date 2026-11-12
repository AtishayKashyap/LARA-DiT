#!/usr/bin/env bash
# Revamp experiments, in priority order. Run one phase at a time:  bash run_revamp.sh <phase>
# Phases: setup | bench | reeval | train <run> | evalckpt <run> | final | collect      (see RUNBOOK.md)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
DATA=${DATA:-/workspace/data/imagenet100}; CACHE=${CACHE:-/workspace/cache}; OUT=${OUT:-/workspace/outputs_v2}
EVAL=${EVAL:-/workspace/evaluation}; REF=${REF:-/workspace/fid_ref}; OLD=${OLD:-/workspace/old_checkpoints}   # holds {baseline,repa,lara_dit,ablation}_final.pt
STEPS=${STEPS:-100000}          # set from benchmark.json before training (RUNBOOK step 2)
CKPT_EVERY=$((STEPS/4))
COMMON=(--data-root "$DATA" --cache-dir "$CACHE" --output-dir "$OUT" --steps "$STEPS" --ckpt-every "$CKPT_EVERY" --use-latents --diag-every 1000 --seed 42)
declare -A RUNS=(
  [r1_baseline]="--mode baseline"
  [r2_repa_patch]="--mode custom --align patch --teachers dinov2,-,- --gammas 0,0,0"
  [r3_lara_patch]="--mode custom --align patch --teachers dinov2,clip,mae --gammas 2,1.5,1"
  [r4_lara_reversed]="--mode custom --align patch --teachers mae,clip,dinov2 --gammas 2,1.5,1"
)
case "${1:-}" in
  setup)
    python -m pytest tests -q
    python scripts/03_precompute_features.py --teacher vae --root "$DATA" --cache "$CACHE"
    for t in dinov2 clip mae; do python scripts/03_precompute_features.py --teacher "$t" --tokens --root "$DATA" --cache "$CACHE"; done
    python scripts/09_make_fid_reference.py --root "$DATA" --out "$REF"
    python scripts/11_teacher_diagnostics.py --cache-dir "$CACHE" --out "$EVAL/teacher_diagnostics.json" ;;
  bench)
    python scripts/10_benchmark.py --data-root "$DATA" --cache-dir "$CACHE" --out "$EVAL/benchmark.json" ;;
  smoke-eval)   # 200 samples, to check the evaluator end to end before spending hours
    python scripts/07_evaluate.py --checkpoint "$OLD/baseline_final.pt" --ref-dir "$REF" --out-root "$EVAL/smoke" --samples 200 --batch-size 100 ;;
  reeval)       # corrected evaluation of the four original checkpoints; seed 0 for all, seed 1 for baseline + LARA
    python scripts/07_evaluate.py --checkpoint "$OLD"/{baseline,repa,lara_dit,ablation}_final.pt --ref-dir "$REF" --out-root "$EVAL" --seed 0
    python scripts/07_evaluate.py --checkpoint "$OLD"/{baseline,lara_dit}_final.pt --ref-dir "$REF" --out-root "$EVAL" --seed 1 ;;
  train)
    run="${2:?run name: ${!RUNS[*]}}"; python scripts/05_train.py ${RUNS[$run]} --run-name "$run" "${COMMON[@]}" 2>&1 | tee -a "$OUT/$run.log" ;;
  evalckpt)     # convergence curve: 5k samples at every saved checkpoint of one run
    run="${2:?run name}"; python scripts/07_evaluate.py --checkpoint "$OUT/$run"/checkpoints/step*.pt --ref-dir "$REF" --out-root "$EVAL" --samples 5000 --seed 0 ;;
  final)        # 10k samples, two sampling seeds, at the last checkpoint of every new run
    for s in 0 1; do python scripts/07_evaluate.py --checkpoint "$OUT"/r*/checkpoints/final.pt --ref-dir "$REF" --out-root "$EVAL" --seed "$s"; done ;;
  collect)
    python scripts/12_collect_results.py --eval-root "$EVAL" --out "$EVAL/results" ;;
  *) sed -n '2,3p' "$0"; exit 1 ;;
esac
