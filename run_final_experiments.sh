#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
DATA=/workspace/data/imagenet100; CACHE=/workspace/cache; OUT=/workspace/outputs
mkdir -p "$OUT"
for mode in baseline repa lara_dit ablation; do
  echo "===== START $mode $(date) ====="
  python scripts/05_train.py --mode "$mode" --data-root "$DATA" --cache-dir "$CACHE" --output-dir "$OUT" --steps 200000 --batch-size 128 --workers 8 --ckpt-every 50000 > "$OUT/${mode}.log" 2>&1
  echo "===== END $mode $(date) ====="
done
echo "All four experiments completed."
