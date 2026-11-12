#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
DATA=/workspace/data/imagenet100; CACHE=/workspace/cache; OUT=/workspace/outputs
python scripts/00_check_gpu.py
python scripts/04_preflight.py
python scripts/01_setup_imagenet100.py --root "$DATA"
python scripts/02_verify_dataset.py --root "$DATA"
for t in dinov2 clip mae; do
  python scripts/03_precompute_features.py --root "$DATA" --cache "$CACHE" --teacher "$t" --split train --batch-size 256 --workers 8 --seed 42
done
python scripts/06_real_smoke.py --mode all --data-root "$DATA" --cache-dir "$CACHE" --batch-size 8
echo "READY. Run ./run_final_experiments.sh to launch the four 200K runs sequentially."
