"""Dump every TensorBoard scalar from the backup event files to one long CSV (run, tag, step, value)."""
import sys
from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results" / "original_runs" / "logs"
out = Path(__file__).resolve().parent.parent / "results" / "original_runs" / "scalars.csv"
rows = []
for run_dir in sorted(p for p in logs.iterdir() if p.is_dir()):
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0}).Reload()  # 0 = keep all points
    for tag in ea.Tags()["scalars"]:
        rows += [(run_dir.name, tag, e.step, e.value) for e in ea.Scalars(tag)]
df = pd.DataFrame(rows, columns=["run", "tag", "step", "value"])
df.to_csv(out, index=False)
print(df.groupby(["run", "tag"]).step.agg(["count", "min", "max"]).to_string())
print(f"wrote {out} ({len(df)} rows)")
