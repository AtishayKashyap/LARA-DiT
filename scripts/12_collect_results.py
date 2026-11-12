#!/usr/bin/env python3
"""Collect every metrics.json under --eval-root into one CSV (one row per run/checkpoint/seed), plus a summary with
mean ± std over sampling seeds for each run/checkpoint. These two files are the source of truth for the paper's tables."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--eval-root',default='/workspace/evaluation'); ap.add_argument('--out',default='results'); a=ap.parse_args()
    rows=[json.loads(p.read_text()) for p in sorted(Path(a.eval_root).glob('*/metrics.json'))]
    if not rows: raise SystemExit('no metrics.json found')
    df=pd.DataFrame(rows); df['ckpt']=df['checkpoint'].map(lambda s: Path(s).stem)
    keep=['run','ckpt','step','seed','samples','sampling_steps','cfg_scale']+[c for c in df.columns if c.startswith(('fid_','precision','recall','is_mean'))]+['checkpoint_sha256','sampling_seconds']
    df[keep].sort_values(['run','step','seed']).to_csv(f'{a.out}_all.csv',index=False)
    metrics=[c for c in keep if c.startswith(('fid_','precision','recall','is_mean'))]
    g=df.groupby(['run','step','samples','sampling_steps'])[metrics].agg(['mean','std','count']); g.columns=['_'.join(c) for c in g.columns]
    g.reset_index().to_csv(f'{a.out}_summary.csv',index=False); print(g.round(3).to_string())
if __name__=='__main__': main()
