#!/usr/bin/env python3
import sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import Config,MODES
from src.models.dit import DiT_B_4,count_parameters
print('Environment OK')
for mode in MODES:
    c=Config(mode=mode); m=DiT_B_4(learn_sigma=False,num_classes=100); n=count_parameters(m)
    print(f'{mode:9s}: params={n:,} ({n/1e6:.3f}M), teachers={c.required_teachers}')
assert n==129686080
print('DiT-B/4 parameter count verified:',n)
print('Preflight PASSED')
