#!/usr/bin/env python3
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config, MODES
from src.models.dit import DiT_B_4, count_parameters

EXPECTED = 129686080

print('=== LARA-DiT project validation ===')
print('Torch:', torch.__version__)
print('CUDA:', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/2**30, 2))
    print('BF16:', torch.cuda.is_bf16_supported())

m = DiT_B_4(learn_sigma=False, num_classes=100)
n = count_parameters(m)
assert n == EXPECTED, (n, EXPECTED)
print('DiT-B/4 parameters:', n)

for mode in MODES:
    c = Config(mode=mode)
    print(mode, 'teachers=', c.required_teachers, 'tiers=', [(t.name, t.blocks, t.teacher, t.lambda_0, t.gamma) for t in c.tiers])

assert Config('repa').tiers[0].blocks == (0,4)
assert Config('lara_dit').tiers[0].blocks == (0,4)
assert Config('lara_dit').tiers[1].blocks == (4,8)
assert Config('lara_dit').tiers[2].blocks == (8,12)
assert [t.teacher for t in Config('lara_dit').tiers] == ['dinov2','clip','mae']
assert [t.teacher for t in Config('ablation').tiers] == ['dinov2','dinov2','dinov2']
print('Research configuration: PASS')
print('Project validation: PASS')
