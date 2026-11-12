#!/usr/bin/env python3
"""Measure throughput on this GPU and project the cost of the planned runs. Run once, after the caches exist.

Trains each setting for --steps steps in a scratch directory (via 05_train.py), times sampling with a randomly
initialised DiT-B/4, and writes benchmark.json with projected GPU-hours.
"""
from __future__ import annotations
import argparse,json,re,shutil,subprocess,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.dit import DiT_B_4
from src.training.diffusion import GaussianDiffusion

SETTINGS={  # name: extra args for 05_train.py
    'baseline_images':['--mode','baseline'],
    'baseline_latents':['--mode','baseline','--use-latents'],
    'lara_patch_latents':['--mode','custom','--align','patch','--teachers','dinov2,clip,mae','--gammas','2,1.5,1','--use-latents'],
}

def train_its(name,extra,a,scratch):
    cmd=[sys.executable,str(Path(__file__).parent/'05_train.py'),*extra,'--steps',str(a.steps),'--ckpt-every',str(10**9),'--run-name',f'bench_{name}',
         '--output-dir',str(scratch),'--data-root',a.data_root,'--cache-dir',a.cache_dir]
    out=subprocess.run(cmd,capture_output=True,text=True)
    if out.returncode: print(out.stdout[-2000:],out.stderr[-4000:]); raise SystemExit(f'{name} failed')
    its=[float(x) for x in re.findall(r'([\d.]+) it/s',out.stdout)]
    return sum(its[1:])/max(1,len(its[1:]))   # drop the first window (warm-up)

def sampling_ips(steps,batch,bf16):
    dev=torch.device('cuda'); m=DiT_B_4(learn_sigma=False).to(dev).eval(); d=GaussianDiffusion(steps=steps).to(dev); y=torch.arange(batch,device=dev)%100
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
        d.sample(m,(batch,4,32,32),y,dev,1.5); torch.cuda.synchronize(); t=time.time()
        d.sample(m,(batch,4,32,32),y,dev,1.5); torch.cuda.synchronize()
    return batch/(time.time()-t)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--steps',type=int,default=400); ap.add_argument('--data-root',default='/workspace/data/imagenet100')
    ap.add_argument('--cache-dir',default='/workspace/cache'); ap.add_argument('--scratch',default='/workspace/bench_tmp'); ap.add_argument('--out',default='benchmark.json'); a=ap.parse_args()
    scratch=Path(a.scratch); res={'device':torch.cuda.get_device_name(0),'train_it_per_s':{},'sample_img_per_s':{}}
    try:
        for n,e in SETTINGS.items(): res['train_it_per_s'][n]=train_its(n,e,a,scratch); print(n,res['train_it_per_s'][n],'it/s')
    finally: shutil.rmtree(scratch,ignore_errors=True)
    for k,bf in [('250steps_bf16',True),('250steps_fp32',False)]: res['sample_img_per_s'][k]=sampling_ips(250,250,bf); print(k,res['sample_img_per_s'][k],'img/s (DiT only, no VAE decode)')
    its=res['train_it_per_s']['lara_patch_latents']; ips=res['sample_img_per_s']['250steps_bf16']
    res['projected_hours']={'train_100k_steps':100000/its/3600,'train_50k_steps':50000/its/3600,'eval_10k_samples':10000/ips/3600,'eval_5k_samples':5000/ips/3600}
    Path(a.out).write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
