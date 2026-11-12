#!/usr/bin/env python3
"""Train one configuration. Presets: baseline / repa / lara_dit / ablation (original runs, global alignment).
Any other design: --mode custom --teachers early,mid,deep ('-' = unaligned tier) --gammas g1,g2,g3.
  REPA-faithful : --mode custom --align patch --teachers dinov2,-,- --gammas 0,0,0
  LARA (patch)  : --mode custom --align patch --teachers dinov2,clip,mae --gammas 2,1.5,1
  reversed      : --mode custom --align patch --teachers mae,clip,dinov2 --gammas 2,1.5,1"""
from __future__ import annotations
import argparse,random,numpy as np,torch,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import Config,custom_tiers
from src.data.dataset import build_dataloader
from src.models.dit import build_dit
from src.models.vae_wrapper import VAEWrapper
from src.models.teachers import TierProjectionHeads
from src.training.diffusion import GaussianDiffusion
from src.training.losses import LossComputer
from src.training.trainer import Trainer

def build_config(a):
    tiers=custom_tiers(a.teachers.split(','),[float(g) for g in a.gammas.split(',')],a.lambda0) if a.mode=='custom' else []
    c=Config(mode=a.mode,tiers=tiers,align=a.align,run_name=a.run_name)
    c.data.data_root=a.data_root; c.data.cache_dir=a.cache_dir; c.data.num_workers=a.workers; c.data.use_latents=a.use_latents
    c.train.output_dir=a.output_dir; c.train.ckpt_every=a.ckpt_every; c.train.total_steps=a.steps; c.train.seed=a.seed; c.train.diag_every=a.diag_every
    return c

def parse(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode',required=True,choices=['baseline','repa','lara_dit','ablation','custom'])
    ap.add_argument('--align',default='global',choices=['global','patch']); ap.add_argument('--teachers',default='dinov2,clip,mae'); ap.add_argument('--gammas',default='2,1.5,1'); ap.add_argument('--lambda0',type=float,default=0.5)
    ap.add_argument('--run-name',default=None); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--steps',type=int,default=200000)
    ap.add_argument('--data-root',default='/workspace/data/imagenet100'); ap.add_argument('--cache-dir',default='/workspace/cache'); ap.add_argument('--output-dir',default='/workspace/outputs')
    ap.add_argument('--use-latents',action='store_true',help='read cached VAE posteriors (03_precompute_features.py --teacher vae) instead of encoding images every step')
    ap.add_argument('--diag-every',type=int,default=0,help='log per-tier gradient norms/cosines every N steps (0=off)')
    ap.add_argument('--resume',default=None); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--ckpt-every',type=int,default=50000)
    return ap.parse_args(argv)

def main():
    a=parse(); c=build_config(a)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    loader=build_dataloader(c,'train'); dev=torch.device('cuda'); model=build_dit(c.model).to(dev); vae=VAEWrapper(c.model.vae_model_id,dev,torch.bfloat16); diffusion=GaussianDiffusion(c.diffusion.num_timesteps,c.diffusion.cosine_s); loss=LossComputer(c.tiers,c.train.total_steps); heads=TierProjectionHeads(c.tiers).to(dev) if c.uses_alignment else None
    print(c.summary()); print('GPU:',torch.cuda.get_device_name(0)); print('VRAM:',torch.cuda.get_device_properties(0).total_memory/2**30)
    Trainer(c,model,vae,diffusion,loss,heads,loader).train(a.resume)
if __name__=='__main__': main()
