#!/usr/bin/env python3
"""Build Inception (clean-fid) reference features once, for 07_evaluate.py.

  train50k.npy : 500 images/class from the training set, ADM-style preprocessing (shorter side -> 256 bicubic, centre crop)
  val5k.npy    : all 5,000 validation images, same preprocessing
  val5k_raw.npy: validation images exactly as the original evaluation saw them (clean-fid on the raw folder),
                 so FID against it reproduces the protocol of the reported numbers
"""
from __future__ import annotations
import argparse,json,shutil,tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF, InterpolationMode
from tqdm import tqdm

def adm_crop(path,size=256):
    im=Image.open(path).convert('RGB'); im=TF.resize(im,size,InterpolationMode.BICUBIC); return TF.center_crop(im,[size,size])

def balanced_subset(samples,per_class,seed):
    rng=np.random.default_rng(seed); by={}
    for i,(_,y) in enumerate(samples): by.setdefault(y,[]).append(i)
    return sorted(j for y in sorted(by) for j in rng.choice(by[y],size=min(per_class,len(by[y])),replace=False))

def feats_of(paths,dev,tmp):
    from cleanfid import fid
    d=Path(tempfile.mkdtemp(dir=tmp))
    try:
        for k,p in enumerate(tqdm(paths,desc='cropping')): adm_crop(p).save(d/f'{k:06d}.png')
        return fid.get_folder_features(str(d),mode='clean',batch_size=100,device=dev).astype(np.float32)
    finally: shutil.rmtree(d)

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root',default='/workspace/data/imagenet100'); ap.add_argument('--out',default='/workspace/fid_ref'); ap.add_argument('--per-class',type=int,default=500)
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--tmp',default='/workspace'); a=ap.parse_args()
    from cleanfid import fid
    dev=torch.device('cuda'); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    tr=ImageFolder(Path(a.root)/'train'); va=ImageFolder(Path(a.root)/'val')
    if tr.classes!=va.classes: raise SystemExit('train/val class folders differ: label indices would not match')
    idx=balanced_subset(tr.samples,a.per_class,a.seed)
    np.save(out/'train50k.npy',feats_of([tr.samples[i][0] for i in idx],dev,a.tmp))
    np.save(out/'val5k.npy',feats_of([p for p,_ in va.samples],dev,a.tmp))
    np.save(out/'val5k_raw.npy',fid.get_folder_features(str(Path(a.root)/'val'),mode='clean',batch_size=100,device=dev).astype(np.float32))
    (out/'reference.json').write_text(json.dumps({'train_indices_seed':a.seed,'per_class':a.per_class,'n_train':len(idx),'n_val':len(va.samples),
        'preprocess':'shorter side 256 bicubic + centre crop 256; clean-fid mode=clean','val5k_raw':'raw val folder, as in the original evaluation','classes':tr.classes},indent=2))
    print('saved',sorted(p.name for p in out.glob('*.npy')))
if __name__=='__main__': main()
