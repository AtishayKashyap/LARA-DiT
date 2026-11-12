from __future__ import annotations
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
from PIL import Image
from .augmentation import paired_train_transform,val_transform

class FeatureBank:
    def __init__(self,path,expected_dim=None,expected_n=None):
        p=Path(path)
        if not p.exists(): raise FileNotFoundError(f"Missing teacher cache: {p}")
        obj=torch.load(p,map_location='cpu')
        if not isinstance(obj,dict) or 'features' not in obj or 'paths' not in obj: raise ValueError(f"Invalid cache format: {p}")
        self.features=obj['features'].float(); self.paths=obj['paths']; self.metadata=obj.get('metadata',{})
        if expected_dim and self.features.shape[1]!=expected_dim: raise ValueError(f"{p}: expected dim {expected_dim}, got {self.features.shape[1]}")
        if expected_n and len(self.features)!=expected_n: raise ValueError(f"{p}: expected {expected_n}, got {len(self.features)}")
        norms=self.features.norm(dim=1)
        if float((norms-1).abs().max())>2e-3: raise ValueError(f"{p}: features are not L2 normalized")
    def __len__(self): return len(self.features)

def open_memmap_bank(npy,paths,seed,shape_tail):
    """Open a float16 .npy cache written by 03_precompute_features.py and check its manifest against the dataset."""
    import json,numpy as np
    npy=Path(npy); meta=npy.with_suffix('.json')
    if not npy.exists() or not meta.exists(): raise FileNotFoundError(f"Missing cache {npy} (+ .json); run 03_precompute_features.py")
    m=json.loads(meta.read_text()); arr=np.load(npy,mmap_mode='r')
    if m['paths']!=paths: raise ValueError(f"Cache order mismatch: {npy}; regenerate")
    if m['augmentation_seed']!=seed: raise ValueError(f"Cache seed mismatch: {npy}")
    if tuple(arr.shape[1:])!=tuple(shape_tail): raise ValueError(f"{npy}: shape {arr.shape} != (*,{shape_tail})")
    return arr

class ImageNet100Dataset(Dataset):
    """Returns (x, label, {tier: target}). x is the paired-augmented image, or -- with use_latents -- the cached
    VAE posterior parameters (8,32,32) of that same view, which the trainer samples from (no encoder, no JPEG decode)."""
    def __init__(self,root,split='train',image_size=256,tiers=None,cache_dir=None,augmentation_seed=42,use_latents=False):
        self.root=Path(root); self.split=split; self.image_size=image_size; self.tiers=tiers or []; self.seed=augmentation_seed
        self.base=datasets.ImageFolder(self.root/split)
        if len(self.base.classes)!=100: raise ValueError(f"Expected 100 classes, found {len(self.base.classes)}")
        self.paths=[str(Path(p).resolve()) for p,_ in self.base.samples]
        self.labels=[y for _,y in self.base.samples]
        self.banks={}; self.latents=None
        if use_latents: self.latents=open_memmap_bank(Path(cache_dir)/'vae'/f'{split}_latents.npy',self.paths,self.seed,(8,32,32))
        for teacher in dict.fromkeys(t.teacher for t in self.tiers):
            tier=next(t for t in self.tiers if t.teacher==teacher)
            if tier.patch: self.banks[teacher]=open_memmap_bank(Path(cache_dir)/teacher/f'{split}_tokens.npy',self.paths,self.seed,(64,tier.teacher_dim))
            else:
                bank=FeatureBank(Path(cache_dir)/teacher/f'{split}.pt',tier.teacher_dim,len(self.paths))
                if bank.paths != self.paths: raise ValueError(f"Cache order mismatch for {teacher}; regenerate caches")
                if bank.metadata.get('augmentation_seed') != self.seed: raise ValueError(f"Cache seed mismatch for {teacher}")
                self.banks[teacher]=bank.features
    def __len__(self): return len(self.base)
    def __getitem__(self,idx):
        if self.latents is not None: x=torch.from_numpy(self.latents[idx].astype('float32'))
        else:
            img=Image.open(self.paths[idx]).convert('RGB')
            x=paired_train_transform(img,idx,self.seed,self.image_size) if self.split=='train' else val_transform(img,self.image_size)
        bank=lambda b: b[idx].float() if torch.is_tensor(b) else torch.from_numpy(b[idx].astype('float32'))   # astype copies out of the read-only memmap
        feats={t.name:bank(self.banks[t.teacher]) for t in self.tiers}
        return x,self.labels[idx],feats

def collate(batch):
    x=torch.stack([b[0] for b in batch]); y=torch.tensor([b[1] for b in batch])
    f={}
    if batch[0][2]:
        for k in batch[0][2]: f[k]=torch.stack([b[2][k] for b in batch])
    return x,y,f

def build_dataloader(cfg,split='train'):
    ds=ImageNet100Dataset(cfg.data.data_root,split,cfg.data.image_size,cfg.tiers if cfg.uses_alignment else [],cfg.data.cache_dir,cfg.data.augmentation_seed,cfg.data.use_latents)
    return DataLoader(ds,batch_size=cfg.train.batch_size,shuffle=(split=='train'),num_workers=cfg.data.num_workers,pin_memory=True,persistent_workers=cfg.data.persistent_workers,prefetch_factor=cfg.data.prefetch_factor,drop_last=(split=='train'),collate_fn=collate)

def denorm(x):
    mean=torch.tensor([.485,.456,.406],device=x.device).view(1,3,1,1); std=torch.tensor([.229,.224,.225],device=x.device).view(1,3,1,1)
    return (x*std+mean).clamp(0,1)
