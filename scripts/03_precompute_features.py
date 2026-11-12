#!/usr/bin/env python3
from __future__ import annotations
import argparse,time,sys
from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.augmentation import paired_train_transform,val_transform
from src.models.teachers import TEACHER_CLASSES

class RawPaths(Dataset):
    def __init__(self,root,split,seed,size):
        from torchvision.datasets import ImageFolder
        ds=ImageFolder(Path(root)/split); self.paths=[str(Path(p).resolve()) for p,_ in ds.samples]; self.labels=[y for _,y in ds.samples]; self.seed=seed; self.size=size; self.split=split
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        im=Image.open(self.paths[i]).convert('RGB')
        x=paired_train_transform(im,i,self.seed,self.size) if self.split=='train' else val_transform(im,self.size)
        return x,self.labels[i],self.paths[i]

def precompute_memmap(a,ds,dl,dev):
    """--tokens: (N,64,768) teacher patch tokens on the student's 8x8 grid. --teacher vae: (N,8,32,32) VAE posterior
    parameters (mean||logvar) of the same paired view, so training can sample latents without running the encoder.
    Both are float16 .npy memmaps plus a .json manifest checked by the dataset (paths, seed, shape)."""
    import json,numpy as np
    if a.teacher=='vae':
        from src.models.vae_wrapper import VAEWrapper
        vae=VAEWrapper(device=dev,dtype=torch.bfloat16); shape=(len(ds),8,32,32); name='latents'
        fn=lambda x: vae.vae.encode(vae.imagenet_to_vae(x).to(dev,torch.bfloat16)).latent_dist.parameters
    else:
        teacher=TEACHER_CLASSES[a.teacher](dev).eval(); shape=(len(ds),64,768); name='tokens'; fn=lambda x: teacher.tokens(x,8)
    out=Path(a.cache)/a.teacher; out.mkdir(parents=True,exist_ok=True); path=out/f'{a.split}_{name}.npy'
    if path.exists() and not a.overwrite: print('exists:',path); return
    arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=shape); paths=[]; pos=0; t0=time.time()
    for x,y,p in dl:
        with torch.no_grad(): f=fn(x.to(dev,non_blocking=True)).float()
        if not torch.isfinite(f).all(): raise RuntimeError(f'non-finite output at {pos}')
        arr[pos:pos+len(x)]=f.cpu().numpy().astype(np.float16); paths.extend(list(p)); pos+=len(x)
    arr.flush(); assert pos==len(ds)
    (out/f'{a.split}_{name}.json').write_text(json.dumps({'teacher':a.teacher,'split':a.split,'augmentation_seed':a.seed,'shape':list(shape),'paths':paths}))
    print(f'saved {path} shape={shape} time={(time.time()-t0)/60:.2f} min')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='/workspace/data/imagenet100'); ap.add_argument('--cache',default='/workspace/cache'); ap.add_argument('--teacher',required=True,choices=[*TEACHER_CLASSES,'vae']); ap.add_argument('--tokens',action='store_true',help='cache 8x8 patch tokens instead of the pooled vector'); ap.add_argument('--split',default='train',choices=['train','val']); ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--overwrite',action='store_true'); a=ap.parse_args()
    if a.teacher=='vae' or a.tokens:
        ds=RawPaths(a.root,a.split,a.seed,256); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,prefetch_factor=4)
        return precompute_memmap(a,ds,dl,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    out=Path(a.cache)/a.teacher; out.mkdir(parents=True,exist_ok=True); path=out/f'{a.split}.pt'
    if path.exists() and not a.overwrite: print('exists:',path); return
    ds=RawPaths(a.root,a.split,a.seed,256); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,prefetch_factor=4)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); teacher=TEACHER_CLASSES[a.teacher](dev).eval(); dim=teacher.output_dim; feats=torch.empty(len(ds),dim,dtype=torch.float32); labels=[]; paths=[]; t0=time.time(); pos=0
    for x,y,p in dl:
        x=x.to(dev,non_blocking=True)
        with torch.no_grad(): f=teacher(x).float().cpu()
        n=len(x); feats[pos:pos+n]=f; labels.extend(y.tolist()); paths.extend(list(p)); pos+=n
    norms=feats.norm(dim=1); feats=torch.nn.functional.normalize(feats,dim=1)
    obj={'features':feats,'labels':torch.tensor(labels),'paths':paths,'metadata':{'teacher':a.teacher,'split':a.split,'augmentation_seed':a.seed,'image_size':256,'count':len(ds)}}
    torch.save(obj,path); print(f'saved {path} shape={tuple(feats.shape)} time={(time.time()-t0)/60:.2f} min max_norm_error={(norms-1).abs().max().item():.2e}')
if __name__=='__main__': main()
