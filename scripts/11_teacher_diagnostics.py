#!/usr/bin/env python3
"""How informative is each cached teacher target? Tests the paper's hypothesis that pooled CLIP/MAE targets are
'easy' because they vary little across images. CPU is enough. Reports, per cache:
  mean_pairwise_cos  : average cosine between different images (high = targets look alike)
  effective_rank     : exp(entropy of normalised singular values) of the centred features
  centroid_acc       : nearest-class-centroid accuracy (how class-discriminative the target is)
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch

def diagnostics(f,labels,n=5000,seed=0):
    f=torch.as_tensor(np.asarray(f),dtype=torch.float32); f=torch.nn.functional.normalize(f.reshape(len(f),-1),dim=1)
    idx=torch.as_tensor(np.random.default_rng(seed).choice(len(f),size=min(n,len(f)),replace=False)); x=f[idx]; y=torch.as_tensor(labels)[idx]
    c=x@x.T; off=c[~torch.eye(len(x),dtype=torch.bool)]
    s=torch.linalg.svdvals(x-x.mean(0)); p=s/s.sum(); erank=float(torch.exp(-(p*torch.log(p+1e-12)).sum()))
    cent=torch.stack([x[y==k].mean(0) for k in range(int(y.max())+1)]); acc=float(((x@torch.nn.functional.normalize(cent,dim=1).T).argmax(1)==y).float().mean())
    return {'mean_pairwise_cos':float(off.mean()),'std_pairwise_cos':float(off.std()),'effective_rank':erank,'centroid_acc':acc,'n':len(x)}

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument('--cache-dir',default='/workspace/cache'); ap.add_argument('--out',default='teacher_diagnostics.json'); a=ap.parse_args()
    res={}
    for t in ['dinov2','clip','mae']:
        g=Path(a.cache_dir)/t/'train.pt'
        if g.exists():
            o=torch.load(g,map_location='cpu'); res[f'{t}_global']=diagnostics(o['features'],o['labels'].numpy())
            tok=Path(a.cache_dir)/t/'train_tokens.npy'
            if tok.exists():   # token caches: diagnostics of the token-mean and of single tokens' spatial variation
                arr=np.load(tok,mmap_mode='r'); sub=np.random.default_rng(0).choice(len(arr),2000,replace=False); s=np.asarray(arr[np.sort(sub)],np.float32)
                res[f'{t}_tokens_mean']=diagnostics(s.mean(1),o['labels'].numpy()[np.sort(sub)],n=2000)
                st=torch.nn.functional.normalize(torch.from_numpy(s),dim=-1); within=(st@st.transpose(1,2)).mean().item()
                res[f'{t}_tokens_mean']['within_image_token_cos']=within   # low = rich spatial structure
        print(t,res.get(f'{t}_global'))
    Path(a.out).write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
