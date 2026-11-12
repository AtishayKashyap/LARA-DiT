#!/usr/bin/env python3
"""Evaluate one or more checkpoints under a single, recorded protocol.

Per checkpoint: generate N class-balanced samples from the EMA weights with a seeded generator (identical noise
streams across checkpoints for the same --seed/--batch-size/--samples), extract clean-fid Inception features once,
then report FID against every reference in --ref-dir (09_make_fid_reference.py), precision/recall against the
train reference, and Inception Score. Everything needed to reproduce the run is written to manifest.json/metrics.json.

  python scripts/07_evaluate.py --checkpoint outputs/*/checkpoints/step*.pt --ref-dir /workspace/fid_ref --samples 10000 --seed 0
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys,time,random
from pathlib import Path
import numpy as np
import torch,torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.dit import build_dit
from src.models.vae_wrapper import VAEWrapper
from src.training.diffusion import GaussianDiffusion
from src.evaluation.metrics import fid_from_features,precision_recall

def inception_score(images_u8,device,splits=10,batch=100):
    """IS from a uint8 (N,H,W,3) array, converted per batch (torchvision Inception-v3; relative use only)."""
    from torchvision.models import inception_v3,Inception_V3_Weights
    from torchvision import transforms
    m=inception_v3(weights=Inception_V3_Weights.DEFAULT,aux_logits=True).to(device).eval(); tf=transforms.Normalize([.485,.456,.406],[.229,.224,.225]); probs=[]
    with torch.no_grad():
        for i in range(0,len(images_u8),batch):
            x=torch.from_numpy(images_u8[i:i+batch]).to(device).permute(0,3,1,2).float()/255
            x=F.interpolate(x,size=(299,299),mode='bilinear',align_corners=False); probs.append(F.softmax(m(tf(x)),1).cpu())
    p=torch.cat(probs); n=len(p)//splits; scores=[]
    for i in range(splits):
        q=p[i*n:(i+1)*n]; py=q.mean(0,keepdim=True); scores.append(torch.exp((q*(q.add(1e-8).log()-py.add(1e-8).log())).sum(1).mean()).item())
    return float(np.mean(scores)),float(np.std(scores))

def sha256(path,chunk=1<<24):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

def evaluate(ckpt_path,a,dev,refs):
    ck=torch.load(ckpt_path,map_location='cpu',weights_only=False); cpath=Path(ckpt_path)
    # <run>/checkpoints/<ckpt>.pt (trainer layout) or flat '<run>_final.pt' (archived copies)
    run=cpath.stem[:-len('_final')] if cpath.stem.endswith('_final') else cpath.parent.parent.name if cpath.parent.name=='checkpoints' else cpath.stem
    out=Path(a.out_root)/f"{run}__{cpath.stem}__seed{a.seed}_n{a.samples}_k{a.sampling_steps}"; gen=out/'generated'
    if (out/'metrics.json').exists() and not a.overwrite: print('done:',out); return json.loads((out/'metrics.json').read_text())
    gen.mkdir(parents=True,exist_ok=True)
    model=build_dit(type('C',(),dict(input_size=32,patch_size=4,in_channels=4,hidden_size=768,depth=12,num_heads=12,mlp_ratio=4,num_classes=100,class_dropout_prob=.1,learn_sigma=False))()).to(dev)
    key='ema' if 'ema' in ck else 'model'; model.load_state_dict(ck[key]); model.eval()
    vae=VAEWrapper(device=dev,dtype=torch.bfloat16); diff=GaussianDiffusion(steps=a.sampling_steps).to(dev)
    manifest={'seed':a.seed,'checkpoint':str(cpath.resolve()),'checkpoint_sha256':sha256(cpath),'step':int(ck.get('step',0)),'mode':ck.get('mode','unknown'),'run':run,'weights':key,
              'cfg_scale':a.cfg_scale,'sampler':'ddpm_ancestral_cosine_respaced','sampling_steps':diff.T,'sampling_batch_size':a.batch_size,'samples':a.samples,'labels':'arange(samples)%100',
              'autocast_bf16':not a.fp32,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # all sampler randomness comes from g; the number of draws is model-independent -> matched noise across checkpoints
    g=torch.Generator(device=dev); g.manual_seed(a.seed); labels=torch.arange(a.samples,device=dev)%100
    imgs=np.empty((a.samples,256,256,3),np.uint8); t0=time.time()
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not a.fp32):
        for s in tqdm(range(0,a.samples,a.batch_size),desc=f'sampling {run}/{cpath.stem}'):
            y=labels[s:s+a.batch_size]; z=diff.sample(model,(len(y),4,32,32),y,dev,a.cfg_scale,generator=g)
            for j,im in enumerate(vae.vae_to_pil(z.float())): imgs[s+j]=np.asarray(im); im.save(gen/f'{s+j:06d}.png')
    sampling_s=time.time()-t0
    from cleanfid import fid as cleanfid
    feats=cleanfid.get_folder_features(str(gen),mode='clean',batch_size=100,device=dev,verbose=False); np.save(out/'inception_feats.npy',feats.astype(np.float32))
    res={**manifest,'sampling_seconds':sampling_s}
    for name,rf in refs.items(): res[f'fid_{name}']=fid_from_features(rf,feats)
    if 'train50k' in refs:
        sub=np.random.default_rng(0).choice(len(refs['train50k']),size=min(len(feats),len(refs['train50k'])),replace=False)
        res.update({f'{k}_train':v for k,v in precision_recall(refs['train50k'][sub],feats).items()})
    res['is_mean'],res['is_std']=inception_score(imgs,dev); res['is_impl']='torchvision inception_v3, 10 splits'
    (out/'metrics.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
    if not a.keep_images:   # keep the first 100 for qualitative grids; features are saved
        for p in sorted(gen.glob('*.png'))[100:]: p.unlink()
    return res

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint',nargs='+',required=True); ap.add_argument('--out-root',default='/workspace/evaluation'); ap.add_argument('--ref-dir',required=True)
    ap.add_argument('--samples',type=int,default=10000); ap.add_argument('--batch-size',type=int,default=250); ap.add_argument('--cfg-scale',type=float,default=1.5)
    ap.add_argument('--sampling-steps',type=int,default=250); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--fp32',action='store_true',help='disable bf16 autocast for the DiT forward')
    ap.add_argument('--keep-images',action='store_true'); ap.add_argument('--overwrite',action='store_true'); a=ap.parse_args()
    if a.samples%100: raise SystemExit('--samples must be a multiple of 100 (class-balanced labels)')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); dev=torch.device('cuda')
    refs={p.stem:np.load(p) for p in sorted(Path(a.ref_dir).glob('*.npy'))}
    if not refs: raise SystemExit(f'no reference features in {a.ref_dir}; run 09_make_fid_reference.py')
    print('references:',{k:v.shape for k,v in refs.items()})
    for c in a.checkpoint: evaluate(c,a,dev,refs)
if __name__=='__main__': main()
