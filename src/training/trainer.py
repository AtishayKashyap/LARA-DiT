from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import os,random,time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

def sample_cached_latents(params,scaling_factor):
    """Draw z ~ N(mean, exp(logvar)) from cached VAE posterior parameters (mean||logvar), as the encoder would."""
    mean,logvar=params.float().chunk(2,dim=1)
    return (mean+torch.exp(0.5*logvar.clamp(-30,20))*torch.randn_like(mean))*scaling_factor

def gradient_diagnostics(terms,params):
    """Norm of each (weighted) loss term's gradient on the shared DiT weights, and its cosine with the denoising
    gradient. Tests whether a tier's alignment pulls against denoising (cf. HASTE's gradient-angle trigger)."""
    def flat(loss): return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float()
                                      for g,p in zip(torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True),params)])
    gd=flat(terms['diffusion']); out={'grad/norm_diffusion':float(gd.norm())}
    for k,v in terms.items():
        if k=='diffusion': continue
        g=flat(v); n=g.norm(); out[f'grad/norm_{k}']=float(n); out[f'grad/cos_{k}']=float(g@gd/(n*gd.norm()+1e-12))
    return out

class EMA:
    def __init__(self,model,decay=0.9999,warmup=1000):
        self.decay=decay; self.warmup=warmup; self.shadow=deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self,model,step):
        d=min(self.decay,(1+step)/(self.warmup+step))
        for a,b in zip(self.shadow.parameters(),model.parameters()): a.mul_(d).add_(b,alpha=1-d)

class Trainer:
    def __init__(self,cfg,model,vae,diffusion,loss,heads,loader):
        self.cfg=cfg; self.model=model; self.vae=vae; self.diffusion=diffusion; self.loss=loss; self.heads=heads; self.loader=loader; self.device=next(model.parameters()).device
        params=list(model.parameters())+(list(heads.parameters()) if heads else [])
        self.opt=torch.optim.AdamW(params,lr=cfg.train.lr,weight_decay=cfg.train.weight_decay,betas=cfg.train.betas)
        self.fp16=cfg.train.use_amp and cfg.train.amp_dtype=='float16' and self.device.type=='cuda'
        self.amp_dtype=torch.float16 if self.fp16 else torch.bfloat16
        self.scaler=torch.amp.GradScaler('cuda',enabled=self.fp16)
        self.ema=EMA(model,cfg.train.ema_decay,cfg.train.ema_warmup_steps) if cfg.train.use_ema else None
        self.run_dir=cfg.run_dir; (self.run_dir/'checkpoints').mkdir(parents=True,exist_ok=True); self.writer=SummaryWriter(self.run_dir/'logs'); cfg.save(self.run_dir/'config.json'); self.step=0; self.epoch=0
    def save(self,tag=None):
        tag=tag or f'step{self.step:08d}'; p=self.run_dir/'checkpoints'/f'{tag}.pt'
        state={'step':self.step,'epoch':self.epoch,'mode':self.cfg.mode,'model':self.model.state_dict(),'optimizer':self.opt.state_dict(),'scaler':self.scaler.state_dict()}
        if self.ema: state['ema']=self.ema.shadow.state_dict()
        if self.heads: state['heads']=self.heads.state_dict()
        state['rng_python']=random.getstate(); state['rng_numpy']=np.random.get_state(); state['rng_torch']=torch.get_rng_state(); state['rng_cuda']=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.save(state,p); print(f'[checkpoint] {p}'); return p
    def resume(self,path):
        p=torch.load(path,map_location=self.device); self.model.load_state_dict(p['model']); self.opt.load_state_dict(p['optimizer']); self.scaler.load_state_dict(p.get('scaler',{})); self.step=p['step']; self.epoch=p.get('epoch',0)
        if self.ema and 'ema' in p: self.ema.shadow.load_state_dict(p['ema'])
        if self.heads and 'heads' in p: self.heads.load_state_dict(p['heads'])
        if p.get('rng_python') is not None: random.setstate(p['rng_python']); np.random.set_state(p['rng_numpy']); torch.set_rng_state(p['rng_torch']);
        if torch.cuda.is_available() and p.get('rng_cuda') is not None: torch.cuda.set_rng_state_all(p['rng_cuda'])
    def train(self,resume=None):
        if resume: self.resume(resume)
        self.model.train(); self.heads.train() if self.heads else None
        it=iter(self.loader); start=time.time(); accum={}
        while self.step<self.cfg.train.total_steps:
            try: images,labels,targets=next(it)
            except StopIteration: self.epoch+=1; it=iter(self.loader); images,labels,targets=next(it)
            images=images.to(self.device,non_blocking=True); labels=labels.to(self.device,non_blocking=True); targets={k:v.to(self.device,non_blocking=True) for k,v in targets.items()}
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=self.amp_dtype,enabled=self.device.type=='cuda' and self.cfg.train.use_amp):
                z=sample_cached_latents(images,self.vae.scaling_factor) if images.shape[1]==8 else self.vae.encode_from_imagenet(images)
                t=self.diffusion.sample_timesteps(len(z),self.device); xt,noise=self.diffusion.add_noise(z,t)
                if self.cfg.uses_alignment: out,hidden=self.model(xt,t,labels,return_hidden=True)
                else: out=self.model(xt,t,labels); hidden=None
                total,logs=self.loss(out,noise,self.step,hidden,self.heads,targets if targets else None)
            if self.cfg.train.diag_every and self.step%self.cfg.train.diag_every==0 and len(self.loss.terms)>1:
                logs.update(gradient_diagnostics(self.loss.terms,[p for p in self.model.parameters() if p.requires_grad]))
            if self.fp16: self.scaler.scale(total).backward(); self.scaler.unscale_(self.opt); params=list(self.model.parameters())+(list(self.heads.parameters()) if self.heads else []); torch.nn.utils.clip_grad_norm_(params,self.cfg.train.grad_clip); self.scaler.step(self.opt); self.scaler.update()
            else: total.backward(); params=list(self.model.parameters())+(list(self.heads.parameters()) if self.heads else []); torch.nn.utils.clip_grad_norm_(params,self.cfg.train.grad_clip); self.opt.step()
            if self.ema: self.ema.update(self.model,self.step)
            self.step+=1
            for k,v in logs.items():
                if k.startswith('grad/'): self.writer.add_scalar(k,v,self.step)   # sparse diagnostics: log as-is, not averaged
                else: accum[k]=accum.get(k,0)+v
            if self.step%self.cfg.train.log_every==0:
                dt=time.time()-start; its=self.cfg.train.log_every/dt; avg={k:v/self.cfg.train.log_every for k,v in accum.items()}; accum={}; start=time.time()
                for k,v in avg.items(): self.writer.add_scalar(k,v,self.step)
                self.writer.add_scalar('perf/it_per_s',its,self.step)
                print(f"step {self.step}/{self.cfg.train.total_steps} total={avg['loss/total']:.4f} diff={avg['loss/diffusion']:.4f} {its:.1f} it/s")
            if self.step%self.cfg.train.ckpt_every==0: self.save()
        self.save('final'); self.writer.close()
