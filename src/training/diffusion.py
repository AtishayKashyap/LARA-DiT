from __future__ import annotations
import math
import torch
import torch.nn as nn

def cosine_betas(T,s=0.008):
    x=torch.linspace(0,T,T+1,dtype=torch.float64)/T; a=torch.cos((x+s)/(1+s)*math.pi/2)**2; a=a/a[0]
    return (1-a[1:]/a[:-1]).clamp(1e-4,0.999).float()
def extract(a,t,shape):
    v=a.to(t.device).gather(0,t).float()
    while v.ndim<len(shape): v=v[...,None]
    return v.expand(shape)

def space_timesteps(T,n):
    # ADM-style even respacing of a T-step schedule down to n steps (always keeps 0 and T-1)
    return sorted({round(i*(T-1)/(n-1)) for i in range(n)}) if n<T else list(range(T))

class GaussianDiffusion(nn.Module):
    def __init__(self,T=1000,s=0.008,steps=None):
        # steps<T gives a respaced sampler over the same trained schedule (Nichol & Dhariwal 2021); training always uses steps=None
        super().__init__(); keep=space_timesteps(T,steps or T); abar_full=torch.cumprod(1-cosine_betas(T,s),0)
        abar=abar_full[keep]; prev=torch.cat([torch.ones(1),abar[:-1]]); b=1-abar/prev; al=1-b; self.T=len(keep)
        self.register_buffer('timestep_map',torch.tensor(keep,dtype=torch.long),persistent=False)
        post=b*(1-prev)/(1-abar); post_log=torch.log(post.clamp(min=1e-20));
        for n,v in {'betas':b,'alphas_cumprod':abar,'sqrt_abar':torch.sqrt(abar),'sqrt_1m_abar':torch.sqrt(1-abar),'sqrt_recip_abar':torch.sqrt(1/abar),'sqrt_recipm1_abar':torch.sqrt(1/abar-1),'post_var':post,'post_log_var':post_log,'post_c1':b*torch.sqrt(prev)/(1-abar),'post_c2':(1-prev)*torch.sqrt(al)/(1-abar)}.items(): self.register_buffer(n,v,persistent=False)
    def sample_timesteps(self,b,device): return torch.randint(0,self.T,(b,),device=device)
    def add_noise(self,x,t,noise=None):
        noise=torch.randn_like(x) if noise is None else noise
        return extract(self.sqrt_abar,t,x.shape)*x+extract(self.sqrt_1m_abar,t,x.shape)*noise,noise
    @torch.no_grad()
    def p_sample(self,model,x,t,y,cfg=1.0,generator=None):
        tm=self.timestep_map.to(t.device)[t]  # model was trained on original timestep indices
        if cfg!=1.0:
            null=torch.full_like(y,model.y_embedder.num_classes); out=model(torch.cat([x,x]),torch.cat([tm,tm]),torch.cat([y,null])); eps_c,eps_u=out[:len(x),:4].float(),out[len(x):,:4].float(); eps=eps_u+cfg*(eps_c-eps_u)
        else: eps=model(x,tm,y)[:,:4].float()
        x0=extract(self.sqrt_recip_abar,t,x.shape)*x-extract(self.sqrt_recipm1_abar,t,x.shape)*eps; x0=x0.clamp(-4,4)
        mean=extract(self.post_c1,t,x.shape)*x0+extract(self.post_c2,t,x.shape)*x
        noise=torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator); nz=(t!=0).float().view(-1,1,1,1); return mean+nz*torch.exp(.5*extract(self.post_log_var,t,x.shape))*noise
    @torch.no_grad()
    def sample(self,model,shape,y,device,cfg=1.0,generator=None,noise=None):
        # generator drives ALL sampler randomness (initial x_T + per-step noise); noise= overrides x_T for matched comparisons
        x=torch.randn(shape,device=device,generator=generator) if noise is None else noise.to(device); model.eval()
        for i in reversed(range(self.T)):
            t=torch.full((shape[0],),i,device=device,dtype=torch.long); x=self.p_sample(model,x,t,y,cfg,generator)
        return x
