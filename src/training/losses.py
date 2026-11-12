from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def align_weight(l0,gamma,step,T): return l0*math.exp(-gamma*step/max(1,T))

def cosine_loss(a,b): return 1-(F.normalize(a.float(),dim=-1)*F.normalize(b.float(),dim=-1)).sum(-1).mean()

class LossComputer(nn.Module):
    def __init__(self,tiers,total_steps): super().__init__(); self.tiers=list(tiers); self.T=total_steps
    def forward(self,out,noise,step,hidden=None,heads=None,targets=None):
        total=F.mse_loss(out.float(),noise.float()); logs={'loss/diffusion':float(total.detach())}
        self.terms={'diffusion':total}   # weighted per-term tensors, read by the trainer's gradient diagnostics
        if self.tiers:
            if hidden is None or heads is None or targets is None: raise RuntimeError('alignment inputs missing')
            for tier in self.tiers:
                h=hidden[tier.terminal_block].float(); h=h if tier.patch else h.mean(1)   # patch: (B,64,768) per token; global: token mean
                p=heads(tier.name,h); q=targets[tier.name].to(p.device)
                if p.shape!=q.shape: raise RuntimeError(f'{tier.name}: {p.shape} vs {q.shape}')
                l=cosine_loss(p,q); w=align_weight(tier.lambda_0,tier.gamma,step,self.T); total=total+w*l; self.terms[tier.name]=w*l
                logs[f'loss/align_{tier.name}']=float(l.detach()); logs[f'weight/lambda_{tier.name}']=w; logs[f'contrib/{tier.name}']=float(l.detach())*w
        logs['loss/total']=float(total.detach()); return total,logs
