from __future__ import annotations
from abc import ABC,abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN=(0.485,0.456,0.406); IMAGENET_STD=(0.229,0.224,0.225)
CLIP_MEAN=(0.48145466,0.4578275,0.40821073); CLIP_STD=(0.26862954,0.26130258,0.27577711)

class BaseTeacher(nn.Module,ABC):
    output_dim:int=-1; input_resolution:int=224
    def __init__(self,device,dtype=torch.float32): super().__init__(); self.device=torch.device(device); self.dtype=dtype
    def freeze(self): self.eval(); [p.requires_grad_(False) for p in self.parameters()]; return self
    def train(self,mode=True): return super().train(False)
    @staticmethod
    def renorm(x,src_m,src_s,dst_m,dst_s):
        sm=torch.tensor(src_m,device=x.device,dtype=x.dtype).view(1,3,1,1); ss=torch.tensor(src_s,device=x.device,dtype=x.dtype).view(1,3,1,1); dm=torch.tensor(dst_m,device=x.device,dtype=x.dtype).view(1,3,1,1); ds=torch.tensor(dst_s,device=x.device,dtype=x.dtype).view(1,3,1,1)
        return ((x*ss+sm)-dm)/ds
    def resize(self,x): return F.interpolate(x,size=(224,224),mode='bicubic',align_corners=False,antialias=True)
    @abstractmethod
    def _forward_features(self,x): ...
    @torch.no_grad()
    def forward(self,x): return F.normalize(self._forward_features(x.to(self.device,self.dtype)).float(),dim=-1)
    @abstractmethod
    def _forward_tokens(self,x): ...
    @torch.no_grad()
    def tokens(self,x,grid=8):
        """Patch tokens resampled to the student's grid x grid token layout, L2-normalised per token: (B, grid*grid, D)."""
        return F.normalize(to_grid(self._forward_tokens(x.to(self.device,self.dtype)).float(),grid),dim=-1)

def to_grid(tok,grid):
    b,n,d=tok.shape; s=int(round(n**.5))
    if s*s!=n: raise ValueError(f"{n} tokens is not a square grid")
    g=tok.transpose(1,2).reshape(b,d,s,s)
    if s!=grid: g=F.interpolate(g,size=(grid,grid),mode='bilinear',align_corners=False,antialias=s>grid)
    return g.flatten(2).transpose(1,2)

class DINOv2Teacher(BaseTeacher):
    output_dim=768
    def __init__(self,device,dtype=torch.float32):
        super().__init__(device,dtype); self.model=torch.hub.load('facebookresearch/dinov2','dinov2_vitb14').to(self.device,dtype); self.freeze()
    def _forward_features(self,x): return self.model(self.resize(x))
    def _forward_tokens(self,x): return self.model.forward_features(self.resize(x))['x_norm_patchtokens']   # 16x16

class CLIPTeacher(BaseTeacher):
    output_dim=512
    def __init__(self,device,dtype=torch.float32):
        super().__init__(device,dtype); import open_clip; model,_,_=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai'); self.model=model.visual.to(self.device,dtype); self.freeze()
    def _forward_features(self,x):
        x=self.renorm(x,IMAGENET_MEAN,IMAGENET_STD,CLIP_MEAN,CLIP_STD); return self.model(self.resize(x))
    def _forward_tokens(self,x):
        # open_clip VisionTransformer returns (pooled, tokens) when output_tokens is set; tokens are ln_post'd, width 768, 7x7
        x=self.renorm(x,IMAGENET_MEAN,IMAGENET_STD,CLIP_MEAN,CLIP_STD); self.model.output_tokens=True
        try: _,tok=self.model(self.resize(x))
        finally: self.model.output_tokens=False
        return tok

class MAETeacher(BaseTeacher):
    output_dim=768
    def __init__(self,device,dtype=torch.float32):
        super().__init__(device,dtype); import timm; self.model=timm.create_model('vit_base_patch16_224.mae',pretrained=True,num_classes=0).to(self.device,dtype); self.freeze()
    def _forward_features(self,x):
        f=self.model.forward_features(self.resize(x)); return f[:,1:,:].mean(1) if f.ndim==3 else f
    def _forward_tokens(self,x): return self.model.forward_features(self.resize(x))[:,self.model.num_prefix_tokens:,:]   # 14x14

TEACHER_CLASSES={'dinov2':DINOv2Teacher,'clip':CLIPTeacher,'mae':MAETeacher}

def repa_mlp(hidden_size,out_dim,width=2048):
    # REPA's projector: 3-layer MLP with SiLU (Yu et al., ICLR 2025), applied per token
    return nn.Sequential(nn.Linear(hidden_size,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,out_dim))

class TierProjectionHeads(nn.Module):
    """Linear heads for global tiers (original runs); REPA MLP heads for patch tiers."""
    def __init__(self,tiers,hidden_size=768):
        super().__init__(); self.heads=nn.ModuleDict()
        for t in tiers:
            if t.patch: self.heads[t.name]=repa_mlp(hidden_size,t.teacher_dim)
            else: self.heads[t.name]=nn.Linear(hidden_size,t.teacher_dim); nn.init.xavier_uniform_(self.heads[t.name].weight); nn.init.zeros_(self.heads[t.name].bias)
    def forward(self,name,x): return F.normalize(self.heads[name](x),dim=-1)
