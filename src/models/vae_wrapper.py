from __future__ import annotations
import torch
import torch.nn as nn

class VAEWrapper(nn.Module):
    def __init__(self,model_name='stabilityai/sd-vae-ft-ema',device='cuda',dtype=torch.bfloat16):
        super().__init__(); from diffusers import AutoencoderKL
        self.device=torch.device(device); self.dtype=dtype
        self.vae=AutoencoderKL.from_pretrained(model_name).to(self.device,dtype=dtype).eval(); self.vae.requires_grad_(False)
        self.scaling_factor=float(getattr(self.vae.config,'scaling_factor',0.18215))
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1),persistent=False); self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1),persistent=False)
    def imagenet_to_vae(self,x): return ((x*self.std.to(x.device,x.dtype)+self.mean.to(x.device,x.dtype))*2-1)
    @torch.no_grad()
    def encode_from_imagenet(self,x):
        posterior=self.vae.encode(self.imagenet_to_vae(x).to(self.device,self.dtype)).latent_dist
        return posterior.sample()*self.scaling_factor
    @torch.no_grad()
    def decode(self,z): return self.vae.decode(z.to(self.device,self.dtype)/self.scaling_factor).sample
    @torch.no_grad()
    def vae_to_pil(self,z):
        from PIL import Image
        x=((self.decode(z)+1)/2).clamp(0,1); arr=(x*255).round().to(torch.uint8).permute(0,2,3,1).cpu().numpy(); return [Image.fromarray(a) for a in arr]
