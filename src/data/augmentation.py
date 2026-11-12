from __future__ import annotations
import math
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225)

def deterministic_train_params(image: Image.Image, index:int, seed:int=42):
    g=torch.Generator(); g.manual_seed(seed+index)
    width,height=image.size; area=width*height
    scale=torch.empty((),generator=g).uniform_(0.8,1.0).item() if False else None
    # torch.empty(..., generator=) is not supported on all builds; use manual random draws.
    u=torch.rand((),generator=g).item(); v=torch.rand((),generator=g).item()
    target=area*(0.8+0.2*u); log_min=math.log(3/4); log_max=math.log(4/3)
    ratio=math.exp(log_min+(log_max-log_min)*v)
    w=round(math.sqrt(target*ratio)); h=round(math.sqrt(target/ratio))
    if not (0<w<=width and 0<h<=height):
        in_ratio=width/height
        if in_ratio<3/4: w=width; h=round(w/(3/4))
        elif in_ratio>4/3: h=height; w=round(h*(4/3))
        else: w=width; h=height
    i=int(torch.randint(0,max(1,height-h+1),(1,),generator=g).item()) if h<=height else 0
    j=int(torch.randint(0,max(1,width-w+1),(1,),generator=g).item()) if w<=width else 0
    flip=bool(torch.rand((),generator=g).item()<0.5)
    return i,j,min(h,height),min(w,width),flip

def paired_train_transform(image,index,seed=42,size=256):
    i,j,h,w,flip=deterministic_train_params(image,index,seed)
    image=TF.resized_crop(image,i,j,h,w,[size,size],InterpolationMode.BICUBIC)
    if flip: image=TF.hflip(image)
    return TF.normalize(TF.to_tensor(image),MEAN,STD)

def val_transform(image,size=256):
    image=TF.resize(image,round(size*1.125),InterpolationMode.BICUBIC)
    image=TF.center_crop(image,[size,size])
    return TF.normalize(TF.to_tensor(image),MEAN,STD)
