from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import json, math

MODES = ("baseline", "repa", "lara_dit", "ablation", "custom")
TEACHER_DIMS: Dict[str,int] = {"dinov2":768, "clip":512, "mae":768}        # global (pooled) targets
PATCH_DIMS: Dict[str,int] = {"dinov2":768, "clip":768, "mae":768}          # per-token targets (CLIP: pre-projection width)
TIER_BLOCKS = {"early":(0,4), "mid":(4,8), "deep":(8,12)}

@dataclass
class TierSpec:
    name: str
    blocks: Tuple[int,int]
    teacher: str
    gamma: float
    lambda_0: float = 0.5
    patch: bool = False   # set by Config from Config.align
    @property
    def terminal_block(self): return self.blocks[1]-1
    @property
    def teacher_dim(self): return (PATCH_DIMS if self.patch else TEACHER_DIMS)[self.teacher]

def custom_tiers(teachers, gammas, lambda_0=0.5, tiers=("early","mid","deep")):
    """e.g. custom_tiers(['mae','clip','dinov2'],[2,1.5,1]); a teacher of '-' leaves that tier unaligned."""
    return [TierSpec(n,TIER_BLOCKS[n],t,float(g),float(lambda_0)) for n,t,g in zip(tiers,teachers,gammas) if t!='-']

def lara_tiers():
    return [TierSpec("early",(0,4),"dinov2",2.0), TierSpec("mid",(4,8),"clip",1.5), TierSpec("deep",(8,12),"mae",1.0)]
def repa_tiers(): return [TierSpec("early",(0,4),"dinov2",0.0)]
def ablation_tiers():
    return [TierSpec("early",(0,4),"dinov2",2.0), TierSpec("mid",(4,8),"dinov2",1.5), TierSpec("deep",(8,12),"dinov2",1.0)]

@dataclass
class ModelConfig:
    input_size:int=32; in_channels:int=4; patch_size:int=4; hidden_size:int=768; depth:int=12; num_heads:int=12; mlp_ratio:float=4.0; num_classes:int=100; class_dropout_prob:float=0.1; learn_sigma:bool=False
    vae_model_id:str="stabilityai/sd-vae-ft-ema"

@dataclass
class DiffusionConfig:
    num_timesteps:int=1000; schedule:str="cosine"; cosine_s:float=0.008; cfg_scale:float=1.5

@dataclass
class DataConfig:
    data_root:str="/workspace/data/imagenet100"; cache_dir:str="/workspace/cache"; image_size:int=256; num_workers:int=8; pin_memory:bool=True; persistent_workers:bool=True; prefetch_factor:int=4; augmentation_seed:int=42; use_latents:bool=False

@dataclass
class TrainConfig:
    batch_size:int=128; total_steps:int=200000; lr:float=1e-4; weight_decay:float=0.0; betas:Tuple[float,float]=(0.9,0.999); grad_clip:float=1.0; seed:int=42; use_amp:bool=True; amp_dtype:str="bfloat16"; use_ema:bool=True; ema_decay:float=0.9999; ema_warmup_steps:int=1000; log_every:int=100; ckpt_every:int=10000; sample_every:int=0; diag_every:int=0; output_dir:str="/workspace/outputs"

@dataclass
class EvalConfig:
    num_samples:int=10000; batch_size:int=64; cfg_scale:float=1.5; sampling_steps:int=1000; is_splits:int=10; lpips_net:str="alex"; hf_percentile:float=75.0

@dataclass
class Config:
    mode:str="lara_dit"; model:ModelConfig=field(default_factory=ModelConfig); diffusion:DiffusionConfig=field(default_factory=DiffusionConfig); data:DataConfig=field(default_factory=DataConfig); train:TrainConfig=field(default_factory=TrainConfig); eval:EvalConfig=field(default_factory=EvalConfig); tiers:List[TierSpec]=field(default_factory=list); run_name:str|None=None
    # align='global': token-mean + linear head vs pooled teacher vector (original runs).
    # align='patch' : per-token MLP head vs 8x8 teacher patch tokens (REPA-faithful projector, Yu et al. 2025).
    align:str="global"
    def __post_init__(self):
        if self.mode not in MODES: raise ValueError(self.mode)
        if self.mode=="custom" and not self.tiers: raise ValueError("custom mode needs explicit tiers")
        if not self.tiers: self.tiers = {"baseline":lambda:[],"repa":repa_tiers,"lara_dit":lara_tiers,"ablation":ablation_tiers}[self.mode]()
        if self.align not in ("global","patch"): raise ValueError(self.align)
        for t in self.tiers: t.patch = self.align=="patch"
        self.run_name = self.run_name or self.mode
        self.validate()
    def validate(self):
        m=self.model
        if m.input_size % m.patch_size: raise ValueError("latent input must be divisible by patch size")
        if m.hidden_size % m.num_heads: raise ValueError("hidden size must divide heads")
        if m.depth != 12 or m.hidden_size != 768 or m.num_heads != 12 or m.patch_size != 4 or m.in_channels != 4: raise ValueError("This experiment fixes DiT-B/4")
        if m.num_classes != 100: raise ValueError("ImageNet-100 requires 100 classes")
        if self.train.batch_size != 128 or self.train.lr != 1e-4: raise ValueError("batch 128 / lr 1e-4 are fixed so all runs stay matched")
        if m.learn_sigma: raise ValueError("This implementation uses the report's noise-MSE objective; learned sigma is disabled")
        for t in self.tiers:
            if not (0<=t.blocks[0]<t.blocks[1]<=m.depth): raise ValueError(f"bad tier {t}")
            if t.teacher not in TEACHER_DIMS: raise ValueError(t.teacher)
    @property
    def uses_alignment(self): return bool(self.tiers)
    @property
    def required_teachers(self): return list(dict.fromkeys(t.teacher for t in self.tiers))
    @property
    def run_dir(self): return Path(self.train.output_dir)/self.run_name
    def alignment_weight(self,tier,step): return tier.lambda_0*math.exp(-tier.gamma*step/max(1,self.train.total_steps))
    def save(self,path): Path(path).write_text(json.dumps(asdict(self),indent=2))
    def summary(self):
        return f"{self.run_name} ({self.mode}, align={self.align}): DiT-B/4 129.686M | batch={self.train.batch_size} steps={self.train.total_steps} lr={self.train.lr} seed={self.train.seed} | tiers="+", ".join(f"{t.name}:{t.teacher}(γ={t.gamma},λ0={t.lambda_0})" for t in self.tiers)
