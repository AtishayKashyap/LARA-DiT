import sys,torch
sys.path.insert(0,'.')
from src.models.dit import DiT_B_4,count_parameters
from src.training.diffusion import GaussianDiffusion
from src.config import Config

def test_model_count():
 m=DiT_B_4(learn_sigma=False); assert count_parameters(m)==129686080

def test_forward_shape():
 m=DiT_B_4(learn_sigma=False); x=torch.randn(2,4,32,32); t=torch.randint(0,1000,(2,)); y=torch.randint(0,100,(2,)); o,h=m(x,t,y,True); assert o.shape==(2,4,32,32); assert len(h)==12 and h[0].shape==(2,64,768)

def test_diffusion():
 d=GaussianDiffusion(); x=torch.randn(2,4,32,32); t=torch.tensor([0,999]); xt,e=d.add_noise(x,t); assert xt.shape==x.shape and e.shape==x.shape

def test_modes():
 for mode in ('baseline','repa','lara_dit','ablation'): Config(mode)

class _Stub(torch.nn.Module):
 # minimal DiT stand-in: p_sample only needs y_embedder.num_classes and an eps output with >=4 channels
 def __init__(self,scale): super().__init__(); self.scale=scale; self.y_embedder=type('Y',(),{'num_classes':100})()
 def forward(self,x,t,y): return self.scale*x

def test_sampler_rng_controlled():
 d=GaussianDiffusion(T=20); y=torch.tensor([0,1]); shape=(2,4,8,8); run=lambda m,s:d.sample(m,shape,y,'cpu',1.5,torch.Generator().manual_seed(s))
 global_before=torch.get_rng_state(); a=run(_Stub(.1),42)
 assert torch.equal(torch.get_rng_state(),global_before)  # sampler no longer touches global RNG
 assert torch.equal(a,run(_Stub(.1),42)) and not torch.equal(a,run(_Stub(.1),43))
 # draw count is model-independent, so two configs with the same seed see identical noise streams
 g1,g2=torch.Generator().manual_seed(42),torch.Generator().manual_seed(42)
 d.sample(_Stub(.1),shape,y,'cpu',1.5,g1); d.sample(_Stub(.9),shape,y,'cpu',1.5,g2); assert torch.equal(g1.get_state(),g2.get_state())
