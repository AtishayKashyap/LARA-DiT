"""CPU checks for the revamp: respaced sampler, patch alignment, configurable tiers, cached latents/tokens, diagnostics."""
import json,math,sys
import numpy as np,torch,pytest
sys.path.insert(0,'.')
from src.config import Config,custom_tiers
from src.training.diffusion import GaussianDiffusion,cosine_betas,space_timesteps
from src.training.losses import LossComputer
from src.training.trainer import sample_cached_latents,gradient_diagnostics
from src.models.teachers import TierProjectionHeads,to_grid
from src.data.dataset import open_memmap_bank

def test_full_schedule_unchanged_by_respacing_code():
    d=GaussianDiffusion(); abar=torch.cumprod(1-cosine_betas(1000),0)
    assert d.T==1000 and torch.equal(d.timestep_map,torch.arange(1000))
    assert torch.allclose(d.alphas_cumprod,abar,atol=1e-6) and torch.allclose(d.betas,cosine_betas(1000),atol=1e-5)

def test_respaced_schedule_keeps_trained_alphas():
    d=GaussianDiffusion(steps=250); keep=space_timesteps(1000,250)
    assert d.T==250 and keep[0]==0 and keep[-1]==999 and len(set(keep))==250
    assert torch.allclose(d.alphas_cumprod,torch.cumprod(1-cosine_betas(1000),0)[keep],atol=1e-6)

def test_respaced_sampler_calls_model_with_original_timesteps():
    seen=[]
    class M(torch.nn.Module):
        y_embedder=type('Y',(),{'num_classes':100})()
        def forward(self,x,t,y): seen.append(int(t[0])); return torch.zeros_like(x)
    GaussianDiffusion(steps=10).sample(M(),(1,4,8,8),torch.tensor([3]),'cpu',1.0,torch.Generator().manual_seed(0))
    assert seen==space_timesteps(1000,10)[::-1]

def test_custom_tiers_and_patch_dims():
    c=Config(mode='custom',tiers=custom_tiers(['mae','clip','dinov2'],[2,1.5,1]),align='patch')
    assert [t.teacher for t in c.tiers]==['mae','clip','dinov2'] and all(t.patch and t.teacher_dim==768 for t in c.tiers)
    r=Config(mode='custom',tiers=custom_tiers(['dinov2','-','-'],[0,0,0]),align='patch')
    assert [(t.name,t.terminal_block,t.gamma) for t in r.tiers]==[('early',3,0.0)]
    assert Config(mode='lara_dit').tiers[1].teacher_dim==512   # global CLIP keeps the 512-d projected embedding

def test_to_grid_resamples_square_token_maps():
    for n in (16,14,7):
        x=torch.ones(2,n*n,5); y=to_grid(x,8); assert y.shape==(2,64,5) and torch.allclose(y,torch.ones_like(y),atol=1e-5)
    with pytest.raises(ValueError): to_grid(torch.ones(1,50,3),8)

def test_patch_loss_and_terms():
    c=Config(mode='custom',tiers=custom_tiers(['dinov2','clip','mae'],[2,1.5,1]),align='patch'); heads=TierProjectionHeads(c.tiers)
    hidden=[torch.randn(2,64,768) for _ in range(12)]; tg={t.name:torch.nn.functional.normalize(torch.randn(2,64,768),dim=-1) for t in c.tiers}
    lc=LossComputer(c.tiers,1000); total,logs=lc(torch.zeros(2,4,32,32),torch.randn(2,4,32,32),0,hidden,heads,tg)
    assert torch.isfinite(total) and set(lc.terms)=={'diffusion','early','mid','deep'}
    assert math.isclose(float(total),sum(float(v) for v in lc.terms.values()),rel_tol=1e-5)
    assert all(0<=logs[f'loss/align_{t}']<=2 for t in ('early','mid','deep'))

def test_global_heads_stay_linear_for_old_checkpoints():
    h=TierProjectionHeads(Config(mode='lara_dit').tiers); assert isinstance(h.heads['mid'],torch.nn.Linear) and h.heads['mid'].out_features==512

def test_cached_latent_sampling():
    mean=torch.randn(3,4,32,32); params=torch.cat([mean,torch.full_like(mean,-30.)],1)
    assert torch.allclose(sample_cached_latents(params,0.18215),mean*0.18215,atol=1e-5)
    z=sample_cached_latents(torch.cat([torch.zeros(2000,4,1,1),torch.zeros(2000,4,1,1)],1),1.0); assert abs(float(z.std())-1)<0.05

def test_memmap_bank_checks_manifest(tmp_path):
    arr=np.lib.format.open_memmap(tmp_path/'train_tokens.npy',mode='w+',dtype=np.float16,shape=(2,64,768)); arr[:]=1; arr.flush()
    (tmp_path/'train_tokens.json').write_text(json.dumps({'paths':['a','b'],'augmentation_seed':42}))
    assert open_memmap_bank(tmp_path/'train_tokens.npy',['a','b'],42,(64,768)).shape==(2,64,768)
    with pytest.raises(ValueError): open_memmap_bank(tmp_path/'train_tokens.npy',['b','a'],42,(64,768))
    with pytest.raises(ValueError): open_memmap_bank(tmp_path/'train_tokens.npy',['a','b'],7,(64,768))
    with pytest.raises(ValueError): open_memmap_bank(tmp_path/'train_tokens.npy',['a','b'],42,(64,512))

def test_gradient_diagnostics():
    w=torch.nn.Parameter(torch.randn(10)); a=(w**2).sum()
    d=gradient_diagnostics({'diffusion':a,'same':2*a,'opp':-a},[w])
    assert abs(d['grad/cos_same']-1)<1e-5 and abs(d['grad/cos_opp']+1)<1e-5 and abs(d['grad/norm_same']-2*d['grad/norm_diffusion'])<1e-4

def test_train_cli_builds_reversed_patch_config():
    import importlib.util
    spec=importlib.util.spec_from_file_location('train','scripts/05_train.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    c=m.build_config(m.parse(['--mode','custom','--align','patch','--teachers','mae,clip,dinov2','--gammas','2,1.5,1','--steps','100000','--seed','1','--use-latents']))
    assert c.train.total_steps==100000 and c.train.seed==1 and c.data.use_latents and [t.teacher for t in c.tiers]==['mae','clip','dinov2']
