> **Note:** this is the original pre-experiment specification, kept for the record. For results, current status and the follow-up experiments, see README.md and RUNBOOK.md.

# LARA-DiT — Project Specification and Validation Contract

This is the clean research implementation for the capstone project. The Mid-Sem report is the starting specification; engineering choices may be changed only when documented and when they preserve the research question.

## Research question
Test whether depth-stratified teacher/tier matching improves convergence and generation quality over vanilla DiT and single-teacher REPA-style alignment under matched compute.

## Fixed experiment
- Student: DiT-B/4, 12 blocks, hidden size 768, 12 heads, patch size 4
- Input: 256x256 RGB -> SD-VAE latent 32x32x4
- Dataset: ImageNet-100, 100 classes, 130,000 train / 5,000 validation
- Baseline: no alignment
- REPA: DINOv2 ViT-B/14 -> blocks 0-3
- LARA-DiT: DINOv2 ViT-B/14 -> blocks 0-3; CLIP ViT-B/32 -> blocks 4-7; MAE ViT-B/16 -> blocks 8-11
- Ablation: DINOv2 -> all three tiers
- Teachers frozen; projection heads trainable
- Alignment: cosine similarity on terminal hidden state of each tier after token averaging and projection
- Weight: lambda_0=0.5; gamma=(2.0,1.5,1.0), with lambda(t)=lambda_0 exp(-gamma*t/T)
- Diffusion: 1000-step cosine DDPM, epsilon prediction, MSE
- Optimizer: AdamW, lr 1e-4, batch 128, seed 42, 200K steps
- Evaluation: FID, Inception Score (10 splits), AlexNet LPIPS on top-25% Laplacian-magnitude regions
- Checkpoints: 50K, 100K, 150K, 200K

## Hardware target
Validated target environment supplied by the user: NVIDIA H100 80GB HBM3, MIG 3g.40GB, ~39.4 GiB visible VRAM, CUDA 12.5, PyTorch 2.4.0 NVIDIA build, BF16 supported.

## Important implementation choices
1. BF16 AMP is used on H100 instead of FP16 GradScaler because the target GPU supports BF16 and the training objective does not require FP16 scaling.
2. Teacher features are precomputed using the same deterministic image view used by the student for each training index. Cache metadata records ordered paths, teacher identity, seed, image size and count.
3. The DiT output is epsilon-only (`learn_sigma=False`). This matches the project loss and avoids an unsupervised learned-variance head.
4. Teachers are never present in the training autograd graph.
5. Full training experiments are sequential; one MIG allocation is never oversubscribed by launching multiple runs concurrently.
6. The full 200K runs are NOT considered validated until the real ImageNet-100 smoke test and throughput benchmark pass on the user's Thapar GPU.
