"""
DiT-B/4 backbone, implemented from scratch per Peebles & Xie (ICCV 2023).

Architecture is fixed across every training mode (Assumption 1 in the report):
12 blocks, hidden 768, 12 heads, patch size 4 over a 32x32x4 SD-VAE latent,
giving 64 tokens per image and ~129.7M parameters.

The one addition over a stock DiT is `return_hidden`, which returns the hidden
state after every block so the tier taps can read blocks 3, 7 and 11 without a
second forward pass. Alignment is training-time only -- at inference this module
is architecturally identical to a standard DiT.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift, broadcast over the token axis."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """Patchify a (B, C, H, W) latent into (B, N, D) tokens via strided conv."""

    def __init__(self, input_size: int, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = input_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.input_size or W != self.input_size:
            raise ValueError(
                f"expected {self.input_size}x{self.input_size} latent, got {H}x{W}"
            )
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            self.mlp[0].weight.dtype
        )
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Class embedding with a learned null token for classifier-free guidance.

    The null token occupies index `num_classes`, so dropping a label to the null
    class during training is what makes CFG possible at sampling time.
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + int(use_cfg_embedding), hidden_size)
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class Attention(nn.Module):
    """Multi-head self-attention using PyTorch scaled_dot_product_attention (flash)."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm zero (AdaLN-Zero) conditioning.

    The 6 modulation parameters (shift/scale/gate for attention and MLP) are
    produced from the conditioning vector c. The final adaLN linear layer is
    zero-initialised, so each block begins training as an identity map.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final AdaLN layer mapping tokens back to patch-shaped output channels."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Fixed 2D sine-cosine positional embeddings, as used by DiT."""

    def _1d(embed_dim: int, pos: np.ndarray) -> np.ndarray:
        omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
        omega = 1.0 / 10000**omega
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _1d(embed_dim // 2, grid[0])
    emb_w = _1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class DiT(nn.Module):
    """Diffusion Transformer (DiT-B/4 by default)."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 100,
        class_dropout_prob: float = 0.1,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.depth = depth

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = self.x_embedder.num_patches
        # Fixed (non-learned) positional embedding, per the DiT reference design.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(hidden_size, self.x_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self._init_weights()

    def _init_weights(self) -> None:
        def _basic(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_basic)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.x_embedder.proj.bias)
        # Re-apply the AdaLN-Zero and FinalLayer zeroing that _basic overwrote.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch^2 * C) -> (B, C, H, W)."""
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W) noisy latent
            t: (B,) diffusion timesteps
            y: (B,) class labels
            return_hidden: also return the hidden state after each block, for the
                alignment tier taps. Costs nothing extra -- the states already exist.

        Returns:
            (B, out_channels, H, W), and optionally a list of `depth` tensors of
            shape (B, N, hidden_size).
        """
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, self.training)

        hidden_states: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x, c)
            if return_hidden:
                hidden_states.append(x)

        out = self.unpatchify(self.final_layer(x, c))
        if return_hidden:
            return out, hidden_states
        return out

    def forward_with_cfg(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float
    ) -> torch.Tensor:
        """Classifier-free guidance forward for sampling.

        Guidance is applied to the epsilon prediction. This project fixes
        `learn_sigma=False`, so the model output is exactly `in_channels`.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return torch.cat([half_eps, half_eps], dim=0)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def DiT_B_4(**kwargs) -> DiT:
    """DiT-B/4: the fixed student backbone for every LARA-DiT experiment."""
    defaults = dict(
        input_size=32, patch_size=4, hidden_size=768, depth=12, num_heads=12
    )
    defaults.update(kwargs)
    return DiT(**defaults)


def build_dit(model_cfg) -> DiT:
    """Construct the student from a ModelConfig."""
    return DiT(
        input_size=model_cfg.input_size,
        patch_size=model_cfg.patch_size,
        in_channels=model_cfg.in_channels,
        hidden_size=model_cfg.hidden_size,
        depth=model_cfg.depth,
        num_heads=model_cfg.num_heads,
        mlp_ratio=model_cfg.mlp_ratio,
        num_classes=model_cfg.num_classes,
        class_dropout_prob=model_cfg.class_dropout_prob,
        learn_sigma=model_cfg.learn_sigma,
    )
