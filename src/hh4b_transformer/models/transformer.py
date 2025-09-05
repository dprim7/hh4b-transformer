from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore


class TransformerBlock(nn.Module):
    """Particle-level transformer block.

    Inputs
    ------
    x: Tensor of shape (batch, length, d_model)
    key_padding_mask: Boolean mask of shape (batch, length)
        True marks padding tokens to be ignored by attention.
    pair_bias: Optional pairwise features used as additive attention bias.
        Shape (batch, length, length, K) or (batch, length-1, length-1, K) if CLS
        is omitted; projected to per-head biases internally.

    Outputs
    -------
    y: Tensor of shape (batch, length, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        use_pair_bias: bool,
        pair_bias_dim: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.use_pair_bias = use_pair_bias
        self.pair_bias_dim = pair_bias_dim
        if use_pair_bias:
            # project K-dim pairwise features to per-head additive bias
            self.pair_projection = nn.Sequential(
                nn.Linear(pair_bias_dim, n_heads),
            )

    def _shape_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        qkv = self.qkv(x)  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        # [B, H, L, Dh]
        q = q.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        pair_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        h = self.norm1(x)
        q, k, v = self._shape_qkv(h)

        B, H, L, Dh = q.shape

        attn_mask: torch.Tensor | None = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] True = pad. Build additive mask [B, 1, 1, L]
            pad = key_padding_mask.view(B, 1, 1, L)
            attn_mask = torch.where(
                pad, torch.finfo(h.dtype).min, torch.zeros(1, dtype=h.dtype, device=h.device)
            )

        if self.use_pair_bias and pair_feats is not None:
            # pair_feats shapes: [B, L-1, L-1, K] (without CLS) or [B, L, L, K]
            if pair_feats.dim() == 4:
                K = pair_feats.size(-1)
                if K != self.pair_bias_dim:
                    # If provided K differs, project last dim anyway
                    pass
                # If missing CLS row/col, pad zeros to make [B, L, L, K]
                if pair_feats.size(1) == L - 1:
                    zeros_row = torch.zeros(
                        (pair_feats.size(0), 1, pair_feats.size(2), pair_feats.size(3)),
                        device=pair_feats.device,
                        dtype=pair_feats.dtype,
                    )
                    zeros_col = torch.zeros(
                        (pair_feats.size(0), L, 1, pair_feats.size(3)),
                        device=pair_feats.device,
                        dtype=pair_feats.dtype,
                    )
                    pair_full = torch.cat([zeros_row, pair_feats], dim=1)
                    pair_full = torch.cat([zeros_col, pair_full], dim=2)
                else:
                    pair_full = pair_feats
                # project to [B, L, L, H] → [B, H, L, L]
                bias_heads = self.pair_projection(pair_full).permute(0, 3, 1, 2)
                if attn_mask is None:
                    attn_mask = bias_heads
                else:
                    attn_mask = attn_mask + bias_heads
            elif pair_feats.dim() == 3:
                # scalar bias [B, L-1, L-1] or [B, L, L] → expand to heads
                if pair_feats.size(1) == L - 1:
                    zeros_row = torch.zeros(
                        (pair_feats.size(0), 1, pair_feats.size(2)),
                        device=pair_feats.device,
                        dtype=pair_feats.dtype,
                    )
                    zeros_col = torch.zeros(
                        (pair_feats.size(0), L, 1), device=pair_feats.device, dtype=pair_feats.dtype
                    )
                    pair_full = torch.cat([zeros_row, pair_feats], dim=1)
                    pair_full = torch.cat([zeros_col, pair_full], dim=2)
                else:
                    pair_full = pair_feats
                bias_heads = pair_full.unsqueeze(1).expand(-1, H, -1, -1)
                attn_mask = bias_heads if attn_mask is None else attn_mask + bias_heads

        # scaled dot-product attention; attn_mask is additive [B, H, L, S]
        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        # [B, H, L, Dh] → [B, L, D]
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)
        attn = self.out_proj(attn)
        x = x + attn

        # Pre-norm FFN
        h = self.norm2(x)
        x = x + self.ffn_dropout(self.ffn(h))
        return x


class HH4bTransformer(nn.Module):
    """Event-level jet-free transformer.

    Inputs
    ------
    x: Tensor of shape (batch, num_particles, feature_dim)
    mask: Boolean mask of shape (batch, num_particles)
    attention_bias: Optional tensor of shape (batch, num_particles, num_particles)

    Outputs
    -------
    logit: Tensor of shape (batch, num_classes)
    """

    def __init__(
        self,
        feature_dim: int = 23,  # TODO: check this
        d_model: int = 256,  # embedding space
        n_heads: int = 16,
        n_layers: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
        pooling: str = "cls",
        num_classes: int = 138,
        use_pair_bias: bool = True,
        pair_bias_dim: int = 256,
    ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # is this present in the paper?
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    dropout,
                    use_pair_bias=use_pair_bias,
                    pair_bias_dim=pair_bias_dim,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.pooling = pooling
        self.num_classes = num_classes

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:

        batch_size = x.size(0)
        x = self.embed(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # mask: True for valid tokens; MultiheadAttention expects True for padding positions.
        # Build key_padding_mask of shape (B, 1 + N_particles)
        pad = ~mask.bool()
        key_padding_mask = torch.cat(
            [torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device), pad], dim=1
        )  # cls is always valid, so all zeros for zeroth position of the maks

        for block in self.blocks:
            x = block(x, key_padding_mask, attention_bias)

        x = self.norm(x)

        if self.pooling == "cls":
            pooled = x[:, 0]
        else:
            valid = mask.float().clamp(min=0.0, max=1.0)
            summed = (x[:, 1:] * valid.unsqueeze(-1)).sum(dim=1)
            denom = valid.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom.unsqueeze(-1)

        logits = self.mlp_head(pooled)

        # softmax not needed for training?

        return logits


# ---------------------------------------------------------
# skeleton for jet-free transformer
# ---------------------------------------------------------

# embedding (particles + interactions)

# N transformer blocks

# MLP head

# 138-dim classification head with softmax

# ---------------------------------------------------------
# skeleton for transformer block
# ---------------------------------------------------------

# 1
# LayerNorm

# P-MHA

# LayerNorm
# 2

# 1+2 (skip connection)

# LayerNorm

# Linear

# GELU

# LayerNorm

# Linear
# 3

# 3 + (1+2) (skip connection)
