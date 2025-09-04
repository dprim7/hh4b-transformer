from __future__ import annotations

import torch # type: ignore
import torch.nn as nn # type: ignore


class SimpleTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(dim, d_model), nn.ReLU(), nn.Dropout(dropout)]
            dim = d_model
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        logit = self.head(h)
        return logit.squeeze(-1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.attn_dropout(attn_out)
        h = self.norm2(x)
        x = x + self.ffn_dropout(self.ffn(h))
        return x


class ParticleTransformer(nn.Module):
    """Event-level transformer approximating the paper's jet-free ParT style.

    Inputs are sequences of particle features: shape (batch, num_particles, in_dim).
    Supports optional boolean masks and CLS/mean pooling. Returns a binary logit.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        pooling: str = "cls",
    ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.pooling = pooling
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, N, F) or (B, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.size(0)
        x = self.embed(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        key_padding_mask: torch.Tensor | None = None
        if mask is not None:
            # mask: True for valid tokens; MultiheadAttention expects True for padding positions.
            # Build key_padding_mask of shape (B, 1 + N)
            pad = ~mask.bool()
            key_padding_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device), pad], dim=1)

        for block in self.blocks:
            x = block(x, key_padding_mask)

        x = self.norm(x)

        if self.pooling == "cls":
            pooled = x[:, 0]
        else:
            if mask is None:
                pooled = x[:, 1:].mean(dim=1)
            else:
                valid = mask.float().clamp(min=0.0, max=1.0)
                summed = (x[:, 1:] * valid.unsqueeze(-1)).sum(dim=1)
                denom = valid.sum(dim=1).clamp(min=1.0)
                pooled = summed / denom.unsqueeze(-1)

        logit = self.head(pooled)
        return logit.squeeze(-1)
