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
