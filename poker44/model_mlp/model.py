"""Deep Sets MLP bot detector (permutation-invariant, no cross-hand attention).

    per-hand features  [B, H, F]
        -> shared per-hand MLP encoder      -> [B, H, D]   (each hand encoded
           (depth layers, LayerNorm+GELU)       independently)
        -> masked attention + mean pooling  -> [B, 2D]     (order-invariant)
        -> classifier head                  -> [B] logit -> sigmoid risk

Unlike the Transformer, hands are encoded independently (no self-attention), so
this is a classic Deep Sets model -- fewer parameters and often a stronger
generalizer in small-data regimes like the ~1.3k v1.13 benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Deep Sets MLP hyperparameters (small defaults for ~1.3k examples)."""

    feature_dim: int
    d_model: int = 64
    depth: int = 2
    attn_dim: int = 48
    head_hidden: int = 64
    dropout: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


class HandEncoder(nn.Module):
    """Shared MLP applied independently to each hand's feature vector."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        layers = []
        in_dim = config.feature_dim
        for _ in range(max(config.depth, 1)):
            layers += [
                nn.Linear(in_dim, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ]
            in_dim = config.d_model
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, H, D]


class AttentionPool(nn.Module):
    """Masked additive-attention pooling over the hand dimension."""

    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(h).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (weights * h).sum(dim=1)


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    summed = (h * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1.0)
    return summed / counts


class BotDetector(nn.Module):
    """Deep Sets bot detector producing one risk logit per batch."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = HandEncoder(config)
        self.attn_pool = AttentionPool(config.d_model, config.attn_dim)
        self.head = nn.Sequential(
            nn.Linear(config.d_model * 2, config.head_hidden),
            nn.LayerNorm(config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features)
        pooled = torch.cat(
            [self.attn_pool(encoded, mask), masked_mean(encoded, mask)], dim=-1
        )
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(features, mask))
