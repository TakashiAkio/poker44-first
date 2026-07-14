"""Reduced-size masked set-Transformer over hands.

    per-hand features  [B, H, F]
        -> input projection                 -> [B, H, D]
        -> N x TransformerEncoder layers    -> [B, H, D]   (self-attention,
           (NO positional encoding,             order-invariant, padding-masked)
        -> masked attention + mean pooling  -> [B, 2D]
        -> classifier head                  -> [B] logit -> sigmoid risk

Defaults are intentionally small (depth=1, d_model=64, dropout=0.3) because the
public v1.13 benchmark only exposes ~1.3k labeled examples; a large Transformer
would overfit. Scale up via ``ModelConfig`` as more data becomes available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Set-Transformer hyperparameters (small defaults for ~1.3k examples)."""

    feature_dim: int
    d_model: int = 64
    depth: int = 1
    n_heads: int = 4
    ff_mult: int = 2
    attn_dim: int = 48
    head_hidden: int = 64
    dropout: float = 0.3

    @property
    def ff_dim(self) -> int:
        return self.d_model * self.ff_mult

    def to_dict(self) -> dict:
        return asdict(self)


class HandSetEncoder(nn.Module):
    """Projects per-hand features and applies padding-masked self-attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(config.feature_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.depth, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        key_padding_mask = mask == 0
        return self.encoder(h, src_key_padding_mask=key_padding_mask)


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
    """Hero-centric set-Transformer producing one risk logit per batch."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = HandSetEncoder(config)
        self.attn_pool = AttentionPool(config.d_model, config.attn_dim)
        self.head = nn.Sequential(
            nn.Linear(config.d_model * 2, config.head_hidden),
            nn.LayerNorm(config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features, mask)
        pooled = torch.cat(
            [self.attn_pool(encoded, mask), masked_mean(encoded, mask)], dim=-1
        )
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(features, mask))
