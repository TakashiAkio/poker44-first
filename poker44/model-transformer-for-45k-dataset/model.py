"""PyTorch bot-detection model.

Architecture (a masked, permutation-invariant set-Transformer over hands):

    per-hand features  [B, H, F]
        -> input projection                 -> [B, H, D]
        -> N x TransformerEncoder layers    -> [B, H, D]   (self-attention
           (NO positional encoding,             lets hands attend to each
            padding-masked)                      other; order-invariant)
        -> masked attention pooling         -> [B, D]
        -> concat with masked mean pooling  -> [B, 2D]
        -> classifier head                  -> [B, 1] logit -> sigmoid risk

Why a set-Transformer: bots reveal themselves through cross-hand consistency
(near-identical sizings / patterns), which self-attention models directly.
Hands have no meaningful order, so positional encodings are intentionally
omitted, making the encoder permutation-invariant. Variable hand counts per
batch are handled with padding + a key-padding mask. Output is one bot-risk
probability per batch, matching the validator contract of one score per group.

The size is fully configurable via ``ModelConfig`` so it can be scaled up as the
amount of training data grows.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Set-Transformer hyperparameters.

    Defaults are deliberately small to suit ~45k v1.13 batch examples and
    resist overfitting on the hardened benchmark. Scale ``d_model`` / ``depth``
    / ``n_heads`` up as the dataset grows.
    """

    feature_dim: int
    d_model: int = 96
    depth: int = 2
    n_heads: int = 4
    ff_mult: int = 2
    attn_dim: int = 64
    head_hidden: int = 96
    dropout: float = 0.2

    @property
    def ff_dim(self) -> int:
        return self.d_model * self.ff_mult

    def to_dict(self) -> dict:
        return asdict(self)


class HandSetEncoder(nn.Module):
    """Projects per-hand features and applies padding-masked self-attention.

    No positional encoding is used, so the encoder is permutation-invariant
    over the hand dimension (hands are an unordered set).
    """

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
        # x: [B, H, F], mask: [B, H] (1 = valid hand, 0 = padding)
        h = self.input_proj(x)  # [B, H, D]
        key_padding_mask = mask == 0  # True where padded -> ignored by attention
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
        # h: [B, H, D], mask: [B, H] (1 = valid hand, 0 = padding)
        scores = self.score(h).squeeze(-1)  # [B, H]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, H, 1]
        return (weights * h).sum(dim=1)  # [B, D]


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
        # features: [B, H, F], mask: [B, H]
        encoded = self.encoder(features, mask)
        pooled_attn = self.attn_pool(encoded, mask)
        pooled_mean = masked_mean(encoded, mask)
        pooled = torch.cat([pooled_attn, pooled_mean], dim=-1)
        return self.head(pooled).squeeze(-1)  # [B] logits

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(features, mask))
