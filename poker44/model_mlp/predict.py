"""Inference wrapper for the trained Deep Sets MLP.

Exposes ``predict(chunks) -> list[float]`` matching the miner-visible
``DetectionSynapse.chunks`` contract: a list of chunk groups (each a list of
hands). Returns one bot-risk score in [0, 1] per chunk group, in order.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from poker44.model_mlp.dataset import Standardizer
from poker44.model_mlp.features import HandFeatureExtractor
from poker44.model_mlp.model import BotDetector, ModelConfig

DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.pt"


class Predictor:
    """Loads a trained checkpoint and scores chunk groups."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
        max_hands: int = 60,
    ):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_ARTIFACT)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_hands = max_hands
        self.extractor = HandFeatureExtractor()
        self.standardizer: Optional[Standardizer] = None
        self.model: Optional[BotDetector] = None
        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}. "
                "Train one with `python -m poker44.model_mlp.train`."
            )
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        config = ModelConfig(**ckpt["model_config"])
        self.model = BotDetector(config).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.standardizer = Standardizer.from_dict(ckpt["standardizer"])

    def _to_tensor(self, hands: List[dict]):
        feats = np.asarray(self.extractor.extract_batch(hands), dtype=np.float32)
        if feats.shape[0] > self.max_hands:
            feats = feats[: self.max_hands]
        if self.standardizer is not None:
            feats = self.standardizer.transform(feats)
        return torch.from_numpy(feats.astype(np.float32))

    @torch.no_grad()
    def predict(self, chunks: List[List[dict]]) -> List[float]:
        """One bot-risk score per chunk group, order preserved."""
        if not chunks:
            return []

        tensors = [self._to_tensor(chunk or []) for chunk in chunks]
        max_h = max(t.shape[0] for t in tensors)
        feat_dim = self.extractor.feature_dim
        batch = len(tensors)

        padded = torch.zeros((batch, max_h, feat_dim), dtype=torch.float32)
        mask = torch.zeros((batch, max_h), dtype=torch.float32)
        for i, t in enumerate(tensors):
            h = t.shape[0]
            padded[i, :h] = t
            mask[i, :h] = 1.0

        padded, mask = padded.to(self.device), mask.to(self.device)
        probs = self.model.predict_proba(padded, mask).cpu().numpy()
        return [float(np.clip(p, 0.0, 1.0)) for p in probs]

    def predict_one(self, hands: List[dict]) -> float:
        return self.predict([hands])[0]
