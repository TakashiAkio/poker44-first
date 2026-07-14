"""Torch dataset, padded collate, standardization and release splitting.

Includes one-time feature caching and optional hand-subsampling augmentation
(valid because the Deep Sets model is permutation-invariant over hands).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from poker44.model_mlp.data import BatchExample
from poker44.model_mlp.features import HandFeatureExtractor


@dataclass
class Standardizer:
    """Feature-wise standardization (fit on train only)."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, feature_rows: np.ndarray) -> "Standardizer":
        mean = feature_rows.mean(axis=0)
        std = feature_rows.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        return cls(mean=np.asarray(d["mean"], dtype=np.float32),
                   std=np.asarray(d["std"], dtype=np.float32))


class BatchHandsDataset(Dataset):
    """Each item is (per-hand feature matrix, label) for one batch example."""

    def __init__(
        self,
        examples: Sequence[BatchExample],
        extractor: Optional[HandFeatureExtractor] = None,
        standardizer: Optional[Standardizer] = None,
        max_hands: int = 60,
        precompute: bool = True,
        augment: bool = False,
        aug_min_frac: float = 0.6,
        aug_min_hands: int = 12,
        seed: int = 44,
    ):
        self.examples = list(examples)
        self.extractor = extractor or HandFeatureExtractor()
        self.standardizer = standardizer
        self.max_hands = max_hands
        self.precompute = precompute
        self.augment = augment
        self.aug_min_frac = aug_min_frac
        self.aug_min_hands = aug_min_hands
        self._rng = np.random.default_rng(seed)
        self._cache: Optional[List[np.ndarray]] = None
        if precompute:
            self._cache = [self._extract_raw(i) for i in range(len(self.examples))]

    def __len__(self) -> int:
        return len(self.examples)

    def _extract_raw(self, idx: int) -> np.ndarray:
        ex = self.examples[idx]
        feats = np.asarray(self.extractor.extract_batch(ex.hands), dtype=np.float32)
        if feats.shape[0] > self.max_hands:
            feats = feats[: self.max_hands]
        return feats

    def raw_features(self, idx: int) -> np.ndarray:
        if self._cache is not None:
            return self._cache[idx]
        return self._extract_raw(idx)

    def _maybe_augment(self, feats: np.ndarray) -> np.ndarray:
        h = feats.shape[0]
        if not self.augment or h <= self.aug_min_hands:
            return feats
        low = max(self.aug_min_hands, int(round(self.aug_min_frac * h)))
        if low >= h:
            return feats
        k = int(self._rng.integers(low, h + 1))
        idx = self._rng.choice(h, size=k, replace=False)
        return feats[np.sort(idx)]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self._maybe_augment(self.raw_features(idx))
        if self.standardizer is not None:
            feats = self.standardizer.transform(feats)
        label = float(self.examples[idx].label)
        return torch.from_numpy(feats.astype(np.float32)), torch.tensor(label)

    def all_feature_rows(self) -> np.ndarray:
        rows = [self.raw_features(i) for i in range(len(self))]
        return np.concatenate(rows, axis=0) if rows else np.zeros(
            (1, self.extractor.feature_dim), dtype=np.float32
        )


def collate_batches(
    items: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length hand sequences and build a validity mask."""
    feats, labels = zip(*items)
    max_h = max(f.shape[0] for f in feats)
    feat_dim = feats[0].shape[1]
    batch = len(feats)

    padded = torch.zeros((batch, max_h, feat_dim), dtype=torch.float32)
    mask = torch.zeros((batch, max_h), dtype=torch.float32)
    for i, f in enumerate(feats):
        h = f.shape[0]
        padded[i, :h] = f
        mask[i, :h] = 1.0
    return padded, mask, torch.stack(labels)


def split_by_release(
    examples: Sequence[BatchExample],
    val_dates: Optional[Sequence[str]] = None,
    val_fraction: float = 0.2,
    seed: int = 44,
) -> Tuple[List[BatchExample], List[BatchExample]]:
    """Split examples into train/val (val_dates -> API split -> latest dates -> random)."""
    examples = list(examples)
    dates = sorted({ex.source_date for ex in examples})

    if not val_dates:
        splits = {(ex.split or "").lower() for ex in examples}
        if "validation" in splits and "train" in splits:
            train = [ex for ex in examples if (ex.split or "").lower() == "train"]
            val = [ex for ex in examples if (ex.split or "").lower() == "validation"]
            return train, val

    if val_dates:
        val_set = set(val_dates)
    elif len(dates) > 1:
        n_val = max(1, int(round(len(dates) * val_fraction)))
        val_set = set(dates[-n_val:])
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(examples))
        rng.shuffle(idx)
        cut = int(len(examples) * (1.0 - val_fraction))
        train_idx, val_idx = set(idx[:cut].tolist()), set(idx[cut:].tolist())
        train = [examples[i] for i in range(len(examples)) if i in train_idx]
        val = [examples[i] for i in range(len(examples)) if i in val_idx]
        return train, val

    train = [ex for ex in examples if ex.source_date not in val_set]
    val = [ex for ex in examples if ex.source_date in val_set]
    return train, val
