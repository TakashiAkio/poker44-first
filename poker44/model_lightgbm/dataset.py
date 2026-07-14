"""Group-level feature aggregation and release splitting for the LightGBM model.

Tree ensembles need a fixed-length vector per example, but a chunk group has a
variable number of hands. We therefore aggregate the per-hand hero features of
a group into order-invariant summary statistics (mean/std/min/max per feature)
plus a normalized hand count. This mirrors the pooling the neural models do,
but in a form LightGBM can consume directly.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from poker44.model_lightgbm.data import BatchExample
from poker44.model_lightgbm.features import HandFeatureExtractor

_AGG_STATS = ("mean", "std", "min", "max")
_MAX_HANDS_NORM = 60.0


def aggregate_feature_names(base_names: Sequence[str]) -> List[str]:
    names = [f"{stat}__{b}" for stat in _AGG_STATS for b in base_names]
    names.append("n_hands_norm")
    return names


def aggregate_matrix(feats: np.ndarray) -> np.ndarray:
    """Turn a group's per-hand matrix [H, F] into one summary vector [4F+1]."""
    if feats.size == 0:
        feats = np.zeros((1, feats.shape[1] if feats.ndim == 2 else 1), dtype=np.float32)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    fmin = feats.min(axis=0)
    fmax = feats.max(axis=0)
    n_hands_norm = np.array([min(feats.shape[0] / _MAX_HANDS_NORM, 1.0)], dtype=np.float32)
    return np.concatenate([mean, std, fmin, fmax, n_hands_norm]).astype(np.float32)


def build_dataset(
    examples: Sequence[BatchExample],
    extractor: Optional[HandFeatureExtractor] = None,
    max_hands: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X [N, 4F+1], y [N]) aggregated feature matrix and labels."""
    extractor = extractor or HandFeatureExtractor()
    rows: List[np.ndarray] = []
    labels: List[int] = []
    for ex in examples:
        feats = np.asarray(extractor.extract_batch(ex.hands), dtype=np.float32)
        if feats.shape[0] > max_hands:
            feats = feats[:max_hands]
        rows.append(aggregate_matrix(feats))
        labels.append(int(ex.label))
    X = np.vstack(rows) if rows else np.zeros((0, 4 * extractor.feature_dim + 1), np.float32)
    return X, np.asarray(labels, dtype=np.int32)


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
