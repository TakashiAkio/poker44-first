"""LightGBM bot-detector wrapper.

Holds a trained ``lightgbm.Booster`` plus the aggregated feature names, and
provides save/load and probability prediction on aggregated group vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import lightgbm as lgb


def default_params(scale_pos_weight: float = 1.0) -> dict:
    """Conservative LightGBM params for the small (~1.3k) benchmark dataset."""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": 44,
    }


@dataclass
class GBMModel:
    """Trained LightGBM booster + metadata for aggregated group features.

    ``metadata`` carries the JSON-safe inference calibration config consumed by
    ``poker44.model_lightgbm.predict.Predictor`` (score-logit, score-remap,
    batch-safety-budget, and an optional param-based calibrator). All fields are
    optional; when absent the predictor emits raw probabilities unchanged.
    """

    booster: lgb.Booster
    feature_names: List[str] = field(default_factory=list)
    best_iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        num_it = self.best_iteration or None
        preds = self.booster.predict(X, num_iteration=num_it)
        return np.asarray(preds, dtype=float)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(
            str(path.with_suffix(".txt")),
            num_iteration=self.best_iteration or None,
        )
        meta = {
            "feature_names": self.feature_names,
            "best_iteration": self.best_iteration,
            "metadata": self.metadata,
        }
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GBMModel":
        path = Path(path)
        booster = lgb.Booster(model_file=str(path.with_suffix(".txt")))
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(
            booster=booster,
            feature_names=meta.get("feature_names", []),
            best_iteration=int(meta.get("best_iteration", 0)),
            metadata=dict(meta.get("metadata") or {}),
        )
