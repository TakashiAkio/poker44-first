"""Validator-aligned evaluation metrics (self-contained copy).

Wraps the subnet's own reward function (``poker44.score.scoring.reward``) so the
model is evaluated with the same objective validators use, plus standard
diagnostics (ROC AUC, average precision, log loss, Brier score).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from poker44.score.scoring import reward


def evaluate(y_score: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute validator reward plus diagnostic metrics."""
    y_score = np.asarray(y_score, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    metrics: Dict[str, float] = {}
    has_both = np.any(y_true == 1) and np.any(y_true == 0)

    rew, detail = reward(y_score, y_true)
    metrics["validator_reward"] = float(rew)
    metrics["ap_score"] = float(detail.get("ap_score", 0.0))
    metrics["bot_recall_at_fpr05"] = float(detail.get("bot_recall", 0.0))
    metrics["human_safety"] = float(detail.get("human_safety_penalty", 0.0))
    metrics["hard_fpr"] = float(detail.get("hard_fpr", 0.0))

    metrics["brier_score"] = float(
        brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0), pos_label=1)
    )

    if has_both:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["avg_precision"] = float(average_precision_score(y_true, y_score))
        eps = 1e-7
        clipped = np.clip(y_score, eps, 1 - eps)
        metrics["log_loss"] = float(log_loss(y_true, clipped, labels=[0, 1]))
    else:
        metrics["roc_auc"] = 0.0
        metrics["avg_precision"] = 0.0
        metrics["log_loss"] = 0.0

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    return " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
