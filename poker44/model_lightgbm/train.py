"""Training entrypoint for the LightGBM bot detector.

    python -m poker44.model_lightgbm.train --num-releases 9

Also exposes ``train_lgbm(...)`` for programmatic use (e.g. compare.py). The
model is written to ``poker44/model_lightgbm/artifacts/model.txt`` (+ .json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

import lightgbm as lgb

from poker44.model_lightgbm.data import BatchExample, BenchmarkClient
from poker44.model_lightgbm.dataset import (
    aggregate_feature_names,
    build_dataset,
    split_by_release,
)
from poker44.model_lightgbm.features import HandFeatureExtractor
from poker44.model_lightgbm.metrics import evaluate, format_metrics
from poker44.model_lightgbm.model import GBMModel, default_params

DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model"


def fit_threshold_remap(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.05,
    temperature: float = 0.25,
) -> Optional[dict]:
    """Fit a ``threshold_logit_v1`` remap so bots cross 0.5 within an FPR budget.

    The threshold is placed at the raw-score quantile that keeps the human
    (negative) false-positive rate at or below ``target_fpr``; the predict-time
    remap then sigmoid-centers that threshold at 0.5.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    negatives = scores[labels == 0]
    if negatives.size == 0:
        return None
    target_fpr = min(max(float(target_fpr), 0.0), 1.0)
    threshold = float(np.quantile(negatives, 1.0 - target_fpr))
    threshold = min(max(threshold, 1e-6), 1.0 - 1e-6)
    return {
        "kind": "threshold_logit_v1",
        "threshold": threshold,
        "temperature": max(float(temperature), 1e-6),
    }


def resolve_dates(client, dates, num_releases):
    if dates:
        return list(dates)
    releases = client.releases(limit=max(num_releases, 1))
    resolved = [r.get("sourceDate") for r in releases if r.get("sourceDate")]
    if not resolved:
        resolved = [client.latest_source_date()]
    return sorted(set(resolved))[-num_releases:]


def train_lgbm(
    train_ex: Sequence[BatchExample],
    val_ex: Sequence[BatchExample],
    *,
    extractor: Optional[HandFeatureExtractor] = None,
    max_hands: int = 60,
    num_boost_round: int = 800,
    early_stopping_rounds: int = 50,
    params: Optional[dict] = None,
    score_logit_bias: float = 0.0,
    score_logit_temperature: float = 1.0,
    calibrate_threshold: bool = False,
    remap_target_fpr: float = 0.05,
    remap_temperature: float = 0.25,
    batch_safety_budget: Optional[dict] = None,
    verbose: bool = True,
) -> Tuple[GBMModel, Dict[str, float]]:
    """Train LightGBM on aggregated group features; return (model, val_metrics).

    Optional calibration is written to ``model.metadata`` and consumed at
    inference by ``Predictor``. Defaults leave metadata empty, so the raw
    probabilities are unchanged (keeping the ensemble blend calibrated).
    """
    extractor = extractor or HandFeatureExtractor()
    feat_names = aggregate_feature_names(extractor.feature_names)

    X_train, y_train = build_dataset(train_ex, extractor, max_hands)
    X_val, y_val = build_dataset(val_ex, extractor, max_hands)

    n_bot = int(y_train.sum())
    n_human = max(len(y_train) - n_bot, 1)
    params = params or default_params(scale_pos_weight=n_human / max(n_bot, 1))

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feat_names)
    valid_sets = [train_set]
    valid_names = ["train"]
    has_val = len(y_val) > 0 and len(np.unique(y_val)) > 1
    if has_val:
        valid_sets.append(lgb.Dataset(X_val, label=y_val, reference=train_set,
                                      feature_name=feat_names))
        valid_names.append("val")

    callbacks = [lgb.log_evaluation(period=50 if verbose else 0)]
    if has_val:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=verbose))

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    model = GBMModel(
        booster=booster,
        feature_names=feat_names,
        best_iteration=int(booster.best_iteration or 0),
    )

    eval_X, eval_y = (X_val, y_val) if len(y_val) else (X_train, y_train)
    eval_scores = model.predict_proba(eval_X)

    metadata: Dict[str, object] = {}
    if abs(float(score_logit_bias)) > 1e-12:
        metadata["score_logit_bias"] = float(score_logit_bias)
    if abs(float(score_logit_temperature) - 1.0) > 1e-12:
        metadata["score_logit_temperature"] = float(score_logit_temperature)
    if calibrate_threshold:
        remap = fit_threshold_remap(eval_scores, eval_y, remap_target_fpr, remap_temperature)
        if remap is not None:
            metadata["score_remap"] = remap
    if isinstance(batch_safety_budget, dict) and batch_safety_budget:
        metadata["batch_safety_budget"] = dict(batch_safety_budget)
    model.metadata = metadata

    val_metrics = evaluate(eval_scores, eval_y)
    return model, val_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Poker44 LightGBM detector")
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--val-dates", nargs="*", default=None)
    parser.add_argument("--num-releases", type=int, default=9)
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--max-hands", type=int, default=60)
    parser.add_argument("--score-logit-bias", type=float, default=0.0)
    parser.add_argument("--score-logit-temperature", type=float, default=1.0)
    parser.add_argument("--calibrate-threshold", action="store_true",
                        help="Fit a threshold_logit_v1 remap so bots cross 0.5 within the FPR budget.")
    parser.add_argument("--remap-target-fpr", type=float, default=0.05)
    parser.add_argument("--remap-temperature", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()

    client = BenchmarkClient()
    dates = resolve_dates(client, args.dates, args.num_releases)
    print(f"Loading benchmark releases: {dates}")
    examples = client.load_examples(dates, use_cache=not args.no_cache)
    print(f"Loaded {len(examples)} batch examples")
    if not examples:
        raise SystemExit("No examples loaded; check network/API availability.")

    train_ex, val_ex = split_by_release(examples, args.val_dates, args.val_fraction, args.seed)
    print(f"Train={len(train_ex)} (bots={sum(ex.label for ex in train_ex)}) Val={len(val_ex)}")

    model, val_metrics = train_lgbm(
        train_ex, val_ex,
        max_hands=args.max_hands,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        score_logit_bias=args.score_logit_bias,
        score_logit_temperature=args.score_logit_temperature,
        calibrate_threshold=args.calibrate_threshold,
        remap_target_fpr=args.remap_target_fpr,
        remap_temperature=args.remap_temperature,
    )
    print("Validation metrics:")
    print("  " + format_metrics(val_metrics))

    model.save(args.out)
    meta = {
        "train_dates": dates,
        "val_dates": sorted({ex.source_date for ex in val_ex}),
        "num_train": len(train_ex),
        "num_val": len(val_ex),
        "best_iteration": model.best_iteration,
        "metadata": model.metadata,
        "val_metrics": val_metrics,
    }
    args.out.with_name(args.out.name + "_run.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Saved model -> {args.out}.txt (+ .json)")
    print(json.dumps(val_metrics, indent=2))


if __name__ == "__main__":
    main()
