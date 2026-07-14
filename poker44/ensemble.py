"""Train an accuracy-maximizing ensemble of the three detectors.

Strategy for the small (~1.3k) v1.13 benchmark, where any single model is
noisy:

1. Multi-seed training of the two torch models (transformer, mlp) -- keep the
   best seed per type by validator reward. LightGBM is deterministic.
2. Blend the per-model validation probabilities with convex weights chosen to
   maximize the validator reward (coarse simplex grid search).
3. Calibrate the blended probability with Platt scaling so scores are usable at
   the 0.5 decision threshold (the reward gates on this) and log-loss/Brier
   improve.

Artifacts written so the miner's ``EnsemblePredictor`` can reload the exact
blend:

- best transformer checkpoint  -> poker44/model_transformer/artifacts/model.pt
- best mlp checkpoint           -> poker44/model_mlp/artifacts/model.pt
- lightgbm model                -> poker44/model_lightgbm/artifacts/model.txt
- blend weights + calibration   -> poker44/artifacts_ensemble/ensemble.json

Usage:

    python -m poker44.ensemble --num-releases 9 --augment --device cuda \
        --epochs 60 --seeds 44 7 123

Note on calibration: Platt is fit on the same validation split used for
reporting, so the calibrated reward is a mildly optimistic estimate. For a fully
unbiased number, hold out a dedicated calibration date via --val-dates and keep
a separate test date.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from poker44.model_transformer.data import BenchmarkClient
from poker44.model_transformer.dataset import split_by_release
from poker44.model_transformer.features import HandFeatureExtractor
from poker44.model_transformer.metrics import evaluate
from poker44.model_transformer.train import resolve_dates, train_transformer
from poker44.model_transformer.predict import DEFAULT_ARTIFACT as TF_ARTIFACT
from poker44.model_mlp.train import train_mlp
from poker44.model_mlp.predict import DEFAULT_ARTIFACT as MLP_ARTIFACT
from poker44.model_lightgbm.train import train_lgbm
from poker44.model_lightgbm.dataset import build_dataset as lgb_build_dataset
from poker44.model_lightgbm.predict import DEFAULT_ARTIFACT as LGB_ARTIFACT

ENSEMBLE_DIR = Path(__file__).resolve().parent / "artifacts_ensemble"
ENSEMBLE_JSON = ENSEMBLE_DIR / "ensemble.json"

_METRIC_COLUMNS = [
    "validator_reward",
    "roc_auc",
    "avg_precision",
    "bot_recall_at_fpr05",
    "brier_score",
    "log_loss",
]


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def _score_torch(
    model,
    standardizer,
    examples,
    extractor: HandFeatureExtractor,
    device: torch.device,
    max_hands: int = 60,
    batch_size: int = 128,
) -> np.ndarray:
    """Return per-example bot probabilities for a trained torch model."""
    model.eval()
    fd = extractor.feature_dim
    rows: List[np.ndarray] = []
    for ex in examples:
        feats = np.asarray(extractor.extract_batch(ex.hands), dtype=np.float32)
        if feats.shape[0] > max_hands:
            feats = feats[:max_hands]
        rows.append(standardizer.transform(feats).astype(np.float32))

    probs: List[float] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        max_h = max(t.shape[0] for t in batch)
        padded = np.zeros((len(batch), max_h, fd), dtype=np.float32)
        mask = np.zeros((len(batch), max_h), dtype=np.float32)
        for j, t in enumerate(batch):
            h = t.shape[0]
            padded[j, :h] = t
            mask[j, :h] = 1.0
        pt = torch.from_numpy(padded).to(device)
        mt = torch.from_numpy(mask).to(device)
        probs.extend(model.predict_proba(pt, mt).cpu().numpy().tolist())
    return np.asarray(probs, dtype=float)


def _score_lgbm(model, examples, max_hands: int = 60) -> np.ndarray:
    X, _ = lgb_build_dataset(examples, None, max_hands)
    return np.asarray(model.predict_proba(X), dtype=float)


# --------------------------------------------------------------------------- #
# Calibration (Platt scaling on the logit of the blended probability)
# --------------------------------------------------------------------------- #
def _to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fit p' = sigmoid(a * logit(p) + b) by minimizing BCE (LBFGS)."""
    if len(np.unique(y)) < 2:
        return {"a": 1.0, "b": 0.0}
    z = torch.tensor(_to_logit(p), dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=200)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(a * z + b, yt)
        loss.backward()
        return loss

    opt.step(closure)
    return {"a": float(a.detach()), "b": float(b.detach())}


def apply_platt(p: np.ndarray, calib: Dict[str, float]) -> np.ndarray:
    z = _to_logit(np.asarray(p, dtype=float))
    return 1.0 / (1.0 + np.exp(-(calib["a"] * z + calib["b"])))


# --------------------------------------------------------------------------- #
# Blend-weight search
# --------------------------------------------------------------------------- #
def search_weights(
    prob_map: Dict[str, np.ndarray], y: np.ndarray, step: float = 0.1
) -> Dict[str, float]:
    """Coarse simplex grid search for convex weights maximizing reward."""
    keys = list(prob_map.keys())
    grid = [round(i * step, 4) for i in range(int(round(1 / step)) + 1)]
    best_reward = -1.0
    best_w = {k: (1.0 if i == 0 else 0.0) for i, k in enumerate(keys)}
    for combo in product(grid, repeat=len(keys)):
        if abs(sum(combo) - 1.0) > 1e-9:
            continue
        blended = sum(w * prob_map[k] for w, k in zip(combo, keys))
        reward = evaluate(blended, y)["validator_reward"]
        if reward > best_reward:
            best_reward = reward
            best_w = dict(zip(keys, [float(c) for c in combo]))
    return best_w


def blend(prob_map: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    total = sum(weights.get(k, 0.0) for k in prob_map)
    if total <= 0:
        total = 1.0
    return sum((weights.get(k, 0.0) / total) * v for k, v in prob_map.items())


# --------------------------------------------------------------------------- #
# Checkpoint saving (schema matches each package's Predictor loader)
# --------------------------------------------------------------------------- #
def _save_torch_ckpt(path: Path, model, standardizer, feature_names, meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
            "standardizer": standardizer.to_dict(),
            "feature_names": feature_names,
            "meta": meta,
        },
        path,
    )


def _print_table(results: Dict[str, Dict[str, float]]) -> None:
    name_w = max(len(n) for n in results) + 2
    header = "model".ljust(name_w) + "".join(c.rjust(20) for c in _METRIC_COLUMNS)
    print("\n" + header)
    print("-" * len(header))
    ranked = sorted(
        results.items(), key=lambda kv: kv[1].get("validator_reward", 0.0), reverse=True
    )
    for name, m in ranked:
        row = name.ljust(name_w)
        for col in _METRIC_COLUMNS:
            row += f"{m.get(col, float('nan')):>20.4f}"
        print(row)
    print(f"\nBest by validator_reward: {ranked[0][0]}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train Poker44 ensemble")
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--val-dates", nargs="*", default=None)
    parser.add_argument("--num-releases", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-hands", type=int, default=60)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44, help="Split determinism seed")
    parser.add_argument("--seeds", nargs="*", type=int, default=[44, 7, 123],
                        help="Seeds for multi-seed torch training (best kept per type)")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--models", nargs="*", default=["transformer", "mlp", "lightgbm"],
                        choices=["transformer", "mlp", "lightgbm"])
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available; use --device cpu.")
        device = torch.device(args.device)
    print(f"Using device: {device}")

    client = BenchmarkClient()
    dates = resolve_dates(client, args.dates, args.num_releases)
    print(f"Loading benchmark releases: {dates}")
    examples = client.load_examples(dates, use_cache=not args.no_cache)
    print(f"Loaded {len(examples)} batch examples")
    if not examples:
        raise SystemExit("No examples loaded; check network/API availability.")

    train_ex, val_ex = split_by_release(examples, args.val_dates, args.val_fraction, args.seed)
    n_bot = sum(int(ex.label) for ex in train_ex)
    print(f"Shared split -> Train={len(train_ex)} (bots={n_bot}) Val={len(val_ex)}")
    if len(val_ex) == 0:
        raise SystemExit("Validation split is empty; provide more dates or --val-dates.")

    extractor = HandFeatureExtractor()
    feature_dim = extractor.feature_dim
    y_val = np.asarray([int(ex.label) for ex in val_ex], dtype=int)

    results: Dict[str, Dict[str, float]] = {}
    val_probs: Dict[str, np.ndarray] = {}

    # ---- Transformer (multi-seed) ---------------------------------------- #
    if "transformer" in args.models:
        best = None  # (reward, probs, model, standardizer)
        for seed in args.seeds:
            print(f"\n=== Training Transformer (seed={seed}) ===")
            model, std, _ = train_transformer(
                train_ex, val_ex, feature_dim=feature_dim, epochs=args.epochs,
                batch_size=args.batch_size, max_hands=args.max_hands, device=device,
                augment=args.augment, seed=seed, num_workers=args.num_workers, verbose=True,
            )
            probs = _score_torch(model, std, val_ex, extractor, device, args.max_hands)
            reward = evaluate(probs, y_val)["validator_reward"]
            print(f"  seed={seed} val reward={reward:.4f}")
            if best is None or reward > best[0]:
                best = (reward, probs, model, std)
        val_probs["transformer"] = best[1]
        results["transformer"] = evaluate(best[1], y_val)
        _save_torch_ckpt(TF_ARTIFACT, best[2], best[3], extractor.feature_names,
                         meta={"component": "transformer", "val_dates": args.val_dates})
        print(f"Saved best transformer -> {TF_ARTIFACT}")

    # ---- MLP (multi-seed) ------------------------------------------------- #
    if "mlp" in args.models:
        best = None
        for seed in args.seeds:
            print(f"\n=== Training Deep Sets MLP (seed={seed}) ===")
            model, std, _ = train_mlp(
                train_ex, val_ex, feature_dim=feature_dim, epochs=args.epochs,
                batch_size=args.batch_size, max_hands=args.max_hands, device=device,
                augment=args.augment, seed=seed, num_workers=args.num_workers, verbose=True,
            )
            probs = _score_torch(model, std, val_ex, extractor, device, args.max_hands)
            reward = evaluate(probs, y_val)["validator_reward"]
            print(f"  seed={seed} val reward={reward:.4f}")
            if best is None or reward > best[0]:
                best = (reward, probs, model, std)
        val_probs["mlp"] = best[1]
        results["mlp"] = evaluate(best[1], y_val)
        _save_torch_ckpt(MLP_ARTIFACT, best[2], best[3], extractor.feature_names,
                         meta={"component": "mlp", "val_dates": args.val_dates})
        print(f"Saved best mlp -> {MLP_ARTIFACT}")

    # ---- LightGBM --------------------------------------------------------- #
    if "lightgbm" in args.models:
        print("\n=== Training LightGBM ===")
        model, _ = train_lgbm(train_ex, val_ex, max_hands=args.max_hands, verbose=True)
        probs = _score_lgbm(model, val_ex, args.max_hands)
        val_probs["lightgbm"] = probs
        results["lightgbm"] = evaluate(probs, y_val)
        model.save(LGB_ARTIFACT)
        print(f"Saved lightgbm -> {LGB_ARTIFACT}.txt")

    # ---- Ensemble blend + calibration ------------------------------------ #
    weights = search_weights(val_probs, y_val)
    ens_raw = blend(val_probs, weights)
    results["ensemble"] = evaluate(ens_raw, y_val)
    print(f"\nBlend weights: {weights}")

    calib = {"a": 1.0, "b": 0.0}
    if not args.no_calibrate:
        calib = fit_platt(ens_raw, y_val)
        ens_cal = apply_platt(ens_raw, calib)
        results["ensemble_calibrated"] = evaluate(ens_cal, y_val)
        print(f"Platt calibration: {calib}")

    print("\n================ VALIDATION COMPARISON ================")
    _print_table(results)

    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "components": list(val_probs.keys()),
        "weights": weights,
        "platt": calib,
        "calibrated": not args.no_calibrate,
        "train_dates": dates,
        "val_dates": sorted({ex.source_date for ex in val_ex}),
        "num_train": len(train_ex),
        "num_val": len(val_ex),
        "seeds": args.seeds,
        "reward_ensemble_raw": results["ensemble"]["validator_reward"],
        "reward_ensemble_calibrated": results.get("ensemble_calibrated", {}).get(
            "validator_reward"
        ),
    }
    ENSEMBLE_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved ensemble manifest -> {ENSEMBLE_JSON}")


if __name__ == "__main__":
    main()
