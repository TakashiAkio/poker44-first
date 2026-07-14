"""Train and compare all three bot-detection models on one validation split.

Loads the benchmark once, builds a single train/validation split (shared by all
models for a fair comparison), trains each model, and prints a table of
validator-aligned metrics on the validation set.

    python -m poker44.compare --num-releases 9 --epochs 60 --augment

Models compared:
  - transformer : poker44.model_transformer (masked set-Transformer)
  - mlp         : poker44.model_mlp          (Deep Sets MLP)
  - lightgbm    : poker44.model_lightgbm     (GBM on aggregated features)
"""

from __future__ import annotations

import argparse
from typing import Dict

import torch

from poker44.model_transformer.data import BenchmarkClient
from poker44.model_transformer.dataset import split_by_release
from poker44.model_transformer.features import HandFeatureExtractor
from poker44.model_transformer.train import resolve_dates, train_transformer
from poker44.model_mlp.train import train_mlp
from poker44.model_lightgbm.train import train_lgbm

_METRIC_COLUMNS = [
    "validator_reward",
    "roc_auc",
    "avg_precision",
    "bot_recall_at_fpr05",
    "brier_score",
    "log_loss",
]


def _print_table(results: Dict[str, Dict[str, float]]) -> None:
    name_w = max(len(n) for n in results) + 2
    header = "model".ljust(name_w) + "".join(c.rjust(20) for c in _METRIC_COLUMNS)
    print("\n" + header)
    print("-" * len(header))
    ranked = sorted(
        results.items(),
        key=lambda kv: kv[1].get("validator_reward", 0.0),
        reverse=True,
    )
    for name, m in ranked:
        row = name.ljust(name_w)
        for col in _METRIC_COLUMNS:
            row += f"{m.get(col, float('nan')):>20.4f}"
        print(row)
    print()
    best = ranked[0][0]
    print(f"Best by validator_reward: {best}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Poker44 detection models")
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--val-dates", nargs="*", default=None)
    parser.add_argument("--num-releases", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=60, help="Torch models epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-hands", type=int, default=60)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--augment", action="store_true",
                        help="Hand-subsampling augmentation for torch models")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--models", nargs="*", default=["transformer", "mlp", "lightgbm"],
                        choices=["transformer", "mlp", "lightgbm"])
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

    feature_dim = HandFeatureExtractor().feature_dim
    results: Dict[str, Dict[str, float]] = {}

    if "transformer" in args.models:
        print("\n=== Training Transformer ===")
        _, _, m = train_transformer(
            train_ex, val_ex, feature_dim=feature_dim, epochs=args.epochs,
            batch_size=args.batch_size, max_hands=args.max_hands, device=device,
            augment=args.augment, seed=args.seed, num_workers=args.num_workers,
            verbose=True,
        )
        results["transformer"] = m

    if "mlp" in args.models:
        print("\n=== Training Deep Sets MLP ===")
        _, _, m = train_mlp(
            train_ex, val_ex, feature_dim=feature_dim, epochs=args.epochs,
            batch_size=args.batch_size, max_hands=args.max_hands, device=device,
            augment=args.augment, seed=args.seed, num_workers=args.num_workers,
            verbose=True,
        )
        results["mlp"] = m

    if "lightgbm" in args.models:
        print("\n=== Training LightGBM ===")
        _, m = train_lgbm(
            train_ex, val_ex, max_hands=args.max_hands, verbose=True,
        )
        results["lightgbm"] = m

    print("\n================ VALIDATION COMPARISON ================")
    _print_table(results)


if __name__ == "__main__":
    main()
