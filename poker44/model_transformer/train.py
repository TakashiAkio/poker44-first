"""Training entrypoint for the set-Transformer bot detector.

    python -m poker44.model_transformer.train --num-releases 9 --epochs 60 --augment

Also exposes ``train_transformer(...)`` for programmatic use (e.g. compare.py).
The checkpoint is written to ``poker44/model_transformer/artifacts/model.pt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from poker44.model_transformer.data import BatchExample, BenchmarkClient
from poker44.model_transformer.dataset import (
    BatchHandsDataset,
    Standardizer,
    collate_batches,
    split_by_release,
)
from poker44.model_transformer.features import HandFeatureExtractor
from poker44.model_transformer.metrics import evaluate, format_metrics
from poker44.model_transformer.model import BotDetector, ModelConfig

DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.pt"


def resolve_dates(client, dates, num_releases):
    if dates:
        return list(dates)
    releases = client.releases(limit=max(num_releases, 1))
    resolved = [r.get("sourceDate") for r in releases if r.get("sourceDate")]
    if not resolved:
        resolved = [client.latest_source_date()]
    return sorted(set(resolved))[-num_releases:]


def build_loaders(train_ex, val_ex, extractor, batch_size, max_hands,
                  num_workers=0, pin_memory=False, augment=False, seed=44):
    train_ds = BatchHandsDataset(
        train_ex, extractor, None, max_hands, augment=augment, seed=seed
    )
    standardizer = Standardizer.fit(train_ds.all_feature_rows())
    train_ds.standardizer = standardizer
    val_ds = BatchHandsDataset(val_ex, extractor, standardizer, max_hands)

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_batches,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, standardizer


def run_epoch(model, loader, device, optimizer=None, pos_weight=None):
    is_train = optimizer is not None
    model.train(is_train)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    non_blocking = device.type == "cuda"

    total_loss = 0.0
    scores: List[float] = []
    labels: List[float] = []
    for feats, mask, y in loader:
        feats = feats.to(device, non_blocking=non_blocking)
        mask = mask.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        with torch.set_grad_enabled(is_train):
            logits = model(feats, mask)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += float(loss.detach()) * y.size(0)
        scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
        labels.extend(y.detach().cpu().tolist())

    metrics = evaluate(np.asarray(scores), np.asarray(labels))
    metrics["loss"] = total_loss / max(len(labels), 1)
    return metrics


def train_transformer(
    train_ex: Sequence[BatchExample],
    val_ex: Sequence[BatchExample],
    *,
    feature_dim: int,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_hands: int = 60,
    device: Optional[torch.device] = None,
    augment: bool = True,
    seed: int = 44,
    config_overrides: Optional[dict] = None,
    num_workers: int = 0,
    verbose: bool = True,
) -> Tuple[BotDetector, Standardizer, Dict[str, float]]:
    """Train the Transformer and return (model, standardizer, best_val_metrics)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = HandFeatureExtractor()
    train_loader, val_loader, standardizer = build_loaders(
        train_ex, val_ex, extractor, batch_size, max_hands,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        augment=augment, seed=seed,
    )

    n_bot = sum(int(ex.label) for ex in train_ex)
    n_human = max(len(train_ex) - n_bot, 1)
    pos_weight = torch.tensor([n_human / max(n_bot, 1)], device=device)

    config = ModelConfig(feature_dim=feature_dim, **(config_overrides or {}))
    model = BotDetector(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_reward = -1.0
    best_state = None
    best_metrics: Dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, pos_weight)
        val_metrics = run_epoch(model, val_loader, device) if len(val_ex) else train_metrics
        scheduler.step()
        if verbose:
            print(
                f"[{epoch:03d}] train loss={train_metrics['loss']:.4f} "
                f"auc={train_metrics['roc_auc']:.3f} | "
                f"val reward={val_metrics['validator_reward']:.4f} "
                f"auc={val_metrics['roc_auc']:.3f} ap={val_metrics['ap_score']:.3f}"
            )
        if val_metrics["validator_reward"] >= best_reward:
            best_reward = val_metrics["validator_reward"]
            best_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, standardizer, best_metrics


def save_checkpoint(path, model, standardizer, extractor, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
            "standardizer": standardizer.to_dict(),
            "feature_names": extractor.feature_names,
            "meta": meta,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Poker44 set-Transformer")
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--val-dates", nargs="*", default=None)
    parser.add_argument("--num-releases", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-hands", type=int, default=60)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--augment", action="store_true", help="Hand-subsampling augmentation")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT)
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
    print(f"Train={len(train_ex)} (bots={sum(ex.label for ex in train_ex)}) Val={len(val_ex)}")

    extractor = HandFeatureExtractor()
    model, standardizer, best_metrics = train_transformer(
        train_ex, val_ex,
        feature_dim=extractor.feature_dim,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, max_hands=args.max_hands, device=device,
        augment=args.augment, seed=args.seed, num_workers=args.num_workers,
        config_overrides=dict(
            d_model=args.d_model, depth=args.depth, n_heads=args.n_heads,
            ff_mult=args.ff_mult, head_hidden=args.head_hidden, dropout=args.dropout,
        ),
    )
    print("Best validation metrics:")
    print("  " + format_metrics(best_metrics))

    save_checkpoint(args.out, model, standardizer, extractor, meta={
        "train_dates": dates,
        "val_dates": sorted({ex.source_date for ex in val_ex}),
        "num_train": len(train_ex),
        "num_val": len(val_ex),
        "best_val_metrics": best_metrics,
    })
    print(f"Saved checkpoint -> {args.out}")
    print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
