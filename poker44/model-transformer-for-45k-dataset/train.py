"""Training entrypoint for the Poker44 bot-detection model.

Usage examples:

    # Train on the latest N releases, validate on the most recent dates
    python -m poker44.model.train --num-releases 8 --epochs 40

    # Train on explicit dates
    python -m poker44.model.train --dates 2026-07-10 2026-07-11 --val-dates 2026-07-12

The trained checkpoint (weights + model config + standardizer + feature names)
is written to ``poker44/model/artifacts/bot_detector.pt`` by default and can be
loaded by ``poker44.model.predict.Predictor``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from poker44.model.data import BatchExample, BenchmarkClient
from poker44.model.dataset import (
    BatchHandsDataset,
    Standardizer,
    collate_batches,
    split_by_release,
)
from poker44.model.features import HandFeatureExtractor
from poker44.model.metrics import evaluate, format_metrics
from poker44.model.model import BotDetector, ModelConfig

DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "bot_detector.pt"


def resolve_dates(
    client: BenchmarkClient,
    dates: Optional[Sequence[str]],
    num_releases: int,
) -> List[str]:
    if dates:
        return list(dates)
    releases = client.releases(limit=max(num_releases, 1))
    resolved = [r.get("sourceDate") for r in releases if r.get("sourceDate")]
    if not resolved:
        resolved = [client.latest_source_date()]
    return sorted(set(resolved))[-num_releases:]


def build_loaders(
    train_ex: Sequence[BatchExample],
    val_ex: Sequence[BatchExample],
    extractor: HandFeatureExtractor,
    batch_size: int,
    max_hands: int,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    train_ds = BatchHandsDataset(train_ex, extractor, None, max_hands)
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
        total_loss += float(loss) * y.size(0)
        scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
        labels.extend(y.detach().cpu().tolist())

    avg_loss = total_loss / max(len(labels), 1)
    metrics = evaluate(np.asarray(scores), np.asarray(labels))
    metrics["loss"] = avg_loss
    return metrics


def save_checkpoint(path: Path, model: BotDetector, standardizer: Standardizer,
                    extractor: HandFeatureExtractor, meta: dict) -> None:
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
    parser = argparse.ArgumentParser(description="Train Poker44 bot detector")
    parser.add_argument("--dates", nargs="*", default=None,
                        help="Explicit source dates (YYYY-MM-DD) to train on")
    parser.add_argument("--val-dates", nargs="*", default=None,
                        help="Explicit source dates to hold out for validation")
    parser.add_argument("--num-releases", type=int, default=8,
                        help="Number of latest releases to use when dates omitted")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # Set-Transformer size (scale up as the dataset grows)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--depth", type=int, default=2, help="Transformer layers")
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--head-hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-hands", type=int, default=60)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Compute device; 'auto' uses CUDA when available")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes")
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit(
                "CUDA requested but not available. Install a CUDA build of torch "
                "or run with --device cpu."
            )
        device = torch.device(args.device)
    print(f"Using device: {device}")

    client = BenchmarkClient()
    dates = resolve_dates(client, args.dates, args.num_releases)
    print(f"Loading benchmark releases: {dates}")
    examples = client.load_examples(dates, use_cache=not args.no_cache)
    print(f"Loaded {len(examples)} batch examples")
    if not examples:
        raise SystemExit("No examples loaded; check network/API availability.")

    train_ex, val_ex = split_by_release(
        examples, args.val_dates, args.val_fraction, args.seed
    )
    n_bot = sum(ex.label for ex in train_ex)
    print(f"Train={len(train_ex)} (bots={n_bot}) Val={len(val_ex)}")

    extractor = HandFeatureExtractor()
    train_loader, val_loader, standardizer = build_loaders(
        train_ex, val_ex, extractor, args.batch_size, args.max_hands,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    n_human = max(len(train_ex) - n_bot, 1)
    pos_weight = torch.tensor([n_human / max(n_bot, 1)], device=device)

    config = ModelConfig(
        feature_dim=extractor.feature_dim,
        d_model=args.d_model,
        depth=args.depth,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
    )
    model = BotDetector(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_reward = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, pos_weight)
        val_metrics = (
            run_epoch(model, val_loader, device)
            if len(val_ex) > 0
            else train_metrics
        )
        scheduler.step()
        print(
            f"[{epoch:03d}] train loss={train_metrics['loss']:.4f} "
            f"auc={train_metrics['roc_auc']:.3f} | "
            f"val reward={val_metrics['validator_reward']:.4f} "
            f"auc={val_metrics['roc_auc']:.3f} ap={val_metrics['ap_score']:.3f}"
        )
        if val_metrics["validator_reward"] >= best_reward:
            best_reward = val_metrics["validator_reward"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = run_epoch(model, val_loader, device) if len(val_ex) else train_metrics
    print("Best validation metrics:")
    print("  " + format_metrics(final_val))

    save_checkpoint(
        args.out,
        model,
        standardizer,
        extractor,
        meta={
            "train_dates": dates,
            "val_dates": sorted({ex.source_date for ex in val_ex}),
            "num_train": len(train_ex),
            "num_val": len(val_ex),
            "best_val_reward": best_reward,
            "final_val_metrics": final_val,
        },
    )
    print(f"Saved checkpoint -> {args.out}")
    print(json.dumps(final_val, indent=2))


if __name__ == "__main__":
    main()
