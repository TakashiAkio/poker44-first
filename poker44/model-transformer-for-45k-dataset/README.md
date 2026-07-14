# Poker44 Bot-Detection Model

A PyTorch model that predicts a bot-risk score in `[0, 1]` per chunk group,
matching the validator scoring contract (`DetectionSynapse.risk_scores`).

## Task framing

Each benchmark chunk object exposes a `chunks` field that is a list of
**batches** (each a list of poker hands) and a `groundTruth` list with one
`0/1` label per batch. Every hand carries `metadata.hero_seat` — the subject
player being judged bot vs human. Hole cards and board cards are obfuscated, so
the model relies purely on **hero behavior**: action types, bet sizings (bb),
pot geometry, street depth, aggression and fold/continuation patterns.

## Architecture

A masked, permutation-invariant **set-Transformer over hands**:

```
per-hand hero features [B, H, F]
  -> input projection               -> [B, H, D]
  -> N x TransformerEncoder layers   -> [B, H, D]  (self-attention, NO
     (padding-masked, no pos-enc)                   positional encoding)
  -> masked attention pooling        -> [B, D]
  + masked mean pooling              -> [B, D]
  -> classifier head                 -> [B, 1] logit -> sigmoid risk
```

Hands are an unordered set, so positional encodings are omitted and a
key-padding mask handles variable hand counts. Self-attention lets hands attend
to each other, capturing the cross-hand consistency (repeated sizings/patterns)
that betrays a bot. Size is configurable via `ModelConfig`
(`d_model`, `depth`, `n_heads`, `ff_mult`, `head_hidden`, `dropout`); defaults
are small (~180K params) to suit ~45K v1.13 examples and can be scaled up as the
dataset grows.

## Modules

- `data.py` — public benchmark API client, caches raw chunks by date.
- `features.py` — hero-centric per-hand feature extraction (`FEATURE_NAMES`).
- `dataset.py` — torch dataset, padded collate, standardization, release split.
- `model.py` — `BotDetector` network + `ModelConfig`.
- `metrics.py` — validator-aligned metrics (reward, AP, recall@FPR, AUC, logloss).
- `train.py` — training entrypoint with checkpointing.
- `predict.py` — `Predictor.predict(chunks) -> list[float]` for the miner.

## Train

```bash
pip install -r requirements.txt
python -m poker44.model.train --num-releases 8 --epochs 40
```

Options:

- `--dates 2026-07-10 2026-07-11` train on explicit dates.
- `--val-dates 2026-07-13` hold out explicit dates (avoids single-date overfit).
- `--out path/to/checkpoint.pt` checkpoint destination.

The checkpoint stores weights, model config, the fitted standardizer, and
feature names.

## Predict

```python
from poker44.model.predict import Predictor

predictor = Predictor()  # loads artifacts/bot_detector.pt
scores = predictor.predict(synapse.chunks)  # list[float], one per chunk group
```

## Notes

- Training uses per-release train/validation separation to catch overfitting to
  a single benchmark date.
- Class imbalance is handled with a positive-class weight in `BCEWithLogitsLoss`.
- Checkpoint selection is by best **validator reward** on validation, not raw
  accuracy, so calibration around the `0.5` threshold is preserved.
