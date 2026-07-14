"""Poker44 AI model for hero-centric poker bot detection.

This package builds a trainable model that consumes benchmark chunk groups
(batches of poker hands) and predicts a bot-risk score in [0, 1] per batch,
matching the validator scoring contract.

Modules:
    data       - public benchmark API client with chunkHash caching
    features   - hero-centric per-hand feature extraction
    dataset    - torch Dataset / padded collate for variable-length batches
    model      - PyTorch encoder + attention pooling + classifier head
    metrics    - validator-aligned evaluation metrics
    train      - training entrypoint
    predict    - inference wrapper exposing predict(chunks) -> list[float]
"""

from poker44.model.features import HandFeatureExtractor, FEATURE_NAMES

__all__ = ["HandFeatureExtractor", "FEATURE_NAMES"]
