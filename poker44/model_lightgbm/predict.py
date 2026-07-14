"""Inference wrapper for the trained LightGBM detector.

Exposes ``predict(chunks) -> list[float]`` matching the miner-visible
``DetectionSynapse.chunks`` contract: a list of chunk groups (each a list of
hands). Returns one bot-risk score in [0, 1] per chunk group, in order.

The scoring pipeline mirrors the reference runtime (``poker44/inference_ref.py``):
``raw_model_scores -> calibrator -> score_remap -> score_logit ->
batch_safety_budget``. Every post-processing stage is driven by the artifact
``metadata`` and is a no-op when unconfigured, so a plain model emits raw
LightGBM probabilities unchanged (which keeps the ensemble blend calibrated).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from poker44.model_lightgbm.dataset import aggregate_matrix
from poker44.model_lightgbm.features import HandFeatureExtractor
from poker44.model_lightgbm.model import GBMModel

DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model"

# Decimal places for miner debug logs (raw / remap / final components).
SCORE_LOG_DECIMALS = 4


class Predictor:
    """Loads a trained LightGBM model and scores chunk groups.

    Post-processing config is read from ``GBMModel.metadata``:

    - ``score_logit_bias`` / ``score_logit_temperature``: logit shift+temperature.
    - ``calibrator``: optional param dict, e.g. ``{"kind": "platt", "a", "b"}``.
    - ``score_remap``: ``{"kind": "threshold_logit_v1", "threshold", "temperature"}``.
    - ``batch_safety_budget``: ``{"kind": "topk_v1", ...}`` per-batch positive cap.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        max_hands: int = 60,
    ):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_ARTIFACT)
        self.max_hands = max_hands
        self.extractor = HandFeatureExtractor()
        self.model: Optional[GBMModel] = None
        self.metadata: Dict[str, Any] = {}
        self.calibrator: Any = None
        self.score_logit_bias = 0.0
        self.score_logit_temperature = 1.0
        self.score_remap: Dict[str, Any] = {}
        self.model_weights: List[float] = [1.0]
        self._load()

    def _load(self) -> None:
        txt = self.checkpoint_path.with_suffix(".txt")
        if not txt.exists():
            raise FileNotFoundError(
                f"Model not found: {txt}. "
                "Train one with `python -m poker44.model_lightgbm.train`."
            )
        self.model = GBMModel.load(self.checkpoint_path)
        self.metadata = dict(self.model.metadata or {})
        self.calibrator = self.metadata.get("calibrator")
        self.score_logit_bias = float(self.metadata.get("score_logit_bias", 0.0) or 0.0)
        self.score_logit_temperature = max(
            float(self.metadata.get("score_logit_temperature", 1.0) or 1.0),
            1e-6,
        )
        score_remap = self.metadata.get("score_remap")
        self.score_remap = dict(score_remap) if isinstance(score_remap, dict) else {}
        self.model_weights = list(self.metadata.get("model_weights") or [1.0])

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-40.0, min(40.0, float(value)))
        return 1.0 / (1.0 + math.exp(-value))

    def _agg(self, hands: List[dict]) -> np.ndarray:
        feats = np.asarray(self.extractor.extract_batch(hands), dtype=np.float32)
        if feats.shape[0] > self.max_hands:
            feats = feats[: self.max_hands]
        return aggregate_matrix(feats)

    def _aligned_rows(self, chunks: List[List[dict]]) -> np.ndarray:
        return np.vstack([self._agg(chunk or []) for chunk in chunks])

    def _raw_model_scores(self, rows: np.ndarray) -> List[float]:
        probs = np.asarray(self.model.predict_proba(rows), dtype=float)
        return [self._clamp01(p) for p in probs]

    def _apply_calibrator(self, scores: List[float]) -> List[float]:
        if not scores or self.calibrator is None:
            return [self._clamp01(value) for value in scores]
        if isinstance(self.calibrator, dict) and self.calibrator.get("kind") == "platt":
            try:
                a = float(self.calibrator.get("a", 1.0))
                b = float(self.calibrator.get("b", 0.0))
            except (TypeError, ValueError):
                return [self._clamp01(value) for value in scores]
            output: List[float] = []
            for value in scores:
                clipped = max(1e-6, min(1.0 - 1e-6, float(value)))
                logit = math.log(clipped / (1.0 - clipped))
                output.append(self._clamp01(self._sigmoid(a * logit + b)))
            return output
        return [self._clamp01(value) for value in scores]

    def _apply_score_remap(self, scores: List[float]) -> List[float]:
        if not scores or not self.score_remap:
            return [self._clamp01(value) for value in scores]
        if self.score_remap.get("kind") != "threshold_logit_v1":
            return [self._clamp01(value) for value in scores]
        try:
            threshold = float(self.score_remap.get("threshold", 0.5))
            temperature = max(float(self.score_remap.get("temperature", 0.25)), 1e-6)
        except (TypeError, ValueError):
            return [self._clamp01(value) for value in scores]
        output: List[float] = []
        for value in scores:
            clipped = max(1e-6, min(1.0 - 1e-6, float(value)))
            adjusted = (clipped - threshold) / temperature
            output.append(self._clamp01(self._sigmoid(adjusted)))
        return output

    def _apply_score_logit(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        if (
            abs(self.score_logit_bias) < 1e-12
            and abs(self.score_logit_temperature - 1.0) < 1e-12
        ):
            return [self._clamp01(value) for value in scores]
        output: List[float] = []
        for score in scores:
            value = max(1e-6, min(1.0 - 1e-6, float(score)))
            logit = math.log(value / (1.0 - value))
            adjusted = (logit + self.score_logit_bias) / self.score_logit_temperature
            output.append(self._clamp01(self._sigmoid(adjusted)))
        return output

    def _apply_batch_safety_budget(self, scores: List[float]) -> List[float]:
        config = self.metadata.get("batch_safety_budget")
        if not scores or not isinstance(config, dict):
            return [self._clamp01(value) for value in scores]
        if config.get("kind") != "topk_v1":
            return [self._clamp01(value) for value in scores]

        count = len(scores)
        try:
            max_positive_count = int(config.get("max_positive_count", 1))
            max_positive_fraction = float(config.get("max_positive_fraction", 0.0) or 0.0)
            positive_floor = float(config.get("positive_floor", 0.501))
            positive_ceiling = float(config.get("positive_ceiling", 0.509))
            negative_ceiling = float(config.get("negative_ceiling", 0.49))
        except (TypeError, ValueError):
            return [self._clamp01(value) for value in scores]

        if max_positive_fraction > 0.0:
            max_positive_count = min(
                max_positive_count,
                max(1, int(math.floor(count * max_positive_fraction))),
            )
        max_positive_count = max(0, min(count, max_positive_count))
        positive_floor = self._clamp01(positive_floor)
        positive_ceiling = self._clamp01(max(positive_floor, positive_ceiling))
        negative_ceiling = min(self._clamp01(negative_ceiling), positive_floor - 1e-6)

        indexed_scores = [(index, self._clamp01(value)) for index, value in enumerate(scores)]
        ranked = sorted(indexed_scores, key=lambda item: item[1], reverse=True)
        output = [0.0 for _ in scores]

        positives = ranked[:max_positive_count]
        negatives = ranked[max_positive_count:]
        if positives:
            denom = max(1, len(positives) - 1)
            for rank, (index, _score) in enumerate(positives):
                relative = 1.0 - (rank / denom if denom else 0.0)
                output[index] = positive_floor + relative * (positive_ceiling - positive_floor)

        if negatives:
            negative_values = [score for _index, score in negatives]
            min_score = min(negative_values)
            max_score = max(negative_values)
            span = max(max_score - min_score, 1e-9)
            for index, score in negatives:
                relative = (score - min_score) / span
                output[index] = max(0.0, min(negative_ceiling, relative * negative_ceiling))

        return [round(self._clamp01(value), 6) for value in output]

    def predict_chunk_scores(self, chunks: List[List[dict]]) -> List[float]:
        """One post-processed bot-risk score per chunk group, order preserved."""
        if not chunks:
            return []
        rows = self._aligned_rows(chunks)
        raw_scores = self._raw_model_scores(rows)
        calibrated_scores = self._apply_calibrator(raw_scores)
        remapped_scores = self._apply_score_remap(calibrated_scores)
        logit_scores = self._apply_score_logit(remapped_scores)
        budgeted_scores = self._apply_batch_safety_budget(logit_scores)
        return [round(self._clamp01(value), 6) for value in budgeted_scores]

    def predict_chunk_score(self, chunk: List[dict]) -> float:
        scores = self.predict_chunk_scores([chunk])
        return scores[0] if scores else 0.5

    # Backwards-compatible aliases for the miner / ensemble contract.
    def predict(self, chunks: List[List[dict]]) -> List[float]:
        return self.predict_chunk_scores(chunks)

    def predict_one(self, hands: List[dict]) -> float:
        return self.predict_chunk_score(hands)

    def _round_score_log_values(self, scores: List[float]) -> List[float]:
        places = int(SCORE_LOG_DECIMALS)
        return [round(float(value), places) for value in scores]

    def debug_score_components(
        self,
        chunks: List[List[dict]],
    ) -> Dict[str, List[float]]:
        if not chunks:
            return {}
        rows = self._aligned_rows(chunks)
        raw_scores = self._raw_model_scores(rows)
        calibrated_scores = self._apply_calibrator(raw_scores)
        remapped_scores = self._apply_score_remap(calibrated_scores)
        logit_scores = self._apply_score_logit(remapped_scores)
        budgeted_scores = self._apply_batch_safety_budget(logit_scores)
        return {
            "raw_scores": self._round_score_log_values(raw_scores),
            "remapped_scores": self._round_score_log_values(remapped_scores),
            "final_scores": self._round_score_log_values(budgeted_scores),
        }

    def benchmark_latency(
        self,
        chunks: List[List[dict]],
        repeats: int = 5,
    ) -> Dict[str, float]:
        if not chunks:
            return {"latency_per_chunk_ms": 0.0, "total_latency_ms": 0.0}
        repeats = max(1, int(repeats))
        started = time.perf_counter()
        for _ in range(repeats):
            self.predict_chunk_scores(chunks)
        elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeats
        return {
            "latency_per_chunk_ms": elapsed_ms / max(len(chunks), 1),
            "total_latency_ms": elapsed_ms,
        }
