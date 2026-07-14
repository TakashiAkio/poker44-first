"""Inference wrapper for the calibrated ensemble.

Loads the blend manifest written by ``poker44.ensemble`` plus each available
component predictor (transformer, mlp, lightgbm), weighted-averages their
per-chunk probabilities, and applies Platt calibration.

Exposes ``predict(chunks) -> list[float]`` matching the miner contract. Missing
components are dropped and the remaining weights renormalized, so the ensemble
degrades gracefully to whatever checkpoints are present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "artifacts_ensemble" / "ensemble.json"

# Lazy import map: component name -> module path providing a ``Predictor``.
_COMPONENT_MODULES = {
    "transformer": "poker44.model_transformer.predict",
    "mlp": "poker44.model_mlp.predict",
    "lightgbm": "poker44.model_lightgbm.predict",
}


class EnsemblePredictor:
    """Weighted, calibrated blend of the three component predictors."""

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        device: Optional[str] = None,
        max_hands: int = 60,
    ):
        self.manifest_path = Path(manifest_path or DEFAULT_MANIFEST)
        self.device = device
        self.max_hands = max_hands
        self.weights: Dict[str, float] = {}
        self.platt: Dict[str, float] = {"a": 1.0, "b": 0.0}
        self.calibrated = False
        self.predictors: Dict[str, object] = {}
        self._load()

    def _load(self) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Ensemble manifest not found: {self.manifest_path}. "
                "Train one with `python -m poker44.ensemble`."
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.weights = dict(manifest.get("weights", {}))
        self.platt = dict(manifest.get("platt", {"a": 1.0, "b": 0.0}))
        self.calibrated = bool(manifest.get("calibrated", False))
        components = manifest.get("components") or list(self.weights.keys())

        import importlib

        for name in components:
            weight = float(self.weights.get(name, 0.0))
            if weight <= 0.0:
                continue
            module_path = _COMPONENT_MODULES.get(name)
            if module_path is None:
                continue
            try:
                module = importlib.import_module(module_path)
                predictor_cls = getattr(module, "Predictor")
                # LightGBM predictor has no device kwarg.
                if name == "lightgbm":
                    self.predictors[name] = predictor_cls(max_hands=self.max_hands)
                else:
                    self.predictors[name] = predictor_cls(
                        device=self.device, max_hands=self.max_hands
                    )
            except Exception as exc:  # noqa: BLE001 - skip unavailable components
                print(f"[ensemble] skipping component '{name}': {exc}")

        if not self.predictors:
            raise RuntimeError(
                "No ensemble components could be loaded; train component models first."
            )

    def _apply_platt(self, p: np.ndarray) -> np.ndarray:
        if not self.calibrated:
            return p
        p = np.clip(p, 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p))
        return 1.0 / (1.0 + np.exp(-(self.platt["a"] * z + self.platt["b"])))

    def predict(self, chunks: List[List[dict]]) -> List[float]:
        """One calibrated ensemble bot-risk score per chunk group, order preserved."""
        if not chunks:
            return []

        active = {n: self.predictors[n] for n in self.predictors}
        total_w = sum(self.weights.get(n, 0.0) for n in active) or 1.0

        blended = np.zeros(len(chunks), dtype=float)
        for name, predictor in active.items():
            probs = np.asarray(predictor.predict(chunks), dtype=float)
            blended += (self.weights.get(name, 0.0) / total_w) * probs

        blended = self._apply_platt(blended)
        return [float(np.clip(p, 0.0, 1.0)) for p in blended]

    def predict_one(self, hands: List[dict]) -> float:
        return self.predict([hands])[0]
