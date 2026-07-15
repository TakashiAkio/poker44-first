"""Poker44 miner: uses the trained set-Transformer model with a heuristic fallback."""

# from __future__ import annotations

import hashlib
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    """
    Reference heuristic miner.

    It aggregates simple behavior signals over each chunk and returns a bot-risk
    score per chunk. The goal is not SOTA accuracy, but a deterministic and
    explainable baseline that is meaningfully better than random.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        repo_root = Path(__file__).resolve().parents[1]

        # Try to load the trained model; fall back to the heuristic otherwise.
        self.model_artifact_path: Optional[Path] = None
        self.predictor = self._load_predictor()

        # Provenance shared by every manifest variant.
        runtime_commit = self._repo_head(repo_root)
        runtime_repo_url = self._normalize_repo_url(self._repo_url(repo_root)) or (
            "https://github.com/Poker44/Poker44-subnet"
        )
        artifact_url = ""
        artifact_sha256 = ""
        if self.predictor is not None and self.model_artifact_path is not None:
            artifact_path = self.model_artifact_path.resolve()
            artifact_url = str(artifact_path)
            if artifact_path.is_file():
                artifact_sha256 = self._sha256_file(artifact_path)

        if self.predictor is not None:
            bt.logging.info("Poker44 Miner started with trained set-Transformer model")
            manifest_defaults = {
                "model_name": "poker44-set-transformer",
                "model_version": "1",
                "framework": "pytorch",
                "license": "MIT",
                "repo_url": runtime_repo_url,
                "repo_commit": runtime_commit,
                "artifact_url": artifact_url,
                "artifact_sha256": artifact_sha256,
                "notes": (
                    "Hero-centric set-Transformer over hands, trained on the public "
                    "Poker44 training benchmark. Falls back to a heuristic if the "
                    "model checkpoint is unavailable."
                ),
                "open_source": True,
                "inference_mode": "local",
                "training_data_statement": (
                    "Trained only on the public Poker44 training benchmark chunks "
                    "(labeled groundTruth). No validator-only evaluation data used."
                ),
                "training_data_sources": ["poker44-public-training-benchmark"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
            }
        else:
            bt.logging.info("Heuristic Poker44 Miner started (no model checkpoint found)")
            manifest_defaults = {
                "model_name": "poker44-reference-heuristic",
                "model_version": "1",
                "framework": "python-heuristic",
                "license": "MIT",
                "repo_url": runtime_repo_url,
                "repo_commit": runtime_commit,
                "notes": "Reference heuristic miner shipped with the Poker44 subnet.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Reference heuristic miner. No training step. Uses only runtime chunk features."
                ),
                "training_data_sources": ["none"],
                "private_data_attestation": (
                    "This reference miner does not train on validator-only evaluation data."
                ),
            }
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=self._implementation_files(
                repo_root, has_predictor=self.predictor is not None
            ),
            defaults=manifest_defaults,
        )
        if artifact_sha256 and self.model_artifact_path is not None:
            self.model_manifest["artifact_basename"] = self.model_artifact_path.name
            self.model_manifest["artifact_sha256"] = artifact_sha256
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        
        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    @staticmethod
    def _normalize_repo_url(url: str) -> str:
        cleaned = str(url or "").strip()
        if not cleaned:
            return ""
        if cleaned.startswith("git@"):
            host_path = cleaned.split(":", 1)
            if len(host_path) == 2:
                host = host_path[0][4:]
                path = host_path[1]
                if path.endswith(".git"):
                    path = path[:-4]
                return f"https://{host}/{path}"
        if cleaned.endswith(".git"):
            cleaned = cleaned[:-4]
        return cleaned

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _repo_head(repo_root: Path) -> str:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return ""

    @staticmethod
    def _repo_url(repo_root: Path) -> str:
        try:
            completed = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return ""

    @classmethod
    def _implementation_files(cls, repo_root: Path, *, has_predictor: bool) -> List[Path]:
        """Files whose contents are hashed into the manifest for transparency."""
        files = [Path(__file__).resolve()]
        if not has_predictor:
            return files
        for relative in (
            "poker44/model_transformer/predict.py",
            "poker44/model_transformer/model.py",
            "poker44/model_transformer/features.py",
            "poker44/model_transformer/dataset.py",
        ):
            candidate = repo_root / relative
            if candidate.exists():
                files.append(candidate.resolve())
        return files

    def _load_predictor(self):
        """Load the best available predictor.

        Preference order: single set-Transformer checkpoint -> heuristic
        fallback (returns None). Any failure degrades to the heuristic so
        model loading never crashes the miner.
        """
        ckpt = os.environ.get("POKER44_MODEL_CKPT")
        try:
            from poker44.model_transformer.predict import Predictor, DEFAULT_ARTIFACT

            ckpt_path = Path(ckpt) if ckpt else DEFAULT_ARTIFACT
            if not ckpt_path.exists():
                bt.logging.warning(
                    f"No model checkpoint at {ckpt_path}; using heuristic fallback."
                )
                return None
            predictor = Predictor(checkpoint_path=ckpt_path)
            self.model_artifact_path = ckpt_path
            bt.logging.info(f"Loaded model checkpoint: {ckpt_path}")
            return predictor
        except Exception as exc:  # noqa: BLE001 - never let model loading crash the miner
            bt.logging.warning(f"Failed to load trained model ({exc}); using heuristic.")
            return None

    def _model_scores(self, chunks: List[list]) -> Optional[List[float]]:
        """Run the trained model; return None on any failure to trigger fallback."""
        if self.predictor is None:
            return None
        try:
            scores = self.predictor.predict(chunks)
            if len(scores) != len(chunks):
                raise ValueError(
                    f"model returned {len(scores)} scores for {len(chunks)} chunks"
                )
            return [round(self._clamp01(float(s)), 6) for s in scores]
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to heuristic
            bt.logging.warning(f"Model inference failed ({exc}); using heuristic.")
            return None

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one bot-risk score per chunk, in order."""
        chunks = synapse.chunks or []
        scores = self._model_scores(chunks)
        source = "model"
        if scores is None:
            scores = [self.score_chunk(chunk) for chunk in chunks]
            source = "heuristic"
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Miner Predictions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks with {source} risks.")
        return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        player_count_signal = 0.0
        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * cls._clamp01(call_ratio / 0.35)
        score += 0.12 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.10 * cls._clamp01(raise_ratio / 0.20)

        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5

        hand_scores = [cls._score_hand(hand) for hand in chunk]
        avg_score = sum(hand_scores) / len(hand_scores)

        return round(cls._clamp01(avg_score), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Random miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
