"""Standalone manifest compliance checker for the Poker44 miner.

Builds the same model_manifest the miner would publish (honouring the
POKER44_MODEL_* environment variables) and prints its compliance status,
without starting the axon or querying the chain.

Usage (PowerShell):

    $env:POKER44_MODEL_REPO_URL   = "https://github.com/<you>/poker44-miner"
    $env:POKER44_MODEL_REPO_COMMIT = "<real sha>"
    python scripts/miner/check_manifest.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)


def _has_trained_artifacts() -> bool:
    """Mirror the miner's predictor availability without importing torch."""
    candidates = [
        REPO_ROOT / "poker44" / "artifacts_ensemble" / "ensemble.json",
        REPO_ROOT / "poker44" / "model_transformer" / "artifacts" / "model.pt",
    ]
    return any(path.exists() for path in candidates)


def _implementation_files(has_predictor: bool) -> list[Path]:
    """Mirror Miner._implementation_files without importing bittensor."""
    files = [REPO_ROOT / "neurons" / "miner.py"]
    if not has_predictor:
        return files
    for relative in (
        "poker44/ensemble_predict.py",
        "poker44/model_transformer/predict.py",
        "poker44/model_transformer/model.py",
        "poker44/model_transformer/features.py",
        "poker44/model_transformer/dataset.py",
        "poker44/model_mlp/predict.py",
        "poker44/model_lightgbm/predict.py",
    ):
        candidate = REPO_ROOT / relative
        if candidate.exists():
            files.append(candidate)
    return files


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def main() -> int:
    has_predictor = _has_trained_artifacts()
    runtime_commit = _git("rev-parse", "HEAD")

    if has_predictor:
        defaults = {
            "model_name": "poker44-set-transformer",
            "model_version": "1",
            "framework": "pytorch",
            "license": "MIT",
            "repo_url": "https://github.com/Poker44/Poker44-subnet",
            "repo_commit": runtime_commit,
            "open_source": True,
            "inference_mode": "local",
            "training_data_statement": (
                "Trained only on the public Poker44 training benchmark chunks."
            ),
            "training_data_sources": ["poker44-public-training-benchmark"],
            "private_data_attestation": (
                "This miner does not train on validator-only evaluation data."
            ),
        }
    else:
        defaults = {
            "model_name": "poker44-reference-heuristic",
            "model_version": "1",
            "framework": "python-heuristic",
            "license": "MIT",
            "repo_url": "https://github.com/Poker44/Poker44-subnet",
            "repo_commit": runtime_commit,
            "open_source": True,
            "inference_mode": "remote",
            "training_data_statement": (
                "Reference heuristic miner. No training step."
            ),
            "training_data_sources": ["none"],
            "private_data_attestation": (
                "This reference miner does not train on validator-only evaluation data."
            ),
        }

    manifest = build_local_model_manifest(
        repo_root=REPO_ROOT,
        implementation_files=_implementation_files(has_predictor),
        defaults=defaults,
    )
    compliance = evaluate_manifest_compliance(manifest)

    print("=== Poker44 miner manifest ===")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print()
    print(f"trained_artifacts_present : {has_predictor}")
    print(f"manifest_digest           : {manifest_digest(manifest)}")
    print(f"status                    : {compliance['status']}")
    print(f"missing_fields            : {compliance['missing_fields']}")
    print(f"policy_violations         : {compliance['policy_violations']}")

    if compliance["status"] != "transparent":
        print()
        print("NOT transparent. Fix the fields/violations above:")
        print("  - set POKER44_MODEL_REPO_URL to YOUR public model repo")
        print("  - set POKER44_MODEL_REPO_COMMIT to a real pushed commit sha")
        return 1

    print()
    print("transparent: manifest meets the current compliance standard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
