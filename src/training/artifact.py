from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import DEFAULT_SCHEMA

REQUIRED_ARTIFACT_VERSION = "1.0"


@dataclass(frozen=True)
class ModelArtifact:
    pipeline: object
    metadata: dict


def make_metadata(metrics: dict, threshold: float, policy_version: str, seed: int, model_name: str) -> dict:
    return {
        "artifact_version": REQUIRED_ARTIFACT_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target_name": DEFAULT_SCHEMA.target_name,
        "target_semantics": "1=default/Charged Off, 0=Fully Paid",
        "classes": [0, 1],
        "positive_class": "default",
        "raw_features": DEFAULT_SCHEMA.raw_features,
        "policy_version": policy_version,
        "decision_threshold": float(threshold),
        "random_seed": seed,
        "model_name": model_name,
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "metrics": metrics,
    }


def save_artifact(path: Path | str, artifact: ModelArtifact) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": artifact.pipeline, "metadata": artifact.metadata}, path)


def load_artifact(path: Path | str) -> ModelArtifact:
    if not Path(path).exists():
        raise RuntimeError(f"Model artifact is missing: {path}")
    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Model artifact cannot be loaded: {exc}") from exc
    artifact = ModelArtifact(pipeline=payload["pipeline"], metadata=payload["metadata"])
    validate_metadata(artifact.metadata)
    return artifact


def validate_metadata(metadata: dict) -> None:
    problems = []
    if metadata.get("artifact_version") != REQUIRED_ARTIFACT_VERSION:
        problems.append("artifact_version")
    if metadata.get("target_semantics") != "1=default/Charged Off, 0=Fully Paid":
        problems.append("target_semantics")
    if metadata.get("raw_features") != DEFAULT_SCHEMA.raw_features:
        problems.append("raw_features")
    if metadata.get("classes") != [0, 1]:
        problems.append("classes")
    if problems:
        raise RuntimeError(f"Model artifact metadata is incompatible: {problems}")
