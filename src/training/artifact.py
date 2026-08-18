from __future__ import annotations

import json
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
TARGET_SEMANTICS = "1=default/Charged Off, 0=Fully Paid"


@dataclass(frozen=True)
class ModelArtifact:
    pipeline: object
    metadata: dict


def make_metadata(metrics: dict, threshold: float, policy_version: str, seed: int, model_name: str) -> dict:
    return {
        "artifact_version": REQUIRED_ARTIFACT_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target_name": DEFAULT_SCHEMA.target_name,
        "target_semantics": TARGET_SEMANTICS,
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
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_metadata(artifact.metadata)
    joblib.dump({"pipeline": artifact.pipeline, "metadata": artifact.metadata}, destination)
    artifact_metadata_path(destination).write_text(json.dumps(artifact.metadata, indent=2), encoding="utf-8")


def artifact_metadata_path(path: Path | str) -> Path:
    return Path(path).with_name("metadata.json")


def load_artifact(path: Path | str, policy_config: dict | None = None) -> ModelArtifact:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise RuntimeError(f"Model artifact is missing: {path}")
    metadata_path = artifact_metadata_path(artifact_path)
    if not metadata_path.is_file():
        raise RuntimeError(f"Model metadata is missing: {metadata_path}")
    try:
        external_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload = joblib.load(artifact_path)
    except Exception as exc:
        raise RuntimeError(f"Model artifact cannot be loaded: {exc}") from exc
    if not isinstance(payload, dict) or "pipeline" not in payload or "metadata" not in payload:
        raise RuntimeError("Model artifact payload must contain pipeline and metadata.")
    if payload["metadata"] != external_metadata:
        raise RuntimeError("Embedded and external model metadata do not match.")
    artifact = ModelArtifact(pipeline=payload["pipeline"], metadata=external_metadata)
    validate_artifact(artifact, policy_config)
    return artifact


def validate_metadata(metadata: dict) -> None:
    problems = []
    if metadata.get("artifact_version") != REQUIRED_ARTIFACT_VERSION:
        problems.append("artifact_version")
    if metadata.get("target_semantics") != TARGET_SEMANTICS:
        problems.append("target_semantics")
    if metadata.get("raw_features") != DEFAULT_SCHEMA.raw_features:
        problems.append("raw_features")
    if metadata.get("classes") != [0, 1]:
        problems.append("classes")
    if metadata.get("target_name") != DEFAULT_SCHEMA.target_name:
        problems.append("target_name")
    if metadata.get("positive_class") != "default":
        problems.append("positive_class")
    versions = metadata.get("library_versions")
    if not isinstance(versions, dict) or "scikit_learn" not in versions:
        problems.append("library_versions")
    try:
        threshold = float(metadata["decision_threshold"])
        if not 0 < threshold < 1:
            problems.append("decision_threshold")
    except (KeyError, TypeError, ValueError):
        problems.append("decision_threshold")
    if problems:
        raise RuntimeError(f"Model artifact metadata is incompatible: {problems}")


def validate_artifact(artifact: ModelArtifact, policy_config: dict | None = None) -> None:
    validate_metadata(artifact.metadata)
    predict_proba = getattr(artifact.pipeline, "predict_proba", None)
    if not callable(predict_proba):
        raise RuntimeError("Model pipeline does not provide predict_proba.")
    classes = getattr(artifact.pipeline, "classes_", None)
    if classes is None or list(classes) != [0, 1]:
        raise RuntimeError("Fitted pipeline classes_ must be exactly [0, 1]; index 1 must mean default.")
    feature_names = getattr(artifact.pipeline, "feature_names_in_", None)
    if feature_names is None or list(feature_names) != DEFAULT_SCHEMA.raw_features:
        raise RuntimeError("Fitted pipeline raw feature schema is incompatible.")
    trained_sklearn = str(artifact.metadata["library_versions"]["scikit_learn"])
    if trained_sklearn.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        raise RuntimeError(
            f"Model scikit-learn version is incompatible: trained={trained_sklearn}, runtime={sklearn.__version__}."
        )
    if policy_config is not None:
        if artifact.metadata["artifact_version"] != str(policy_config.get("artifact_version")):
            raise RuntimeError("Model artifact version does not match policy config.")
        if artifact.metadata["policy_version"] != policy_config.get("version"):
            raise RuntimeError("Model policy_version does not match policy config.")
        if float(artifact.metadata["decision_threshold"]) != float(policy_config.get("decision_threshold")):
            raise RuntimeError("Model decision_threshold does not match policy config and risk bands.")
