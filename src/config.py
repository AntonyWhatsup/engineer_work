from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "model.joblib"
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "policy_rules.yaml"


@dataclass(frozen=True)
class ModelSchema:
    target_name: str
    raw_features: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    allowed_terms: list[int]


DEFAULT_SCHEMA = ModelSchema(
    target_name="target",
    raw_features=[
        "loan_amnt",
        "term",
        "annual_inc",
        "dti",
        "fico_range_low",
        "emp_length",
        "home_ownership",
        "purpose",
        "verification_status",
    ],
    categorical_features=[
        "home_ownership",
        "purpose",
        "verification_status",
    ],
    numeric_features=[
        "loan_amnt",
        "term",
        "annual_inc",
        "dti",
        "fico_range_low",
        "emp_length",
    ],
    allowed_terms=[36, 60],
)


def load_policy_config(path: Path | str = DEFAULT_POLICY_PATH) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
