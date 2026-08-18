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
        policy = yaml.safe_load(fh)
    if not isinstance(policy, dict):
        raise RuntimeError(f"Policy config must be a mapping: {path}")
    return policy


def validate_policy_config(path: Path | str = DEFAULT_POLICY_PATH) -> dict:
    policy = load_policy_config(path)
    required = {"version", "artifact_version", "decision_threshold", "risk_bands", "rules"}
    missing = sorted(required - policy.keys())
    if missing:
        raise RuntimeError(f"Policy config is missing keys: {missing}")
    try:
        threshold = float(policy["decision_threshold"])
        lower = float(policy["risk_bands"]["lower_risk_max"])
        elevated = float(policy["risk_bands"]["elevated_risk_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Policy thresholds must be numeric and complete.") from exc
    if not 0 < threshold < 1 or not 0 <= lower < elevated <= 1:
        raise RuntimeError("Policy thresholds must satisfy 0 <= lower < elevated <= 1 and 0 < decision < 1.")
    if threshold != elevated:
        raise RuntimeError("Policy decision_threshold must equal risk_bands.elevated_risk_max.")
    if not isinstance(policy["rules"], list) or not policy["rules"]:
        raise RuntimeError("Policy config must contain at least one rule.")
    invalid_rules = [
        rule.get("id", "<missing-id>")
        for rule in policy["rules"]
        if not isinstance(rule, dict)
        or rule.get("version") != policy["version"]
        or rule.get("type") not in {"hard_stop", "soft_flag"}
    ]
    if invalid_rules:
        raise RuntimeError(f"Policy rules have invalid type or version: {invalid_rules}")
    return policy
