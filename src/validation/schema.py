from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import DEFAULT_SCHEMA, ModelSchema

ALLOWED_CATEGORIES = {
    "home_ownership": {"RENT", "OWN", "MORTGAGE", "OTHER"},
    "purpose": {"debt_consolidation", "credit_card", "home_improvement", "major_purchase", "other"},
    "verification_status": {"Verified", "Source Verified", "Not Verified"},
    "employment_type": {"UOP_UNDEFINED", "UOP_FIXED", "B2B_GE_12M", "B2B_LT_12M", "OTHER"},
}


@dataclass(frozen=True)
class ValidatedApplication:
    loan_amnt: float
    term: int
    annual_inc: float
    dti: float
    fico_range_low: float
    emp_length: float
    home_ownership: str
    purpose: str
    verification_status: str
    employment_type: str
    apr: float
    num_dependents: int
    residency_covers_term: bool

    def to_model_frame(self, schema: ModelSchema = DEFAULT_SCHEMA) -> pd.DataFrame:
        values = {feature: getattr(self, feature) for feature in schema.raw_features}
        return pd.DataFrame([values], columns=schema.raw_features)


def _parse_float(raw: Any, field: str, errors: list[str]) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        errors.append(f"{field}: must be a number")
        return 0.0
    if not math.isfinite(value):
        errors.append(f"{field}: must be finite")
        return 0.0
    return value


def _parse_int(raw: Any, field: str, errors: list[str]) -> int:
    value = _parse_float(raw, field, errors)
    if not float(value).is_integer():
        errors.append(f"{field}: must be an integer")
    return int(value)


def validate_application(payload: dict[str, Any]) -> tuple[ValidatedApplication | None, list[str]]:
    errors: list[str] = []
    required = [
        "loan_amnt",
        "term",
        "annual_inc",
        "dti",
        "fico_range_low",
        "emp_length",
        "home_ownership",
        "purpose",
        "verification_status",
        "employment_type",
        "apr",
        "num_dependents",
        "residency_covers_term",
    ]
    for field in required:
        if payload.get(field) in (None, ""):
            errors.append(f"{field}: is required")

    loan_amnt = _parse_float(payload.get("loan_amnt"), "loan_amnt", errors)
    term = _parse_int(payload.get("term"), "term", errors)
    annual_inc = _parse_float(payload.get("annual_inc"), "annual_inc", errors)
    dti = _parse_float(payload.get("dti"), "dti", errors)
    fico = _parse_float(payload.get("fico_range_low"), "fico_range_low", errors)
    emp_length = _parse_float(payload.get("emp_length"), "emp_length", errors)
    apr = _parse_float(payload.get("apr"), "apr", errors)
    num_dependents = _parse_int(payload.get("num_dependents"), "num_dependents", errors)

    home_ownership = str(payload.get("home_ownership", ""))
    purpose = str(payload.get("purpose", ""))
    verification_status = str(payload.get("verification_status", ""))
    employment_type = str(payload.get("employment_type", ""))
    residency_raw = payload.get("residency_covers_term")
    residency_covers_term = str(residency_raw).lower() in {"true", "1", "yes", "on"}

    if loan_amnt < 0:
        errors.append("loan_amnt: must be non-negative")
    if term not in DEFAULT_SCHEMA.allowed_terms:
        errors.append("term: must be one of 36 or 60")
    if annual_inc <= 0:
        errors.append("annual_inc: must be positive")
    if not 0 <= dti <= 100:
        errors.append("dti: must be between 0 and 100")
    if not 300 <= fico <= 850:
        errors.append("fico_range_low: must be between 300 and 850")
    if not 0 <= emp_length <= 50:
        errors.append("emp_length: must be between 0 and 50")
    if not 0 <= apr <= 100:
        errors.append("apr: must be between 0 and 100")
    if not 0 <= num_dependents <= 20:
        errors.append("num_dependents: must be between 0 and 20")

    for field, value in {
        "home_ownership": home_ownership,
        "purpose": purpose,
        "verification_status": verification_status,
        "employment_type": employment_type,
    }.items():
        if value not in ALLOWED_CATEGORIES[field]:
            errors.append(f"{field}: unsupported category")

    if errors:
        return None, errors

    return (
        ValidatedApplication(
            loan_amnt=loan_amnt,
            term=term,
            annual_inc=annual_inc,
            dti=dti,
            fico_range_low=fico,
            emp_length=emp_length,
            home_ownership=home_ownership,
            purpose=purpose,
            verification_status=verification_status,
            employment_type=employment_type,
            apr=apr,
            num_dependents=num_dependents,
            residency_covers_term=residency_covers_term,
        ),
        [],
    )


def validate_model_frame(frame: pd.DataFrame, schema: ModelSchema = DEFAULT_SCHEMA) -> None:
    missing = [field for field in schema.raw_features if field not in frame.columns]
    extra = [field for field in frame.columns if field not in schema.raw_features]
    if missing or extra:
        raise ValueError(f"Model schema mismatch. Missing={missing}; extra={extra}")
