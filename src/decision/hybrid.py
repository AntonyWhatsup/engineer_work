from __future__ import annotations

from dataclasses import dataclass

from src.rules.engine import PolicyResult

VALID_STATUSES = {"LOWER_RISK", "ELEVATED_RISK", "HIGH_RISK", "MANUAL_REVIEW", "POLICY_STOP"}


@dataclass(frozen=True)
class DecisionResult:
    status: str
    probability_default: float | None
    threshold: float
    risk_band: str | None
    hard_stop_reasons: list[str]
    soft_flag_reasons: list[str]


def choose_threshold(y_true, probability_default, min_recall_default: float = 0.70) -> float:
    candidates = sorted(set(float(p) for p in probability_default))
    best_threshold = 0.5
    best_precision = -1.0
    for threshold in candidates:
        predicted = [1 if p >= threshold else 0 for p in probability_default]
        paired = list(zip(y_true, predicted, strict=True))
        tp = sum(1 for y, pred in paired if y == 1 and pred == 1)
        fp = sum(1 for y, pred in paired if y == 0 and pred == 1)
        fn = sum(1 for y, pred in paired if y == 1 and pred == 0)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        if recall >= min_recall_default and precision > best_precision:
            best_threshold = threshold
            best_precision = precision
    return float(best_threshold)


def make_decision(probability_default: float, policy_result: PolicyResult, policy_config: dict) -> DecisionResult:
    threshold = float(policy_config["decision_threshold"])
    if policy_result.hard_stops:
        status = "MANUAL_REVIEW"
        if any(rule["id"].endswith("HARD_65") for rule in policy_result.hard_stops):
            status = "POLICY_STOP"
        return DecisionResult(
            status=status,
            probability_default=None,
            threshold=threshold,
            risk_band=None,
            hard_stop_reasons=[rule["user_message"] for rule in policy_result.hard_stops],
            soft_flag_reasons=[rule["user_message"] for rule in policy_result.soft_flags],
        )

    bands = policy_config["risk_bands"]
    if probability_default <= float(bands["lower_risk_max"]):
        status = "LOWER_RISK"
    elif probability_default <= float(bands["elevated_risk_max"]):
        status = "ELEVATED_RISK"
    else:
        status = "HIGH_RISK"
    return DecisionResult(
        status=status,
        probability_default=probability_default,
        threshold=threshold,
        risk_band=status,
        hard_stop_reasons=[],
        soft_flag_reasons=[rule["user_message"] for rule in policy_result.soft_flags],
    )
