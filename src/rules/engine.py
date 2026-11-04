from __future__ import annotations

import operator
from dataclasses import dataclass

from src.decision.payment import annuity_payment
from src.validation.schema import ValidatedApplication

OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "in": lambda observed, target: observed in target,
}


@dataclass(frozen=True)
class PolicyResult:
    indicators: dict[str, float | bool]
    hard_stops: list[dict]
    soft_flags: list[dict]


def calculate_policy_indicators(
    application: ValidatedApplication, policy_config: dict | None = None
) -> dict[str, float | bool]:
    monthly_income = application.annual_inc / 12
    existing_debt_service = monthly_income * application.dti / 100
    new_payment = annuity_payment(application.loan_amnt, application.apr, application.term)
    total_debt_service = existing_debt_service + new_payment
    dsti_after_new_loan = total_debt_service / monthly_income * 100
    cash_left = monthly_income - total_debt_service

    base_min = 2000.0
    per_dep = 500.0
    if policy_config is not None:
        base_min = float(policy_config.get("min_disposable_income_base", base_min))
        per_dep = float(policy_config.get("min_disposable_income_per_dependent", per_dep))

    required_disposable_income = base_min + (application.num_dependents * per_dep)
    disposable_income_deficit = cash_left - required_disposable_income

    return {
        "monthly_income": monthly_income,
        "existing_debt_service": existing_debt_service,
        "new_monthly_payment": new_payment,
        "dsti_after_new_loan": dsti_after_new_loan,
        "cash_left": cash_left,
        "residency_covers_term": application.residency_covers_term,
        "employment_type": application.employment_type,
        "num_dependents": application.num_dependents,
        "required_disposable_income": required_disposable_income,
        "disposable_income_deficit": disposable_income_deficit,
    }


def evaluate_policy(application: ValidatedApplication, policy_config: dict) -> PolicyResult:
    indicators = calculate_policy_indicators(application, policy_config)
    hard_stops: list[dict] = []
    soft_flags: list[dict] = []
    for rule in policy_config["rules"]:
        op = OPS[rule["operator"]]
        observed = indicators[rule["field"]]
        if op(observed, rule["threshold"]):
            payload = {**rule, "observed_value": observed}
            if rule["type"] == "hard_stop":
                hard_stops.append(payload)
            elif rule["type"] == "soft_flag":
                soft_flags.append(payload)
    return PolicyResult(indicators=indicators, hard_stops=hard_stops, soft_flags=soft_flags)
