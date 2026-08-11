from __future__ import annotations

import math


def annuity_payment(principal: float, annual_percentage_rate: float, term_months: int) -> float:
    if not math.isfinite(principal) or not math.isfinite(annual_percentage_rate):
        raise ValueError("principal and APR must be finite")
    if principal < 0:
        raise ValueError("principal must be non-negative")
    if term_months <= 0:
        raise ValueError("term_months must be positive")
    if annual_percentage_rate < 0:
        raise ValueError("APR must be non-negative")
    if principal == 0:
        return 0.0
    monthly_rate = annual_percentage_rate / 100 / 12
    if monthly_rate == 0:
        return principal / term_months
    return principal * monthly_rate / (1 - (1 + monthly_rate) ** (-term_months))
