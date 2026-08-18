from __future__ import annotations

import pandas as pd

from src.config import DEFAULT_SCHEMA

FINAL_STATUSES = {"Fully Paid", "Charged Off"}
LEAKAGE_COLUMNS = {
    "grade",
    "sub_grade",
    "int_rate",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "total_pymnt",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "out_prncp",
    "out_prncp_inv",
    "settlement_status",
    "hardship_flag",
}


def map_target(status: str) -> int:
    if status == "Charged Off":
        return 1
    if status == "Fully Paid":
        return 0
    raise ValueError(f"Unsupported loan_status for final-outcome training: {status!r}")


def prepare_lendingclub_frame(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.loc[raw["loan_status"].isin(FINAL_STATUSES)].copy()
    data["target"] = data["loan_status"].map(map_target).astype(int)
    term_text = data["term"].astype(str).str.extract(r"(\d+)")[0]
    data["term"] = term_text.astype(float).astype("Int64")
    if "emp_length" in data.columns:
        emp = data["emp_length"].astype(str).str.extract(r"(\d+)")[0].fillna("0")
        data["emp_length"] = emp.astype(float)
    required = DEFAULT_SCHEMA.raw_features + ["target"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    overlap = sorted(set(DEFAULT_SCHEMA.raw_features) & LEAKAGE_COLUMNS)
    if overlap:
        raise ValueError(f"Raw feature schema contains leakage-prone columns: {overlap}")
    return data[required + (["issue_d"] if "issue_d" in data.columns else [])].dropna(subset=["target"])


def split_train_validation_test(data: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "issue_d" in data.columns:
        parsed_dates = pd.to_datetime(data["issue_d"], format="mixed", errors="coerce", utc=True)
        if parsed_dates.isna().any():
            raise ValueError("issue_d contains missing or unparseable dates.")
        ordered = data.assign(issue_d=parsed_dates).sort_values("issue_d")
    else:
        ordered = data.sample(frac=1.0, random_state=seed)
    n = len(ordered)
    if n < 30:
        # Tiny fixtures need all classes in each split more than time fidelity.
        shuffled = data.sample(frac=1.0, random_state=seed)
        splits = (
            shuffled.iloc[: int(n * 0.6)],
            shuffled.iloc[int(n * 0.6) : int(n * 0.8)],
            shuffled.iloc[int(n * 0.8) :],
        )
        _validate_split_classes(splits)
        return splits
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    splits = ordered.iloc[:train_end], ordered.iloc[train_end:val_end], ordered.iloc[val_end:]
    _validate_split_classes(splits)
    return splits


def _validate_split_classes(splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    for name, split in zip(("train", "validation", "test"), splits, strict=True):
        classes = set(split["target"].dropna().astype(int).unique())
        if classes != {0, 1}:
            raise ValueError(f"{name} split must contain both target classes 0 and 1; found {sorted(classes)}.")
