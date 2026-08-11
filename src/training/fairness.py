from __future__ import annotations

import pandas as pd


def subgroup_default_metrics(
    frame: pd.DataFrame, group_column: str, target_column: str, score_column: str
) -> pd.DataFrame:
    required = {group_column, target_column, score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing subgroup evaluation columns: {sorted(missing)}")
    rows = []
    for group, group_df in frame.groupby(group_column, dropna=False):
        rows.append(
            {
                "group": group,
                "n": int(len(group_df)),
                "default_rate": float(group_df[target_column].mean()),
                "mean_probability_default": float(group_df[score_column].mean()),
            }
        )
    return pd.DataFrame(rows)
