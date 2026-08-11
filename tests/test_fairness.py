from __future__ import annotations

import pandas as pd

from src.training.fairness import subgroup_default_metrics


def test_subgroup_evaluation_synthetic():
    frame = pd.DataFrame(
        {
            "segment": ["A", "A", "B", "B"],
            "target": [0, 1, 1, 1],
            "probability_default": [0.2, 0.7, 0.6, 0.8],
        }
    )
    result = subgroup_default_metrics(frame, "segment", "target", "probability_default")
    assert set(result["group"]) == {"A", "B"}
    assert result.loc[result["group"] == "B", "default_rate"].item() == 1.0
