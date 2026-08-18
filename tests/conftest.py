from __future__ import annotations

import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator

from src.config import load_policy_config
from src.training.artifact import ModelArtifact, make_metadata, save_artifact
from src.training.metrics import evaluate_predictions
from src.training.pipeline import build_pipeline, candidate_models


def synthetic_training_frame(rows: int = 48) -> pd.DataFrame:
    records = []
    for i in range(rows):
        default = i % 4 == 0
        records.append(
            {
                "loan_amnt": 5000 + i * 250,
                "term": 60 if i % 3 == 0 else 36,
                "annual_inc": 90000 - i * 800 if default else 55000 + i * 900,
                "dti": 45 + (i % 10) if default else 8 + (i % 12),
                "fico_range_low": 610 + (i % 20) if default else 690 + (i % 40),
                "emp_length": i % 11,
                "home_ownership": ["RENT", "OWN", "MORTGAGE"][i % 3],
                "purpose": ["debt_consolidation", "credit_card", "home_improvement", "major_purchase"][i % 4],
                "verification_status": ["Verified", "Source Verified", "Not Verified"][i % 3],
                "target": 1 if default else 0,
            }
        )
    return pd.DataFrame(records)


@pytest.fixture()
def valid_payload() -> dict:
    return {
        "loan_amnt": "12000",
        "term": "36",
        "annual_inc": "72000",
        "dti": "18",
        "fico_range_low": "710",
        "emp_length": "6",
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
        "verification_status": "Verified",
        "employment_type": "UOP_FIXED",
        "apr": "9.5",
        "num_dependents": "1",
        "residency_covers_term": "true",
    }


@pytest.fixture()
def artifact_path(tmp_path):
    data = synthetic_training_frame()
    x = data.drop(columns=["target"])
    y = data["target"]
    base = build_pipeline(candidate_models(7)["random_forest"])
    base.fit(x, y)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid", cv=3)
    calibrated.fit(x, y)
    probs = calibrated.predict_proba(x)[:, 1]
    policy = load_policy_config()
    threshold = float(policy["decision_threshold"])
    metadata = make_metadata(
        {"synthetic_smoke": evaluate_predictions(y, probs, threshold)},
        threshold=threshold,
        policy_version=policy["version"],
        seed=7,
        model_name="synthetic_random_forest",
    )
    path = tmp_path / "model.joblib"
    save_artifact(path, ModelArtifact(calibrated, metadata))
    return path
