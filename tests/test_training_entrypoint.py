from __future__ import annotations

import pandas as pd

from src.config import load_policy_config
from src.train import main
from src.training.artifact import artifact_metadata_path, load_artifact


def test_official_training_cli_entrypoint_smoke(tmp_path, monkeypatch):
    rows = []
    for index in range(60):
        default = index % 4 == 0
        rows.append(
            {
                "loan_status": "Charged Off" if default else "Fully Paid",
                "loan_amnt": 5000 + index * 100,
                "term": " 60 months" if index % 3 == 0 else " 36 months",
                "annual_inc": 45000 + index * 700,
                "dti": 35 + index % 10 if default else 8 + index % 8,
                "fico_range_low": 620 + index % 20 if default else 690 + index % 30,
                "emp_length": f"{index % 11} years",
                "home_ownership": ["RENT", "OWN", "MORTGAGE"][index % 3],
                "purpose": ["debt_consolidation", "credit_card", "home_improvement"][index % 3],
                "verification_status": ["Verified", "Source Verified", "Not Verified"][index % 3],
                "issue_d": (pd.Timestamp("2018-01-01") + pd.DateOffset(months=index)).strftime("%b-%Y"),
            }
        )
    csv_path = tmp_path / "accepted_loans.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    output = tmp_path / "artifacts"

    monkeypatch.setattr(
        "sys.argv",
        ["src.train", "--data", str(csv_path), "--output", str(output), "--seed", "11"],
    )
    main()

    model_path = output / "model.joblib"
    artifact = load_artifact(model_path, load_policy_config())
    assert artifact.pipeline.classes_.tolist() == [0, 1]
    assert artifact_metadata_path(model_path).is_file()
    assert (output / "metrics.json").is_file()
