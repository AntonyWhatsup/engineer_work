from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import load_policy_config
from src.decision.hybrid import choose_threshold
from src.training.artifact import ModelArtifact, make_metadata, save_artifact
from src.training.data import prepare_lendingclub_frame, split_train_validation_test
from src.training.metrics import evaluate_predictions, write_results
from src.training.pipeline import build_pipeline, candidate_models


def train_from_csv(data_path: Path, output_dir: Path, seed: int = 42) -> ModelArtifact:
    raw = pd.read_csv(data_path, low_memory=False)
    prepared = prepare_lendingclub_frame(raw)
    train_df, val_df, test_df = split_train_validation_test(prepared, seed=seed)

    x_train, y_train = train_df.drop(columns=["target", "issue_d"], errors="ignore"), train_df["target"]
    x_val, y_val = val_df.drop(columns=["target", "issue_d"], errors="ignore"), val_df["target"]
    x_test, y_test = test_df.drop(columns=["target", "issue_d"], errors="ignore"), test_df["target"]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    model_scores: dict[str, dict] = {}
    fitted_candidates = {}
    for name, estimator in candidate_models(seed).items():
        pipeline = build_pipeline(estimator)
        probability_default = cross_val_predict(pipeline, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
        model_scores[name] = evaluate_predictions(y_train, probability_default, threshold=0.5)
        pipeline.fit(x_train, y_train)
        fitted_candidates[name] = pipeline

    selected_name = "random_forest"
    selected_pipeline = fitted_candidates[selected_name]
    calibrator = CalibratedClassifierCV(FrozenEstimator(selected_pipeline), method="sigmoid")
    calibrator.fit(x_val, y_val)

    val_probability_default = calibrator.predict_proba(x_val)[:, 1]
    threshold = choose_threshold(y_val.tolist(), val_probability_default.tolist(), min_recall_default=0.70)
    test_probability_default = calibrator.predict_proba(x_test)[:, 1]

    metrics = {
        "candidate_cv_train": model_scores,
        "validation": evaluate_predictions(y_val, val_probability_default, threshold),
        "test": evaluate_predictions(y_test, test_probability_default, threshold),
    }
    policy = load_policy_config()
    metadata = make_metadata(metrics, threshold, policy["version"], seed, selected_name)
    artifact = ModelArtifact(pipeline=calibrator, metadata=metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_artifact(output_dir / "model.joblib", artifact)
    write_results(output_dir, metrics, y_test, test_probability_default)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibrated LendingClub default-risk model.")
    parser.add_argument("--data", required=True, type=Path, help="Path to accepted_loans.csv")
    parser.add_argument("--output", required=True, type=Path, help="Directory for model and metrics artifacts")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    train_from_csv(args.data, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
