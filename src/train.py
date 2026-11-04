from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator

from src.config import load_policy_config
from src.decision.hybrid import choose_threshold
from src.training.artifact import ModelArtifact, make_metadata, save_artifact
from src.training.data import prepare_lendingclub_frame, split_train_validation_test
from src.training.metrics import evaluate_predictions, write_results
from src.training.pipeline import build_pipeline, candidate_models


def select_production_model(candidates_val_metrics: dict[str, dict]) -> tuple[str, str]:
    eligible = {name: metrics for name, metrics in candidates_val_metrics.items() if name != "hist_gradient_boosting"}
    sorted_eligible = sorted(
        eligible.items(),
        key=lambda item: (item[1]["pr_auc_default"], item[1]["roc_auc"]),
        reverse=True,
    )
    best_name, best_metrics = sorted_eligible[0]
    best_pr_auc = best_metrics["pr_auc_default"]

    rf_metrics = eligible.get("random_forest")
    if rf_metrics is not None and (best_pr_auc - rf_metrics["pr_auc_default"] <= 0.005):
        selected_name = "random_forest"
        rationale = (
            f"Selected 'random_forest' (PR-AUC={rf_metrics['pr_auc_default']:.4f}, ROC-AUC={rf_metrics['roc_auc']:.4f}) "
            f"under the interpretability rule: within 0.005 PR-AUC of the top candidate '{best_name}' "
            f"(PR-AUC={best_pr_auc:.4f}, ROC-AUC={best_metrics['roc_auc']:.4f}) to enable exact TreeSHAP explanations. "
            f"hist_gradient_boosting served as an uncalibrated/calibrated benchmark only."
        )
    else:
        selected_name = best_name
        rationale = (
            f"Selected '{selected_name}' based on highest validation PR-AUC ({best_pr_auc:.4f}) "
            f"with ROC-AUC ({best_metrics['roc_auc']:.4f}) as tie-breaker. "
            f"hist_gradient_boosting served as an uncalibrated/calibrated benchmark only."
        )
    return selected_name, rationale


def train_from_csv(data_path: Path, output_dir: Path, seed: int = 42) -> ModelArtifact:
    raw = pd.read_csv(data_path, low_memory=False)
    prepared = prepare_lendingclub_frame(raw)
    train_df, val_df, test_df = split_train_validation_test(prepared, seed=seed)

    x_train, y_train = train_df.drop(columns=["target", "issue_d"], errors="ignore"), train_df["target"]
    x_val, y_val = val_df.drop(columns=["target", "issue_d"], errors="ignore"), val_df["target"]
    x_test, y_test = test_df.drop(columns=["target", "issue_d"], errors="ignore"), test_df["target"]

    policy = load_policy_config()
    min_recall_default = float(policy.get("min_recall_default", 0.70))

    comparison_records: list[dict] = []
    fitted_uncalibrated: dict[str, object] = {}
    fitted_calibrated: dict[str, object] = {}
    candidates_val_metrics: dict[str, dict] = {}

    for name, estimator in candidate_models(seed).items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        fitted_uncalibrated[name] = pipeline

        p_uncal_val = pipeline.predict_proba(x_val)[:, 1]
        metrics_uncal = evaluate_predictions(y_val, p_uncal_val, threshold=0.5)
        comparison_records.append(
            {
                "candidate": name,
                "calibrated": False,
                "is_benchmark_only": name == "hist_gradient_boosting",
                **metrics_uncal,
            }
        )

        calibrator = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid", cv=3)
        calibrator.fit(x_val, y_val)
        fitted_calibrated[name] = calibrator

        p_cal_val = calibrator.predict_proba(x_val)[:, 1]
        metrics_cal = evaluate_predictions(y_val, p_cal_val, threshold=0.5)
        comparison_records.append(
            {
                "candidate": name,
                "calibrated": True,
                "is_benchmark_only": name == "hist_gradient_boosting",
                **metrics_cal,
            }
        )
        candidates_val_metrics[name] = metrics_cal

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv(reports_dir / "model_comparison.csv", index=False)

    selected_name, selection_rationale = select_production_model(candidates_val_metrics)
    production_calibrator = fitted_calibrated[selected_name]

    val_probability_default = production_calibrator.predict_proba(x_val)[:, 1]
    derived_threshold = choose_threshold(y_val, val_probability_default, min_recall_default=min_recall_default)

    config_threshold = float(policy["decision_threshold"])
    if round(derived_threshold, 4) != round(config_threshold, 4):
        threshold = derived_threshold
    else:
        threshold = config_threshold

    test_probability_default = production_calibrator.predict_proba(x_test)[:, 1]

    metrics = {
        "candidate_validation": candidates_val_metrics,
        "validation": evaluate_predictions(y_val, val_probability_default, threshold),
        "test": evaluate_predictions(y_test, test_probability_default, threshold),
    }
    metadata = make_metadata(
        metrics,
        threshold,
        policy["version"],
        seed,
        selected_name,
        selection_rationale=selection_rationale,
    )
    artifact = ModelArtifact(pipeline=production_calibrator, metadata=metadata)
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
