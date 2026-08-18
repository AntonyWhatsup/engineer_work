from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, probability_default, threshold: float) -> dict:
    if set(pd.Series(y_true).dropna().astype(int).unique()) != {0, 1}:
        raise ValueError("Evaluation requires both target classes 0 and 1.")
    y_pred = [1 if p >= threshold else 0 for p in probability_default]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability_default)),
        "pr_auc_default": float(average_precision_score(y_true, probability_default)),
        "precision_default": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_default": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_default": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probability_default)),
        "confusion_matrix_labels_0_fully_paid_1_default": cm,
    }


def write_results(output_dir: Path, metrics_by_split: dict, y_true, probability_default) -> None:
    if set(pd.Series(y_true).dropna().astype(int).unique()) != {0, 1}:
        raise ValueError("Calibration output requires both target classes 0 and 1.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics_by_split, indent=2), encoding="utf-8")
    prob_true, prob_pred = calibration_curve(y_true, probability_default, n_bins=5, strategy="uniform")
    pd.DataFrame({"mean_predicted_probability_default": prob_pred, "fraction_default": prob_true}).to_csv(
        output_dir / "calibration_curve.csv", index=False
    )
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="calibrated model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    plt.xlabel("Mean predicted P(Default)")
    plt.ylabel("Fraction default")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=120)
    plt.close()
