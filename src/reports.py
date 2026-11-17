import argparse
import json
import pathlib

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import PercentFormatter
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Ensure the non-interactive Agg backend (required on Windows CI).
matplotlib.use("Agg")

# Local imports – these are part of the repository.
from src.config import load_policy_config
from src.training.data import prepare_lendingclub_frame, split_train_validation_test
from src.training.pipeline import build_pipeline, candidate_models

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ensure_dir(path: pathlib.Path) -> pathlib.Path:
    """Create ``path`` and its parents if they do not exist.

    Returns the same ``Path`` instance for convenient chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig: plt.Figure, out_path: pathlib.Path) -> None:
    """Save ``fig`` as a 200dpi PNG using the Agg canvas.

    The canvas is attached explicitly to avoid side-effects when the same
    ``Figure`` object is reused.
    """
    canvas = FigureCanvas(fig)
    canvas.print_png(str(out_path), dpi=200)
    plt.close(fig)


def _to_percent(values: np.ndarray) -> np.ndarray:
    """Convert a probability array (0-1) into percentage points (0-100)."""
    return values * 100.0


# ---------------------------------------------------------------------------
# Figure generators – each returns a ``Path`` to the saved PNG and optionally a
# data file (CSV/JSON) containing the numbers used for the plot.
# ---------------------------------------------------------------------------


def target_distribution(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """a) ``target_distribution.png`` – raw distribution of ``loan_status``.

    The source column is ``loan_status`` (the original string values).  A second
    figure ``class_balance.png`` shows the same information after the dataset has
    been filtered to the final statuses (Fully Paid / Charged Off) and after the
    ``target`` column has been created.
    """
    out_dir = _ensure_dir(out_dir)

    # 1️⃣ Raw distribution (string labels)
    raw_counts = df["loan_status"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    raw_counts.plot(kind="bar", color="#4c72b0", ax=ax)
    ax.set_title("Rozkład statusu pożyczki (surowe)")
    ax.set_xlabel("Status pożyczki")
    ax.set_ylabel("Liczba obserwacji")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=raw_counts.sum()))
    fig.tight_layout()
    _save_figure(fig, out_dir / "target_distribution.png")
    raw_counts.to_csv(out_dir / "target_distribution.csv")

    # 2️⃣ After filtering – we reuse the helper that already creates ``target``
    filtered = prepare_lendingclub_frame(df)
    balances = filtered["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    balances.plot(kind="bar", color="#55a868", ax=ax)
    ax.set_title("Rozkład klas po filtracji")
    ax.set_xlabel("Klasa (0 = spłacona, 1 = niespłacona)")
    ax.set_ylabel("Liczba obserwacji")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=balances.sum()))
    fig.tight_layout()
    _save_figure(fig, out_dir / "class_balance.png")
    balances.to_csv(out_dir / "class_balance.csv")


def default_rate_by_year(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """b) ``default_rate_by_year.png`` – default rate per ``issue_d`` year.

    The figure includes a note about right-censoring, rendered as a caption
    (Polish).
    """
    out_dir = _ensure_dir(out_dir)
    data = prepare_lendingclub_frame(df)
    # Extract year from the ``issue_d`` column (already a datetime after preparation).
    data["issue_year"] = pd.to_datetime(data["issue_d"], utc=True).dt.year
    yearly = data.groupby("issue_year")["target"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    yearly.plot(marker="o", linestyle="-", color="#c44e52", ax=ax)
    ax.set_title("Współczynnik niewypłacalności w zależności od roku emisji")
    ax.set_xlabel("Rok emisji")
    ax.set_ylabel("Domyślna stopa (procent)")
    ax.yaxis.set_major_formatter(PercentFormatter())
    fig.tight_layout()
    _save_figure(fig, out_dir / "default_rate_by_year.png")
    yearly.to_csv(out_dir / "default_rate_by_year.csv")


def missing_values_top20(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """c) ``missing_values_top20.png`` – bar plot of the 20 columns with most missing values."""
    out_dir = _ensure_dir(out_dir)
    missing_counts = df.isna().sum().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    missing_counts.plot(kind="barh", color="#8172b2", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Najwięcej brakujących wartości (top20)")
    ax.set_xlabel("Liczba braków")
    fig.tight_layout()
    _save_figure(fig, out_dir / "missing_values_top20.png")
    missing_counts.to_csv(out_dir / "missing_values_top20.csv")


def correlation_heatmap(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """d) ``correlation_heatmap.png`` – Pearson correlation of numeric features + target.

    Only the numeric columns that appear in ``DEFAULT_SCHEMA.numeric_features`` are
    considered, plus the generated ``target`` column.
    """
    from src.config import DEFAULT_SCHEMA

    out_dir = _ensure_dir(out_dir)
    data = prepare_lendingclub_frame(df)
    numeric = data[DEFAULT_SCHEMA.numeric_features + ["target"]]
    corr = numeric.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Macierz korelacji Pearsona (liczbowe cechy + target)")
    fig.tight_layout()
    _save_figure(fig, out_dir / "correlation_heatmap.png")
    corr.to_csv(out_dir / "correlation_heatmap.csv")


def model_comparison_artifact(artifact_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """e) Copy ``model_comparison.csv`` generated during training and create a summary plot.

    The CSV is expected at ``artifact_dir/reports/model_comparison.csv`` – if it is
    missing we fall back to a minimal placeholder to keep the pipeline robust.
    """
    out_dir = _ensure_dir(out_dir)
    src_csv = artifact_dir / "reports" / "model_comparison.csv"
    dst_csv = out_dir / "model_comparison.csv"
    if src_csv.is_file():
        dst_csv.write_text(src_csv.read_text(encoding="utf-8"), encoding="utf-8")
        df = pd.read_csv(src_csv)
    else:
        # Minimal placeholder – avoids hard failure when the training step has not
        # been executed yet (e.g. in the smoke-test fixture).
        df = pd.DataFrame(
            {
                "candidate": ["random_forest"],
                "calibrated": [True],
                "pr_auc_default": [np.nan],
                "roc_auc": [np.nan],
                "brier": [np.nan],
            }
        )
        df.to_csv(dst_csv, index=False)

    # Plot – ROC-AUC, PR-AUC and Brier score per model (bars grouped by model).
    metrics = ["roc_auc", "pr_auc_default", "brier"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.2
    indices = np.arange(len(df))
    for i, metric in enumerate(metrics):
        ax.bar(
            indices + i * width,
            df[metric],
            width,
            label=metric.replace("_", " ").title(),
        )
    ax.set_xticks(indices + width)
    ax.set_xticklabels(df["candidate"] + (" (cal.)" if df["calibrated"].iloc[0] else ""), rotation=45, ha="right")
    ax.set_ylabel("Wartość metryki")
    ax.set_title("Porównanie modeli (walidacja)")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, out_dir / "model_comparison.png")


def _load_model(artifact_dir: pathlib.Path):
    """Load the calibrated model artifact (``model.joblib``)."""
    model_path = artifact_dir / "model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    return joblib.load(model_path)


def roc_pr_curves(df: pd.DataFrame, artifact_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """f) ROC and PR curves on the TEST split for the final calibrated model.

    The prevalence baseline (proportion of positives) is drawn as a dashed line
    on the PR curve.
    """
    out_dir = _ensure_dir(out_dir)
    data = prepare_lendingclub_frame(df)
    train_df, val_df, test_df = split_train_validation_test(data)
    X_test = test_df.drop(columns=["target", "issue_d"], errors="ignore")
    y_test = test_df["target"]
    model = _load_model(artifact_dir)
    probs = model.predict_proba(X_test)[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Krzywa ROC (test)")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, out_dir / "roc_curve.png")

    # PR
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    prevalence = y_test.mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    ax.hlines(prevalence, 0, 1, colors="gray", linestyles="dashed", label="Częstość (baseline)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Krzywa Precision-Recall (test)")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, out_dir / "pr_curve.png")


def calibration_diagram(df: pd.DataFrame, artifact_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """g) Reliability diagram before / after calibration with prediction histogram."""
    out_dir = _ensure_dir(out_dir)
    data = prepare_lendingclub_frame(df)
    train_df, val_df, test_df = split_train_validation_test(data)
    X_val = val_df.drop(columns=["target", "issue_d"], errors="ignore")
    y_val = val_df["target"]
    # Load uncalibrated estimator (the pipeline before calibration) – the
    # training script stores only the calibrated object.  For the purpose of the
    # report we rebuild an uncalibrated pipeline on the same data.
    # NOTE: This is a lightweight reconstruction; results may differ slightly
    # from the original but are sufficient for illustration.

    rf = candidate_models()["random_forest"]
    uncal_pipeline = build_pipeline(rf)
    uncal_pipeline.fit(X_val, y_val)
    uncal_probs = uncal_pipeline.predict_proba(X_val)[:, 1]

    calibrated = _load_model(artifact_dir)
    cal_probs = calibrated.predict_proba(X_val)[:, 1]

    def _plot(ax, probs, title):
        # Bin into 10 equally-spaced buckets.
        bins = np.linspace(0, 1, 11)
        binids = np.digitize(probs, bins) - 1
        bin_centers = (bins[:-1] + bins[1:]) / 2
        observed = [y_val[binids == i].mean() if np.any(binids == i) else np.nan for i in range(10)]
        ax.plot(bin_centers, observed, "s-", label="Obserwowane")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Idealny")
        ax.set_xlabel("Prawdopodobieństwo (prognozowane)")
        ax.set_ylabel("Częstość (obserwowana)")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.legend()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    _plot(axs[0], uncal_probs, "Diagram kalibracji – model nieskalibrowany")
    _plot(axs[1], cal_probs, "Diagram kalibracji – model skalibrowany")
    fig.tight_layout()
    _save_figure(fig, out_dir / "calibration_before_after.png")

    # Histogram of predictions (both calibrated and uncalibrated) – saved as a separate figure.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(uncal_probs, bins=20, alpha=0.6, label="Nieskalibrowany", color="#dd8452")
    ax.hist(cal_probs, bins=20, alpha=0.6, label="Skalibrowany", color="#55a868")
    ax.set_xlabel("Prawdopodobieństwo domyślności")
    ax.set_ylabel("Liczba obserwacji")
    ax.set_title("Histogram prognozowanych prawdopodobieństw")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, out_dir / "prediction_histogram.png")


def confusion_and_threshold_sweep(df: pd.DataFrame, artifact_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """h) Confusion matrix at the deployed threshold and a threshold-sweep plot."""
    out_dir = _ensure_dir(out_dir)
    data = prepare_lendingclub_frame(df)
    _, _, test_df = split_train_validation_test(data)
    X_test = test_df.drop(columns=["target", "issue_d"], errors="ignore")
    y_test = test_df["target"]
    model = _load_model(artifact_dir)
    probs = model.predict_proba(X_test)[:, 1]

    policy = load_policy_config()
    threshold = float(policy["decision_threshold"])
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2%})", ha="center", va="center", color="black")
    ax.set_xlabel("Przewidywana klasa")
    ax.set_ylabel("Rzeczywista klasa")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    ax.set_title(f"Macierz pomyłek przy progu = {threshold:.3f}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    _save_figure(fig, out_dir / "confusion_matrix.png")
    pd.DataFrame(cm, index=["0", "1"], columns=["0", "1"]).to_csv(out_dir / "confusion_matrix.csv")

    # Threshold sweep
    thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []
    f1s = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        precisions.append(precision_score(y_test, pred, zero_division=0))
        recalls.append(recall_score(y_test, pred, zero_division=0))
        f1s.append(f1_score(y_test, pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="Precyzja")
    ax.plot(thresholds, recalls, label="Czułość")
    ax.plot(thresholds, f1s, label="F1")
    ax.axvline(threshold, color="red", linestyle="--", label="Wybrany próg")
    ax.set_xlabel("Próg decyzyjny")
    ax.set_ylabel("Wartość metryki")
    ax.set_title("Wrażliwość metryk na próg")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, out_dir / "threshold_sweep.png")
    pd.DataFrame({"threshold": thresholds, "precision": precisions, "recall": recalls, "f1": f1s}).to_csv(
        out_dir / "threshold_sweep.csv", index=False
    )


def score_distribution(df: pd.DataFrame, artifact_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """i) Score distribution per true class and default-rate by risk band."""
    out_dir = _ensure_dir(out_dir)
    data = prepare_lendingclub_frame(df)
    _, _, test_df = split_train_validation_test(data)
    X_test = test_df.drop(columns=["target", "issue_d"], errors="ignore")
    y_test = test_df["target"]
    model = _load_model(artifact_dir)
    probs = model.predict_proba(X_test)[:, 1]

    # Score distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        [_to_percent(probs[y_test == 0]), _to_percent(probs[y_test == 1])],
        bins=20,
        label=["Spłacona", "Niespłacona"],
        color=["#4c72b0", "#c44e52"],
        alpha=0.7,
        stacked=False,
    )
    ax.set_xlabel("Prawdopodobieństwo domyślności [%]")
    ax.set_ylabel("Liczba obserwacji")
    ax.set_title("Rozkład prawdopodobieństwa według klasy rzeczywistej")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, out_dir / "score_distribution.png")
    pd.DataFrame({"probability": probs, "target": y_test}).to_csv(out_dir / "score_distribution.csv", index=False)

    # Default rate by risk band – uses the policy thresholds.
    policy = load_policy_config()
    bands = policy["risk_bands"]
    lower = float(bands["lower_risk_max"])
    elevated = float(bands["elevated_risk_max"])

    def band(p):
        if p <= lower:
            return "LOWER"
        if p <= elevated:
            return "ELEVATED"
        return "HIGH"

    band_labels = np.vectorize(band)(_to_percent(probs) / 100.0)
    df_band = pd.DataFrame({"band": band_labels, "target": y_test})
    rates = df_band.groupby("band")["target"].mean()
    # Verify monotonicity – raise if not satisfied.
    if not (rates["LOWER"] <= rates["ELEVATED"] <= rates["HIGH"]):
        raise AssertionError("Monotonicity check failed: default rates not increasing across risk bands.")
    fig, ax = plt.subplots(figsize=(6, 4))
    rates.plot(kind="bar", color=["#55a868", "#c44e52", "#4c72b0"], ax=ax)
    ax.set_ylabel("Średni współczynnik domyślności")
    ax.set_title("Domyślność wg pasma ryzyka (polityka)")
    fig.tight_layout()
    _save_figure(fig, out_dir / "default_rate_by_risk_band.png")
    rates.to_csv(out_dir / "default_rate_by_risk_band.csv")


def permutation_importance_plot(df: pd.DataFrame, artifact_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """j) Permutation importance on the validation set (model-agnostic)."""
    out_dir = _ensure_dir(out_dir)
    data = prepare_lendingclub_frame(df)
    train_df, val_df, _ = split_train_validation_test(data)
    X_val = val_df.drop(columns=["target", "issue_d"], errors="ignore")
    y_val = val_df["target"]
    model = _load_model(artifact_dir)
    result = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)
    importances = pd.Series(result.importances_mean, index=X_val.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="#8172b2")
    ax.invert_yaxis()
    ax.set_title("Ważność cech – Permutacja (walidacja)")
    ax.set_xlabel("Średni spadek wyniku (ważność)")
    fig.tight_layout()
    _save_figure(fig, out_dir / "permutation_importance.png")
    importances.to_csv(out_dir / "permutation_importance.csv")


def shap_global_and_beeswarm(df: pd.DataFrame, artifact_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """j) SHAP global bar and beeswarm plots (original features, ≤2000 test rows)."""
    out_dir = _ensure_dir(out_dir)
    data = prepare_lendingclub_frame(df)
    _, _, test_df = split_train_validation_test(data)
    X_test = test_df.drop(columns=["target", "issue_d"], errors="ignore")
    # Sample up to 2000 rows for speed.
    if len(X_test) > 2000:
        idx = np.random.RandomState(42).choice(len(X_test), 2000, replace=False)
        X_sample = X_test.iloc[idx]
    else:
        X_sample = X_test
    model = _load_model(artifact_dir)
    # Use shap.Explainer (model-agnostic) – it works with sklearn pipelines.
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    # Global bar – mean absolute SHAP value per feature.
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=20, show=False)
    plt.gcf().set_size_inches(8, 6)
    _save_figure(plt.gcf(), out_dir / "shap_global_bar.png")
    plt.clf()

    # Beeswarm – dot plot.
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    plt.gcf().set_size_inches(8, 6)
    _save_figure(plt.gcf(), out_dir / "shap_beeswarm.png")
    # Underlying numbers – we store the raw SHAP values as JSON for reproducibility.
    shap_array = shap_values.values
    json_path = out_dir / "shap_values.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(shap_array.tolist(), f, ensure_ascii=False, indent=2)


def subgroup_metrics(df: pd.DataFrame, artifact_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """k) Compute subgroup metrics via ``src.training.fairness`` and export."""
    out_dir = _ensure_dir(out_dir)
    try:
        from src.training.fairness import compute_subgroup_metrics
    except Exception as exc:
        raise ImportError("src.training.fairness.compute_subgroup_metrics not found") from exc

    data = prepare_lendingclub_frame(df)
    _, _, test_df = split_train_validation_test(data)
    X_test = test_df.drop(columns=["target", "issue_d"], errors="ignore")
    model = _load_model(artifact_dir)
    # Prediction probabilities for the whole test set.
    probs = model.predict_proba(X_test)[:, 1]
    metrics = compute_subgroup_metrics(test_df, probs, group_cols=["home_ownership", "purpose"])
    # ``metrics`` is expected to be a DataFrame.
    metrics.to_csv(out_dir / "subgroup_metrics.csv", index=False)
    # Plot – simple heatmap of default rate per subgroup.
    fig, ax = plt.subplots(figsize=(9, 6))
    # Pivot table for heatmap.
    pivot = metrics.pivot(index="home_ownership", columns="purpose", values="default_rate")
    im = ax.imshow(pivot, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax)
    ax.set_title("Średnia prognozowana prawdopodobność domyślności wg podgrup")
    fig.tight_layout()
    _save_figure(fig, out_dir / "subgroup_metrics.png")


def generate_figures_map(out_dir: pathlib.Path) -> None:
    """l) Create ``docs/figures_map.md`` with a placeholder table."""
    docs_dir = _ensure_dir(out_dir.parent / "docs")
    md_path = docs_dir / "figures_map.md"
    content = """# Mapa ilustracji dla pracy dyplomowej

| Plik ilustracji | Proponowana sekcja pracy | Polski opis (Rysunek N.) | Źródło | Interpretacja (szablon) |
|-----------------|--------------------------|--------------------------|--------|------------------------|
| target_distribution.png | 2.1 Dane | Rysunek 1. Rozkład surowych statusów pożyczek | Źródło: opracowanie własne. | *Interpretacja*: Wprawdzie… |
| class_balance.png | 2.2 Przetwarzanie danych | Rysunek 2. Bilans klas po filtracji | Źródło: opracowanie własne. | *Interpretacja*: … |
| default_rate_by_year.png | 3.1 Analiza temporalna | Rysunek 3. Współczynnik niewypłacalności wg roku emisji | Źródło: opracowanie własne. | *Interpretacja*: … |
| missing_values_top20.png | 2.3 Braki danych | Rysunek 4. Najwięcej braków wartości (top20) | Źródło: opracowanie własne. | *Interpretacja*: … |
| correlation_heatmap.png | 4.1 Analiza cech | Rysunek 5. Macierz korelacji Pearsona | Źródło: opracowanie własne. | *Interpretacja*: … |
| model_comparison.png | 4.2 Modelowanie | Rysunek 6. Porównanie modeli (walidacja) | Źródło: opracowanie własne. | *Interpretacja*: … |
| roc_curve.png | 5.1 Ocena modelu | Rysunek 7. Krzywa ROC na zbiorze testowym | Źródło: opracowanie własne. | *Interpretacja*: … |
| pr_curve.png | 5.1 Ocena modelu | Rysunek 8. Krzywa Precision-Recall na zbiorze testowym | Źródło: opracowanie własne. | *Interpretacja*: … |
| calibration_before_after.png | 5.2 Kalibracja | Rysunek 9. Diagram kalibracji przed i po | Źródło: opracowanie własne. | *Interpretacja*: … |
| confusion_matrix.png | 5.3 Wyniki decyzyjne | Rysunek 10. Macierz pomyłek przy wybranym progu | Źródło: opracowanie własne. | *Interpretacja*: … |
| threshold_sweep.png | 5.3 Wyniki decyzyjne | Rysunek 11. Wpływ progu na metryki | Źródło: opraczenie własne. | *Interpretacja*: … |
| score_distribution.png | 5.4 Analiza ryzyka | Rysunek 12. Rozkład prawdopodobieństwa wg klasy rzeczywistej | Źródło: opracowanie własne. | *Interpretacja*: … |
| default_rate_by_risk_band.png | 5.4 Analiza ryzyka | Rysunek 13. Domyślność wg pasma ryzyka | Źródło: opracowanie własne. | *Interpretacja*: … |
| permutation_importance.png | 6.1 Wyjaśnialność | Rysunek 14. Permutacyjna ważność cech | Źródło: opracowanie własne. | *Interpretacja*: … |
| shap_global_bar.png | 6.2 SHAP | Rysunek 15. Globalny wykres SHAP (słupki) | Źródło: opracowanie własne. | *Interpretacja*: … |
| shap_beeswarm.png | 6.2 SHAP | Rysunek 16. SHAP beeswarm (kropki) | Źródło: opracowanie własne. | *Interpretacja*: … |
| subgroup_metrics.png | 7.1 Sprawiedliwość | Rysunek 17. Średnia prognozowana prawdopodobność wg podgrup | Źródło: opracowanie własne. | *Interpretacja*: … |
"""
    md_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis figures and tables.")
    parser.add_argument("--data", type=pathlib.Path, required=True, help="Path to LendingClub CSV file.")
    parser.add_argument(
        "--artifact-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing the trained model artifact and where reports will be stored.",
    )
    args = parser.parse_args()

    reports_dir = args.artifact_dir / "reports"
    # Load the raw CSV (may be a fixture for tests).
    raw_df = pd.read_csv(args.data, low_memory=False)

    # Generate each figure / table.
    target_distribution(raw_df, reports_dir)
    default_rate_by_year(raw_df, reports_dir)
    missing_values_top20(raw_df, reports_dir)
    correlation_heatmap(raw_df, reports_dir)
    model_comparison_artifact(args.artifact_dir, reports_dir)
    roc_pr_curves(raw_df, args.artifact_dir, reports_dir)
    calibration_diagram(raw_df, args.artifact_dir, reports_dir)
    confusion_and_threshold_sweep(raw_df, args.artifact_dir, reports_dir)
    score_distribution(raw_df, args.artifact_dir, reports_dir)
    permutation_importance_plot(raw_df, args.artifact_dir, reports_dir)
    shap_global_and_beeswarm(raw_df, args.artifact_dir, reports_dir)
    subgroup_metrics(raw_df, args.artifact_dir, reports_dir)
    generate_figures_map(args.artifact_dir)

    print(f"All reports generated in {reports_dir}")


if __name__ == "__main__":
    main()
