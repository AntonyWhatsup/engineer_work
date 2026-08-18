from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock

import numpy as np

from src.config import DEFAULT_SCHEMA
from src.validation.schema import ValidatedApplication

_SHAP_LOCK = Lock()


@dataclass(frozen=True)
class Explanation:
    base_value_default: float | None
    top_features: list[dict]
    note: str


def explain_default_prediction(artifact, application: ValidatedApplication, top_n: int = 8) -> Explanation:
    frame = application.to_model_frame(DEFAULT_SCHEMA)
    pipeline = artifact.pipeline
    # CalibratedClassifierCV exposes calibrated_classifiers_; explain the fitted base estimator when possible.
    fitted = getattr(pipeline, "calibrated_classifiers_", None)
    if fitted:
        estimator = fitted[0].estimator
    else:
        estimator = pipeline
    preprocessor = estimator.named_steps.get("preprocessor") if hasattr(estimator, "named_steps") else None
    model = estimator.named_steps.get("model") if hasattr(estimator, "named_steps") else None
    if preprocessor is None or model is None:
        return Explanation(None, [], "SHAP unavailable for this artifact type.")
    try:
        # Avoid concurrent native SHAP/Numba initialization and execution. No shared plots or files are created.
        with _SHAP_LOCK:
            os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
            import shap

            transformed = preprocessor.transform(frame)
            feature_names = list(preprocessor.get_feature_names_out())
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(transformed)
            expected_value = explainer.expected_value
        if isinstance(shap_values, list):
            values = shap_values[1][0]
            expected = float(np.ravel(expected_value)[1])
        elif getattr(shap_values, "ndim", 0) == 3:
            values = shap_values[0, :, 1]
            expected = float(np.ravel(expected_value)[1])
        else:
            values = shap_values[0]
            expected = float(np.ravel(expected_value)[0])
        order = np.argsort(np.abs(values))[::-1][:top_n]
        top_features = [{"feature": feature_names[i], "contribution_to_default": float(values[i])} for i in order]
        return Explanation(expected, top_features, "SHAP explains the ML default-risk prediction only.")
    except Exception as exc:
        return Explanation(None, [], f"SHAP unavailable: {exc}")
