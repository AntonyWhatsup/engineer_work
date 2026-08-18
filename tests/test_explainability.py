from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src.explainability.shap_explainer import explain_default_prediction
from src.training.artifact import load_artifact
from src.validation.schema import validate_application


def test_explainability_smoke_and_isolated(artifact_path, valid_payload):
    artifact = load_artifact(artifact_path)
    app, errors = validate_application(valid_payload)
    assert not errors

    def run_once():
        return explain_default_prediction(artifact, app)

    with ThreadPoolExecutor(max_workers=2) as pool:
        explanations = list(pool.map(lambda _: run_once(), range(2)))

    assert all(
        "ML default-risk prediction" in explanation.note or "SHAP unavailable" in explanation.note
        for explanation in explanations
    )
    assert all(not hasattr(explanation, "path") for explanation in explanations)
    assert explanations[0] == explanations[1]
    if "SHAP unavailable" not in explanations[0].note:
        assert explanations[0].base_value_default is not None
        assert explanations[0].top_features
