from __future__ import annotations

import pytest

from src.explainability.shap_explainer import Explanation
from web_app.app import _explanation_chart, create_app


def test_fail_fast_without_model(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        create_app(tmp_path / "missing.joblib", testing=True)


def test_health_reports_not_ready_when_explicitly_requested(tmp_path):
    app = create_app(tmp_path / "missing.joblib", testing=True, allow_not_ready=True)
    response = app.test_client().get("/health")
    assert response.status_code == 503
    assert response.get_json()["status"] == "not-ready"


def test_flask_get_health_and_post(artifact_path, valid_payload):
    app = create_app(artifact_path, testing=True)
    client = app.test_client()
    assert client.get("/").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["target"] == "P(Default)"
    response = client.post("/", data=valid_payload)
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "P(Default)" in body
    assert "DSS recommendation" in body


def test_invalid_input_no_stack_trace(artifact_path, valid_payload):
    app = create_app(artifact_path, testing=True)
    response = app.test_client().post("/", data={**valid_payload, "loan_amnt": "abc", "dti": "150"})
    assert response.status_code == 400
    body = response.get_data(as_text=True)
    assert "Validation errors" in body
    assert "Traceback" not in body


def test_status_css_mapping_present(artifact_path):
    app = create_app(artifact_path, testing=True)
    response = app.test_client().get("/")
    body = response.get_data(as_text=True)
    for status in ["LOWER_RISK", "ELEVATED_RISK", "HIGH_RISK", "MANUAL_REVIEW", "POLICY_STOP"]:
        assert f"status-{status}" in body


def test_user_guide_and_example_are_visible_with_form(artifact_path):
    app = create_app(artifact_path, testing=True)
    body = app.test_client().get("/").get_data(as_text=True)
    assert 'id="risk-form"' in body
    assert 'aria-label="Довідник полів"' in body
    assert 'id="guide-toggle"' in body
    assert 'id="fill-example"' in body
    assert "P(Default)" in body
    assert "12000" in body


def test_result_contains_real_probability_and_shap_charts(artifact_path, valid_payload):
    app = create_app(artifact_path, testing=True)
    body = app.test_client().post("/", data=valid_payload).get_data(as_text=True)
    assert "probability-ring" in body
    assert "risk-marker" in body
    assert "Фінансові індикатори" in body
    assert "Що вплинуло на прогноз ML" in body


def test_explanation_chart_scales_real_contributions():
    chart = _explanation_chart(
        Explanation(
            base_value_default=0.2,
            top_features=[
                {"feature": "dti", "contribution_to_default": 0.4},
                {"feature": "income", "contribution_to_default": -0.2},
            ],
            note="test",
        )
    )
    assert chart[0]["bar_percent"] == 100
    assert chart[0]["direction"] == "increases risk"
    assert chart[1]["bar_percent"] == 50
    assert chart[1]["direction"] == "reduces risk"
