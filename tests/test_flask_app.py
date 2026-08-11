from __future__ import annotations

import pytest

from web_app.app import create_app


def test_fail_fast_without_model(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        create_app(tmp_path / "missing.joblib", testing=True)


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
