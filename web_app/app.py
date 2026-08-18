from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.config import DEFAULT_MODEL_PATH
from src.explainability.shap_explainer import explain_default_prediction
from src.inference.service import CreditRiskService
from src.validation.schema import validate_application


def _form_payload(form) -> dict:
    return {
        "loan_amnt": form.get("loan_amnt"),
        "term": form.get("term"),
        "annual_inc": form.get("annual_inc"),
        "dti": form.get("dti"),
        "fico_range_low": form.get("fico_range_low"),
        "emp_length": form.get("emp_length"),
        "home_ownership": form.get("home_ownership"),
        "purpose": form.get("purpose"),
        "verification_status": form.get("verification_status"),
        "employment_type": form.get("employment_type"),
        "apr": form.get("apr"),
        "num_dependents": form.get("num_dependents"),
        "residency_covers_term": form.get("residency_covers_term"),
    }


def _explanation_chart(explanation) -> list[dict]:
    if not explanation.top_features:
        return []
    maximum = max(abs(float(item["contribution_to_default"])) for item in explanation.top_features) or 1.0
    return [
        {
            **item,
            "direction": "increases risk" if item["contribution_to_default"] >= 0 else "reduces risk",
            "bar_percent": round(abs(float(item["contribution_to_default"])) / maximum * 100, 1),
        }
        for item in explanation.top_features
    ]


def create_app(
    artifact_path: str | Path | None = None,
    testing: bool = False,
    policy_path: str | Path | None = None,
    allow_not_ready: bool = False,
) -> Flask:
    app = Flask(__name__)
    model_path = Path(artifact_path or os.environ.get("MODEL_ARTIFACT_PATH", DEFAULT_MODEL_PATH))
    service = None
    readiness_error = None
    try:
        service = CreditRiskService(model_path, policy_path)
    except RuntimeError as exc:
        if not allow_not_ready:
            raise
        readiness_error = str(exc)
    app.config["credit_risk_service"] = service
    app.config["TESTING"] = testing

    @app.get("/health")
    def health():
        if service is None:
            return jsonify({"status": "not-ready", "error": readiness_error}), 503
        return jsonify(
            {
                "status": "ready",
                "artifact_version": service.artifact.metadata["artifact_version"],
                "target": "P(Default)",
            }
        )

    @app.route("/", methods=["GET", "POST"])
    def index():
        if service is None:
            return jsonify({"status": "not-ready", "error": readiness_error}), 503
        if request.method == "GET":
            return render_template("index.html", success=False, data={})

        raw_payload = _form_payload(request.form)
        application, errors = validate_application(raw_payload)
        if errors or application is None:
            return render_template("index.html", success=False, data=raw_payload, errors=errors), 400

        result = service.predict(application)
        explanation = explain_default_prediction(service.artifact, application)
        probability_default_percent = round(result.probability_default * 100, 1)
        return render_template(
            "index.html",
            success=True,
            data=application,
            result=result,
            explanation=explanation,
            explanation_chart=_explanation_chart(explanation),
            probability_default_percent=probability_default_percent,
            probability_marker_percent=min(max(probability_default_percent, 0), 100),
            risk_bands=service.policy_config["risk_bands"],
            threshold_percent=round(result.decision.threshold * 100, 1),
            dsti_bar_percent=min(max(result.policy.indicators["dsti_after_new_loan"], 0), 100),
            decision_probability_percent=(
                None
                if result.decision.probability_default is None
                else round(result.decision.probability_default * 100, 1)
            ),
        )

    return app


app = None


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    create_app().run(debug=debug, port=int(os.environ.get("PORT", "5000")), use_reloader=False)
