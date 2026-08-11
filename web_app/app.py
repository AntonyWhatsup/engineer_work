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


def create_app(artifact_path: str | Path | None = None, testing: bool = False) -> Flask:
    app = Flask(__name__)
    model_path = Path(artifact_path or os.environ.get("MODEL_ARTIFACT_PATH", DEFAULT_MODEL_PATH))
    service = CreditRiskService(model_path)
    app.config["credit_risk_service"] = service
    app.config["TESTING"] = testing

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ready",
                "artifact_version": service.artifact.metadata["artifact_version"],
                "target": "P(Default)",
            }
        )

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "GET":
            return render_template("index.html", success=False, data={})

        raw_payload = _form_payload(request.form)
        application, errors = validate_application(raw_payload)
        if errors or application is None:
            return render_template("index.html", success=False, data=raw_payload, errors=errors), 400

        result = service.predict(application)
        explanation = explain_default_prediction(service.artifact, application)
        return render_template(
            "index.html",
            success=True,
            data=application,
            result=result,
            explanation=explanation,
            probability_default_percent=round(result.probability_default * 100, 1),
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
    create_app().run(debug=debug, port=int(os.environ.get("PORT", "5000")))
