from __future__ import annotations

import math

import pandas as pd
import pytest

from src.config import load_policy_config, validate_policy_config
from src.decision.hybrid import choose_threshold, make_decision
from src.decision.payment import annuity_payment
from src.rules.engine import evaluate_policy
from src.training.artifact import ModelArtifact, load_artifact, save_artifact
from src.training.data import map_target, prepare_lendingclub_frame, split_train_validation_test
from src.training.metrics import evaluate_predictions
from src.validation.schema import validate_application


def test_target_mapping_default_positive_class():
    assert map_target("Charged Off") == 1
    assert map_target("Fully Paid") == 0
    with pytest.raises(ValueError):
        map_target("Current")


def test_prepare_lendingclub_frame_filters_and_schema():
    raw = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Current"],
            "loan_amnt": [1000, 2000, 3000],
            "term": [" 36 months", " 60 months", " 36 months"],
            "annual_inc": [50000, 40000, 60000],
            "dti": [10, 30, 20],
            "fico_range_low": [700, 620, 710],
            "emp_length": ["10+ years", "< 1 year", "2 years"],
            "home_ownership": ["RENT", "OWN", "MORTGAGE"],
            "purpose": ["credit_card", "other", "other"],
            "verification_status": ["Verified", "Not Verified", "Verified"],
        }
    )
    prepared = prepare_lendingclub_frame(raw)
    assert prepared["target"].tolist() == [0, 1]
    assert prepared["emp_length"].tolist() == [10.0, 0.0]
    assert list(prepared.drop(columns=["target"]).columns) == [
        "loan_amnt",
        "term",
        "annual_inc",
        "dti",
        "fico_range_low",
        "emp_length",
        "home_ownership",
        "purpose",
        "verification_status",
    ]


def test_emp_length_mapping_and_nan_preservation():
    raw = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Fully Paid", "Charged Off"],
            "loan_amnt": [1000, 2000, 3000, 4000],
            "term": [" 36 months", " 60 months", " 36 months", " 36 months"],
            "annual_inc": [50000, 40000, 60000, 70000],
            "dti": [10, 30, 20, 15],
            "fico_range_low": [700, 620, 710, 680],
            "emp_length": ["< 1 year", "1 year", None, "10+ years"],
            "home_ownership": ["RENT", "OWN", "MORTGAGE", "RENT"],
            "purpose": ["credit_card", "other", "other", "credit_card"],
            "verification_status": ["Verified", "Not Verified", "Verified", "Verified"],
        }
    )
    prepared = prepare_lendingclub_frame(raw)
    assert prepared["emp_length"].iloc[0] == 0.0
    assert prepared["emp_length"].iloc[1] == 1.0
    assert pd.isna(prepared["emp_length"].iloc[2])
    assert prepared["emp_length"].iloc[3] == 10.0


def test_annuity_payment_zero_and_regular_rate():
    assert annuity_payment(12000, 0, 12) == pytest.approx(1000)
    assert annuity_payment(100000, 12, 360) == pytest.approx(1028.61, rel=1e-3)
    with pytest.raises(ValueError):
        annuity_payment(-1, 10, 12)


def test_schema_validation_collects_errors(valid_payload):
    bad = {**valid_payload, "annual_inc": "0", "fico_range_low": "999", "loan_amnt": "nan"}
    app, errors = validate_application(bad)
    assert app is None
    assert any("annual_inc" in error for error in errors)
    assert any("fico_range_low" in error for error in errors)
    assert any("finite" in error for error in errors)


def test_policy_rules_and_hard_stop(valid_payload):
    app, errors = validate_application({**valid_payload, "loan_amnt": "200000", "annual_inc": "30000"})
    assert not errors
    result = evaluate_policy(app, load_policy_config())
    assert result.hard_stops
    decision = make_decision(0.1, result, load_policy_config())
    assert decision.status in {"POLICY_STOP", "MANUAL_REVIEW"}
    assert decision.probability_default is None


def test_threshold_not_fixed_to_half():
    threshold = choose_threshold([0, 0, 1, 1], [0.05, 0.2, 0.35, 0.9], min_recall_default=1.0)
    assert threshold != 0.5
    assert math.isclose(threshold, 0.35)


def test_artifact_metadata_and_inference_schema(artifact_path, valid_payload):
    artifact = load_artifact(artifact_path)
    assert artifact.metadata["target_semantics"] == "1=default/Charged Off, 0=Fully Paid"
    app, errors = validate_application(valid_payload)
    assert not errors
    frame = app.to_model_frame()
    probability_default = artifact.pipeline.predict_proba(frame)[0][1]
    assert 0 <= probability_default <= 1


def test_issue_date_is_parsed_before_chronological_sorting():
    rows = []
    dates = ["Dec-2020", "Jan-2019", "Feb-2019", "Jan-2020", "Mar-2020"] * 6
    for index, date in enumerate(dates):
        rows.append({"issue_d": date, "target": index % 2})
    train, validation, test = split_train_validation_test(pd.DataFrame(rows))
    assert train["issue_d"].max() <= validation["issue_d"].min()
    assert validation["issue_d"].max() <= test["issue_d"].min()


def test_split_rejects_single_class_partition():
    frame = pd.DataFrame({"target": [0] * 20 + [1] * 10, "issue_d": pd.date_range("2020-01-01", periods=30)})
    with pytest.raises(ValueError, match="both target classes"):
        split_train_validation_test(frame)


def test_metrics_reject_single_class():
    with pytest.raises(ValueError, match="both target classes"):
        evaluate_predictions([0, 0], [0.1, 0.2], 0.42)


def test_policy_threshold_matches_risk_bands():
    policy = validate_policy_config()
    assert policy["decision_threshold"] == policy["risk_bands"]["elevated_risk_max"]


def test_artifact_without_predict_proba_is_blocked(artifact_path, tmp_path):
    artifact = load_artifact(artifact_path)
    bad_path = tmp_path / "bad" / "model.joblib"
    save_artifact(bad_path, ModelArtifact(object(), artifact.metadata))
    with pytest.raises(RuntimeError, match="predict_proba"):
        load_artifact(bad_path)


def test_unused_inputs_as_policy_rules(valid_payload):
    policy = load_policy_config()
    # 1. employment_type: B2B_LT_12M or OTHER -> manual_review soft flag
    app_b2b, errors = validate_application({**valid_payload, "employment_type": "B2B_LT_12M"})
    assert not errors
    res_b2b = evaluate_policy(app_b2b, policy)
    assert any(flag["id"] == "EMPLOYMENT_TYPE_REVIEW" for flag in res_b2b.soft_flags)

    app_other, errors = validate_application({**valid_payload, "employment_type": "OTHER"})
    assert not errors
    res_other = evaluate_policy(app_other, policy)
    assert any(flag["id"] == "EMPLOYMENT_TYPE_REVIEW" for flag in res_other.soft_flags)

    app_fixed, errors = validate_application({**valid_payload, "employment_type": "UOP_FIXED"})
    assert not errors
    res_fixed = evaluate_policy(app_fixed, policy)
    assert not any(flag["id"] == "EMPLOYMENT_TYPE_REVIEW" for flag in res_fixed.soft_flags)

    # 2. disposable income scales with num_dependents
    # annual_inc: 36000 -> monthly = 3000. dti=0 -> existing=0. loan=1000, 36m, apr=0 -> payment = 27.78.
    # cash_left = 3000 - 28 = 2972.
    # With 0 dependents: min required = 2000 -> deficit = 2972 - 2000 = +972 >= 0 (No hard stop)
    # With 3 dependents: min required = 2000 + 3*500 = 3500 -> deficit = 2972 - 3500 = -528 < 0 (Hard stop!)
    low_inc_payload = {
        **valid_payload,
        "annual_inc": "36000",
        "loan_amnt": "1000",
        "dti": "0",
        "apr": "0",
        "num_dependents": "0",
    }
    app_dep0, _ = validate_application(low_inc_payload)
    res_dep0 = evaluate_policy(app_dep0, policy)
    assert not any(stop["id"] == "DISPOSABLE_INCOME_REVIEW" for stop in res_dep0.hard_stops)

    app_dep3, _ = validate_application({**low_inc_payload, "num_dependents": "3"})
    res_dep3 = evaluate_policy(app_dep3, policy)
    assert any(stop["id"] == "DISPOSABLE_INCOME_REVIEW" for stop in res_dep3.hard_stops)


def test_currency_conversion_to_model_frame(valid_payload):
    app, errors = validate_application(valid_payload)
    assert not errors
    # Base PLN values
    frame_pln = app.to_model_frame()
    assert frame_pln["loan_amnt"].iloc[0] == 12000.0
    assert frame_pln["annual_inc"].iloc[0] == 72000.0

    # Converted to USD
    frame_usd = app.to_model_frame(fx_pln_per_usd=4.0)
    assert frame_usd["loan_amnt"].iloc[0] == 3000.0
    assert frame_usd["annual_inc"].iloc[0] == 18000.0
    # Other features unchanged
    assert frame_usd["dti"].iloc[0] == 18.0
    assert frame_usd["fico_range_low"].iloc[0] == 710.0


def test_corrupt_artifact_and_metadata_mismatch_are_blocked(artifact_path):
    metadata_path = artifact_path.with_name("metadata.json")
    metadata_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="do not match"):
        load_artifact(artifact_path)
