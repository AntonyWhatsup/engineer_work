# Credit Default Risk DSS Prototype

Educational decision-support prototype for credit default-risk assessment. The system estimates:

```text
target = 1 means default / Charged Off
target = 0 means Fully Paid
predict_proba(...)[1] = P(Default)
```

It is not a bank production system and does not issue an official credit decision. The final decision remains with a human reviewer.

## Architecture

- `src/train.py` trains a reproducible sklearn pipeline and writes `artifacts/model.joblib`.
- `src/training/` contains LendingClub preparation, preprocessing, model comparison, metrics, metadata, and optional subgroup evaluation helpers.
- `src/inference/` loads a validated artifact and returns calibrated `P(Default)`.
- `src/validation/`, `src/rules/`, and `src/decision/` keep server validation, policy indicators, and hybrid recommendation logic separate.
- `src/explainability/` explains the ML default-risk prediction only. Policy rules are not mixed into SHAP.
- `web_app/` is a Flask UI over the domain services.
- `config/policy_rules.yaml` contains demonstration thresholds and rule messages.

## Environment

Use Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

The full LendingClub CSV is not committed. Place the accepted-loans CSV at a local path, for example:

```text
data/accepted_loans.csv
```

Training keeps only finished loans with `loan_status` equal to `Fully Paid` or `Charged Off`. The target is formed exactly as `Charged Off -> 1`, `Fully Paid -> 0`. Leakage-prone lender/outcome fields such as `grade`, `sub_grade`, `int_rate`, recovery/payment columns, and hardship/settlement columns are not part of the default raw feature schema.

APR is collected by the web form only for the payment calculator and policy indicators. APR is not passed into the ML pipeline.

## Training

```bash
python -m src.train --data data/accepted_loans.csv --output artifacts/
```

The training command compares Logistic Regression, Decision Tree, and Random Forest on the same preprocessing. Random Forest is calibrated on validation data. The decision threshold is selected on validation data, not test data. The untouched test split is used for final metrics.

Generated files include:

- `artifacts/model.joblib`
- `artifacts/metrics.json`
- `artifacts/calibration_curve.csv`
- `artifacts/calibration_curve.png`

Do not commit large datasets, generated model files, or random plots unless there is a separate review reason.

## Run Flask

The app fails fast if `artifacts/model.joblib` is missing, corrupted, or metadata-incompatible.

```bash
python run_app.py
```

Optional:

```bash
set MODEL_ARTIFACT_PATH=artifacts\model.joblib
set FLASK_DEBUG=1
python run_app.py
```

Health check:

```bash
curl http://127.0.0.1:5000/health
```

## Hybrid Algorithm

1. Validate all submitted fields server-side.
2. Calculate policy indicators, including annuity payment, DSTI after the new loan, disposable income, and whether residence rights cover the term.
3. Evaluate configured hard-stop and soft-flag policy rules.
4. If a hard stop fires, return `MANUAL_REVIEW` or `POLICY_STOP` with reasons.
5. If no hard stop fires, compute calibrated `P(Default)`.
6. Compare `P(Default)` with configured risk bands and threshold.
7. Return one DSS recommendation: `LOWER_RISK`, `ELEVATED_RISK`, `HIGH_RISK`, `MANUAL_REVIEW`, or `POLICY_STOP`.
8. SHAP explains only the ML prediction for the default class. Rule effects are displayed separately.

The DSTI 50% and 65% thresholds are demonstration credit-policy assumptions for the prototype, not mandatory KNF thresholds.

## Tests And Checks

```bash
python -m pytest
ruff check .
ruff format --check .
```

Tests use small synthetic fixtures and do not require the full LendingClub dataset.

## Model Artifact Metadata

The saved artifact includes artifact version, training time, target semantics, class labels, raw feature schema, policy version, decision threshold, library versions, random seed, selected model, and validation/test metrics. Flask validates this metadata during startup.

## Fairness And Limitations

The repository does not claim full fairness validation because the default schema does not include a suitable, ethically acceptable set of sensitive attributes. A subgroup evaluation helper exists for controlled analysis, and tests cover it on synthetic data. Real deployment would need a legally reviewed fairness protocol, proxy-variable analysis, monitoring for historical bias, and human appeal processes.

Methodological limitations remain: LendingClub data may not represent current Polish lending, macroeconomic drift can reduce validity, proxy variables can encode historical bias, and synthetic tests cannot substitute for full data validation.
