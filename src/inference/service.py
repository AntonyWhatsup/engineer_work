from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import DEFAULT_MODEL_PATH, DEFAULT_SCHEMA, validate_policy_config
from src.decision.hybrid import DecisionResult, make_decision
from src.rules.engine import PolicyResult, evaluate_policy
from src.training.artifact import ModelArtifact, load_artifact
from src.validation.schema import ValidatedApplication, validate_model_frame


@dataclass(frozen=True)
class InferenceResult:
    probability_default: float
    decision: DecisionResult
    policy: PolicyResult
    metadata: dict


class CreditRiskService:
    def __init__(self, artifact_path: Path | str = DEFAULT_MODEL_PATH, policy_path: Path | str | None = None):
        self.policy_config = validate_policy_config(policy_path) if policy_path else validate_policy_config()
        self.artifact: ModelArtifact = load_artifact(artifact_path, self.policy_config)

    def predict(self, application: ValidatedApplication) -> InferenceResult:
        frame = application.to_model_frame(DEFAULT_SCHEMA)
        validate_model_frame(frame, DEFAULT_SCHEMA)
        probabilities = self.artifact.pipeline.predict_proba(frame)[0]
        probability_default = float(probabilities[1])
        policy = evaluate_policy(application, self.policy_config)
        decision = make_decision(probability_default, policy, self.policy_config)
        return InferenceResult(
            probability_default=probability_default,
            decision=decision,
            policy=policy,
            metadata=self.artifact.metadata,
        )
