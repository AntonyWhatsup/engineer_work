from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import DEFAULT_SCHEMA


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        [
            ("num", numeric, DEFAULT_SCHEMA.numeric_features),
            ("cat", categorical, DEFAULT_SCHEMA.categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_models(seed: int = 42) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "decision_tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=80, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=-1
        ),
    }


def build_pipeline(model) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
