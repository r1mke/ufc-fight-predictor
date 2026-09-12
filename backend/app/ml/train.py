"""Trains and evaluates 3 models x 2 targets (winner, method) on the
leakage-safe training_table.parquet, and writes:
  - models/<target>_<model_name>.joblib   (trained pipeline)
  - models/feature_columns.json           (encoded column order - required to
                                            build aligned live feature vectors)
  - models/metrics.json                   (everything the API and future
                                            paper need: per-model metrics, CV,
                                            ANOVA feature importance, Pearson
                                            correlations)
"""

import json

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import (
    DEFAULT_MODEL_VARIANT, METHOD_CLASSES, MODEL_NAMES, TARGETS, models_dir_for, training_table_path,
)
from app.ml.evaluate import (
    anova_feature_importance,
    cross_validate_grouped,
    evaluate_classifier,
    group_train_test_split,
    pearson_correlation_pairs,
)
from app.ml.features import DIFF_NUMERIC_FEATURES, FEATURE_COLUMNS_BY_VARIANT, encode_features

import joblib

RANDOM_STATE = 42
CLASS_LABELS = {
    "winner": [False, True],
    "method": METHOD_CLASSES,
}


def build_model(model_name: str, random_state=RANDOM_STATE):
    if model_name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ])
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=random_state, verbosity=-1,
        )
    raise ValueError(f"unknown model_name: {model_name}")


def prepare_features(table: pd.DataFrame, variant: str = DEFAULT_MODEL_VARIANT):
    X_raw = table[FEATURE_COLUMNS_BY_VARIANT[variant]].copy()
    bool_cols = X_raw.select_dtypes(include="bool").columns
    X_raw[bool_cols] = X_raw[bool_cols].astype(int)
    X = encode_features(X_raw)
    return X, X.columns.tolist()


def run(variant: str = DEFAULT_MODEL_VARIANT):
    models_dir = models_dir_for(variant)
    models_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(training_table_path(variant))

    X, feature_columns = prepare_features(table, variant=variant)
    groups = table["fight_url"]
    if variant == "diff":
        numeric_cols = [f"{f}_diff" for f in DIFF_NUMERIC_FEATURES]
    else:
        numeric_cols = [f"{f}_a" for f in DIFF_NUMERIC_FEATURES] + [f"{f}_b" for f in DIFF_NUMERIC_FEATURES]

    (models_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2))

    correlation_pairs = pearson_correlation_pairs(X[numeric_cols])

    metrics = {"targets": {}, "correlation_pairs": correlation_pairs, "feature_columns": feature_columns}

    for target in TARGETS:
        y = table["winner_is_a"] if target == "winner" else table["method_class"]
        labels = CLASS_LABELS[target]

        X_train, X_test, y_train, y_test, groups_train = group_train_test_split(X, y, groups, random_state=RANDOM_STATE)

        anova = anova_feature_importance(X_train[numeric_cols], y_train)

        target_result = {
            "labels": [str(l) for l in labels],
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "class_distribution_train": {str(k): int(v) for k, v in y_train.value_counts().items()},
            "class_distribution_test": {str(k): int(v) for k, v in y_test.value_counts().items()},
            "anova_feature_importance": anova,
            "models": {},
        }

        for model_name in MODEL_NAMES:
            cv_scores = cross_validate_grouped(
                lambda: build_model(model_name), X_train, y_train, groups_train, n_splits=5, random_state=RANDOM_STATE
            )
            model = build_model(model_name)
            model.fit(X_train, y_train)

            model_metrics = evaluate_classifier(model, X_test, y_test, labels=labels)
            model_metrics["cv_scores"] = cv_scores
            model_metrics["cv_mean"] = sum(cv_scores) / len(cv_scores)
            model_metrics["cv_std"] = pd.Series(cv_scores).std()

            joblib.dump(model, models_dir / f"{target}_{model_name}.joblib")
            target_result["models"][model_name] = model_metrics

            print(
                f"[{variant}/{target}/{model_name}] acc={model_metrics['accuracy']:.4f} "
                f"macro_f1={model_metrics['macro_f1']:.4f} log_loss={model_metrics['log_loss']:.4f} "
                f"cv={model_metrics['cv_mean']:.4f}+-{model_metrics['cv_std']:.4f}"
            )

        metrics["targets"][target] = target_result

    metrics_path = models_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    import sys

    run(variant=sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_VARIANT)
