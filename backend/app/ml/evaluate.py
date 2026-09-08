"""Evaluation helpers shared across all trained models: grouped CV/split (so a
fight's two augmented mirror-rows never end up on opposite sides of a split),
confusion matrix, log-loss/calibration, ANOVA feature importance, and Pearson
feature correlations - the same methodology used in the reference Musicle
evaluation paper.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def group_train_test_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series, test_size=0.2, random_state=42):
    """Splits by fight_url group, not by row - each fight's original and
    swapped augmented rows always land together, never split across train/test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (
        X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True),
        groups.iloc[train_idx].reset_index(drop=True),
    )


def cross_validate_grouped(model_factory, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits=5, random_state=42):
    """Returns per-fold accuracy for a grouped stratified k-fold CV."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, test_idx in skf.split(X, y, groups):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        scores.append(float(accuracy_score(y.iloc[test_idx], preds)))
    return scores


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series, labels) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    # align predict_proba's column order (model.classes_) with the requested
    # label order - indexed by position, not pandas column selection, since a
    # bool label list (winner target: [False, True]) is misread as a row mask.
    class_index = {cls: i for i, cls in enumerate(model.classes_)}
    proba_aligned = proba[:, [class_index[l] for l in labels]]

    report = classification_report(y_test, preds, labels=labels, output_dict=True, zero_division=0)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, labels=labels, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_test, proba_aligned, labels=labels)),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=labels).tolist(),
        "confusion_matrix_labels": [str(l) for l in labels],
        "classification_report": {str(k): v for k, v in report.items()},
        "test_size": int(len(y_test)),
    }


def anova_feature_importance(X_numeric: pd.DataFrame, y: pd.Series) -> list:
    """One-way ANOVA F-statistic per feature, normalized 0-100 (higher = more
    discriminative between classes)."""
    f_stats, _ = f_classif(X_numeric, y)
    f_stats = np.nan_to_num(f_stats, nan=0.0)
    max_f = f_stats.max() if f_stats.max() > 0 else 1.0
    normalized = (f_stats / max_f) * 100.0
    ranked = sorted(
        zip(X_numeric.columns, normalized.tolist()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [{"feature": feat, "score": round(score, 2)} for feat, score in ranked]


def pearson_correlation_pairs(X_numeric: pd.DataFrame, top_n=15) -> list:
    corr = X_numeric.corr(method="pearson")
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r):
                pairs.append({"feature_a": cols[i], "feature_b": cols[j], "r": round(float(r), 3)})
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
    return pairs[:top_n]
