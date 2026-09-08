import json

import joblib
import pandas as pd

from app.config import METRICS_JSON, MODEL_NAMES, MODELS_DIR, TARGETS
from app.ml.features import build_pairwise_row, encode_features
from app.services.fighter_service import FighterService, get_fighter_service


class PredictionService:
    def __init__(self, fighter_service: FighterService):
        self.fighter_service = fighter_service
        self.feature_columns = json.loads((MODELS_DIR / "feature_columns.json").read_text())
        self.metrics = json.loads(METRICS_JSON.read_text())
        self.models = {
            (target, name): joblib.load(MODELS_DIR / f"{target}_{name}.joblib")
            for target in TARGETS
            for name in MODEL_NAMES
        }

    def model_metrics(self, target: str, model_name: str) -> dict:
        m = self.metrics["targets"][target]["models"][model_name]
        return {
            "model_name": model_name,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "log_loss": m["log_loss"],
            "cv_mean": m["cv_mean"],
            "cv_std": m["cv_std"],
            "test_size": m["test_size"],
        }

    def top_features(self, target: str, n: int = 8) -> list[dict]:
        return self.metrics["targets"][target]["anova_feature_importance"][:n]

    def build_feature_row(self, fighter_a_id: str, fighter_b_id: str, weight_class: str, is_title_fight: bool) -> pd.DataFrame:
        a_stats, _ = self.fighter_service.current_stats_dict(fighter_a_id)
        b_stats, _ = self.fighter_service.current_stats_dict(fighter_b_id)

        row = build_pairwise_row(a_stats, b_stats)
        row["weight_class"] = weight_class
        row["is_title_fight"] = is_title_fight

        df = pd.DataFrame([row])
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)
        return encode_features(df, reference_columns=self.feature_columns)

    def predict(self, fighter_a_id: str, fighter_b_id: str, weight_class: str, model_name: str, is_title_fight: bool = False) -> dict:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"unknown model_name: {model_name}")

        X = self.build_feature_row(fighter_a_id, fighter_b_id, weight_class, is_title_fight)

        winner_model = self.models[("winner", model_name)]
        method_model = self.models[("method", model_name)]

        winner_proba = winner_model.predict_proba(X)[0]
        winner_classes = list(winner_model.classes_)
        a_win_probability = float(winner_proba[winner_classes.index(True)])

        method_proba = method_model.predict_proba(X)[0]
        method_classes = list(method_model.classes_)
        method_probabilities = {cls: float(p) for cls, p in zip(method_classes, method_proba)}

        return {
            "a_win_probability": a_win_probability,
            "b_win_probability": 1.0 - a_win_probability,
            "method_probabilities": method_probabilities,
        }


_instance: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    global _instance
    if _instance is None:
        _instance = PredictionService(get_fighter_service())
    return _instance


def reset_instance() -> None:
    """Drops the cached singleton so the next call to get_prediction_service()
    reloads models/metrics.json from disk - used after a retrain or a model
    version restore."""
    global _instance
    _instance = None
