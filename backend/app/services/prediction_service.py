import json

import joblib
import pandas as pd

from app.config import DEFAULT_MODEL_VARIANT, MODEL_NAMES, MODEL_VARIANTS, TARGETS, models_dir_for
from app.ml.features import build_pairwise_row, build_pairwise_row_concat, encode_features
from app.services.fighter_service import FighterService, get_fighter_service

ROW_BUILDERS = {"diff": build_pairwise_row, "concat": build_pairwise_row_concat}


class _VariantModels:
    """Lazily-loaded feature_columns/metrics/models for one feature-engineering
    variant. The default ("diff") variant is loaded eagerly by
    PredictionService.__init__ exactly as before; other variants are only
    loaded the first time they're actually requested, so a deployment that
    never trained an experimental variant behaves identically to before."""

    def __init__(self, variant: str):
        models_dir = models_dir_for(variant)
        feature_columns_path = models_dir / "feature_columns.json"
        metrics_path = models_dir / "metrics.json"
        if not feature_columns_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(
                f"variant '{variant}' has not been trained yet - expected {feature_columns_path} and {metrics_path}"
            )

        self.feature_columns = json.loads(feature_columns_path.read_text())
        self.metrics = json.loads(metrics_path.read_text())
        self.models = {
            (target, name): joblib.load(models_dir / f"{target}_{name}.joblib")
            for target in TARGETS
            for name in MODEL_NAMES
        }


class PredictionService:
    def __init__(self, fighter_service: FighterService):
        self.fighter_service = fighter_service
        self._variants: dict[str, _VariantModels] = {DEFAULT_MODEL_VARIANT: _VariantModels(DEFAULT_MODEL_VARIANT)}

    def _variant_models(self, variant: str) -> _VariantModels:
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        if variant not in self._variants:
            self._variants[variant] = _VariantModels(variant)
        return self._variants[variant]

    def model_metrics(self, target: str, model_name: str, variant: str = DEFAULT_MODEL_VARIANT) -> dict:
        m = self._variant_models(variant).metrics["targets"][target]["models"][model_name]
        return {
            "model_name": model_name,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "log_loss": m["log_loss"],
            "cv_mean": m["cv_mean"],
            "cv_std": m["cv_std"],
            "test_size": m["test_size"],
        }

    def top_features(self, target: str, n: int = 8, variant: str = DEFAULT_MODEL_VARIANT) -> list[dict]:
        return self._variant_models(variant).metrics["targets"][target]["anova_feature_importance"][:n]

    def get_metrics(self, variant: str = DEFAULT_MODEL_VARIANT) -> dict:
        return self._variant_models(variant).metrics

    def build_feature_row(
        self, fighter_a_id: str, fighter_b_id: str, weight_class: str, is_title_fight: bool, variant: str = DEFAULT_MODEL_VARIANT
    ) -> pd.DataFrame:
        a_stats, _ = self.fighter_service.current_stats_dict(fighter_a_id)
        b_stats, _ = self.fighter_service.current_stats_dict(fighter_b_id)

        row = ROW_BUILDERS[variant](a_stats, b_stats)
        row["weight_class"] = weight_class
        row["is_title_fight"] = is_title_fight

        df = pd.DataFrame([row])
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)
        return encode_features(df, reference_columns=self._variant_models(variant).feature_columns)

    def predict(
        self,
        fighter_a_id: str,
        fighter_b_id: str,
        weight_class: str,
        model_name: str,
        is_title_fight: bool = False,
        variant: str = DEFAULT_MODEL_VARIANT,
    ) -> dict:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"unknown model_name: {model_name}")

        variant_models = self._variant_models(variant)
        X = self.build_feature_row(fighter_a_id, fighter_b_id, weight_class, is_title_fight, variant=variant)

        winner_model = variant_models.models[("winner", model_name)]
        method_model = variant_models.models[("method", model_name)]

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
