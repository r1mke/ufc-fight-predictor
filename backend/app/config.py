import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PENDING_DIR = DATA_DIR / "pending"
MODELS_DIR = BASE_DIR / "models"
MODELS_ARCHIVE_DIR = BASE_DIR / "models_archive"

RAW_FIGHTERS_CSV = RAW_DIR / "ufc_fighters_final.csv"
RAW_FIGHTS_CSV = RAW_DIR / "ufc_gold_dataset_final.csv"
USER_SUBMITTED_FIGHTERS_CSV = RAW_DIR / "user_submitted_fighters.csv"
USER_SUBMITTED_FIGHTS_CSV = RAW_DIR / "user_submitted_fights.csv"

PENDING_FIGHTS_CSV = PENDING_DIR / "pending_fights.csv"

FIGHTERS_CLEAN_PARQUET = PROCESSED_DIR / "fighters_clean.parquet"
FIGHTS_CLEAN_PARQUET = PROCESSED_DIR / "fights_clean.parquet"
TRAINING_TABLE_PARQUET = PROCESSED_DIR / "training_table.parquet"

METRICS_JSON = MODELS_DIR / "metrics.json"
IMPUTATION_JSON = MODELS_DIR / "imputation.json"

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

MODEL_NAMES = ["logistic_regression", "random_forest", "lightgbm"]
TARGETS = ["winner", "method"]
METHOD_CLASSES = ["KO_TKO", "Submission", "Decision"]

# Feature-engineering variants, kept fully separate on disk so the default
# ("diff") production models/pipeline are never touched by an experiment.
# "diff": current production approach, (fighter_a - fighter_b) numeric features.
# "concat": experimental - fighter_a and fighter_b numeric features kept
#   separate instead of pre-subtracted, so the model learns its own comparison.
MODEL_VARIANTS = ["diff", "concat"]
DEFAULT_MODEL_VARIANT = "diff"


def models_dir_for(variant: str) -> Path:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if variant == DEFAULT_MODEL_VARIANT:
        return MODELS_DIR
    return MODELS_DIR / "variants" / variant


def training_table_path(variant: str) -> Path:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if variant == DEFAULT_MODEL_VARIANT:
        return TRAINING_TABLE_PARQUET
    return PROCESSED_DIR / f"training_table_{variant}.parquet"

SCRAPE_BASE_URL = "http://ufcstats.com"
SCRAPE_EVENTS_URL = f"{SCRAPE_BASE_URL}/statistics/events/completed"
