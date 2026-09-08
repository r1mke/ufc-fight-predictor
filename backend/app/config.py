import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

RAW_FIGHTERS_CSV = RAW_DIR / "ufc_fighters_final.csv"
RAW_FIGHTS_CSV = RAW_DIR / "ufc_gold_dataset_final.csv"

FIGHTERS_CLEAN_PARQUET = PROCESSED_DIR / "fighters_clean.parquet"
FIGHTS_CLEAN_PARQUET = PROCESSED_DIR / "fights_clean.parquet"
TRAINING_TABLE_PARQUET = PROCESSED_DIR / "training_table.parquet"

METRICS_JSON = MODELS_DIR / "metrics.json"
IMPUTATION_JSON = MODELS_DIR / "imputation.json"

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")

MODEL_NAMES = ["logistic_regression", "random_forest", "lightgbm"]
TARGETS = ["winner", "method"]
METHOD_CLASSES = ["KO_TKO", "Submission", "Decision"]
