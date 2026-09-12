"""Point-in-time feature engineering, shared by training and live inference.

The central rule: a fighter's "form" stats (strikes landed, takedowns, control
time, knockdowns, submission attempts) may only be aggregated from fights that
happened STRICTLY BEFORE the fight being featurized. The current, present-day
career averages in ufc_fighters_final.csv (SLpM, Str_Acc, ...) already include
fights that happened after any given historical fight, so they are NEVER used
as model features - only the per-fight numbers reconstructed here are. Those
career-average columns are still useful for display (fighter profile cards),
just not as ML inputs.

This module is used identically by app/ml/train.py (building the historical
training_table.parquet) and app/services/prediction_service.py (building the
live feature vector for a hypothetical future fight) so both paths compute
features the exact same way.
"""

from datetime import datetime

import numpy as np
import pandas as pd

NUMERIC_FORM_STATS = ["sig_landed", "sig_att", "td_landed", "td_att", "sub_att", "ctrl_sec", "kd"]

STATIC_NUMERIC_FEATURES = ["height_in", "reach_in", "weight_lbs", "age_years"]
FORM_NUMERIC_FEATURES = [
    "prior_fight_count", "prior_wins", "prior_losses", "win_streak", "days_since_last_fight",
] + [f"prior_avg_{c}" for c in NUMERIC_FORM_STATS]

# All numeric features that get turned into an (a - b) difference per fight.
DIFF_NUMERIC_FEATURES = STATIC_NUMERIC_FEATURES + FORM_NUMERIC_FEATURES

CATEGORICAL_FEATURES = ["stance_a", "stance_b", "weight_class"]
FLAG_FEATURES = [
    "is_title_fight",
    "reach_missing_a", "reach_missing_b",
    "height_missing_a", "height_missing_b",
    "weight_missing_a", "weight_missing_b",
    "is_debut_a", "is_debut_b",
]


def build_history_long(fights_clean: pd.DataFrame) -> pd.DataFrame:
    """One row per (fight, fighter-perspective): 2x rows of fights_clean."""
    f1_cols = {
        "F1_Sig_Landed": "sig_landed", "F1_Sig_Att": "sig_att",
        "F1_TD_Landed": "td_landed", "F1_TD_Att": "td_att",
        "F1_Sub_Att": "sub_att", "F1_Ctrl_Sec": "ctrl_sec", "F1_KD": "kd",
    }
    f2_cols = {
        "F2_Sig_Landed": "sig_landed", "F2_Sig_Att": "sig_att",
        "F2_TD_Landed": "td_landed", "F2_TD_Att": "td_att",
        "F2_Sub_Att": "sub_att", "F2_Ctrl_Sec": "ctrl_sec", "F2_KD": "kd",
    }

    base_cols = ["fight_url", "event_date", "weight_class", "is_title_fight", "method_class"]

    f1 = fights_clean[base_cols + list(f1_cols.keys()) + ["fighter_1", "fighter_2", "winner_is_f1"]].rename(
        columns={**f1_cols, "fighter_1": "fighter_name", "fighter_2": "opponent_name", "winner_is_f1": "won"}
    )
    f2 = fights_clean[base_cols + list(f2_cols.keys()) + ["fighter_1", "fighter_2", "winner_is_f1"]].rename(
        columns={**f2_cols, "fighter_2": "fighter_name", "fighter_1": "opponent_name", "winner_is_f1": "won"}
    )
    f2["won"] = ~f2["won"]

    long_df = pd.concat([f1, f2], ignore_index=True)
    return long_df


def _compute_win_streaks(won_series: pd.Series) -> pd.Series:
    """Win streak ENTERING each fight (before that fight's own result is known).
    Assumes won_series is already in chronological order (caller pre-sorts)."""
    streaks = []
    current = 0
    for won in won_series:
        streaks.append(current)
        current = current + 1 if won else 0
    return pd.Series(streaks, index=won_series.index)


def compute_point_in_time_stats(history_long: pd.DataFrame) -> pd.DataFrame:
    """For every (fighter, fight) row, aggregates over that fighter's STRICTLY
    EARLIER fights only. Uses the cumsum-minus-current-row trick: a cumulative
    sum through and including the current row, minus the current row's own
    value, equals the sum over prior rows only.
    """
    df = history_long.sort_values(["fighter_name", "event_date"]).copy()
    g = df.groupby("fighter_name")

    df["prior_fight_count"] = g.cumcount()

    won_int = df["won"].astype(int)
    cum_wins = g["won"].apply(lambda s: s.astype(int).cumsum()).reset_index(level=0, drop=True)
    df["prior_wins"] = cum_wins - won_int
    df["prior_losses"] = df["prior_fight_count"] - df["prior_wins"]

    # df is already sorted by [fighter_name, event_date], so grouping preserves
    # chronological order within each fighter's group without re-sorting.
    df["win_streak"] = df.groupby("fighter_name", group_keys=False)["won"].apply(_compute_win_streaks)

    prev_date = g["event_date"].shift(1)
    df["days_since_last_fight"] = (df["event_date"] - prev_date).dt.days
    df["days_since_last_fight"] = df["days_since_last_fight"].fillna(-1)

    for col in NUMERIC_FORM_STATS:
        # Some historical fights (e.g. manually added or scraped-but-unparsed
        # rows) may have this stat missing. A missing value must not silently
        # bias the average - it contributes neither to the sum nor to the
        # count of fights used as the denominator (NOT the same as
        # prior_fight_count, which counts every prior fight regardless of
        # whether this particular stat is known for it).
        filled_col = df[col].fillna(0)
        cum_col = filled_col.groupby(df["fighter_name"]).cumsum()
        prior_sum = cum_col - filled_col

        known = df[col].notna().astype(int)
        cum_known = known.groupby(df["fighter_name"]).cumsum()
        prior_known_count = cum_known - known

        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"prior_avg_{col}"] = (prior_sum / prior_known_count.replace(0, np.nan)).fillna(0)

    df["is_debut"] = df["prior_fight_count"] == 0

    keep = ["fighter_name", "fight_url", "event_date", "prior_fight_count", "prior_wins",
            "prior_losses", "win_streak", "days_since_last_fight", "is_debut"] + \
           [f"prior_avg_{c}" for c in NUMERIC_FORM_STATS]
    return df[keep]


def aggregate_current_stats(fighter_history: pd.DataFrame) -> dict:
    """Same aggregation as compute_point_in_time_stats, but for a SINGLE
    fighter's FULL history evaluated as of today (i.e. every recorded fight
    counts as 'prior'). Used for live predictions of hypothetical future fights.
    """
    n = len(fighter_history)
    if n == 0:
        result = {"prior_fight_count": 0, "prior_wins": 0, "prior_losses": 0,
                   "win_streak": 0, "days_since_last_fight": -1, "is_debut": True}
        for c in NUMERIC_FORM_STATS:
            result[f"prior_avg_{c}"] = 0.0
        return result

    sorted_hist = fighter_history.sort_values("event_date")
    wins = int(sorted_hist["won"].sum())
    streak = 0
    for won in sorted_hist["won"]:
        streak = streak + 1 if won else 0

    last_date = sorted_hist["event_date"].max()
    days_since = (pd.Timestamp(datetime.now()) - last_date).days

    result = {
        "prior_fight_count": n, "prior_wins": wins, "prior_losses": n - wins,
        "win_streak": streak, "days_since_last_fight": days_since, "is_debut": False,
    }
    for c in NUMERIC_FORM_STATS:
        mean = sorted_hist[c].mean()  # skipna=True by default - already NaN-aware
        result[f"prior_avg_{c}"] = float(mean) if pd.notna(mean) else 0.0
    return result


def compute_primary_weight_class(history_long: pd.DataFrame) -> dict:
    """Fighter's most common weight class across their career, used only to
    condition height imputation (not fed to the model as a feature)."""
    modes = history_long.groupby("fighter_name")["weight_class"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"
    )
    return modes.to_dict()


def fit_imputation_params(fighters_clean: pd.DataFrame, primary_weight_class: dict) -> dict:
    known = fighters_clean.dropna(subset=["height_in", "reach_in"])
    slope, intercept = np.polyfit(known["height_in"], known["reach_in"], 1)

    fc = fighters_clean.copy()
    fc["primary_weight_class"] = fc["fighter_name"].map(primary_weight_class).fillna("Unknown")
    height_median_by_wc = fc.dropna(subset=["height_in"]).groupby("primary_weight_class")["height_in"].median().to_dict()
    global_height_median = float(fc["height_in"].median())
    weight_median_by_wc = fc.dropna(subset=["weight_lbs"]).groupby("primary_weight_class")["weight_lbs"].median().to_dict()
    global_weight_median = float(fc["weight_lbs"].median())

    global_age_median_years = 30.0  # fallback used only if dob is missing at feature time

    return {
        "reach_height_slope": float(slope),
        "reach_height_intercept": float(intercept),
        "height_median_by_weight_class": {k: float(v) for k, v in height_median_by_wc.items()},
        "global_height_median": global_height_median,
        "weight_median_by_weight_class": {k: float(v) for k, v in weight_median_by_wc.items()},
        "global_weight_median": global_weight_median,
        "global_age_median_years": global_age_median_years,
    }


def apply_static_imputation(fighters_clean: pd.DataFrame, primary_weight_class: dict, params: dict) -> pd.DataFrame:
    fc = fighters_clean.copy()
    fc["primary_weight_class"] = fc["fighter_name"].map(primary_weight_class).fillna("Unknown")

    fc["height_missing"] = fc["height_in"].isna()
    height_fallback = fc["primary_weight_class"].map(params["height_median_by_weight_class"]).fillna(
        params["global_height_median"]
    )
    fc["height_in"] = fc["height_in"].fillna(height_fallback)

    fc["reach_missing"] = fc["reach_in"].isna()
    reach_fallback = params["reach_height_slope"] * fc["height_in"] + params["reach_height_intercept"]
    fc["reach_in"] = fc["reach_in"].fillna(reach_fallback)

    fc["weight_missing"] = fc["weight_lbs"].isna()
    weight_fallback = fc["primary_weight_class"].map(params["weight_median_by_weight_class"]).fillna(
        params["global_weight_median"]
    )
    fc["weight_lbs"] = fc["weight_lbs"].fillna(weight_fallback)

    return fc


def age_years_at(dob, event_date, fallback_years: float) -> float:
    if pd.isna(dob) or pd.isna(event_date):
        return fallback_years
    return (event_date - dob).days / 365.25


def build_pairwise_row(a_stats: dict, b_stats: dict) -> dict:
    """Given two dicts of identically-named raw feature values for fighter A
    and fighter B, returns the (a - b) diff feature dict used as model input."""
    row = {}
    for feat in DIFF_NUMERIC_FEATURES:
        row[f"{feat}_diff"] = a_stats[feat] - b_stats[feat]
    row["stance_a"] = a_stats["stance"]
    row["stance_b"] = b_stats["stance"]
    row["reach_missing_a"] = a_stats["reach_missing"]
    row["reach_missing_b"] = b_stats["reach_missing"]
    row["height_missing_a"] = a_stats["height_missing"]
    row["height_missing_b"] = b_stats["height_missing"]
    row["weight_missing_a"] = a_stats["weight_missing"]
    row["weight_missing_b"] = b_stats["weight_missing"]
    row["is_debut_a"] = a_stats["is_debut"]
    row["is_debut_b"] = b_stats["is_debut"]
    return row


def build_pairwise_row_concat(a_stats: dict, b_stats: dict) -> dict:
    """Same inputs as build_pairwise_row(), but keeps fighter A's and fighter
    B's raw values as separate columns instead of pre-computing (a - b). Used
    by the "concat" variant, which lets the model learn its own comparison
    instead of being forced into a linear difference."""
    row = {}
    for feat in DIFF_NUMERIC_FEATURES:
        row[f"{feat}_a"] = a_stats[feat]
        row[f"{feat}_b"] = b_stats[feat]
    row["stance_a"] = a_stats["stance"]
    row["stance_b"] = b_stats["stance"]
    row["reach_missing_a"] = a_stats["reach_missing"]
    row["reach_missing_b"] = b_stats["reach_missing"]
    row["height_missing_a"] = a_stats["height_missing"]
    row["height_missing_b"] = b_stats["height_missing"]
    row["weight_missing_a"] = a_stats["weight_missing"]
    row["weight_missing_b"] = b_stats["weight_missing"]
    row["is_debut_a"] = a_stats["is_debut"]
    row["is_debut_b"] = b_stats["is_debut"]
    return row


ALL_FEATURE_COLUMNS = [f"{f}_diff" for f in DIFF_NUMERIC_FEATURES] + CATEGORICAL_FEATURES + FLAG_FEATURES

# Alternative feature schema: same underlying per-fighter numbers, but given to
# the model as separate fighter_a/fighter_b columns instead of a pre-computed
# (a - b) difference. See build_pairwise_row_concat() and variant="concat" on
# build_training_table()/build_and_save().
ALL_FEATURE_COLUMNS_CONCAT = (
    [f"{f}_a" for f in DIFF_NUMERIC_FEATURES]
    + [f"{f}_b" for f in DIFF_NUMERIC_FEATURES]
    + CATEGORICAL_FEATURES
    + FLAG_FEATURES
)

FEATURE_COLUMNS_BY_VARIANT = {"diff": ALL_FEATURE_COLUMNS, "concat": ALL_FEATURE_COLUMNS_CONCAT}


def encode_features(df: pd.DataFrame, reference_columns=None):
    """One-hot encodes categorical columns. If reference_columns is given
    (the training set's final encoded column list), the result is reindexed
    to exactly match it - this is what keeps live inference and training
    features aligned (same columns, same order, unseen categories -> 0)."""
    encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
    if reference_columns is not None:
        encoded = encoded.reindex(columns=reference_columns, fill_value=0)
    return encoded


def _dedupe_fighters_by_record(fighters_clean: pd.DataFrame) -> pd.DataFrame:
    """A handful of names belong to two different real fighters (different
    Fighter_URL). Fights only carry names, not URLs, so we can't disambiguate
    per-fight - as a documented limitation we attribute all of a name's fight
    history to whichever of the duplicate profiles has the larger recorded
    record (wins+losses+draws). Affects 7 names / ~0.3% of fighters."""
    fc = fighters_clean.copy()
    fc["_record_total"] = fc["wins"] + fc["losses"] + fc["draws"]
    return fc.sort_values("_record_total", ascending=False).drop_duplicates("fighter_name", keep="first")


def build_training_table(fighters_clean: pd.DataFrame, fights_clean: pd.DataFrame, variant: str = "diff"):
    """Assembles the full (augmented, leakage-safe) training table. Returns
    (training_df, imputation_params) - the latter must be persisted and reused
    at inference time.

    variant="diff" (default, unchanged): numeric features are (a - b) differences.
    variant="concat": numeric features are fighter_a and fighter_b raw values,
    kept separate instead of pre-subtracted."""
    if variant not in FEATURE_COLUMNS_BY_VARIANT:
        raise ValueError(f"unknown variant: {variant}")
    history_long = build_history_long(fights_clean)
    prior_stats = compute_point_in_time_stats(history_long)
    primary_wc = compute_primary_weight_class(history_long)

    imputation_params = fit_imputation_params(fighters_clean, primary_wc)
    fighters_imputed = apply_static_imputation(fighters_clean, primary_wc, imputation_params)
    fighters_dedup = _dedupe_fighters_by_record(fighters_imputed)

    static_cols = fighters_dedup[["fighter_name", "height_in", "reach_in", "weight_lbs", "stance", "dob", "height_missing", "reach_missing", "weight_missing"]]
    per_fighter_fight = prior_stats.merge(static_cols, on="fighter_name", how="left")
    per_fighter_fight["age_years"] = per_fighter_fight.apply(
        lambda r: age_years_at(r["dob"], r["event_date"], imputation_params["global_age_median_years"]), axis=1
    )
    per_fighter_fight = per_fighter_fight.drop(columns=["dob"])

    raw_cols = [c for c in per_fighter_fight.columns if c not in ("fighter_name", "fight_url", "event_date")]

    side_a = per_fighter_fight.rename(columns={**{c: f"{c}_a" for c in raw_cols}, "fighter_name": "fighter_1"})
    side_a = side_a.drop(columns=["event_date"])
    side_b = per_fighter_fight.rename(columns={**{c: f"{c}_b" for c in raw_cols}, "fighter_name": "fighter_2"})
    side_b = side_b.drop(columns=["event_date"])

    merged = fights_clean.merge(side_a, on=["fight_url", "fighter_1"], how="left")
    merged = merged.merge(side_b, on=["fight_url", "fighter_2"], how="left")
    merged["winner_is_a"] = merged["winner_is_f1"]

    for feat in DIFF_NUMERIC_FEATURES:
        merged[f"{feat}_diff"] = merged[f"{feat}_a"] - merged[f"{feat}_b"]

    swap_pairs = [(f"{c}_a", f"{c}_b") for c in raw_cols]

    swapped = merged.copy()
    for a_col, b_col in swap_pairs:
        swapped[a_col] = merged[b_col]
        swapped[b_col] = merged[a_col]
    for feat in DIFF_NUMERIC_FEATURES:
        swapped[f"{feat}_diff"] = -merged[f"{feat}_diff"]
    swapped["winner_is_a"] = ~merged["winner_is_a"]

    full = pd.concat([merged, swapped], ignore_index=True)

    feature_columns = FEATURE_COLUMNS_BY_VARIANT[variant]
    if variant == "diff":
        dropna_subset = [f"{f}_diff" for f in DIFF_NUMERIC_FEATURES]
    else:
        dropna_subset = [f"{f}_a" for f in DIFF_NUMERIC_FEATURES] + [f"{f}_b" for f in DIFF_NUMERIC_FEATURES]

    keep = feature_columns + ["winner_is_a", "method_class", "fight_url", "event_date"]
    training_table = full[keep].dropna(subset=dropna_subset)

    return training_table.reset_index(drop=True), imputation_params


def build_and_save(variant: str = "diff"):
    """Reads the cleaned parquet files, builds the training table, and writes
    training_table.parquet (or training_table_<variant>.parquet for a
    non-default variant) + imputation.json. Shared by the CLI entry point
    below and app.services.retrain_service (which always uses the default
    "diff" variant - the automated admin retrain flow is unaffected by
    variant support)."""
    import json

    from app.config import FIGHTERS_CLEAN_PARQUET, FIGHTS_CLEAN_PARQUET, IMPUTATION_JSON, training_table_path

    fighters = pd.read_parquet(FIGHTERS_CLEAN_PARQUET)
    fights = pd.read_parquet(FIGHTS_CLEAN_PARQUET)

    table, imputation_params = build_training_table(fighters, fights, variant=variant)
    table.to_parquet(training_table_path(variant), index=False)
    # Imputation params depend only on fighters_clean, not on the feature
    # schema, so they're identical across variants - written once, shared.
    IMPUTATION_JSON.write_text(json.dumps(imputation_params, indent=2))
    return table


if __name__ == "__main__":
    import sys

    variant = sys.argv[1] if len(sys.argv) > 1 else "diff"
    table = build_and_save(variant=variant)

    print(f"variant={variant}: training_table: {len(table)} rows, {len(FEATURE_COLUMNS_BY_VARIANT[variant])} raw feature columns")
    print(f"winner_is_a distribution:\n{table['winner_is_a'].value_counts()}")
    print(f"method_class distribution:\n{table['method_class'].value_counts()}")
