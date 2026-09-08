import pandas as pd

from app.ml.features import (
    apply_static_imputation,
    compute_point_in_time_stats,
    fit_imputation_params,
)


def _history_row(fighter_name, fight_url, date, won, sig_landed):
    return {
        "fighter_name": fighter_name,
        "fight_url": fight_url,
        "event_date": pd.Timestamp(date),
        "won": won,
        "sig_landed": sig_landed,
        "sig_att": sig_landed * 2,
        "td_landed": 0,
        "td_att": 0,
        "sub_att": 0,
        "ctrl_sec": 0,
        "kd": 0,
    }


def test_point_in_time_stats_excludes_current_fight():
    """The whole point of this module: a fight's prior-stat features must be
    computed only from that fighter's EARLIER fights, never from itself."""
    history = pd.DataFrame([
        _history_row("A", "f1", "2020-01-01", won=True, sig_landed=10),
        _history_row("A", "f2", "2020-02-01", won=False, sig_landed=20),
        _history_row("A", "f3", "2020-03-01", won=True, sig_landed=30),
    ])

    stats = compute_point_in_time_stats(history).set_index("fight_url")

    debut = stats.loc["f1"]
    assert debut["prior_fight_count"] == 0
    assert debut["is_debut"]
    assert debut["prior_avg_sig_landed"] == 0
    assert debut["days_since_last_fight"] == -1

    second = stats.loc["f2"]
    assert second["prior_fight_count"] == 1
    assert second["prior_wins"] == 1
    assert second["prior_avg_sig_landed"] == 10  # only fight f1, not f2's own 20
    assert second["win_streak"] == 1  # entering f2, coming off a win in f1
    assert second["days_since_last_fight"] == 31  # Jan 1 -> Feb 1

    third = stats.loc["f3"]
    assert third["prior_fight_count"] == 2
    assert third["prior_wins"] == 1
    assert third["prior_losses"] == 1
    # average of f1 (10) and f2 (20) only - NOT including f3's own 30
    assert third["prior_avg_sig_landed"] == 15
    assert third["win_streak"] == 0  # entering f3, coming off a loss in f2


def test_win_streak_resets_on_loss():
    history = pd.DataFrame([
        _history_row("A", "f1", "2020-01-01", won=True, sig_landed=1),
        _history_row("A", "f2", "2020-02-01", won=True, sig_landed=1),
        _history_row("A", "f3", "2020-03-01", won=False, sig_landed=1),
        _history_row("A", "f4", "2020-04-01", won=True, sig_landed=1),
    ])
    stats = compute_point_in_time_stats(history).set_index("fight_url")
    assert stats.loc["f1", "win_streak"] == 0
    assert stats.loc["f2", "win_streak"] == 1
    assert stats.loc["f3", "win_streak"] == 2
    assert stats.loc["f4", "win_streak"] == 0  # f3 was a loss


def test_stats_are_independent_per_fighter():
    history = pd.DataFrame([
        _history_row("A", "f1", "2020-01-01", won=True, sig_landed=100),
        _history_row("B", "f2", "2020-01-15", won=True, sig_landed=5),
    ])
    stats = compute_point_in_time_stats(history).set_index("fight_url")
    # B's debut stats must not be contaminated by A's fight history
    assert stats.loc["f2", "prior_fight_count"] == 0
    assert stats.loc["f2", "is_debut"]


def _fighters_df():
    return pd.DataFrame({
        "fighter_name": ["A", "B", "C", "D"],
        "height_in": [70.0, 72.0, None, 68.0],
        "reach_in": [72.0, 74.0, 70.0, None],
        "weight_lbs": [155.0, 170.0, 155.0, None],
        "wins": [1, 1, 1, 1],
        "losses": [0, 0, 0, 0],
        "draws": [0, 0, 0, 0],
        "dob": [pd.Timestamp("1990-01-01")] * 4,
    })


def test_imputation_fills_missing_height_reach_weight():
    fighters = _fighters_df()
    primary_wc = {"A": "Lightweight", "B": "Welterweight", "C": "Lightweight", "D": "Lightweight"}

    params = fit_imputation_params(fighters, primary_wc)
    imputed = apply_static_imputation(fighters, primary_wc, params).set_index("fighter_name")

    # C's height was missing -> filled from Lightweight median of known heights (A=70, D=68)
    assert imputed.loc["C", "height_missing"]
    assert imputed.loc["C", "height_in"] == 69.0
    # C's reach was already known -> untouched
    assert imputed.loc["C", "reach_in"] == 70.0

    # D's height was known (68) -> untouched, D's reach was missing -> regression fallback
    assert not imputed.loc["D", "height_missing"]
    assert imputed.loc["D", "reach_missing"]
    # reach ~ height fit exactly through A(70,72) and B(72,74) -> reach = height + 2
    assert imputed.loc["D", "reach_in"] == 70.0

    # D's weight was missing -> Lightweight median of known weights (A=155, C=155)
    assert imputed.loc["D", "weight_missing"]
    assert imputed.loc["D", "weight_lbs"] == 155.0

    # untouched fighters keep their original values and missing=False
    assert not imputed.loc["A", "height_missing"]
    assert not imputed.loc["A", "reach_missing"]
    assert imputed.loc["A", "height_in"] == 70.0
