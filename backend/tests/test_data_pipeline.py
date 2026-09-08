from app.ml.data_pipeline import (
    is_title_fight,
    normalize_method,
    normalize_weight_class,
    parse_height_to_inches,
    parse_percent,
    parse_reach_to_inches,
    parse_weight_lbs,
)


def test_parse_height_to_inches():
    assert parse_height_to_inches("5' 8\"") == 68.0
    assert parse_height_to_inches("6' 2\"") == 74.0
    assert parse_height_to_inches("") is None
    assert parse_height_to_inches(None) is None


def test_parse_reach_to_inches():
    assert parse_reach_to_inches('66.0"') == 66.0
    assert parse_reach_to_inches("") is None


def test_parse_weight_lbs():
    assert parse_weight_lbs("155 lbs.") == 155.0
    assert parse_weight_lbs("") is None


def test_parse_percent():
    assert parse_percent("20%") == 0.20
    assert parse_percent("0%") == 0.0
    assert parse_percent("") is None


def test_normalize_weight_class_canonical_categories():
    assert normalize_weight_class("Lightweight Bout") == "Lightweight"
    assert normalize_weight_class("UFC Lightweight Title Bout") == "Lightweight"
    assert normalize_weight_class("Women's Bantamweight Bout") == "Women's Bantamweight"
    # substring trap: "Light Heavyweight" must not match plain "Heavyweight" first
    assert normalize_weight_class("Light Heavyweight Bout") == "Light Heavyweight"
    assert normalize_weight_class("UFC Heavyweight Title Bout") == "Heavyweight"


def test_normalize_weight_class_unknown_fallback():
    assert normalize_weight_class("UFC Superfight Championship Bout") == "Unknown"
    assert normalize_weight_class("") == "Unknown"
    assert normalize_weight_class(None) == "Unknown"


def test_is_title_fight():
    assert is_title_fight("UFC Lightweight Title Bout") is True
    assert is_title_fight("Lightweight Bout") is False


def test_normalize_method_collapses_to_three_classes():
    assert normalize_method("KO/TKO") == "KO_TKO"
    assert normalize_method("TKO - Doctor's Stoppage") == "KO_TKO"
    assert normalize_method("Submission") == "Submission"
    assert normalize_method("Decision - Unanimous") == "Decision"
    assert normalize_method("Decision - Split") == "Decision"
    assert normalize_method("Decision - Majority") == "Decision"


def test_normalize_method_drops_ambiguous_outcomes():
    assert normalize_method("Overturned") is None
    assert normalize_method("Could Not Continue") is None
    assert normalize_method("DQ") is None
    assert normalize_method("Other") is None
