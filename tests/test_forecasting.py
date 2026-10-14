import numpy as np
import pandas as pd
import pytest

from src.forecasting.data import (
    FEATURES,
    build_features,
    calendar_features,
    clean_volumes,
    features_from_history,
    load_volumes,
    normalize_commodity,
)
from src.forecasting.predict import DemandForecaster
from src.forecasting.train import regression_metrics


def test_clean_volumes_fills_missing_days_and_clips_negatives():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-03", "2024-01-03"],
            "commodity": ["chemicals"] * 3,
            "teu": [5, -2, 4],
        }
    )
    df = clean_volumes(raw)
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert df["teu"].tolist() == [5, 0, 4]  # gap filled with 0, -2 clipped to 0


def test_load_volumes_accepts_column_aliases(tmp_path):
    path = tmp_path / "export.csv"
    path.write_text("Date,Series,Volume\n2026-01-01,Palm Oil,13\n2026-01-02,CPO,11\n")
    df = load_volumes(path)
    assert set(df["commodity"]) == {"palm-oil"}
    assert df["teu"].sum() == 24


def test_load_volumes_rejects_unknown_format(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,amount\n2018-01-01,1\n")
    with pytest.raises(ValueError):
        load_volumes(path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Palm Oil", "palm-oil"),
        ("CPO", "palm-oil"),
        ("chemical", "chemicals"),
        ("resin", "plastics"),
    ],
)
def test_normalize_commodity(raw, expected):
    assert normalize_commodity(raw) == expected


def test_holiday_feature_counts_days_to_chinese_new_year():
    feats = calendar_features(pd.to_datetime(["2026-02-17", "2026-02-14", "2026-07-01"]))
    assert feats["days_to_holiday"].tolist() == [0, 3, 15]


def test_synthetic_data_dips_around_holidays(volumes_df):
    daily = volumes_df.groupby("date")["teu"].sum()
    near_cny = daily.loc["2026-02-15":"2026-02-19"].mean()
    normal = daily.loc["2026-01-05":"2026-01-30"].mean()
    assert near_cny < 0.6 * normal


def test_inference_features_match_training_features(volumes_df):
    """Recursive forecasting is only valid if both feature paths agree exactly."""
    categories = sorted(volumes_df["commodity"].unique())
    feats = build_features(volumes_df, categories)
    row = feats.iloc[100]
    history = (
        volumes_df[
            (volumes_df["commodity"] == row["commodity"]) & (volumes_df["date"] < row["date"])
        ]
        .sort_values("date")["teu"]
        .to_numpy(dtype=float)
    )
    online = features_from_history(history, row["date"], categories.index(row["commodity"]))
    for name in FEATURES:
        assert online[name] == pytest.approx(float(row[name])), name


def test_model_beats_seasonal_naive_baseline(trained_model):
    assert trained_model.model_path.exists()
    assert trained_model.metrics["mae"] < trained_model.baseline_metrics["mae"]
    assert trained_model.metrics["r2"] > 0.3


def test_forecast_shape_and_values(trained_model):
    forecaster = DemandForecaster.load(trained_model.model_path)
    df = forecaster.forecast("Palm Oil", horizon_days=10)
    assert len(df) == 10
    assert (df["forecast"] >= 0).all()
    assert pd.to_datetime(df["date"]).is_monotonic_increasing


def test_forecast_rejects_unknown_commodity_and_bad_horizon(trained_model):
    forecaster = DemandForecaster.load(trained_model.model_path)
    with pytest.raises(KeyError):
        forecaster.forecast("bananas", 7)
    with pytest.raises(ValueError):
        forecaster.forecast("chemicals", 0)


def test_regression_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0 and m["rmse"] == 0 and m["r2"] == 1


def test_grid_search_picks_a_candidate(volumes_df, tmp_path):
    from src.forecasting.train import SEARCH_SPACE, train

    result = train(volumes_df, model_path=tmp_path / "m.joblib", holdout_days=28, tune=True)
    assert result.bundle["params"]["num_leaves"] in SEARCH_SPACE["num_leaves"]
    assert result.bundle["params"]["learning_rate"] in SEARCH_SPACE["learning_rate"]
