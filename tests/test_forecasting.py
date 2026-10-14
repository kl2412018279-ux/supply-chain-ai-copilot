import numpy as np
import pandas as pd
import pytest

from src.forecasting.data import (
    FEATURES,
    build_features,
    clean_sales,
    features_from_history,
    load_sales,
)
from src.forecasting.predict import DemandForecaster
from src.forecasting.train import regression_metrics


def test_clean_sales_fills_missing_days_and_clips_negatives():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-03", "2024-01-03"],
            "sku": ["A", "A", "A"],
            "sales": [5, -2, 4],
        }
    )
    df = clean_sales(raw)
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert df["sales"].tolist() == [5, 0, 4]  # gap filled with 0, -2 clipped to 0


def test_load_sales_accepts_kaggle_format(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("date,store,item,sales\n2018-01-01,1,1,13\n2018-01-02,1,1,11\n")
    df = load_sales(path)
    assert set(df["sku"]) == {"S1-I1"}
    assert df["sales"].sum() == 24


def test_load_sales_rejects_unknown_format(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,qty\n2018-01-01,1\n")
    with pytest.raises(ValueError):
        load_sales(path)


def test_inference_features_match_training_features(sales_df):
    """Recursive forecasting is only valid if both feature paths agree exactly."""
    skus = sorted(sales_df["sku"].unique())
    feats = build_features(sales_df, skus)
    row = feats.iloc[100]
    history = (
        sales_df[(sales_df["sku"] == row["sku"]) & (sales_df["date"] < row["date"])]
        .sort_values("date")["sales"]
        .to_numpy(dtype=float)
    )
    online = features_from_history(history, row["date"], skus.index(row["sku"]))
    for name in FEATURES:
        assert online[name] == pytest.approx(float(row[name])), name


def test_model_beats_seasonal_naive_baseline(trained_model):
    assert trained_model.model_path.exists()
    assert trained_model.metrics["mae"] < trained_model.baseline_metrics["mae"]
    assert trained_model.metrics["r2"] > 0.5


def test_forecast_shape_and_values(trained_model):
    forecaster = DemandForecaster.load(trained_model.model_path)
    df = forecaster.forecast("sku-101", horizon_days=10)
    assert len(df) == 10
    assert (df["forecast"] >= 0).all()
    assert pd.to_datetime(df["date"]).is_monotonic_increasing


def test_forecast_rejects_unknown_sku_and_bad_horizon(trained_model):
    forecaster = DemandForecaster.load(trained_model.model_path)
    with pytest.raises(KeyError):
        forecaster.forecast("SKU-999", 7)
    with pytest.raises(ValueError):
        forecaster.forecast("SKU-101", 0)


def test_regression_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0 and m["rmse"] == 0 and m["r2"] == 1


def test_grid_search_picks_a_candidate(sales_df, tmp_path):
    from src.forecasting.train import SEARCH_SPACE, train

    result = train(sales_df, model_path=tmp_path / "m.joblib", holdout_days=28, tune=True)
    assert result.bundle["params"]["num_leaves"] in SEARCH_SPACE["num_leaves"]
    assert result.bundle["params"]["learning_rate"] in SEARCH_SPACE["learning_rate"]
