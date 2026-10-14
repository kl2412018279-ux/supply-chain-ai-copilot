import pandas as pd
import pytest

from src.delays.data import RISK_LABELS, generate_shipments, risk_class
from src.delays.model import DelayInput, DelayModel, ModelNotTrainedError, train, upcoming_shipments


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("delay") / "delay_model.joblib"
    history = generate_shipments(start="2025-06-01", end="2026-06-30", per_day=8)
    train(history, model_path=path)
    return DelayModel.load(path)


def test_risk_class_thresholds():
    assert risk_class(pd.Series([0, 11.9, 12.0, 48.0, 48.1])).tolist() == [
        "Low", "Low", "Medium", "Medium", "High",
    ]  # fmt: skip


def test_history_drivers_push_delays_in_the_right_direction():
    df = generate_shipments(start="2025-01-01", end="2025-12-31", per_day=6)
    assert set(df["risk"]) == set(RISK_LABELS)
    assert (
        df[df.customs_inspection == 1].delay_hours.mean()
        > df[df.customs_inspection == 0].delay_hours.mean()
    )
    sea = df[df["mode"] == "sea"]
    assert sea[sea.monsoon == 1].delay_hours.mean() > sea[sea.monsoon == 0].delay_hours.mean() + 15
    assert (
        df.groupby("mode").delay_hours.mean()["sea"]
        > df.groupby("mode").delay_hours.mean()["truck"]
    )


def test_model_beats_baselines(model):
    m, b = model.metrics, model.bundle["baselines"]
    assert m["macro_f1"] > b["majority_per_mode"]["macro_f1"] + 0.15
    assert m["high_recall"] > 0.3 and b["majority_class"]["high_recall"] == 0


def test_bad_conditions_score_higher_than_good(model):
    calm = model.predict(DelayInput(destination="Shah Alam", eta="2026-07-08", congestion=0.6))
    storm = model.predict(
        DelayInput(
            destination="Kuching", carrier="nusantara", eta="2026-12-10", congestion=0.9,
            commodity="chemicals", customs_inspection=True, containers=2,
        )
    )  # fmt: skip
    assert calm.risk == "Low" and storm.risk == "High"
    assert storm.risk_score > calm.risk_score
    assert storm.demurrage_exposure_rm > calm.demurrage_exposure_rm
    assert sum(storm.probabilities.values()) == pytest.approx(1, abs=0.01)


def test_explanations_name_the_real_drivers(model):
    out = model.predict(
        DelayInput(
            destination="Kuching", carrier="nusantara", eta="2026-12-10", congestion=0.9,
            customs_inspection=True,
        )
    )  # fmt: skip
    factors = [d.factor for d in out.drivers]
    assert "Monsoon season" in factors and "Customs inspection" in factors
    assert out.drivers == sorted(out.drivers, key=lambda d: d.contribution, reverse=True)


def test_features_derived_from_eta(model):
    f = model.features_for(DelayInput(destination="Penang", carrier="cempaka", eta="2026-02-15"))
    assert f["mode"] == "truck" and f["distance_km"] == 370
    assert f["monsoon"] == 1 and f["days_to_holiday"] == 2 and f["weekend_arrival"] == 1


def test_invalid_inputs(model):
    with pytest.raises(ValueError):
        model.predict(
            DelayInput(destination="Kuching", carrier="hijau")
        )  # trucks don't go to Borneo
    with pytest.raises(KeyError):
        model.predict(DelayInput(destination="KL", carrier="ghost-trucks"))
    with pytest.raises(ModelNotTrainedError):
        DelayModel.load("/nonexistent/model.joblib")


def test_upcoming_manifest_is_scored(model):
    manifest = upcoming_shipments(start="2026-12-01", days=7)
    scored = model.score_frame(manifest)
    assert scored["eta"].min() == "2026-12-01" and scored["eta"].max() <= "2026-12-07"
    assert {"p_low", "p_medium", "p_high", "risk", "risk_score"} <= set(scored.columns)
    assert "delay_hours" not in scored.columns  # outcomes are not known yet


def test_heavy_rain_raises_risk_and_is_explained(model):
    base = dict(destination="shah-alam", eta="2026-06-10", congestion=0.6)
    dry = model.predict(DelayInput(**base, rain_mm=0))
    wet = model.predict(DelayInput(**base, rain_mm=80))
    score = lambda p: p.probabilities["Medium"] / 2 + p.probabilities["High"]  # noqa: E731
    assert score(wet) > score(dry)
    assert wet.weather.source == "entered by user" and wet.weather.rain_mm == 80
    assert "Heavy rain" in [d.factor for d in wet.drivers]


def test_rain_defaults_to_offline_climatology(model):
    pred = model.predict(DelayInput(destination="kuching", carrier="nusantara", eta="2026-12-10"))
    assert "offline" in pred.weather.source and pred.weather.rain_mm > 0


def test_stale_model_is_detected(model, tmp_path):
    import joblib

    from src.delays.model import model_is_current

    path = tmp_path / "old.joblib"
    assert not model_is_current(path)  # missing
    joblib.dump({**model.bundle, "features": ["congestion"]}, path)
    assert not model_is_current(path)  # trained before a feature was added
    joblib.dump(model.bundle, path)
    assert model_is_current(path)
