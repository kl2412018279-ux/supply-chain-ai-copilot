"""Delay-risk classifier (Low / Medium / High) with per-shipment explanations.

* Random forest on one-hot categoricals + numeric drivers, class-balanced so the rare
  High class is not ignored.
* Time-based split (train on the past, test on the most recent 20%), benchmarked
  against two baselines: always the majority class, and the majority class per mode.
* Explanations: each situational driver (congestion, monsoon, holiday week, customs
  inspection, weekend arrival, carrier, dangerous goods) is switched to its favourable
  value and the drop in risk score is reported; that drop is the driver's contribution.
* Demurrage exposure: expected port days beyond free time x daily demurrage rate.

    python -m src.delays.model
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import PROJECT_ROOT, get_settings
from src.costing.rates import get_rate_card
from src.delays.data import (
    DISTANCE_KM,
    FEATURES_CATEGORICAL,
    FEATURES_NUMERIC,
    RISK_LABELS,
    congestion_series,
    generate_shipments,
)
from src.delays.weather import WeatherReading, get_rain
from src.forecasting.data import _days_to_nearest_holiday

logger = logging.getLogger(__name__)
HISTORY_CSV = PROJECT_ROOT / "data" / "raw" / "shipment_history.csv"
FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL
RISK_WEIGHTS = {"Low": 0.0, "Medium": 0.5, "High": 1.0}


class ModelNotTrainedError(RuntimeError):
    pass


class DelayInput(BaseModel):
    destination: str = Field(description="Delivery point, e.g. 'Kuching', 'Penang'.")
    carrier: str = Field(default="muara-fleet", description="Carrier key from the rate card.")
    commodity: str = Field(default="general")
    port: str = Field(default="westport", description="westport or northport")
    eta: str | None = Field(default=None, description="Expected arrival date YYYY-MM-DD.")
    congestion: float | None = Field(
        default=None, ge=0, le=1, description="Berth occupancy 0-1; default: typical for date."
    )
    customs_inspection: bool = False
    rain_mm: float | None = Field(
        default=None, ge=0, le=500, description="Rain at destination on ETA; default: live weather."
    )
    containers: int = Field(default=1, ge=1, le=100)
    container_size: str = Field(default="40")


class Driver(BaseModel):
    factor: str
    detail: str
    contribution: float = Field(description="Drop in risk score (0-1) if this were favourable")


class DelayPrediction(BaseModel):
    weather: WeatherReading
    destination_label: str
    carrier_label: str
    risk: str
    probabilities: dict[str, float]
    risk_score: float = Field(description="P(Medium) x 0.5 + P(High), 0-1")
    expected_delay_hours: float
    demurrage_exposure_rm: float
    drivers: list[Driver]
    features: dict[str, Any]


# ---------------------------------------------------------------------------- training


def _pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAL)],
        remainder="passthrough",
    )
    clf = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "high_recall": round(
            float(recall_score(y_true, y_pred, labels=["High"], average="macro", zero_division=0)),
            4,
        ),
    }


def train(
    history: pd.DataFrame | None = None,
    model_path: str | Path | None = None,
    test_frac: float = 0.2,
) -> dict[str, Any]:
    df = history if history is not None else pd.read_csv(HISTORY_CSV)
    df = df.sort_values("date", ignore_index=True)
    cut = int(len(df) * (1 - test_frac))
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]

    model = _pipeline().fit(train_df[FEATURES], train_df["risk"])
    pred = model.predict(test_df[FEATURES])
    metrics = _metrics(test_df["risk"], pred)

    majority = train_df["risk"].mode()[0]
    by_mode = train_df.groupby("mode")["risk"].agg(lambda s: s.mode()[0])
    baselines = {
        "majority_class": _metrics(test_df["risk"], [majority] * len(test_df)),
        "majority_per_mode": _metrics(test_df["risk"], test_df["mode"].map(by_mode)),
    }
    cm = confusion_matrix(test_df["risk"], pred, labels=RISK_LABELS).tolist()

    final = _pipeline().fit(df[FEATURES], df["risk"])
    class_delay = df.groupby("risk")["delay_hours"].mean().round(1).to_dict()
    bundle = {
        "model": final,
        "metrics": metrics,
        "baselines": baselines,
        "confusion_matrix": {"labels": RISK_LABELS, "matrix": cm},
        "class_mean_delay_hours": class_delay,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    model_path = Path(model_path or get_settings().delay_model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(".tmp")
    joblib.dump(bundle, tmp)
    tmp.replace(model_path)
    return bundle


# --------------------------------------------------------------------------- inference


class DelayModel:
    def __init__(self, bundle: dict[str, Any]):
        self.bundle = bundle
        self.model: Pipeline = bundle["model"]
        self.classes: list[str] = list(self.model.classes_)
        self.metrics = bundle["metrics"]

    @classmethod
    def load(cls, path: str | Path | None = None) -> DelayModel:
        path = Path(path or get_settings().delay_model_path)
        if not path.exists():
            raise ModelNotTrainedError(
                f"No delay model at {path}. Run `python -m src.delays.model`."
            )
        return cls(joblib.load(path))

    def _proba(self, rows: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(rows[FEATURES])
        return p[:, [self.classes.index(c) for c in RISK_LABELS]]

    @staticmethod
    def _score(p: np.ndarray) -> np.ndarray:
        return p @ np.array([RISK_WEIGHTS[c] for c in RISK_LABELS])

    def features_for(self, inp: DelayInput) -> dict[str, Any]:
        return self._features(inp)[0]

    def _features(self, inp: DelayInput) -> tuple[dict[str, Any], WeatherReading]:
        rates = get_rate_card()
        dest = rates.resolve_destination(inp.destination)
        carriers = {k: v for k, v in rates.data["carriers"].items() if not k.startswith("_")}
        if inp.carrier not in carriers:
            raise KeyError(f"Unknown carrier '{inp.carrier}'. Known: {', '.join(carriers)}")
        mode = carriers[inp.carrier]["modes"][0]
        if mode not in rates.destinations[dest]["modes"]:
            raise ValueError(f"{carriers[inp.carrier]['label']} does not serve {dest} by {mode}.")
        eta = pd.Timestamp(inp.eta) if inp.eta else pd.Timestamp.today().normalize()
        congestion = inp.congestion
        if congestion is None:
            congestion = float(congestion_series(pd.DatetimeIndex([eta]))[inp.port].iloc[0])
        from src.forecasting.data import normalize_commodity  # noqa: PLC0415

        commodity = normalize_commodity(inp.commodity)
        if inp.rain_mm is not None:
            weather = WeatherReading(
                rain_mm=inp.rain_mm, source="entered by user", date=eta.strftime("%Y-%m-%d")
            )
        else:
            weather = get_rain(dest, eta.date())
        features = {
            "distance_km": DISTANCE_KM[dest],
            "congestion": round(congestion, 3),
            "monsoon": int(eta.month in (11, 12, 1, 2)),
            "days_to_holiday": int(_days_to_nearest_holiday(pd.DatetimeIndex([eta]))[0]),
            "customs_inspection": int(inp.customs_inspection),
            "dangerous_goods": int(commodity == "chemicals"),
            "weekend_arrival": int(eta.dayofweek >= 5),
            "containers": inp.containers,
            "rain_mm": round(weather.rain_mm, 1),
            "mode": mode,
            "carrier": inp.carrier,
            "port": inp.port,
            "commodity": commodity,
        }
        return features, weather

    def _favourable(self, f: dict[str, Any]) -> dict[str, tuple[dict[str, Any], str]]:
        """Each situational driver set to its favourable value, with a human description."""
        rates = get_rate_card()
        carriers = {k: v for k, v in rates.data["carriers"].items() if not k.startswith("_")}
        alt: dict[str, tuple[dict[str, Any], str]] = {
            "Port congestion": (
                {"congestion": min(f["congestion"], 0.65)},
                f"berth occupancy {f['congestion']:.0%}",
            ),
            "Monsoon season": ({"monsoon": 0}, "north-east monsoon (Nov-Feb)"),
            "Heavy rain": ({"rain_mm": 0.0}, f"{f['rain_mm']:.0f} mm rain at destination"),
            "Holiday week": (
                {"days_to_holiday": 15},
                f"{f['days_to_holiday']} day(s) from Chinese New Year / Hari Raya",
            ),
            "Customs inspection": ({"customs_inspection": 0}, "selected for physical inspection"),
            "Weekend arrival": ({"weekend_arrival": 0}, "arrives on a weekend"),
            "Dangerous goods": ({"dangerous_goods": 0}, "DG documentation and handling"),
        }
        peers = [k for k, v in carriers.items() if f["mode"] in v["modes"] and k != f["carrier"]]
        if peers:
            rows = pd.DataFrame([{**f, "carrier": p} for p in peers])
            best = peers[int(np.argmin(self._score(self._proba(rows))))]
            alt["Carrier"] = (
                {"carrier": best},
                f"{carriers[f['carrier']]['label']} (vs {carriers[best]['label']})",
            )
        return alt

    def predict(self, inp: DelayInput) -> DelayPrediction:
        f, weather = self._features(inp)
        p = self._proba(pd.DataFrame([f]))[0]
        score = float(self._score(p[None, :])[0])

        # Symmetric attribution: average the effect of removing the factor from the actual
        # situation and of adding it to the all-favourable situation. Single-factor removal
        # alone underrates drivers when several push risk to saturation.
        favourable = {
            name: (change, detail)
            for name, (change, detail) in self._favourable(f).items()
            if not all(f.get(k) == v for k, v in change.items())
        }
        best_case = dict(f)
        for change, _ in favourable.values():
            best_case.update(change)
        names = list(favourable)
        without = [{**f, **favourable[n][0]} for n in names]
        with_only = [{**best_case, **{k: f[k] for k in favourable[n][0]}} for n in names]
        if names:
            scores = self._score(self._proba(pd.DataFrame(without + with_only + [best_case])))
            base_score = scores[-1]
            drivers = []
            for i, n in enumerate(names):
                contribution = 0.5 * ((score - scores[i]) + (scores[len(names) + i] - base_score))
                if contribution > 0.02:
                    drivers.append(
                        Driver(
                            factor=n, detail=favourable[n][1], contribution=round(contribution, 3)
                        )
                    )
            drivers.sort(key=lambda d: d.contribution, reverse=True)
        else:
            drivers = []

        means = self.bundle["class_mean_delay_hours"]
        expected_hours = float(sum(p[i] * means.get(c, 0) for i, c in enumerate(RISK_LABELS)))
        dem = get_rate_card().data["demurrage"]
        rate = dem["rm_per_day"].get(str(inp.container_size).replace("ft", ""), 240.0)
        # Planned port dwell of 1 day + delay; anything beyond the free days is charged.
        chargeable = sum(
            p[i] * max(0.0, 1 + means.get(c, 0) / 24 - dem["free_days"])
            for i, c in enumerate(RISK_LABELS)
        )
        rates = get_rate_card()
        return DelayPrediction(
            weather=weather,
            destination_label=rates.destinations[rates.resolve_destination(inp.destination)][
                "label"
            ],
            carrier_label=rates.data["carriers"][inp.carrier]["label"],
            risk=RISK_LABELS[int(np.argmax(p))],
            probabilities={c: round(float(p[i]), 3) for i, c in enumerate(RISK_LABELS)},
            risk_score=round(score, 3),
            expected_delay_hours=round(expected_hours, 1),
            demurrage_exposure_rm=round(chargeable * rate * inp.containers, 2),
            drivers=drivers[:4],
            features=f,
        )

    def score_frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        p = self._proba(rows)
        out = rows.copy()
        for i, c in enumerate(RISK_LABELS):
            out[f"p_{c.lower()}"] = p[:, i].round(3)
        out["risk"] = [RISK_LABELS[i] for i in p.argmax(axis=1)]
        out["risk_score"] = self._score(p).round(3)
        return out


def upcoming_shipments(start: str | None = None, days: int = 14, seed: int = 21) -> pd.DataFrame:
    """A synthetic manifest of shipments arriving in the next ``days`` days (from today)."""
    start = start or pd.Timestamp.today().strftime("%Y-%m-%d")
    df = generate_shipments(
        start=start,
        end=(pd.Timestamp(start) + pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d"),
        per_day=3,
        seed=seed,
    )
    return df.drop(columns=["delay_hours", "risk"]).rename(columns={"date": "eta"})


@lru_cache
def get_delay_model() -> DelayModel:
    return DelayModel.load()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train the delay-risk classifier.")
    parser.add_argument("--regenerate", action="store_true", help="regenerate the history CSV")
    args = parser.parse_args()
    if args.regenerate or not HISTORY_CSV.exists():
        generate_shipments().to_csv(HISTORY_CSV, index=False)
    bundle = train()
    m, b = bundle["metrics"], bundle["baselines"]
    path = get_settings().delay_model_path
    print(f"Saved {path} (train {bundle['n_train']}, test {bundle['n_test']})")
    for name, v in {"random forest": m, **b}.items():
        print(
            f"  {name:<18} acc={v['accuracy']:.3f}  macro-F1={v['macro_f1']:.3f}  "
            f"High recall={v['high_recall']:.3f}"
        )


if __name__ == "__main__":
    main()
