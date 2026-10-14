"""Inference for the trained container-volume model (recursive multi-day forecasts)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import get_settings
from src.forecasting.data import features_from_history, normalize_commodity

MAX_HORIZON_DAYS = 90


class ModelNotTrainedError(RuntimeError):
    pass


class DemandForecaster:
    def __init__(self, bundle: dict[str, Any]):
        self.bundle = bundle
        self.model = bundle["model"]
        self.features: list[str] = bundle["features"]
        self.commodities: list[str] = bundle["categories"]
        self.metrics: dict[str, float] = bundle.get("metrics", {})

    @classmethod
    def load(cls, path: str | Path | None = None) -> DemandForecaster:
        path = Path(path or get_settings().model_path)
        if not path.exists():
            raise ModelNotTrainedError(
                f"No model at {path}. Train one with `python -m src.forecasting.train`."
            )
        return cls(joblib.load(path))

    def resolve_commodity(self, commodity: str) -> str:
        normalized = normalize_commodity(commodity)
        if normalized in self.commodities:
            return normalized
        raise KeyError(
            f"Unknown commodity '{commodity}'. Known commodities: {', '.join(self.commodities)}"
        )

    def forecast(self, commodity: str, horizon_days: int = 14) -> pd.DataFrame:
        """Forecast daily TEU for the ``horizon_days`` after the last observed date."""
        if not 1 <= horizon_days <= MAX_HORIZON_DAYS:
            raise ValueError(f"horizon_days must be between 1 and {MAX_HORIZON_DAYS}.")
        commodity = self.resolve_commodity(commodity)
        commodity_id = self.commodities.index(commodity)
        history = np.asarray(self.bundle["history"][commodity], dtype=float)
        date = pd.Timestamp(self.bundle["last_dates"][commodity])

        rows = []
        for _ in range(horizon_days):
            date += pd.Timedelta(days=1)
            feats = features_from_history(history, date, commodity_id)
            x = pd.DataFrame([feats], columns=self.features)
            pred = max(float(self.model.predict(x)[0]), 0.0)
            rows.append({"date": date.strftime("%Y-%m-%d"), "forecast": round(pred, 1)})
            history = np.append(history, pred)
        return pd.DataFrame(rows)

    def recent_history(self, commodity: str, days: int = 28) -> pd.DataFrame:
        commodity = self.resolve_commodity(commodity)
        values = self.bundle["history"][commodity][-days:]
        end = pd.Timestamp(self.bundle["last_dates"][commodity])
        dates = pd.date_range(end=end, periods=len(values), freq="D")
        return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "teu": values})


@lru_cache
def get_forecaster() -> DemandForecaster:
    return DemandForecaster.load()
