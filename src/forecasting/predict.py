"""Inference for the trained demand model (recursive multi-day forecasts)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import get_settings
from src.forecasting.data import features_from_history

MAX_HORIZON_DAYS = 90


class ModelNotTrainedError(RuntimeError):
    pass


class DemandForecaster:
    def __init__(self, bundle: dict[str, Any]):
        self.bundle = bundle
        self.model = bundle["model"]
        self.features: list[str] = bundle["features"]
        self.skus: list[str] = bundle["sku_categories"]
        self.metrics: dict[str, float] = bundle.get("metrics", {})

    @classmethod
    def load(cls, path: str | Path | None = None) -> DemandForecaster:
        path = Path(path or get_settings().model_path)
        if not path.exists():
            raise ModelNotTrainedError(
                f"No model at {path}. Train one with `python -m src.forecasting.train`."
            )
        return cls(joblib.load(path))

    def resolve_sku(self, sku: str) -> str:
        normalized = sku.strip().upper()
        for known in self.skus:
            if known.upper() == normalized:
                return known
        raise KeyError(f"Unknown SKU '{sku}'. Known SKUs: {', '.join(self.skus)}")

    def forecast(self, sku: str, horizon_days: int = 14) -> pd.DataFrame:
        """Forecast daily demand for the ``horizon_days`` after the last observed date."""
        if not 1 <= horizon_days <= MAX_HORIZON_DAYS:
            raise ValueError(f"horizon_days must be between 1 and {MAX_HORIZON_DAYS}.")
        sku = self.resolve_sku(sku)
        sku_id = self.skus.index(sku)
        history = np.asarray(self.bundle["history"][sku], dtype=float)
        date = pd.Timestamp(self.bundle["last_dates"][sku])

        rows = []
        for _ in range(horizon_days):
            date += pd.Timedelta(days=1)
            feats = features_from_history(history, date, sku_id)
            x = pd.DataFrame([feats], columns=self.features)
            pred = max(float(self.model.predict(x)[0]), 0.0)
            rows.append({"date": date.strftime("%Y-%m-%d"), "forecast": round(pred, 1)})
            history = np.append(history, pred)
        return pd.DataFrame(rows)

    def recent_history(self, sku: str, days: int = 28) -> pd.DataFrame:
        sku = self.resolve_sku(sku)
        values = self.bundle["history"][sku][-days:]
        end = pd.Timestamp(self.bundle["last_dates"][sku])
        dates = pd.date_range(end=end, periods=len(values), freq="D")
        return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "sales": values})


@lru_cache
def get_forecaster() -> DemandForecaster:
    return DemandForecaster.load()
