"""Sales data loading, synthetic data generation and feature engineering.

The canonical in-memory format is a long DataFrame with columns ``date``, ``sku``,
``sales``. Loaders accept either that format or the Kaggle *Store Item Demand
Forecasting Challenge* format (``date, store, item, sales``).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

LAGS: tuple[int, ...] = (1, 7, 14, 28)
ROLL_WINDOWS: tuple[int, ...] = (7, 28)
CALENDAR_FEATURES = ["dayofweek", "month", "dayofyear", "weekofyear", "is_weekend", "is_month_end"]
LAG_FEATURES = [f"lag_{k}" for k in LAGS]
ROLL_FEATURES = [f"roll_mean_{w}" for w in ROLL_WINDOWS] + ["roll_std_7"]
FEATURES = ["sku_id", *CALENDAR_FEATURES, *LAG_FEATURES, *ROLL_FEATURES]
MIN_HISTORY = max(max(LAGS), max(ROLL_WINDOWS))


def generate_sample_data(
    skus: Sequence[str] = ("SKU-101", "SKU-102", "SKU-103", "SKU-104", "SKU-105"),
    start: str = "2023-01-01",
    periods: int = 731,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic daily demand with trend, weekly + yearly seasonality and promotions."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="D")
    t = np.arange(periods)
    frames = []
    for i, sku in enumerate(skus):
        base = 40 + 15 * i
        trend = 1 + (0.0004 + 0.0002 * i) * t
        weekly = 1 + 0.25 * np.isin(dates.dayofweek, [4, 5]) - 0.15 * (dates.dayofweek == 6)
        yearly = 1 + 0.2 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365.25 + i)
        promo = 1 + 0.5 * (rng.random(periods) < 0.03)
        lam = base * trend * weekly * yearly * promo
        frames.append(pd.DataFrame({"date": dates, "sku": sku, "sales": rng.poisson(lam)}))
    return pd.concat(frames, ignore_index=True)


def load_sales(path: str | Path) -> pd.DataFrame:
    """Load sales CSV into canonical ``date, sku, sales`` format."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "sku" not in df.columns:
        if {"store", "item"}.issubset(df.columns):  # Kaggle store-item format
            df["sku"] = "S" + df["store"].astype(str) + "-I" + df["item"].astype(str)
        elif "item" in df.columns:
            df["sku"] = df["item"].astype(str)
        else:
            raise ValueError("Sales data needs a 'sku' column or Kaggle 'store'/'item' columns.")
    missing = {"date", "sales"} - set(df.columns)
    if missing:
        raise ValueError(f"Sales data is missing columns: {sorted(missing)}")
    return clean_sales(df[["date", "sku", "sales"]])


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, aggregate duplicates, fill missing days with 0 and clip negatives."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["sku"] = df["sku"].astype(str)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0).clip(lower=0)
    df = df.groupby(["sku", "date"], as_index=False)["sales"].sum()

    filled = []
    for sku, g in df.groupby("sku"):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(idx).rename_axis("date").reset_index()
        g["sku"] = sku
        g["sales"] = g["sales"].fillna(0)
        filled.append(g)
    return pd.concat(filled, ignore_index=True)[["date", "sku", "sales"]].sort_values(
        ["sku", "date"], ignore_index=True
    )


def calendar_features(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.DatetimeIndex(dates)
    return pd.DataFrame(
        {
            "dayofweek": d.dayofweek,
            "month": d.month,
            "dayofyear": d.dayofyear,
            "weekofyear": d.isocalendar().week.to_numpy().astype(int),
            "is_weekend": (d.dayofweek >= 5).astype(int),
            "is_month_end": d.is_month_end.astype(int),
        }
    )


def build_features(df: pd.DataFrame, sku_categories: Sequence[str]) -> pd.DataFrame:
    """Add model features. Rows without full lag history are dropped."""
    df = df.sort_values(["sku", "date"], ignore_index=True).copy()
    sku_index = {s: i for i, s in enumerate(sku_categories)}
    df["sku_id"] = df["sku"].map(sku_index).astype("Int64")
    df = df[df["sku_id"].notna()].copy()
    df["sku_id"] = df["sku_id"].astype(int)

    cal = calendar_features(df["date"])
    df[CALENDAR_FEATURES] = cal.to_numpy()

    grouped = df.groupby("sku")["sales"]
    for k in LAGS:
        df[f"lag_{k}"] = grouped.shift(k)
    shifted = grouped.shift(1)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = shifted.groupby(df["sku"]).transform(
            lambda s, w=w: s.rolling(w).mean()
        )
    df["roll_std_7"] = shifted.groupby(df["sku"]).transform(lambda s: s.rolling(7).std())
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def features_from_history(history: np.ndarray, date: pd.Timestamp, sku_id: int) -> dict[str, float]:
    """Feature row for ``date`` given the sales ``history`` that ends the day before.

    Mirrors :func:`build_features` exactly, which makes recursive forecasting possible.
    """
    if len(history) < MIN_HISTORY:
        raise ValueError(f"Need at least {MIN_HISTORY} days of history, got {len(history)}.")
    row: dict[str, float] = {"sku_id": sku_id}
    row.update(calendar_features(pd.DatetimeIndex([date])).iloc[0].to_dict())
    for k in LAGS:
        row[f"lag_{k}"] = float(history[-k])
    for w in ROLL_WINDOWS:
        row[f"roll_mean_{w}"] = float(np.mean(history[-w:]))
    row["roll_std_7"] = float(np.std(history[-7:], ddof=1))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic sample sales CSV.")
    parser.add_argument("--out", default="data/raw/sample_sales.csv")
    args = parser.parse_args()
    df = generate_sample_data()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
