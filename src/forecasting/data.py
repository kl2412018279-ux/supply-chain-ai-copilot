"""Container-volume data: loading, synthetic generation and feature engineering.

The canonical in-memory format is a long DataFrame with columns ``date``,
``commodity`` and ``teu`` (twenty-foot equivalent units handled that day).
Loaders also accept common aliases (``series``/``sku``/``item`` and
``volume``/``sales``/``qty``) so real operational exports can be dropped in.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

COMMODITIES: tuple[str, ...] = ("chemicals", "palm-oil", "plastics", "general")
COMMODITY_ALIASES: dict[str, str] = {
    "chemical": "chemicals",
    "chem": "chemicals",
    "dg": "chemicals",
    "dangerous goods": "chemicals",
    "palm oil": "palm-oil",
    "palm": "palm-oil",
    "palm_oil": "palm-oil",
    "cpo": "palm-oil",
    "plastic": "plastics",
    "resin": "plastics",
    "polymer": "plastics",
    "general cargo": "general",
    "gc": "general",
}

# Major Malaysian holidays that slow port and haulage activity.
HOLIDAYS: dict[str, list[str]] = {
    "chinese_new_year": ["2024-02-10", "2025-01-29", "2026-02-17", "2027-02-06"],
    "hari_raya_aidilfitri": ["2024-04-10", "2025-03-31", "2026-03-20", "2027-03-10"],
}
_HOLIDAY_DATES = pd.DatetimeIndex([d for dates in HOLIDAYS.values() for d in dates])

LAGS: tuple[int, ...] = (1, 7, 14, 28)
ROLL_WINDOWS: tuple[int, ...] = (7, 28)
CALENDAR_FEATURES = [
    "dayofweek",
    "month",
    "dayofyear",
    "weekofyear",
    "is_weekend",
    "is_month_end",
    "days_to_holiday",
]
LAG_FEATURES = [f"lag_{k}" for k in LAGS]
ROLL_FEATURES = [f"roll_mean_{w}" for w in ROLL_WINDOWS] + ["roll_std_7"]
FEATURES = ["commodity_id", *CALENDAR_FEATURES, *LAG_FEATURES, *ROLL_FEATURES]
MIN_HISTORY = max(max(LAGS), max(ROLL_WINDOWS))
HOLIDAY_HORIZON = 15  # days_to_holiday is capped at this many days


def normalize_commodity(name: str) -> str:
    key = " ".join(name.strip().lower().replace("_", " ").split())
    if key.replace(" ", "-") in COMMODITIES:
        return key.replace(" ", "-")
    return COMMODITY_ALIASES.get(key, key.replace(" ", "-"))


def _days_to_nearest_holiday(dates: pd.DatetimeIndex) -> np.ndarray:
    """Absolute distance in days to the nearest major holiday, capped."""
    diffs = np.abs(
        (dates.values[:, None] - _HOLIDAY_DATES.values[None, :]).astype("timedelta64[D]")
    )
    return np.minimum(diffs.min(axis=1).astype(int), HOLIDAY_HORIZON)


def generate_sample_data(
    commodities: Sequence[str] = COMMODITIES,
    start: str = "2024-01-01",
    end: str = "2026-09-30",
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic daily TEU handled per commodity for a mid-sized Port Klang 3PL.

    Patterns: weekday peaks / quiet Sundays, holiday dips around Chinese New Year and
    Hari Raya, a palm-oil harvest peak (Sep-Nov), year-end peaks for plastics and
    general cargo, and steady growth.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D")
    t = np.arange(len(dates))
    doy = dates.dayofyear.to_numpy()
    weekly = np.select([dates.dayofweek == 6, dates.dayofweek == 5], [0.25, 0.7], default=1.0)
    near_holiday = _days_to_nearest_holiday(dates)
    holiday_dip = np.where(near_holiday <= 3, 0.35, np.where(near_holiday <= 7, 0.8, 1.0))

    profile = {  # base TEU/day, yearly-seasonality amplitude, peak day-of-year, growth/day
        "chemicals": (14.0, 0.10, 100, 0.0004),
        "palm-oil": (10.0, 0.35, 290, 0.0002),
        "plastics": (8.0, 0.15, 330, 0.0005),
        "general": (18.0, 0.12, 340, 0.0003),
    }
    frames = []
    for name in commodities:
        base, amp, peak, growth = profile.get(name, (10.0, 0.1, 180, 0.0003))
        yearly = 1 + amp * np.cos(2 * np.pi * (doy - peak) / 365.25)
        lam = base * weekly * holiday_dip * yearly * (1 + growth * t)
        frames.append(pd.DataFrame({"date": dates, "commodity": name, "teu": rng.poisson(lam)}))
    return pd.concat(frames, ignore_index=True)


def load_volumes(path: str | Path) -> pd.DataFrame:
    """Load a CSV of daily volumes into canonical ``date, commodity, teu`` format."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for alias in ("series", "sku", "item"):
        if "commodity" not in df.columns and alias in df.columns:
            df = df.rename(columns={alias: "commodity"})
    for alias in ("volume", "containers", "sales", "qty"):
        if "teu" not in df.columns and alias in df.columns:
            df = df.rename(columns={alias: "teu"})
    missing = {"date", "commodity", "teu"} - set(df.columns)
    if missing:
        raise ValueError(f"Volume data is missing columns: {sorted(missing)}")
    df["commodity"] = df["commodity"].astype(str).map(normalize_commodity)
    return clean_volumes(df[["date", "commodity", "teu"]])


def clean_volumes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, aggregate duplicates, fill missing days with 0 and clip negatives."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["commodity"] = df["commodity"].astype(str)
    df["teu"] = pd.to_numeric(df["teu"], errors="coerce").fillna(0).clip(lower=0)
    df = df.groupby(["commodity", "date"], as_index=False)["teu"].sum()

    filled = []
    for commodity, g in df.groupby("commodity"):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(idx).rename_axis("date").reset_index()
        g["commodity"] = commodity
        g["teu"] = g["teu"].fillna(0)
        filled.append(g)
    return pd.concat(filled, ignore_index=True)[["date", "commodity", "teu"]].sort_values(
        ["commodity", "date"], ignore_index=True
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
            "days_to_holiday": _days_to_nearest_holiday(d),
        }
    )


def build_features(df: pd.DataFrame, categories: Sequence[str]) -> pd.DataFrame:
    """Add model features. Rows without full lag history are dropped."""
    df = df.sort_values(["commodity", "date"], ignore_index=True).copy()
    index = {c: i for i, c in enumerate(categories)}
    df["commodity_id"] = df["commodity"].map(index).astype("Int64")
    df = df[df["commodity_id"].notna()].copy()
    df["commodity_id"] = df["commodity_id"].astype(int)

    cal = calendar_features(df["date"])
    df[CALENDAR_FEATURES] = cal.to_numpy()

    grouped = df.groupby("commodity")["teu"]
    for k in LAGS:
        df[f"lag_{k}"] = grouped.shift(k)
    shifted = grouped.shift(1)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = shifted.groupby(df["commodity"]).transform(
            lambda s, w=w: s.rolling(w).mean()
        )
    df["roll_std_7"] = shifted.groupby(df["commodity"]).transform(lambda s: s.rolling(7).std())
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def features_from_history(
    history: np.ndarray, date: pd.Timestamp, commodity_id: int
) -> dict[str, float]:
    """Feature row for ``date`` given the volume ``history`` that ends the day before.

    Mirrors :func:`build_features` exactly, which makes recursive forecasting possible.
    """
    if len(history) < MIN_HISTORY:
        raise ValueError(f"Need at least {MIN_HISTORY} days of history, got {len(history)}.")
    row: dict[str, float] = {"commodity_id": commodity_id}
    row.update(calendar_features(pd.DatetimeIndex([date])).iloc[0].to_dict())
    for k in LAGS:
        row[f"lag_{k}"] = float(history[-k])
    for w in ROLL_WINDOWS:
        row[f"roll_mean_{w}"] = float(np.mean(history[-w:]))
    row["roll_std_7"] = float(np.std(history[-7:], ddof=1))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic container-volume CSV.")
    parser.add_argument("--out", default="data/raw/container_volumes.csv")
    args = parser.parse_args()
    df = generate_sample_data()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
