"""Synthetic shipment history with delay outcomes.

Delay (hours beyond the promised delivery) is driven by the factors ops teams name:
port congestion (berth occupancy), north-east monsoon for sea legs (Nov-Feb), the
week around Chinese New Year / Hari Raya, customs inspection, dangerous goods,
weekend arrival, distance and the carrier's own reliability, plus noise.
Risk classes: Low < 12 h, Medium 12-48 h, High > 48 h.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.data import _days_to_nearest_holiday
from src.hauliers.data import DESTINATIONS, MODES, PROFILES

DISTANCE_KM = {
    "shah-alam": 40, "kuala-lumpur": 55, "nilai": 75, "seremban": 90, "melaka": 160,
    "ipoh": 230, "kuantan": 290, "pasir-gudang": 360, "penang-prai": 370,
    "kuching": 1050, "kota-kinabalu": 1650,
}  # fmt: skip
COMMODITIES = ["chemicals", "palm-oil", "plastics", "general"]
RISK_LABELS = ["Low", "Medium", "High"]
FEATURES_NUMERIC = [
    "distance_km", "congestion", "monsoon", "days_to_holiday", "customs_inspection",
    "dangerous_goods", "weekend_arrival", "containers",
]  # fmt: skip
FEATURES_CATEGORICAL = ["mode", "carrier", "port", "commodity"]


def congestion_series(dates: pd.DatetimeIndex, seed: int = 11) -> pd.DataFrame:
    """Daily berth-occupancy index per terminal (0-1): AR(1) noise + holiday / year-end peaks."""
    rng = np.random.default_rng(seed)
    near = _days_to_nearest_holiday(dates)
    seasonal = 0.06 * np.isin(dates.month, [11, 12]) + 0.08 * (near <= 7)
    out = {}
    for port, level in (("westport", 0.72), ("northport", 0.66)):
        noise = np.zeros(len(dates))
        for i in range(1, len(dates)):
            noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 0.03)
        out[port] = np.clip(level + seasonal + noise, 0.4, 0.99)
    return pd.DataFrame(out, index=dates)


def risk_class(delay_hours: np.ndarray | pd.Series) -> pd.Series:
    """Low: < 12 h, Medium: 12-48 h, High: > 48 h."""
    hours = pd.Series(delay_hours, dtype=float)
    return pd.Series(
        np.select([hours < 12, hours <= 48], ["Low", "Medium"], default="High"), index=hours.index
    )


def generate_shipments(
    start: str = "2025-01-01", end: str = "2026-09-30", per_day: float = 8.0, seed: int = 3
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    cong = congestion_series(dates)
    near = _days_to_nearest_holiday(dates)
    carriers = list(PROFILES)
    weights = np.array([PROFILES[c][3] for c in carriers])
    rows = []
    for di, date in enumerate(dates):
        for _ in range(rng.poisson(per_day)):
            carrier = rng.choice(carriers, p=weights / weights.sum())
            mode = MODES[carrier]
            dest = rng.choice(DESTINATIONS[mode])
            commodity = rng.choice(COMMODITIES, p=[0.3, 0.2, 0.2, 0.3])
            port = "westport" if rng.random() < 0.65 else "northport"
            eta = date + pd.Timedelta(days=int(rng.integers(0, 3)))
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "carrier": carrier,
                    "mode": mode,
                    "destination": dest,
                    "distance_km": DISTANCE_KM[dest],
                    "commodity": commodity,
                    "port": port,
                    "congestion": round(float(cong[port].iloc[di]), 3),
                    "monsoon": int(date.month in (11, 12, 1, 2)),
                    "days_to_holiday": int(near[di]),
                    "customs_inspection": int(
                        rng.random() < (0.25 if commodity == "chemicals" else 0.08)
                    ),
                    "dangerous_goods": int(commodity == "chemicals"),
                    "weekend_arrival": int(eta.dayofweek >= 5),
                    "containers": int(rng.integers(1, 4)),
                }
            )
    df = pd.DataFrame(rows)
    reliability = df["carrier"].map({c: p[0] for c, p in PROFILES.items()})
    base = df["mode"].map({"truck": 0.0, "rail": 4.0, "sea": 16.0})
    delay = (
        base
        + 160 * np.clip(df["congestion"] - 0.76, 0, None)  # queues explode near capacity
        + df["monsoon"] * np.where(df["mode"] == "sea", 30, 1)
        + np.where(df["days_to_holiday"] <= 3, 22, np.where(df["days_to_holiday"] <= 7, 7, 0))
        + df["customs_inspection"] * rng.uniform(18, 45, len(df))
        + df["dangerous_goods"] * 4
        + df["weekend_arrival"] * 8
        + df["distance_km"] / 150
        + (1 - reliability) * 50
        + rng.exponential(5, len(df))
        - 12
    )
    df["delay_hours"] = np.round(np.clip(delay, 0, None), 1)
    df["risk"] = risk_class(df["delay_hours"]).to_numpy()
    df.insert(0, "shipment_id", [f"S{i:06d}" for i in range(1, len(df) + 1)])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic shipment delay history.")
    parser.add_argument("--out", default="data/raw/shipment_history.csv")
    args = parser.parse_args()
    df = generate_shipments()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} shipments to {args.out}")
    print(df["risk"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
