"""Synthetic delivery logs for the (fictional) hauliers and carriers in the rate card.

Each row is one job: promised vs actual delivery, damage, claims. Profiles differ by
carrier so the scorecard has something real to separate; seasonality adds monsoon
(Nov-Feb) delays for sea legs and holiday-week congestion for everyone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.data import _days_to_nearest_holiday

# carrier: (on-time probability, mean late hours when late, damage rate, jobs per day)
PROFILES: dict[str, tuple[float, float, float, float]] = {
    "muara-fleet": (0.93, 7.0, 0.006, 2.2),
    "cempaka": (0.81, 14.0, 0.021, 1.6),
    "hijau": (0.95, 6.0, 0.005, 1.0),
    "raillink": (0.88, 18.0, 0.010, 0.6),
    "nusantara": (0.86, 30.0, 0.012, 0.5),
    "borneo-express": (0.93, 20.0, 0.008, 0.3),
}
MODES = {
    "muara-fleet": "truck",
    "cempaka": "truck",
    "hijau": "truck",
    "raillink": "rail",
    "nusantara": "sea",
    "borneo-express": "sea",
}
DESTINATIONS = {
    "truck": ["shah-alam", "kuala-lumpur", "nilai", "seremban", "melaka", "ipoh", "kuantan",
              "pasir-gudang", "penang-prai"],
    "rail": ["ipoh", "penang-prai"],
    "sea": ["kuching", "kota-kinabalu"],
}  # fmt: skip
COMMODITIES = ["chemicals", "palm-oil", "plastics", "general"]


def generate_deliveries(
    start: str = "2025-01-01", end: str = "2026-09-30", seed: int = 7
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    near_holiday = _days_to_nearest_holiday(dates)
    rows = []
    n = 0
    for carrier, (p_on_time, late_mean, damage, per_day) in PROFILES.items():
        mode = MODES[carrier]
        for date, hol in zip(dates, near_holiday, strict=True):
            for _ in range(rng.poisson(per_day * (0.4 if date.dayofweek == 6 else 1.0))):
                p = p_on_time
                if hol <= 5:
                    p -= 0.10
                if mode == "sea" and date.month in (11, 12, 1, 2):  # north-east monsoon
                    p -= 0.12
                late = rng.random() > p
                late_hours = float(np.round(rng.exponential(late_mean), 1)) if late else 0.0
                n += 1
                rows.append(
                    {
                        "delivery_id": f"D{n:06d}",
                        "date": date.strftime("%Y-%m-%d"),
                        "haulier": carrier,
                        "mode": mode,
                        "destination": rng.choice(DESTINATIONS[mode]),
                        "commodity": rng.choice(COMMODITIES, p=[0.3, 0.2, 0.2, 0.3]),
                        "containers": int(rng.integers(1, 4)),
                        "late_hours": late_hours,
                        "on_time": int(late_hours <= 2.0),
                        "damaged": int(rng.random() < damage),
                    }
                )
    df = pd.DataFrame(rows)
    df["claim_rm"] = np.where(
        df["damaged"] == 1, np.round(rng.uniform(800, 12000, len(df)), 2), 0.0
    )
    return df.sort_values(["date", "delivery_id"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic haulier delivery logs.")
    parser.add_argument("--out", default="data/raw/haulier_deliveries.csv")
    args = parser.parse_args()
    df = generate_deliveries()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} deliveries to {args.out}")


if __name__ == "__main__":
    main()
