"""Read-only access to historical container volumes, backed by PostgreSQL or a CSV file.

The agent never writes SQL itself: it calls a fixed, parameterised summary query,
which removes the SQL-injection / data-exfiltration risk of text-to-SQL.

Load the CSV into PostgreSQL with::

    python -m src.forecasting.volume_repository --load data/raw/container_volumes.csv
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import pandas as pd
from pydantic import BaseModel

from src.config import get_settings
from src.db import PgConnection
from src.forecasting.data import load_volumes, normalize_commodity

logger = logging.getLogger(__name__)

VOLUMES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS container_volumes (
    date      DATE    NOT NULL,
    commodity TEXT    NOT NULL,
    teu       NUMERIC NOT NULL CHECK (teu >= 0),
    PRIMARY KEY (commodity, date)
);
ALTER TABLE container_volumes ENABLE ROW LEVEL SECURITY;
"""


class CommoditySummary(BaseModel):
    commodity: str
    total_teu: float
    avg_daily_teu: float
    avg_weekly_teu: float
    peak_date: str
    peak_teu: float


class VolumeSummary(BaseModel):
    start: str
    end: str
    days: int
    total_teu: float
    by_commodity: list[CommoditySummary]


class VolumeRepository(Protocol):
    def commodities(self) -> list[str]: ...

    def summary(
        self, commodity: str | None = None, start: date | None = None, end: date | None = None
    ) -> VolumeSummary: ...


def _summarize(df: pd.DataFrame) -> VolumeSummary:
    if df.empty:
        raise LookupError("No container volumes found for the requested filter.")
    by_commodity = []
    for commodity, g in df.groupby("commodity"):
        peak = g.loc[g["teu"].idxmax()]
        by_commodity.append(
            CommoditySummary(
                commodity=commodity,
                total_teu=round(float(g["teu"].sum()), 1),
                avg_daily_teu=round(float(g["teu"].mean()), 1),
                avg_weekly_teu=round(float(g["teu"].mean()) * 7, 1),
                peak_date=pd.Timestamp(peak["date"]).strftime("%Y-%m-%d"),
                peak_teu=float(peak["teu"]),
            )
        )
    return VolumeSummary(
        start=df["date"].min().strftime("%Y-%m-%d"),
        end=df["date"].max().strftime("%Y-%m-%d"),
        days=int(df["date"].nunique()),
        total_teu=round(float(df["teu"].sum()), 1),
        by_commodity=by_commodity,
    )


class CsvVolumeRepository:
    def __init__(self, path: str | Path):
        self.df = load_volumes(path)

    def commodities(self) -> list[str]:
        return sorted(self.df["commodity"].unique())

    def summary(self, commodity=None, start=None, end=None) -> VolumeSummary:
        df = self.df
        if commodity:
            df = df[df["commodity"] == normalize_commodity(commodity)]
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        return _summarize(df)


class PostgresVolumeRepository:
    def __init__(self, dsn: str):

        self._conn = PgConnection(dsn)
        self._conn.execute(VOLUMES_SCHEMA_SQL)

    def load_dataframe(self, df: pd.DataFrame) -> int:
        rows = [(r.date.date(), r.commodity, float(r.teu)) for r in df.itertuples(index=False)]
        with self._conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO container_volumes (date, commodity, teu) VALUES (%s, %s, %s)
                   ON CONFLICT (commodity, date) DO UPDATE SET teu = EXCLUDED.teu""",
                rows,
            )
        return len(rows)

    def count(self) -> int:
        return int(self._conn.execute("SELECT count(*) FROM container_volumes").fetchone()[0])

    def commodities(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT commodity FROM container_volumes ORDER BY commodity"
        )
        return [r[0] for r in rows]

    def summary(self, commodity=None, start=None, end=None) -> VolumeSummary:
        rows = self._conn.execute(
            """
            SELECT date, commodity, teu::float FROM container_volumes
            WHERE (%(c)s::text IS NULL OR commodity = %(c)s::text)
              AND (%(start)s::date IS NULL OR date >= %(start)s::date)
              AND (%(end)s::date IS NULL OR date <= %(end)s::date)
            ORDER BY commodity, date
            """,
            {
                "c": normalize_commodity(commodity) if commodity else None,
                "start": start,
                "end": end,
            },
        ).fetchall()
        df = pd.DataFrame(rows, columns=["date", "commodity", "teu"])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return _summarize(df)


@lru_cache
def get_volume_repository() -> VolumeRepository:
    settings = get_settings()
    if settings.database_url:
        try:
            repo = PostgresVolumeRepository(settings.database_url)
            if repo.count() == 0 and settings.volumes_csv_path.exists():
                repo.load_dataframe(load_volumes(settings.volumes_csv_path))
            return repo
        except Exception as exc:
            logger.warning("PostgreSQL unavailable (%s); reading volumes from CSV.", exc)
    return CsvVolumeRepository(settings.volumes_csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a container-volume CSV into PostgreSQL.")
    parser.add_argument("--load", default=str(get_settings().volumes_csv_path))
    args = parser.parse_args()
    dsn = get_settings().database_url
    if not dsn:
        raise SystemExit("Set DATABASE_URL first.")
    n = PostgresVolumeRepository(dsn).load_dataframe(load_volumes(args.load))
    print(f"Loaded {n:,} rows into the container_volumes table.")


if __name__ == "__main__":
    main()
