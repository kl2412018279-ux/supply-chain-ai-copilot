"""Read-only access to historical sales, backed by PostgreSQL or a CSV file.

The agent never writes SQL itself: it calls a fixed, parameterised summary query,
which removes the SQL-injection / data-exfiltration risk of text-to-SQL.

Load the CSV into PostgreSQL with::

    python -m src.forecasting.sales_repository --load data/raw/sample_sales.csv
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
from src.forecasting.data import load_sales

logger = logging.getLogger(__name__)

SALES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sales (
    date  DATE    NOT NULL,
    sku   TEXT    NOT NULL,
    sales NUMERIC NOT NULL CHECK (sales >= 0),
    PRIMARY KEY (sku, date)
);
"""


class SkuSummary(BaseModel):
    sku: str
    total_units: float
    avg_daily_units: float
    peak_date: str
    peak_units: float


class SalesSummary(BaseModel):
    start: str
    end: str
    days: int
    total_units: float
    by_sku: list[SkuSummary]


class SalesRepository(Protocol):
    def skus(self) -> list[str]: ...

    def summary(
        self, sku: str | None = None, start: date | None = None, end: date | None = None
    ) -> SalesSummary: ...


def _summarize(df: pd.DataFrame) -> SalesSummary:
    if df.empty:
        raise LookupError("No sales found for the requested filter.")
    by_sku = []
    for sku, g in df.groupby("sku"):
        peak = g.loc[g["sales"].idxmax()]
        by_sku.append(
            SkuSummary(
                sku=sku,
                total_units=round(float(g["sales"].sum()), 1),
                avg_daily_units=round(float(g["sales"].mean()), 1),
                peak_date=pd.Timestamp(peak["date"]).strftime("%Y-%m-%d"),
                peak_units=float(peak["sales"]),
            )
        )
    return SalesSummary(
        start=df["date"].min().strftime("%Y-%m-%d"),
        end=df["date"].max().strftime("%Y-%m-%d"),
        days=int(df["date"].nunique()),
        total_units=round(float(df["sales"].sum()), 1),
        by_sku=by_sku,
    )


class CsvSalesRepository:
    def __init__(self, path: str | Path):
        self.df = load_sales(path)

    def skus(self) -> list[str]:
        return sorted(self.df["sku"].unique())

    def summary(self, sku=None, start=None, end=None) -> SalesSummary:
        df = self.df
        if sku:
            df = df[df["sku"].str.upper() == sku.strip().upper()]
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        return _summarize(df)


class PostgresSalesRepository:
    def __init__(self, dsn: str):
        import psycopg  # noqa: PLC0415

        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute(SALES_SCHEMA_SQL)

    def load_dataframe(self, df: pd.DataFrame) -> int:
        rows = [(r.date.date(), r.sku, float(r.sales)) for r in df.itertuples(index=False)]
        with self._conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO sales (date, sku, sales) VALUES (%s, %s, %s)
                   ON CONFLICT (sku, date) DO UPDATE SET sales = EXCLUDED.sales""",
                rows,
            )
        return len(rows)

    def count(self) -> int:
        return int(self._conn.execute("SELECT count(*) FROM sales").fetchone()[0])

    def skus(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT DISTINCT sku FROM sales ORDER BY sku")]

    def summary(self, sku=None, start=None, end=None) -> SalesSummary:
        rows = self._conn.execute(
            """
            SELECT date, sku, sales::float FROM sales
            WHERE (%(sku)s::text IS NULL OR upper(sku) = upper(%(sku)s::text))
              AND (%(start)s::date IS NULL OR date >= %(start)s::date)
              AND (%(end)s::date IS NULL OR date <= %(end)s::date)
            ORDER BY sku, date
            """,
            {"sku": sku.strip() if sku else None, "start": start, "end": end},
        ).fetchall()
        df = pd.DataFrame(rows, columns=["date", "sku", "sales"])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return _summarize(df)


@lru_cache
def get_sales_repository() -> SalesRepository:
    settings = get_settings()
    if settings.database_url:
        try:
            repo = PostgresSalesRepository(settings.database_url)
            if repo.count() == 0 and settings.sales_csv_path.exists():
                repo.load_dataframe(load_sales(settings.sales_csv_path))
            return repo
        except Exception as exc:
            logger.warning("PostgreSQL unavailable (%s); reading sales from CSV.", exc)
    return CsvSalesRepository(settings.sales_csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a sales CSV into PostgreSQL.")
    parser.add_argument("--load", default=str(get_settings().sales_csv_path))
    args = parser.parse_args()
    dsn = get_settings().database_url
    if not dsn:
        raise SystemExit("Set DATABASE_URL first.")
    n = PostgresSalesRepository(dsn).load_dataframe(load_sales(args.load))
    print(f"Loaded {n:,} rows into the sales table.")


if __name__ == "__main__":
    main()
