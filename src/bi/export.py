"""Power BI feed: clean, flat tables for a management dashboard.

Power BI connects straight to the same PostgreSQL database (Supabase) the platform
uses. This module writes report-ready tables with a ``bi_`` prefix:

* ``bi_jobs``        one row per completed job: revenue, cost, margin and each leak (RM)
* ``bi_free_time``   today's import containers on the demurrage / detention clock
* ``bi_deliveries``  delivery log with carrier and destination names, month, on-time flag
* ``bi_volumes``     daily TEU per commodity
* ``bi_carriers``, ``bi_destinations``  lookup (dimension) tables
* ``bi_refresh``     when the feed was last rebuilt

Rebuilt on app start and by ``python -m src.bi.export``; ``--csv DIR`` writes the same
tables as CSV files for Power BI without a database. See docs/powerbi.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.costing.rates import get_rate_card
from src.db import PgConnection
from src.forecasting.data import load_volumes
from src.hauliers.scorecard import DELIVERIES_CSV
from src.operations.freetime import generate_board
from src.operations.margin import get_ledger

logger = logging.getLogger(__name__)

LEAK_COLUMNS = {
    "Unbilled waiting time": "leak_waiting_rm",
    "Demurrage / detention absorbed": "leak_demurrage_rm",
    "Late-delivery penalty": "leak_late_penalty_rm",
    "Damage claims": "leak_damage_rm",
    "Diesel above quote": "leak_diesel_rm",
}


def _board(today: dt.date) -> pd.DataFrame:
    return generate_board(today)


def _ledger() -> pd.DataFrame:
    return get_ledger()


def build_tables(today: dt.date | None = None) -> dict[str, pd.DataFrame]:
    """All ``bi_`` tables as DataFrames (no database needed)."""
    today = today or dt.date.today()
    rates = get_rate_card()
    carriers = {k: v for k, v in rates.data["carriers"].items() if not k.startswith("_")}

    jobs = _ledger().rename(columns={**LEAK_COLUMNS, "job": "job_id"})
    jobs["date"] = pd.to_datetime(jobs["date"]).dt.date
    jobs["month"] = pd.to_datetime(jobs["date"]).dt.to_period("M").dt.to_timestamp().dt.date
    jobs["margin_pct"] = (jobs["margin_rm"] / jobs["revenue_rm"] * 100).round(2)
    jobs["loss_making"] = jobs["margin_rm"] < 0
    jobs = jobs[
        ["job_id", "date", "month", "customer", "commodity", "destination", "mode", "haulier",
         "containers", "revenue_rm", "planned_cost_rm", "actual_cost_rm", "planned_margin_rm",
         "margin_rm", "margin_pct", "leakage_rm", *LEAK_COLUMNS.values(), "loss_making"]
    ].round(2)  # fmt: skip

    board = _board(today)
    free_time = board[
        ["container", "size", "customer", "commodity", "destination", "port", "status", "clock",
         "discharged", "gate_out", "planned_move", "last_free_day", "days_left", "urgency",
         "daily_rate_rm", "accrued_rm", "projected_rm", "action"]
    ].assign(snapshot_date=today)  # fmt: skip

    deliveries = pd.read_csv(DELIVERIES_CSV, parse_dates=["date"])
    deliveries["month"] = deliveries["date"].dt.to_period("M").dt.to_timestamp().dt.date
    deliveries["date"] = deliveries["date"].dt.date
    deliveries["on_time"] = deliveries["on_time"].astype(bool)
    deliveries["damaged"] = deliveries["damaged"].astype(bool)

    volumes = load_volumes(get_settings().volumes_csv_path)
    volumes = volumes.reset_index() if "date" not in volumes.columns else volumes
    volumes = volumes[["date", "commodity", "teu"]].copy()
    volumes["date"] = pd.to_datetime(volumes["date"]).dt.date

    carrier_dim = pd.DataFrame(
        [
            {"haulier": k, "carrier_name": v["label"], "mode": v["modes"][0],
             "price_factor": v.get("price_factor", 1.0), "fleet": v.get("fleet", "")}
            for k, v in carriers.items()
        ]
    )  # fmt: skip
    dest_dim = pd.DataFrame(
        [
            {"destination": k, "destination_name": d["label"], "km": d["km"],
             "modes": ", ".join(d["modes"]), "lat": d["lat"], "lon": d["lon"]}
            for k, d in rates.destinations.items()
        ]
    )  # fmt: skip
    refresh = pd.DataFrame(
        [{"refreshed_at": dt.datetime.now(dt.timezone.utc), "jobs": len(jobs),
          "containers": len(free_time), "source": "example data"}]
    )  # fmt: skip
    return {
        "bi_jobs": jobs,
        "bi_free_time": free_time,
        "bi_deliveries": deliveries,
        "bi_volumes": volumes,
        "bi_carriers": carrier_dim,
        "bi_destinations": dest_dim,
        "bi_refresh": refresh,
    }


def _pg_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMPTZ"
    sample = series.dropna()
    if len(sample) and isinstance(sample.iloc[0], dt.datetime):
        return "TIMESTAMPTZ"
    if len(sample) and isinstance(sample.iloc[0], dt.date):
        return "DATE"
    return "TEXT"


def _clean(value):
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NaT:
        return None
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


BI_READER = "powerbi_reader"  # optional read-only login, see docs/powerbi/readonly_role.sql


def write_postgres(tables: dict[str, pd.DataFrame], dsn: str) -> None:
    """Replace each ``bi_`` table in one transaction per table.

    Row Level Security is on, so Supabase's public API (anon / authenticated keys)
    cannot read the tables; the Power BI reader login, if it exists, gets a read policy.
    """
    conn = PgConnection(dsn)
    has_reader = bool(
        conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (BI_READER,)).fetchone()
    )
    for name, df in tables.items():
        cols = ", ".join(f'"{c}" {_pg_type(df[c])}' for c in df.columns)
        placeholders = ", ".join(["%s"] * len(df.columns))
        rows = [tuple(_clean(v) for v in row) for row in df.itertuples(index=False)]
        with conn.raw.transaction(), conn.raw.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{name}"')
            cur.execute(f'CREATE TABLE "{name}" ({cols})')
            cur.execute(f'ALTER TABLE "{name}" ENABLE ROW LEVEL SECURITY')
            if has_reader:
                cur.execute(f'GRANT SELECT ON "{name}" TO {BI_READER}')
                cur.execute(
                    f'CREATE POLICY bi_read ON "{name}" FOR SELECT TO {BI_READER} USING (true)'
                )
            if rows:
                cur.executemany(f'INSERT INTO "{name}" VALUES ({placeholders})', rows)
    conn.close()


def write_csv(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def refresh(today: dt.date | None = None) -> dict[str, int]:
    """Rebuild the feed in PostgreSQL (no-op without ``DATABASE_URL``)."""
    dsn = get_settings().database_url
    if not dsn:
        return {}
    tables = build_tables(today)
    write_postgres(tables, dsn)
    counts = {k: len(v) for k, v in tables.items()}
    logger.info("Power BI feed refreshed: %s", counts)
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Rebuild the Power BI feed (bi_ tables).")
    parser.add_argument("--csv", metavar="DIR", help="write CSV files instead of PostgreSQL")
    args = parser.parse_args()
    if args.csv:
        for path in write_csv(build_tables(), args.csv):
            print(path)
    elif not get_settings().database_url:
        raise SystemExit("DATABASE_URL is not set; use --csv DIR to export files instead.")
    else:
        print(refresh())


if __name__ == "__main__":
    main()
