import datetime as dt
import os

import pandas as pd
import pytest

from src.bi.export import LEAK_COLUMNS, build_tables, write_csv

TODAY = dt.date(2026, 10, 17)


@pytest.fixture(scope="module")
def tables():
    return build_tables(TODAY)


def test_all_tables_present_and_non_empty(tables):
    assert set(tables) == {"bi_jobs", "bi_free_time", "bi_deliveries", "bi_volumes",
                           "bi_carriers", "bi_destinations", "bi_refresh"}  # fmt: skip
    assert all(len(df) for df in tables.values())


def test_jobs_add_up(tables):
    jobs = tables["bi_jobs"]
    leaks = jobs[list(LEAK_COLUMNS.values())].sum(axis=1)
    assert ((jobs["leakage_rm"] - leaks).abs() < 0.05).all()
    assert ((jobs["revenue_rm"] - jobs["actual_cost_rm"] - jobs["margin_rm"]).abs() < 0.05).all()
    assert (jobs["loss_making"] == (jobs["margin_rm"] < 0)).all()
    assert jobs["job_id"].is_unique


def test_keys_join_to_dimensions(tables):
    carriers = set(tables["bi_carriers"]["haulier"])
    destinations = set(tables["bi_destinations"]["destination"])
    for name in ("bi_jobs", "bi_deliveries"):
        assert set(tables[name]["haulier"]) <= carriers
        assert set(tables[name]["destination"]) <= destinations


def test_free_time_snapshot(tables):
    ft = tables["bi_free_time"]
    assert (ft["snapshot_date"] == TODAY).all()
    assert (ft["projected_rm"] >= ft["accrued_rm"]).all()


def test_csv_export(tables, tmp_path):
    paths = write_csv(tables, tmp_path)
    assert {p.name for p in paths} >= {"bi_jobs.csv", "bi_free_time.csv"}
    back = pd.read_csv(tmp_path / "bi_jobs.csv")
    assert len(back) == len(tables["bi_jobs"])


@pytest.mark.integration
def test_write_postgres(tables):
    dsn = os.getenv("TEST_DATABASE_URL", "")
    if not dsn:
        pytest.skip("TEST_DATABASE_URL not set")
    from src.bi.export import write_postgres
    from src.db import PgConnection

    write_postgres(tables, dsn)
    conn = PgConnection(dsn)
    n = conn.execute("SELECT count(*), sum(revenue_rm) FROM bi_jobs").fetchone()
    assert n[0] == len(tables["bi_jobs"])
    assert n[1] == pytest.approx(tables["bi_jobs"]["revenue_rm"].sum(), rel=1e-6)
    types = dict(conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'bi_jobs'"
    ).fetchall())  # fmt: skip
    assert types["date"] == "date" and types["loss_making"] == "boolean"
