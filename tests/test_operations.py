import datetime as dt

import pandas as pd
import pytest

from src.extraction.iso6346 import is_valid
from src.operations.freetime import Status, generate_board, price_board, summarise
from src.operations.margin import LEAKS, job_ledger, margin_summary, weekly_trend

TODAY = dt.date(2026, 10, 14)


def _box(status: Status, discharged: dt.date, planned: dt.date, gate_out=None, size="40"):
    return {
        "container": "MSKU1234565", "size": size, "customer": "X", "commodity": "general",
        "destination": "shah-alam", "port": "westport", "status": status.value,
        "discharged": discharged, "gate_out": gate_out, "planned_move": planned,
    }  # fmt: skip


def test_demurrage_clock_and_charges():
    # Discharged 10 Oct, 3 free days -> last free day 12 Oct. On 14 Oct: 2 charged days.
    row = price_board(
        pd.DataFrame([_box(Status.awaiting_haulier, dt.date(2026, 10, 10), dt.date(2026, 10, 16))]),
        TODAY,
    ).iloc[0]
    assert row["clock"] == "Demurrage"
    assert row["last_free_day"] == dt.date(2026, 10, 12)
    assert row["days_left"] == -2 and row["urgency"] == "Overdue"
    assert row["accrued_rm"] == 2 * (240 + 40)  # demurrage + port storage, 40ft
    assert row["projected_rm"] == 4 * (240 + 40)
    assert row["action"] == "Book a haulier"


def test_detention_clock_after_gate_out():
    row = price_board(
        pd.DataFrame(
            [_box(Status.empty_return, dt.date(2026, 10, 8), dt.date(2026, 10, 16),
                  gate_out=dt.date(2026, 10, 12), size="20")]
        ),
        TODAY,
    ).iloc[0]  # fmt: skip
    assert row["clock"] == "Detention"
    assert row["last_free_day"] == dt.date(2026, 10, 16)  # 5 free days from gate-out
    assert row["accrued_rm"] == 0 and row["projected_rm"] == 0
    assert row["urgency"] == "2-3 days"


def test_board_is_deterministic_and_sorted():
    a, b = generate_board(TODAY), generate_board(TODAY)
    pd.testing.assert_frame_equal(a, b)
    assert all(is_valid(c) for c in a["container"])
    order = ["Overdue", "Today", "Tomorrow", "2-3 days", "OK"]
    ranks = a["urgency"].map(order.index)
    assert ranks.is_monotonic_increasing
    assert (a["projected_rm"] >= a["accrued_rm"]).all()


def test_summary_counts():
    board = generate_board(TODAY)
    s = summarise(board, TODAY)
    assert s.containers == len(board)
    assert s.overdue == int((board["days_left"] < 0).sum())
    assert sum(s.by_status.values()) == len(board)
    assert len(s.top) <= 8


@pytest.fixture(scope="module")
def ledger():
    return job_ledger()


def test_ledger_margin_identity(ledger):
    assert len(ledger) > 1000
    leak_sum = ledger[LEAKS].sum(axis=1)
    assert ((ledger["actual_cost_rm"] - ledger["planned_cost_rm"] - leak_sum).abs() < 0.05).all()
    assert (ledger[LEAKS] >= 0).all().all()
    assert ((ledger["margin_rm"] - (ledger["revenue_rm"] - ledger["actual_cost_rm"])).abs()
            < 0.05).all()  # fmt: skip


def test_margin_summary(ledger):
    s = margin_summary(90, ledger)
    assert s.jobs > 0 and s.revenue_rm > 0
    assert s.margin_pct < s.planned_margin_pct  # leaks only ever cost money
    assert s.leakage_rm == pytest.approx(sum(s.leakage_by_reason.values()), abs=1)
    assert [c["margin_pct"] for c in s.by_customer] == sorted(
        c["margin_pct"] for c in s.by_customer
    )


def test_weekly_trend_has_full_weeks(ledger):
    w = weekly_trend(ledger, weeks=13)
    assert len(w) == 13
    assert (w["revenue_rm"] > 0).all()
    assert w["week_ending"].is_monotonic_increasing
