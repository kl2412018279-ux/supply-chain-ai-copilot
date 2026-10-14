"""Free-time board: which import containers are about to start costing money.

Every import box at Port Klang runs on two clocks:

* **Demurrage + port storage** from discharge until it gates out of the port,
  free for the first few days, then charged per container per day.
* **Detention** from gate-out until the empty is returned to the line's depot.

The board lists live containers with their last free day, what they have already
accrued, what they will cost if moved on the planned date, and the next action
for the ops team. The container list is synthetic (seeded by date) because there
is no terminal or TMS feed in the demo; the clock and charge logic is the real part.
"""

from __future__ import annotations

import datetime as dt
import json
from enum import Enum
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.config import PROJECT_ROOT
from src.costing.rates import get_rate_card
from src.extraction.iso6346 import check_digit

OPERATIONS_PATH = PROJECT_ROOT / "data" / "reference" / "operations.json"
OWNER_CODES = ["MSKU", "MSCU", "CMAU", "ONEU", "EVRU", "HLXU", "OOLU", "TGHU"]


class Status(str, Enum):
    customs_hold = "Customs hold"
    awaiting_haulier = "Awaiting haulier"
    pickup_booked = "Pickup booked"
    empty_return = "Empty not returned"


ACTIONS = {
    Status.customs_hold: "Chase K1 release / inspection slot",
    Status.awaiting_haulier: "Book a haulier",
    Status.pickup_booked: "Confirm pickup slot",
    Status.empty_return: "Return empty to depot",
}


@lru_cache
def operations_reference() -> dict[str, Any]:
    return json.loads(OPERATIONS_PATH.read_text(encoding="utf-8"))


def _container_number(rng: np.random.Generator) -> str:
    first10 = f"{rng.choice(OWNER_CODES)}{rng.integers(0, 10**6):06d}"
    return f"{first10}{check_digit(first10)}"


def generate_board(today: dt.date | None = None, n: int = 30, seed: int = 7) -> pd.DataFrame:
    """Synthetic live containers for ``today`` (same day -> same board)."""
    today = today or dt.date.today()
    ref = operations_reference()
    customers = {k: v for k, v in ref["customers"].items() if not k.startswith("_")}
    rates = get_rate_card()
    trucked = [k for k, d in rates.destinations.items() if "truck" in d["modes"]]
    rng = np.random.default_rng(seed + today.toordinal())
    statuses = list(Status)
    rows = []
    for _ in range(n):
        commodity = str(rng.choice(list(customers), p=[0.3, 0.3, 0.25, 0.15]))
        status = statuses[rng.choice(4, p=[0.15, 0.3, 0.3, 0.25])]
        size = str(rng.choice(["20", "40"], p=[0.35, 0.65]))
        discharged = today - dt.timedelta(days=int(rng.integers(0, 5)))
        gate_out = None
        planned = None
        if status is Status.empty_return:
            gate_out = discharged + dt.timedelta(days=int(rng.integers(1, 4)))
            if gate_out > today:
                gate_out = today - dt.timedelta(days=1)
            planned = gate_out + dt.timedelta(days=int(rng.integers(2, 9)))
        elif status is Status.pickup_booked:
            planned = today + dt.timedelta(days=int(rng.integers(0, 4)))
        elif status is Status.awaiting_haulier:
            planned = today + dt.timedelta(days=int(rng.integers(1, 5)))  # earliest truck
        else:  # customs hold: release + pickup is uncertain
            planned = today + dt.timedelta(days=int(rng.integers(2, 6)))
        planned = max(planned, today)
        rows.append(
            {
                "container": _container_number(rng),
                "size": size,
                "customer": str(rng.choice(customers[commodity])),
                "commodity": commodity,
                "destination": str(rng.choice(trucked)),
                "port": str(rng.choice(["westport", "northport"], p=[0.7, 0.3])),
                "status": status.value,
                "discharged": discharged,
                "gate_out": gate_out,
                "planned_move": planned,
            }
        )
    return price_board(pd.DataFrame(rows), today)


def price_board(board: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    """Add the clock, charges and next action to a container list."""
    ref = operations_reference()
    free, per_day = ref["free_time"], ref["charges_rm_per_day"]
    out = []
    for row in board.to_dict("records"):
        size = row["size"]
        at_port = row["status"] != Status.empty_return.value
        if at_port:
            clock = "Demurrage"
            start = row["discharged"]
            last_free = start + dt.timedelta(days=free["demurrage_free_days"] - 1)
            storage_free = start + dt.timedelta(days=free["port_storage_free_days"] - 1)
            daily = per_day["demurrage"][size]
        else:
            clock = "Detention"
            start = row["gate_out"]
            last_free = start + dt.timedelta(days=free["detention_free_days"] - 1)
            storage_free = None
            daily = per_day["detention"][size]

        def charged_days(until: dt.date, free_until: dt.date | None) -> int:
            return max(0, (until - free_until).days) if free_until else 0

        storage_daily = per_day["port_storage"][size] if at_port else 0.0
        accrued = (
            charged_days(today, last_free) * daily
            + charged_days(today, storage_free) * storage_daily
        )
        projected = (
            charged_days(row["planned_move"], last_free) * daily
            + charged_days(row["planned_move"], storage_free) * storage_daily
        )
        days_left = (last_free - today).days
        out.append(
            {
                **row,
                "clock": clock,
                "last_free_day": last_free,
                "days_left": days_left,
                "daily_rate_rm": daily + storage_daily,
                "accrued_rm": round(accrued, 2),
                "projected_rm": round(projected, 2),
                "urgency": _urgency(days_left),
                "action": ACTIONS[Status(row["status"])],
            }
        )
    df = pd.DataFrame(out)
    order = {"Overdue": 0, "Today": 1, "Tomorrow": 2, "2-3 days": 3, "OK": 4}
    return (
        df.assign(_o=df["urgency"].map(order))
        .sort_values(["_o", "projected_rm"], ascending=[True, False])
        .drop(columns="_o")
        .reset_index(drop=True)
    )


def _urgency(days_left: int) -> str:
    if days_left < 0:
        return "Overdue"
    if days_left == 0:
        return "Today"
    if days_left == 1:
        return "Tomorrow"
    return "2-3 days" if days_left <= 3 else "OK"


class FreeTimeSummary(BaseModel):
    as_of: str
    containers: int
    overdue: int
    due_within_48h: int
    accruing_rm_per_day: float
    accrued_rm: float
    projected_rm: float
    by_status: dict[str, int]
    top: list[dict[str, Any]]


def summarise(board: pd.DataFrame, today: dt.date, top: int = 8) -> FreeTimeSummary:
    overdue = board["days_left"] < 0
    cols = ["container", "size", "customer", "status", "clock", "last_free_day", "days_left",
            "accrued_rm", "projected_rm", "action"]  # fmt: skip
    head = board[cols].head(top).copy()
    head["last_free_day"] = head["last_free_day"].astype(str)
    return FreeTimeSummary(
        as_of=today.isoformat(),
        containers=len(board),
        overdue=int(overdue.sum()),
        due_within_48h=int(board["days_left"].between(0, 1).sum()),
        accruing_rm_per_day=round(float(board.loc[overdue, "daily_rate_rm"].sum()), 2),
        accrued_rm=round(float(board["accrued_rm"].sum()), 2),
        projected_rm=round(float(board["projected_rm"].sum()), 2),
        by_status=board["status"].value_counts().to_dict(),
        top=head.to_dict("records"),
    )


def free_time_summary(today: dt.date | None = None) -> FreeTimeSummary:
    today = today or dt.date.today()
    return summarise(generate_board(today), today)
