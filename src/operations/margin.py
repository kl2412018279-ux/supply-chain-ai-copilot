"""Job profitability: what each completed job earned and where margin leaked.

Revenue is the rate-card quote for the job (what the customer was billed). Planned
cost is the quote's cost, with haulage lines scaled by the subcontractor's price
factor when the job went to an outside haulier. Actual cost adds the leaks a 3PL
fights every week:

* **Unbilled waiting time**: truck idle at the port or customer beyond the free hours.
* **Demurrage / detention absorbed**: line charges paid but not recharged.
* **Late-delivery penalty**: credit note when delivery is more than a day late.
* **Damage claims**: from the delivery log.
* **Diesel above quote**: fuel bought above the price the job was quoted on.

Jobs come from ``data/raw/haulier_deliveries.csv``; the leak events are simulated
per job (seeded) because the demo has no TMS or accounting feed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.costing.quote import Mode, QuoteRequest, build_quote
from src.costing.rates import get_rate_card
from src.hauliers.scorecard import DELIVERIES_CSV
from src.operations.freetime import operations_reference

HAULAGE = {"fuel", "truck", "driver", "tolls"}
LEAKS = ["Unbilled waiting time", "Demurrage / detention absorbed", "Late-delivery penalty",
         "Damage claims", "Diesel above quote"]  # fmt: skip


@lru_cache(maxsize=512)
def _job_quote(destination: str, commodity: str, mode: str, containers: int, carrier: str):
    rates = get_rate_card()
    q = build_quote(
        QuoteRequest(
            destination=destination, cargo_type=commodity, containers=containers,
            mode=Mode(mode),
        ),
        rates,
    )  # fmt: skip
    factor = rates.data["carriers"].get(carrier, {}).get("price_factor", 1.0)
    haulage = sum(i.amount_rm for i in q.line_items if i.category in HAULAGE)
    fuel = sum(i.amount_rm for i in q.line_items if i.category == "fuel")
    planned = q.cost_rm + haulage * (factor - 1)
    return q.price_rm, planned, fuel


def job_ledger(deliveries: pd.DataFrame | None = None, seed: int = 11) -> pd.DataFrame:
    """One row per job: revenue, planned and actual cost, margin and each leak in RM."""
    df = deliveries if deliveries is not None else pd.read_csv(DELIVERIES_CSV)
    ref = operations_reference()
    leak, per_day = ref["leakage"], ref["charges_rm_per_day"]
    customers = {k: v for k, v in ref["customers"].items() if not k.startswith("_")}
    rng = np.random.default_rng(seed)
    rows = []
    for job in df.to_dict("records"):
        try:
            revenue, planned, fuel = _job_quote(
                job["destination"], job["commodity"], job["mode"], int(job["containers"]),
                job["haulier"],
            )  # fmt: skip
        except (KeyError, ValueError):
            continue
        n = int(job["containers"])
        waiting_h = rng.gamma(1.4, 2.2) if rng.random() < 0.3 else 0.0
        waiting = max(0.0, waiting_h - leak["free_waiting_hours"]) * leak["idle_truck_rm_per_hour"]
        absorbed = (
            rng.integers(1, 4) * per_day["detention"]["40"] * n if rng.random() < 0.05 else 0.0
        )
        late = (
            revenue * leak["late_penalty_pct"] / 100
            if job["late_hours"] > leak["late_penalty_after_hours"]
            else 0.0
        )
        diesel = fuel * max(0.0, rng.normal(0.01, 0.025)) if job["mode"] == "truck" else 0.0
        leaks = dict(zip(LEAKS, [waiting, absorbed, late, float(job["claim_rm"]), diesel],
                         strict=True))  # fmt: skip
        actual = planned + sum(leaks.values())
        names = customers[job["commodity"]]
        rows.append(
            {
                "date": job["date"],
                "job": job["delivery_id"],
                "customer": names[int(job["delivery_id"][1:]) % len(names)],
                "commodity": job["commodity"],
                "destination": job["destination"],
                "mode": job["mode"],
                "haulier": job["haulier"],
                "containers": n,
                "revenue_rm": round(revenue, 2),
                "planned_cost_rm": round(planned, 2),
                "actual_cost_rm": round(actual, 2),
                **{k: round(v, 2) for k, v in leaks.items()},
            }
        )
    ledger = pd.DataFrame(rows)
    ledger["date"] = pd.to_datetime(ledger["date"])
    ledger["planned_margin_rm"] = ledger["revenue_rm"] - ledger["planned_cost_rm"]
    ledger["margin_rm"] = ledger["revenue_rm"] - ledger["actual_cost_rm"]
    ledger["leakage_rm"] = ledger[LEAKS].sum(axis=1)
    return ledger


@lru_cache
def get_ledger() -> pd.DataFrame:
    return job_ledger()


def _pct(margin: float, revenue: float) -> float:
    return round(margin / revenue * 100, 1) if revenue else 0.0


class MarginSummary(BaseModel):
    period: str
    jobs: int
    revenue_rm: float
    margin_rm: float
    margin_pct: float
    planned_margin_pct: float
    leakage_rm: float
    leakage_by_reason: dict[str, float]
    loss_making_jobs: int
    by_customer: list[dict[str, Any]]
    worst_lanes: list[dict[str, Any]]


def margin_summary(days: int = 90, ledger: pd.DataFrame | None = None) -> MarginSummary:
    """Margin over the last ``days`` days of the job log, and where it leaked."""
    ledger = get_ledger() if ledger is None else ledger
    end = ledger["date"].max()
    window = ledger[ledger["date"] > end - pd.Timedelta(days=days)]
    rev, mar = window["revenue_rm"].sum(), window["margin_rm"].sum()
    by_cust = (
        window.groupby("customer")[["revenue_rm", "margin_rm", "leakage_rm"]].sum()
        .assign(margin_pct=lambda d: (d["margin_rm"] / d["revenue_rm"] * 100).round(1))
        .sort_values("margin_pct")
        .reset_index()
    )  # fmt: skip
    lanes = (
        window.assign(lane=window["destination"] + " · " + window["mode"])
        .groupby("lane")[["revenue_rm", "margin_rm", "leakage_rm"]].sum()
        .assign(margin_pct=lambda d: (d["margin_rm"] / d["revenue_rm"] * 100).round(1))
        .sort_values("margin_pct")
        .head(5)
        .reset_index()
    )  # fmt: skip
    return MarginSummary(
        period=f"{(end - pd.Timedelta(days=days - 1)).date()} to {end.date()}",
        jobs=len(window),
        revenue_rm=round(float(rev), 2),
        margin_rm=round(float(mar), 2),
        margin_pct=_pct(mar, rev),
        planned_margin_pct=_pct(window["planned_margin_rm"].sum(), rev),
        leakage_rm=round(float(window["leakage_rm"].sum()), 2),
        leakage_by_reason={k: round(float(window[k].sum()), 2) for k in LEAKS},
        loss_making_jobs=int((window["margin_rm"] < 0).sum()),
        by_customer=by_cust.round(2).to_dict("records"),
        worst_lanes=lanes.round(2).to_dict("records"),
    )


def weekly_trend(ledger: pd.DataFrame | None = None, weeks: int = 13) -> pd.DataFrame:
    """Revenue, margin % and leakage for the last ``weeks`` 7-day periods of the job log."""
    ledger = get_ledger() if ledger is None else ledger
    end = ledger["date"].max()
    back = (end - ledger["date"]).dt.days // 7
    recent = ledger[back < weeks]
    w = (
        recent.groupby(back[back < weeks])[["revenue_rm", "margin_rm", "leakage_rm"]].sum()
        .sort_index(ascending=False)
    )  # fmt: skip
    w.insert(0, "week_ending", [(end - pd.Timedelta(days=7 * int(k))).date() for k in w.index])
    w["margin_pct"] = (w["margin_rm"] / w["revenue_rm"] * 100).round(1)
    return w.reset_index(drop=True)
