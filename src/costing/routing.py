"""Route & carrier recommendation: balance price, transit time, reliability and cargo safety.

Every (mode, carrier) combination that serves the destination becomes an option:

* **price**: the rate-card quote, with the carrier's price factor applied to the
  haulage lines (fuel, truck, driver, linehaul) before margin;
* **transit days**: door-to-door, by mode, plus the carrier's speed adjustment;
* **reliability**: the carrier's *historical* on-time rate from the delivery log;
* **safety**: the carrier's *historical* damage rate (share of jobs with a cargo
  damage claim) from the same delivery log.

Each criterion is min-max normalised across the options (0 = best) and combined with
the user's priority weights. Options that no other option beats on every criterion
are flagged Pareto-optimal; a dominated option is never the right answer.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

from src.costing.quote import Mode, QuoteRequest, build_quote
from src.costing.rates import RateCard, get_rate_card

HAULAGE_CATEGORIES = {"fuel", "truck", "driver", "linehaul"}
CRITERIA = ("cost", "speed", "reliability", "safety")


class Priorities(BaseModel):
    """Relative importance of each criterion; normalised to sum to 1."""

    cost: float = Field(default=0.4, ge=0)
    speed: float = Field(default=0.2, ge=0)
    reliability: float = Field(default=0.25, ge=0)
    safety: float = Field(default=0.15, ge=0, description="Low cargo damage / claims risk")

    def normalised(self) -> dict[str, float]:
        total = sum(getattr(self, c) for c in CRITERIA)
        if total <= 0:
            return dict.fromkeys(CRITERIA, 0.25)
        return {c: getattr(self, c) / total for c in CRITERIA}


class RouteOption(BaseModel):
    carrier: str
    carrier_label: str
    mode: Mode
    fleet: str
    price_rm: float
    cost_rm: float
    transit_days: int
    on_time_rate: float
    damage_rate: float = Field(description="Share of historical jobs with cargo damage")
    criterion_scores: dict[str, float] = Field(
        description="0 = best option on that criterion, 1 = worst"
    )
    fit: float = Field(description="0-100, higher is a better match for the priorities")
    pareto_optimal: bool
    rank: int = 0


class RouteRecommendation(BaseModel):
    destination: str
    destination_label: str
    priorities: dict[str, float]
    options: list[RouteOption]
    recommended: RouteOption
    explanation: str


def transit_days(
    rates: RateCard,
    destination: str,
    mode: Mode,
    carrier: dict[str, Any],
    road_drive_hours: float | None = None,
) -> int:
    t = rates.data["transit"]
    km = rates.destinations[destination]["km"]
    if mode is Mode.truck and road_drive_hours:
        # Real drive time (routing API) + port turnaround, within one driver shift per day.
        hours = road_drive_hours + rates.value("truck", "port_turnaround_hours")
        days = math.ceil(hours / rates.value("truck", "shift_hours"))
    elif mode is Mode.truck:
        days = math.ceil(km / t["truck_shift_km_per_day"])
    elif mode is Mode.rail:
        days = t["rail_terminal_dwell_days"] + math.ceil(km / t["rail_km_per_day"])
    else:
        days = t["sea_port_days"] + math.ceil(km / t["sea_km_per_day"])
    return max(1, days + int(carrier.get("transit_days_delta", 0)))


def _days(n: int) -> str:
    return f"{n} day" if n == 1 else f"{n} days"


def _pct(new: float, old: float) -> str:
    change = (new - old) / old * 100 if old else 0.0
    if round(change) == 0:
        return "same price"
    return f"{abs(change):.0f}% {'cheaper' if change < 0 else 'more expensive'}"


def _versus(best: RouteOption, ref: RouteOption, which: str) -> str:
    """Plain-terms trade-off of ``best`` against an alternative, e.g. 1 day slower, 18% cheaper."""
    day_diff = best.transit_days - ref.transit_days
    timing = (
        f"{_days(day_diff)} slower"
        if day_diff > 0
        else f"{_days(-day_diff)} faster"
        if day_diff < 0
        else "same transit time"
    )
    parts = [
        timing,
        _pct(best.price_rm, ref.price_rm),
        f"on time {best.on_time_rate:.0%} vs {ref.on_time_rate:.0%}",
    ]
    if round(best.damage_rate, 3) != round(ref.damage_rate, 3):
        parts.append(f"damage rate {best.damage_rate:.1%} vs {ref.damage_rate:.1%}")
    return (
        f" Versus the {which} option ({ref.carrier_label}, {ref.mode.value}): "
        + ", ".join(parts)
        + "."
    )


def recommend(
    req: QuoteRequest,
    priorities: Priorities | None = None,
    rates: RateCard | None = None,
    on_time: dict[str, float] | None = None,
    damage: dict[str, float] | None = None,
) -> RouteRecommendation:
    rates = rates or get_rate_card()
    priorities = priorities or Priorities()
    if on_time is None:
        from src.hauliers.scorecard import on_time_rates  # noqa: PLC0415

        on_time = on_time_rates()
    if damage is None:
        from src.hauliers.scorecard import damage_rates  # noqa: PLC0415

        damage = damage_rates()
    from src.costing.road_routing import with_road_route  # noqa: PLC0415

    req, _ = with_road_route(req, rates)
    dest_key = rates.resolve_destination(req.destination)
    dest = rates.destinations[dest_key]
    carriers = {k: v for k, v in rates.data["carriers"].items() if not k.startswith("_")}

    raw: list[dict[str, Any]] = []
    for mode_name in dest["modes"]:
        mode = Mode(mode_name)
        base = build_quote(req.model_copy(update={"mode": mode}), rates)
        haulage = sum(i.amount_rm for i in base.line_items if i.category in HAULAGE_CATEGORIES)
        for key, carrier in carriers.items():
            if mode_name not in carrier["modes"]:
                continue
            cost = base.cost_rm + haulage * (carrier["price_factor"] - 1)
            raw.append(
                {
                    "carrier": key,
                    "carrier_label": carrier["label"],
                    "mode": mode,
                    "fleet": carrier.get("fleet", ""),
                    "cost_rm": round(cost, 2),
                    "price_rm": round(cost * (1 + base.margin_pct / 100), 2),
                    "transit_days": transit_days(
                        rates, dest_key, mode, carrier, req.road_drive_hours
                    ),
                    "on_time_rate": round(on_time.get(key, 0.9), 4),
                    "damage_rate": round(damage.get(key, 0.01), 4),
                }
            )
    if not raw:
        raise ValueError(f"No carrier serves {dest['label']}.")

    values = {
        "cost": [o["price_rm"] for o in raw],
        "speed": [o["transit_days"] for o in raw],
        "reliability": [1 - o["on_time_rate"] for o in raw],
        "safety": [o["damage_rate"] for o in raw],
    }
    weights = priorities.normalised()
    options: list[RouteOption] = []
    for i, o in enumerate(raw):
        scores = {}
        for c in CRITERIA:
            lo, hi = min(values[c]), max(values[c])
            scores[c] = round((values[c][i] - lo) / (hi - lo), 3) if hi > lo else 0.0
        penalty = sum(weights[c] * scores[c] for c in CRITERIA)
        dominated = any(
            all(values[c][j] <= values[c][i] for c in CRITERIA)
            and any(values[c][j] < values[c][i] for c in CRITERIA)
            for j in range(len(raw))
            if j != i
        )
        options.append(
            RouteOption(
                **o,
                criterion_scores=scores,
                fit=round(100 * (1 - penalty), 1),
                pareto_optimal=not dominated,
            )
        )
    options.sort(key=lambda o: (-o.fit, o.price_rm))
    for rank, o in enumerate(options, start=1):
        o.rank = rank
    best = options[0]
    explanation = (
        f"Recommended: {best.carrier_label} by {best.mode.value} — {_days(best.transit_days)}, "
        f"RM{best.price_rm:,.0f}, {best.on_time_rate:.0%} on time, "
        f"{best.damage_rate:.1%} damage rate."
    )
    # Explain the trade-off against the alternatives that beat it on one criterion.
    alternatives = [
        ("cheapest", min(options, key=lambda o: (o.price_rm, o.transit_days)), "price_rm", -1),
        ("fastest", min(options, key=lambda o: (o.transit_days, o.price_rm)), "transit_days", -1),
        ("most reliable", max(options, key=lambda o: (o.on_time_rate, -o.price_rm)),
         "on_time_rate", 1),
        ("safest", min(options, key=lambda o: (o.damage_rate, o.price_rm)), "damage_rate", -1),
    ]  # fmt: skip
    shown: list[RouteOption] = []
    for which, ref, attr, better in alternatives:
        if ref is best or ref in shown or len(shown) == 2:
            continue
        if (getattr(ref, attr) - getattr(best, attr)) * better > 0:
            explanation += _versus(best, ref, which)
            shown.append(ref)
    if not shown and len(options) > 1:
        explanation += " It is the best option on every criterion."
    return RouteRecommendation(
        destination=dest_key,
        destination_label=dest["label"],
        priorities={c: round(w, 3) for c, w in weights.items()},
        options=options,
        recommended=best,
        explanation=explanation,
    )
