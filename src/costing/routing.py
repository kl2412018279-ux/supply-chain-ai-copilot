"""Route & carrier recommendation: balance price, transit time, CO2e and reliability.

Every (mode, carrier) combination that serves the destination becomes an option:

* **price**: the rate-card quote, with the carrier's price factor applied to the
  haulage lines (fuel, truck, driver, linehaul) before margin;
* **transit days**: door-to-door, by mode, plus the carrier's speed adjustment;
* **CO2e**: the quote's emissions scaled by the carrier's fleet factor;
* **reliability**: the carrier's *historical* on-time rate from the delivery log.

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
CRITERIA = ("cost", "speed", "co2", "reliability")


class Priorities(BaseModel):
    """Relative importance of each criterion; normalised to sum to 1."""

    cost: float = Field(default=0.4, ge=0)
    speed: float = Field(default=0.2, ge=0)
    co2: float = Field(default=0.2, ge=0)
    reliability: float = Field(default=0.2, ge=0)

    def normalised(self) -> dict[str, float]:
        total = self.cost + self.speed + self.co2 + self.reliability
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
    co2e_kg: float
    transit_days: int
    on_time_rate: float
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
    return f"{abs(change):.0f}% {'cheaper' if change < 0 else 'more expensive'}"


def recommend(
    req: QuoteRequest,
    priorities: Priorities | None = None,
    rates: RateCard | None = None,
    on_time: dict[str, float] | None = None,
) -> RouteRecommendation:
    rates = rates or get_rate_card()
    priorities = priorities or Priorities()
    if on_time is None:
        from src.hauliers.scorecard import on_time_rates  # noqa: PLC0415

        on_time = on_time_rates()
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
                    "co2e_kg": round(base.co2e_kg * carrier["co2_factor"], 1),
                    "transit_days": transit_days(
                        rates, dest_key, mode, carrier, req.road_drive_hours
                    ),
                    "on_time_rate": round(on_time.get(key, 0.9), 4),
                }
            )
    if not raw:
        raise ValueError(f"No carrier serves {dest['label']}.")

    values = {
        "cost": [o["price_rm"] for o in raw],
        "speed": [o["transit_days"] for o in raw],
        "co2": [o["co2e_kg"] for o in raw],
        "reliability": [1 - o["on_time_rate"] for o in raw],
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
        f"RM{best.price_rm:,.0f}, {best.co2e_kg:,.0f} kg CO2e, {best.on_time_rate:.0%} on time."
    )
    # Explain the trade-off against the most relevant alternative.
    fastest = min(options, key=lambda o: (o.transit_days, o.price_rm))
    cheapest = min(options, key=lambda o: o.price_rm)
    greenest = min(options, key=lambda o: o.co2e_kg)
    if fastest.transit_days < best.transit_days:
        ref, which = fastest, "fastest"
    elif cheapest.price_rm < best.price_rm:
        ref, which = cheapest, "cheapest"
    elif greenest.co2e_kg < best.co2e_kg:
        ref, which = greenest, "lowest-CO2"
    else:
        ref, which = None, ""
    if ref is not None:
        day_diff = best.transit_days - ref.transit_days
        timing = (
            f"{_days(day_diff)} slower"
            if day_diff > 0
            else f"{_days(-day_diff)} faster"
            if day_diff < 0
            else "same transit time"
        )
        co2_change = (best.co2e_kg - ref.co2e_kg) / ref.co2e_kg * 100 if ref.co2e_kg else 0
        explanation += (
            f" Versus the {which} option ({ref.carrier_label}, {ref.mode.value}): {timing}, "
            f"{_pct(best.price_rm, ref.price_rm)}, "
            f"{abs(co2_change):.0f}% {'less' if co2_change < 0 else 'more'} CO2e, "
            f"and {(best.on_time_rate - ref.on_time_rate) * 100:+.0f} pts on-time rate."
        )
    elif len(options) > 1:
        explanation += " It is the best option on every criterion."
    return RouteRecommendation(
        destination=dest_key,
        destination_label=dest["label"],
        priorities={c: round(w, 3) for c, w in weights.items()},
        options=options,
        recommended=best,
        explanation=explanation,
    )
