"""Job costing and quotation for container moves out of Port Klang.

A quote is built bottom-up from the rate card:

* **Fuel** - round trip litres (loaded out, empty back) x diesel price.
* **Truck** - maintenance & tyres per km, fixed costs and driver time per hour.
* **Tolls**, **port handling** (Westports tariff), **customs** (K1/K2 declaration).
* **Cargo** surcharges - dangerous-goods chemicals, palm-oil tank handling.
* **Warehousing** - storage per pallet-day plus handling.
* **Margin** on top of cost gives the quoted price.

Diesel litres are reported alongside the price so fuel exposure is visible.
:func:`compare_modes` sets every offered mode side by side with its door-to-door
transit days, so the trade-off is plain: how much extra per day saved.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from src.costing.rates import RateCard, get_rate_card


class Mode(str, Enum):
    truck = "truck"
    rail = "rail"
    sea = "sea"


class QuoteRequest(BaseModel):
    destination: str = Field(description="Delivery point, e.g. 'Shah Alam', 'Penang', 'Kuching'.")
    container_size: str = Field(default="40", description="'20' or '40' (feet).")
    containers: int = Field(default=1, ge=1, le=100)
    cargo_type: str = Field(
        default="general", description="general, plastics, palm-oil or chemicals."
    )
    cargo_weight_kg: float = Field(
        default=18000, gt=0, le=30000, description="Cargo weight per container (kg)."
    )
    origin: str = Field(default="westport", description="'westport' or 'northport'.")
    mode: Mode | None = Field(default=None, description="truck, rail or sea; default picks best.")
    customs_clearance: bool = True
    road_distance_km: float | None = Field(
        default=None, gt=0, description="Real one-way road distance (e.g. from a routing API)."
    )
    road_drive_hours: float | None = Field(
        default=None, gt=0, description="Real one-way drive time in hours (routing API)."
    )
    warehouse_pallets: int = Field(default=0, ge=0, le=5000)
    warehouse_days: int = Field(default=0, ge=0, le=365)

    @field_validator("container_size", mode="before")
    @classmethod
    def _size(cls, v: object) -> str:
        size = str(v).lower().replace("ft", "").replace("'", "").strip()
        if size not in {"20", "40"}:
            raise ValueError("container_size must be '20' or '40'")
        return size

    @field_validator("cargo_type", "origin", mode="before")
    @classmethod
    def _lower(cls, v: object) -> str:
        from src.forecasting.data import normalize_commodity  # noqa: PLC0415

        return normalize_commodity(str(v))


class LineItem(BaseModel):
    category: str
    description: str
    amount_rm: float


class Quote(BaseModel):
    destination: str
    destination_label: str
    origin_label: str
    mode: Mode
    container_size: str
    containers: int
    cargo_type: str
    distance_km: float
    distance_source: str = "rate-card estimate"
    line_items: list[LineItem]
    cost_rm: float
    margin_pct: float
    margin_rm: float
    price_rm: float
    price_per_container_rm: float
    cost_per_teu_rm: float
    diesel_litres: float
    warnings: list[str] = Field(default_factory=list)


class QuoteComparison(BaseModel):
    quotes: list[Quote]
    cheapest_mode: Mode
    fastest_mode: Mode
    transit_days: dict[str, int] = Field(description="Door-to-door transit days per mode.")
    premium_rm_per_day_saved: float | None = Field(
        default=None,
        description="Extra RM paid per transit day saved by choosing the fastest mode over the "
        "cheapest one; None when the cheapest mode is also the fastest.",
    )


def _round(x: float) -> float:
    return round(x + 1e-9, 2)


def build_quote(req: QuoteRequest, rates: RateCard | None = None) -> Quote:
    rates = rates or get_rate_card()
    dest_key = rates.resolve_destination(req.destination)
    dest = rates.destinations[dest_key]
    if req.origin not in rates.origins:
        raise ValueError(f"Unknown origin '{req.origin}'. Use: {', '.join(rates.origins)}")
    if req.cargo_type not in rates.cargo_types:
        raise ValueError(
            f"Unknown cargo type '{req.cargo_type}'. Use: {', '.join(rates.cargo_types)}"
        )
    origin = rates.origins[req.origin]
    mode = req.mode or Mode(dest["modes"][0])
    if mode.value not in dest["modes"]:
        raise ValueError(f"{mode.value} is not offered to {dest['label']} (use {dest['modes']}).")

    n = req.containers
    box = rates.container(req.container_size)
    cargo = rates.cargo_types[req.cargo_type]
    km = max(dest["km"] + origin["km_offset"], 5)
    distance_source = "rate-card estimate"
    one_way_drive_h: float | None = None
    if mode is Mode.truck and req.road_distance_km:
        km, distance_source = req.road_distance_km, "road route (OpenRouteService)"
        one_way_drive_h = req.road_drive_hours
    gross_kg = req.cargo_weight_kg + box["tare_kg"]
    warnings: list[str] = []
    max_gross = rates.value("truck", "max_gross_weight_kg")
    if gross_kg > max_gross:
        warnings.append(
            f"Gross weight {gross_kg:,.0f} kg per container exceeds the {max_gross:,.0f} kg "
            "haulage limit at Port Klang; split the load or use a side loader permit."
        )

    items: list[LineItem] = []

    def add(category: str, description: str, amount: float) -> None:
        if amount:
            items.append(
                LineItem(category=category, description=description, amount_rm=_round(amount))
            )

    peninsular_diesel = rates.value("fuel", "diesel_rm_per_litre_peninsular")
    l_loaded = rates.value("fuel", "litres_per_km_loaded")
    l_empty = rates.value("fuel", "litres_per_km_empty")
    speed = rates.value("truck", "avg_speed_kmh")

    def truck_leg(leg_km: float, label: str) -> tuple[float, float]:
        """Round-trip trucking cost lines for ``leg_km`` one way; returns (litres, hours)."""
        litres = leg_km * (l_loaded + l_empty) * n
        drive = 2 * one_way_drive_h if one_way_drive_h and leg_km == km else 2 * leg_km / speed
        hours = (drive + rates.value("truck", "port_turnaround_hours")) * n
        add(
            "fuel",
            f"{label}: diesel {litres:,.0f} L @ RM{peninsular_diesel:.2f}/L",
            litres * peninsular_diesel,
        )
        add(
            "truck",
            f"{label}: maintenance & tyres {2 * leg_km * n:,.0f} km",
            2 * leg_km * n * rates.value("truck", "maintenance_tyres_rm_per_km"),
        )
        add(
            "truck",
            f"{label}: fixed truck cost {hours:,.1f} h",
            hours * rates.value("truck", "fixed_cost_rm_per_hour"),
        )
        add(
            "driver",
            f"{label}: driver {hours:,.1f} h + allowance",
            hours * rates.value("truck", "driver_rm_per_hour")
            + n * rates.value("truck", "driver_trip_allowance_rm"),
        )
        return litres, hours

    litres = 0.0
    if mode is Mode.truck:
        litres, _ = truck_leg(km, "Haulage")
        add("tolls", f"Tolls (round trip, {n} x)", 2 * dest["toll_rm"] * n)
    elif mode is Mode.rail:
        first_last = rates.data["rail"]["first_last_mile_km"]
        litres, _ = truck_leg(first_last, "First/last-mile truck")
        rail_km = max(km - first_last, 0)
        add(
            "linehaul",
            f"Rail linehaul {rail_km:,.0f} km x {n}",
            rail_km * rates.data["rail"]["rm_per_container_km"][req.container_size] * n,
        )
    else:  # sea to East Malaysia
        add(
            "linehaul",
            f"Coastal shipping Port Klang -> {dest['label']} x {n}",
            rates.data["sea"]["rm_per_container"][req.container_size] * n,
        )
        add(
            "port",
            f"Destination port handling ({req.container_size}ft x {n})",
            box["port_handling_rm"] * n,
        )
        warnings.append(
            "Port-to-port only: delivery inside East Malaysia is quoted separately "
            "(local diesel is priced differently)."
        )

    add("port", f"Port Klang handling ({req.container_size}ft x {n})", box["port_handling_rm"] * n)
    if req.customs_clearance:
        add("customs", "Customs declaration (K1/K2)", rates.value("customs", "declaration_rm"))
        add(
            "customs",
            f"Customs inspection coordination x {n}",
            n * rates.value("customs", "per_container_rm"),
        )
    haulage_base = sum(
        i.amount_rm for i in items if i.category in {"fuel", "truck", "driver", "linehaul"}
    )
    if cargo["haulage_surcharge_pct"]:
        add(
            "cargo",
            f"{cargo['label']} surcharge {cargo['haulage_surcharge_pct']}% of haulage",
            haulage_base * cargo["haulage_surcharge_pct"] / 100,
        )
    add("cargo", f"{cargo['label']} handling fee x {n}", cargo["fixed_fee_rm"] * n)
    if req.warehouse_pallets and req.warehouse_days:
        mult = cargo["warehouse_multiplier"]
        pallet_days = req.warehouse_pallets * req.warehouse_days
        add(
            "warehouse",
            f"Storage {req.warehouse_pallets} pallets x {req.warehouse_days} days"
            + (f" (x{mult} DG store)" if mult != 1 else ""),
            pallet_days * rates.value("warehouse", "storage_rm_per_pallet_day") * mult,
        )
        add(
            "warehouse",
            f"Pallet handling in/out x {req.warehouse_pallets}",
            req.warehouse_pallets * rates.value("warehouse", "handling_rm_per_pallet"),
        )

    cost = sum(i.amount_rm for i in items)
    margin_pct = float(rates.data["margin_pct"])
    margin = cost * margin_pct / 100
    price = cost + margin
    return Quote(
        destination=dest_key,
        destination_label=dest["label"],
        origin_label=origin["label"],
        mode=mode,
        container_size=req.container_size,
        containers=n,
        cargo_type=req.cargo_type,
        distance_km=km,
        distance_source=distance_source,
        line_items=items,
        cost_rm=_round(cost),
        margin_pct=margin_pct,
        margin_rm=_round(margin),
        price_rm=_round(price),
        price_per_container_rm=_round(price / n),
        cost_per_teu_rm=_round(cost / (n * box["teu"])),
        diesel_litres=round(litres, 1),
        warnings=warnings,
    )


def compare_modes(req: QuoteRequest, rates: RateCard | None = None) -> QuoteComparison:
    """Quote every transport mode offered to the destination, with the price/speed trade-off."""
    from src.costing.routing import transit_days  # noqa: PLC0415 (routing imports this module)

    rates = rates or get_rate_card()
    dest_key = rates.resolve_destination(req.destination)
    dest = rates.destinations[dest_key]
    quotes = [build_quote(req.model_copy(update={"mode": Mode(m)}), rates) for m in dest["modes"]]
    days = {
        q.mode.value: transit_days(rates, dest_key, q.mode, {}, req.road_drive_hours)
        for q in quotes
    }
    cheapest = min(quotes, key=lambda q: q.price_rm)
    fastest = min(quotes, key=lambda q: (days[q.mode.value], q.price_rm))
    premium = None
    saved = days[cheapest.mode.value] - days[fastest.mode.value]
    if fastest is not cheapest and saved > 0:
        premium = round((fastest.price_rm - cheapest.price_rm) / saved, 2)
    return QuoteComparison(
        quotes=quotes,
        cheapest_mode=cheapest.mode,
        fastest_mode=fastest.mode,
        transit_days=days,
        premium_rm_per_day_saved=premium,
    )
