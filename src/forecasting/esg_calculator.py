"""Shipping (Scope 3, upstream transport) emission calculator.

Emission = distance (km) x weight (tonnes) x mode factor (kg CO2e per tonne-km).
Factors are typical well-to-wheel averages in the range published by the GLEC
Framework / UK DEFRA conversion factors; override them for audited reporting.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

EMISSION_FACTORS_KG_PER_TKM: dict[str, float] = {
    "truck": 0.105,
    "rail": 0.028,
    "sea": 0.016,
    "air": 0.500,
}


class TransportMode(str, Enum):
    truck = "truck"
    rail = "rail"
    sea = "sea"
    air = "air"


class EmissionResult(BaseModel):
    distance_km: float = Field(gt=0)
    weight_kg: float = Field(gt=0)
    mode: TransportMode
    factor_kg_per_tkm: float
    co2e_kg: float
    alternatives_co2e_kg: dict[str, float]
    lowest_emission_mode: TransportMode


def calculate_carbon(distance_km: float, weight_kg: float, transport_mode: str = "truck") -> float:
    """kg CO2e for moving ``weight_kg`` over ``distance_km`` using ``transport_mode``."""
    if distance_km <= 0 or weight_kg <= 0:
        raise ValueError("distance_km and weight_kg must be positive.")
    mode = transport_mode.strip().lower()
    if mode not in EMISSION_FACTORS_KG_PER_TKM:
        raise ValueError(
            f"Unknown transport mode '{transport_mode}'. "
            f"Choose one of: {', '.join(EMISSION_FACTORS_KG_PER_TKM)}."
        )
    return round(distance_km * (weight_kg / 1000.0) * EMISSION_FACTORS_KG_PER_TKM[mode], 2)


def calculate_emissions(
    distance_km: float, weight_kg: float, transport_mode: str = "truck"
) -> EmissionResult:
    """Emissions for the chosen mode plus a comparison against every other mode."""
    co2e = calculate_carbon(distance_km, weight_kg, transport_mode)
    alternatives = {
        m: calculate_carbon(distance_km, weight_kg, m) for m in EMISSION_FACTORS_KG_PER_TKM
    }
    lowest = min(alternatives, key=alternatives.__getitem__)
    mode = transport_mode.strip().lower()
    return EmissionResult(
        distance_km=distance_km,
        weight_kg=weight_kg,
        mode=TransportMode(mode),
        factor_kg_per_tkm=EMISSION_FACTORS_KG_PER_TKM[mode],
        co2e_kg=co2e,
        alternatives_co2e_kg=alternatives,
        lowest_emission_mode=TransportMode(lowest),
    )
