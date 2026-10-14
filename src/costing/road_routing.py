"""Real road routing for trucks via OpenRouteService (free API key, HGV profile).

When ``ORS_API_KEY`` is set, truck quotes use the actual road distance and drive time
from Westport/Northport to the destination, and the dashboard can draw the road on
the map with turn-by-turn steps. Without a key (or on any API error) everything falls
back to the rate card's approximate distance table, so the app never breaks.

Free key: https://openrouteservice.org/dev/#/signup
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import httpx
from pydantic import BaseModel, Field

from src.config import get_settings
from src.costing.rates import RateCard, get_rate_card

logger = logging.getLogger(__name__)

ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-hgv/geojson"
TIMEOUT_S = 10.0


class RouteStep(BaseModel):
    instruction: str
    road: str = ""
    distance_km: float
    duration_min: float


class RoadRoute(BaseModel):
    distance_km: float
    drive_hours: float
    steps: list[RouteStep] = Field(default_factory=list)
    geometry: list[tuple[float, float]] = Field(
        default_factory=list, description="(lat, lon) points along the road"
    )
    source: str = "OpenRouteService (driving-hgv)"


class RoutingUnavailable(RuntimeError):
    pass


def parse_ors_geojson(payload: dict[str, Any]) -> RoadRoute:
    """Parse an ORS ``/directions/{profile}/geojson`` response."""
    try:
        feature = payload["features"][0]
        props = feature["properties"]
        summary = props["summary"]
        steps = [
            RouteStep(
                instruction=step.get("instruction", ""),
                road=step.get("name", "") if step.get("name", "-") != "-" else "",
                distance_km=round(step.get("distance", 0) / 1000, 2),
                duration_min=round(step.get("duration", 0) / 60, 1),
            )
            for segment in props.get("segments", [])
            for step in segment.get("steps", [])
        ]
        coords = feature.get("geometry", {}).get("coordinates", [])
        return RoadRoute(
            distance_km=round(summary["distance"] / 1000, 1),
            drive_hours=round(summary["duration"] / 3600, 2),
            steps=steps,
            geometry=[(lat, lon) for lon, lat, *_ in coords],
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RoutingUnavailable(f"Unexpected OpenRouteService response: {exc}") from exc


def fetch_road_route(
    origin: tuple[float, float],
    destination: tuple[float, float],
    api_key: str,
    client: httpx.Client | None = None,
) -> RoadRoute:
    """Call ORS for a truck route between two (lat, lon) points."""
    body = {
        "coordinates": [[origin[1], origin[0]], [destination[1], destination[0]]],
        "instructions": True,
        "language": "en",
    }
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    own_client = client is None
    client = client or httpx.Client(timeout=TIMEOUT_S)
    try:
        resp = client.post(ORS_URL, json=body, headers=headers)
        if resp.status_code != 200:
            raise RoutingUnavailable(f"OpenRouteService HTTP {resp.status_code}: {resp.text[:200]}")
        return parse_ors_geojson(resp.json())
    except httpx.HTTPError as exc:
        raise RoutingUnavailable(f"OpenRouteService unreachable: {exc}") from exc
    finally:
        if own_client:
            client.close()


@lru_cache(maxsize=64)
def _cached_route(origin: str, destination: str, api_key: str) -> RoadRoute:
    rates = get_rate_card()
    o = rates.origins[origin]
    d = rates.destinations[destination]
    return fetch_road_route((o["lat"], o["lon"]), (d["lat"], d["lon"]), api_key)


def road_route(origin: str, destination: str, rates: RateCard | None = None) -> RoadRoute | None:
    """Truck route from a Port Klang terminal to a destination, or ``None`` if unavailable.

    Only for road destinations (East Malaysia goes by sea). Results are cached per pair.
    """
    key = get_settings().ors_api_key
    rates = rates or get_rate_card()
    dest_key = rates.resolve_destination(destination)
    if not key or "truck" not in rates.destinations[dest_key]["modes"]:
        return None
    try:
        return _cached_route(origin, dest_key, key)
    except RoutingUnavailable as exc:
        logger.warning("Road routing unavailable (%s); using rate-card distance.", exc)
        return None


def with_road_route(req, rates: RateCard | None = None):
    """Return ``(request, route)``: the request carries the real road distance when available."""
    if req.road_distance_km:
        return req, None
    route = road_route(req.origin, req.destination, rates)
    if route is None:
        return req, None
    return (
        req.model_copy(
            update={"road_distance_km": route.distance_km, "road_drive_hours": route.drive_hours}
        ),
        route,
    )


def main_roads(route: RoadRoute, limit: int = 5) -> list[str]:
    """The named roads that carry most of the distance, in driving order."""
    totals: dict[str, float] = {}
    for step in route.steps:
        if step.road:
            totals[step.road] = totals.get(step.road, 0) + step.distance_km
    major = {r for r, km in sorted(totals.items(), key=lambda kv: -kv[1])[:limit]}
    ordered: list[str] = []
    for step in route.steps:
        if step.road in major and step.road not in ordered:
            ordered.append(step.road)
    return ordered
