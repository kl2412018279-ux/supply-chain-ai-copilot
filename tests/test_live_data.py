"""Road routing (OpenRouteService) and weather (Open-Meteo) with mocked HTTP."""

from __future__ import annotations

import datetime as dt
import json

import httpx
import pytest

from src.costing import road_routing
from src.costing.quote import Mode, QuoteRequest, build_quote
from src.costing.rates import get_rate_card
from src.costing.road_routing import (
    RoutingUnavailable,
    fetch_road_route,
    main_roads,
    parse_ors_geojson,
    with_road_route,
)
from src.costing.routing import transit_days
from src.delays import weather
from src.delays.weather import fetch_rain, get_rain, typical_rain

ORS_SAMPLE = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "summary": {"distance": 372_400.0, "duration": 16_920.0},
                "segments": [
                    {
                        "steps": [
                            {"instruction": "Head east", "name": "Jalan Pelabuhan Utara",
                             "distance": 2_100.0, "duration": 180.0},
                            {"instruction": "Merge onto E1", "name": "North-South Expressway",
                             "distance": 330_000.0, "duration": 14_400.0},
                            {"instruction": "Keep left", "name": "-",
                             "distance": 400.0, "duration": 60.0},
                            {"instruction": "Turn right", "name": "Federal Route 1",
                             "distance": 39_900.0, "duration": 2_280.0},
                        ]
                    }
                ],
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[101.32, 3.0], [101.5, 3.8], [100.37, 5.39]],
            },
        }
    ],
}  # fmt: skip


def test_parse_ors_geojson():
    route = parse_ors_geojson(ORS_SAMPLE)
    assert route.distance_km == pytest.approx(372.4)
    assert route.drive_hours == pytest.approx(4.7)
    assert route.geometry[0] == (3.0, 101.32)  # converted to (lat, lon)
    assert route.steps[2].road == ""  # ORS uses "-" for unnamed roads
    assert main_roads(route, limit=2) == ["North-South Expressway", "Federal Route 1"]


def test_parse_rejects_bad_payload():
    with pytest.raises(RoutingUnavailable):
        parse_ors_geojson({"features": []})


def test_fetch_road_route_sends_lon_lat_and_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ORS_SAMPLE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    route = fetch_road_route((3.0, 101.3), (5.4, 100.4), "k123", client=client)
    assert seen["auth"] == "k123"
    assert seen["body"]["coordinates"] == [[101.3, 3.0], [100.4, 5.4]]
    assert route.distance_km == pytest.approx(372.4)


def test_fetch_road_route_http_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(403)))
    with pytest.raises(RoutingUnavailable):
        fetch_road_route((3.0, 101.3), (5.4, 100.4), "bad", client=client)


def test_no_key_keeps_rate_card_distance():
    req = QuoteRequest(destination="penang-prai")
    same, route = with_road_route(req)
    assert route is None and same.road_distance_km is None
    assert build_quote(same).distance_source == "rate-card estimate"


def test_road_route_feeds_quote_and_transit(monkeypatch):
    rates = get_rate_card()
    route = parse_ors_geojson(ORS_SAMPLE)
    monkeypatch.setattr(road_routing, "road_route", lambda *a, **k: route)
    req, got = with_road_route(QuoteRequest(destination="penang-prai", mode=Mode.truck))
    assert got is route and req.road_distance_km == pytest.approx(372.4)

    q = build_quote(req, rates)
    assert q.distance_km == pytest.approx(372.4)
    assert q.distance_source.startswith("road route")

    fleet = rates.data["carriers"]["muara-fleet"]
    slow = transit_days(rates, "penang-prai", Mode.truck, fleet, road_drive_hours=30)
    fast = transit_days(rates, "penang-prai", Mode.truck, fleet, road_drive_hours=2)
    assert slow > fast >= 1


def test_road_distance_ignored_for_sea():
    req = QuoteRequest(destination="kuching", road_distance_km=50, road_drive_hours=1)
    assert build_quote(req).distance_source == "rate-card estimate"


def _meteo(day: dt.date, mm: float | None, calls: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.url.host}:{request.url.params['start_date']}")
        return httpx.Response(
            200, json={"daily": {"time": [day.isoformat()], "precipitation_sum": [mm]}}
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


TODAY = dt.date(2026, 10, 14)


@pytest.mark.parametrize(
    ("eta", "host", "source", "day"),
    [
        (TODAY + dt.timedelta(days=3), "api.open-meteo.com", "forecast", None),
        (dt.date(2026, 3, 1), "archive-api.open-meteo.com", "observed", None),
        (dt.date(2027, 2, 1), "archive-api.open-meteo.com", "in 2026", dt.date(2026, 2, 1)),
    ],
)
def test_fetch_rain_picks_the_right_source(eta, host, source, day):
    calls: list[str] = []
    day = day or eta
    reading = fetch_rain(3.1, 101.4, eta, today=TODAY, client=_meteo(day, 12.5, calls))
    assert reading.rain_mm == 12.5
    assert source in reading.source
    assert calls == [f"{host}:{day.isoformat()}"]


def test_fetch_rain_missing_value_raises():
    with pytest.raises(ValueError):
        fetch_rain(3.1, 101.4, TODAY, today=TODAY, client=_meteo(TODAY, None, []))


def test_get_rain_falls_back_when_disabled():
    reading = get_rain("kuching", dt.date(2026, 12, 10))
    assert reading.rain_mm == typical_rain("kuching", dt.date(2026, 12, 10))
    assert "offline" in reading.source


def test_get_rain_falls_back_on_error(monkeypatch):
    import dataclasses

    from src.config import get_settings

    enabled = dataclasses.replace(get_settings(), weather_enabled=True)
    monkeypatch.setattr(weather, "get_settings", lambda: enabled)
    monkeypatch.setattr(weather, "_failed_at", None)

    def boom(*_a):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(weather, "_cached", boom)
    reading = get_rain("kuantan", dt.date(2026, 11, 20))
    assert reading.rain_mm == typical_rain("kuantan", dt.date(2026, 11, 20))
    assert weather._failed_at is not None  # later calls skip the network for a while
