"""Rainfall at the destination on the ETA date, from Open-Meteo (free, no API key).

* ETA within the next 16 days  -> Open-Meteo **forecast**
* ETA in the past              -> Open-Meteo **archive** (observed reanalysis)
* ETA further ahead            -> archive for the **same date last year** (seasonal proxy)
* offline / disabled / error   -> typical daily rain for that month and region

The synthetic training history uses the same monthly climatology, so live and
offline values are on the same scale. Open-Meteo's free tier is for non-commercial
use; a commercial deployment needs their paid plan or another weather provider.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from functools import lru_cache

import httpx
from pydantic import BaseModel

from src.config import get_settings

logger = logging.getLogger(__name__)
_RETRY_AFTER_S = 600  # after a failure, use the offline estimate for 10 minutes
_failed_at: float | None = None

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HORIZON_DAYS = 15
ARCHIVE_LAG_DAYS = 5
TIMEOUT_S = 8.0

# Approximate mean daily rainfall (mm) by month. West coast: inter-monsoon peaks in
# Apr and Oct-Nov; east coast and Borneo: wettest in the north-east monsoon (Nov-Feb).
# Rounded illustrative values for the synthetic data and offline fallback only.
CLIMATOLOGY_MM: dict[str, list[float]] = {
    "west": [6, 6, 8, 9, 7, 5, 5, 6, 7, 9, 10, 8],
    "east": [12, 6, 6, 5, 5, 5, 5, 6, 7, 9, 18, 20],
    "borneo": [18, 13, 10, 8, 7, 6, 6, 6, 8, 10, 12, 16],
}
REGIONS = {"kuantan": "east", "kuching": "borneo", "kota-kinabalu": "borneo"}


class WeatherReading(BaseModel):
    rain_mm: float
    source: str
    date: str


def region_of(destination: str) -> str:
    return REGIONS.get(destination, "west")


def typical_rain(destination: str, date: dt.date) -> float:
    return float(CLIMATOLOGY_MM[region_of(destination)][date.month - 1])


def _daily_precipitation(
    url: str, lat: float, lon: float, day: dt.date, client: httpx.Client
) -> float:
    resp = client.get(
        url,
        params={
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "daily": "precipitation_sum",
            "timezone": "Asia/Kuala_Lumpur",
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
        },
    )
    resp.raise_for_status()
    daily = resp.json()["daily"]
    idx = daily["time"].index(day.isoformat())
    value = daily["precipitation_sum"][idx]
    if value is None:
        raise ValueError("no precipitation value for that date")
    return float(value)


def fetch_rain(
    lat: float,
    lon: float,
    eta: dt.date,
    today: dt.date | None = None,
    client: httpx.Client | None = None,
) -> WeatherReading:
    """Rain (mm) for ``eta`` at a location, choosing forecast / archive / last-year proxy."""
    today = today or dt.date.today()
    own = client is None
    client = client or httpx.Client(timeout=TIMEOUT_S)
    try:
        if (
            today - dt.timedelta(days=ARCHIVE_LAG_DAYS)
            <= eta
            <= today + dt.timedelta(days=FORECAST_HORIZON_DAYS)
        ):
            mm = _daily_precipitation(FORECAST_URL, lat, lon, eta, client)
            return WeatherReading(rain_mm=mm, source="Open-Meteo forecast", date=eta.isoformat())
        if eta < today:
            mm = _daily_precipitation(ARCHIVE_URL, lat, lon, eta, client)
            return WeatherReading(
                rain_mm=mm, source="Open-Meteo archive (observed)", date=eta.isoformat()
            )
        # Most recent past year with the same calendar date (29 Feb -> 28 Feb).
        proxy = eta.replace(day=28) if (eta.month, eta.day) == (2, 29) else eta
        while proxy > today - dt.timedelta(days=ARCHIVE_LAG_DAYS):
            proxy = proxy.replace(year=proxy.year - 1)
        mm = _daily_precipitation(ARCHIVE_URL, lat, lon, proxy, client)
        return WeatherReading(
            rain_mm=mm,
            source=f"Open-Meteo archive, same date in {proxy.year}",
            date=proxy.isoformat(),
        )
    finally:
        if own:
            client.close()


@lru_cache(maxsize=256)
def _cached(lat: float, lon: float, eta: str, today: str) -> WeatherReading:
    return fetch_rain(lat, lon, dt.date.fromisoformat(eta), dt.date.fromisoformat(today))


def get_rain(destination: str, eta: dt.date) -> WeatherReading:
    """Best available rain value for the destination on ``eta`` (never raises)."""
    fallback = WeatherReading(
        rain_mm=typical_rain(destination, eta),
        source="typical for the month (offline estimate)",
        date=eta.isoformat(),
    )
    global _failed_at
    if not get_settings().weather_enabled:
        return fallback
    if _failed_at is not None and time.monotonic() - _failed_at < _RETRY_AFTER_S:
        return fallback
    from src.costing.rates import get_rate_card  # noqa: PLC0415

    dest = get_rate_card().destinations.get(destination)
    if not dest:
        return fallback
    try:
        return _cached(dest["lat"], dest["lon"], eta.isoformat(), dt.date.today().isoformat())
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
        logger.warning("Weather unavailable (%s); using typical rainfall.", exc)
        _failed_at = time.monotonic()
        return fallback
