"""Tools the copilot agent can call.

Each tool is a plain, typed Python function with a Pydantic argument schema, so the
same function serves the LangGraph agent (via ``build_langchain_tools``), the
offline rule-based agent, the FastAPI endpoints and the unit tests.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from src.costing.quote import QuoteRequest, build_quote, compare_modes
from src.costing.road_routing import main_roads, with_road_route
from src.costing.routing import Priorities, recommend
from src.delays.model import DelayInput, get_delay_model
from src.forecasting.esg_calculator import TransportMode, calculate_emissions
from src.forecasting.predict import MAX_HORIZON_DAYS, get_forecaster
from src.forecasting.volume_repository import get_volume_repository
from src.guardrails.schemas import Source, ToolCall
from src.rag.retriever import get_retriever


class ForecastArgs(BaseModel):
    commodity: str = Field(description="chemicals, palm-oil, plastics or general.")
    horizon_days: int = Field(
        default=14, ge=1, le=MAX_HORIZON_DAYS, description="Number of days to forecast."
    )


class SpendArgs(ForecastArgs):
    destination: str = Field(default="Shah Alam", description="Typical delivery point.")
    container_size: str = Field(default="20", description="'20' or '40' (feet).")


class RouteArgs(QuoteRequest):
    priority_cost: float = Field(default=0.4, ge=0, description="Weight for low price")
    priority_speed: float = Field(default=0.2, ge=0, description="Weight for short transit")
    priority_co2: float = Field(default=0.2, ge=0, description="Weight for low emissions")
    priority_reliability: float = Field(default=0.2, ge=0, description="Weight for on-time history")


class ScorecardArgs(BaseModel):
    haulier: str | None = Field(
        default=None, description="Optional carrier key or name; omit to rank all hauliers."
    )


class EmissionArgs(BaseModel):
    distance_km: float = Field(gt=0, description="Shipping distance in kilometres.")
    weight_kg: float = Field(gt=0, description="Shipment weight in kilograms.")
    mode: TransportMode = Field(default=TransportMode.truck, description="truck, rail, sea or air.")


class DocumentSearchArgs(BaseModel):
    query: str = Field(
        description="What to look up in the operations manual: customs, dangerous goods, "
        "palm oil, plastics, haulage, demurrage, warehousing, pricing or ESG rules."
    )
    k: int = Field(default=4, ge=1, le=10)


class VolumeSummaryArgs(BaseModel):
    commodity: str | None = Field(default=None, description="Optional commodity filter.")
    start_date: date | None = Field(default=None, description="Optional start date (YYYY-MM-DD).")
    end_date: date | None = Field(default=None, description="Optional end date (YYYY-MM-DD).")


def forecast_container_volume(commodity: str, horizon_days: int = 14) -> dict[str, Any]:
    forecaster = get_forecaster()
    df = forecaster.forecast(commodity, horizon_days)
    return {
        "commodity": forecaster.resolve_commodity(commodity),
        "horizon_days": horizon_days,
        "start_date": df["date"].iloc[0],
        "end_date": df["date"].iloc[-1],
        "total_teu": round(float(df["forecast"].sum()), 1),
        "avg_daily_teu": round(float(df["forecast"].mean()), 1),
        "peak_day": df.loc[df["forecast"].idxmax(), "date"],
        "daily": df.to_dict(orient="records"),
        "model_validation_mae_teu": round(forecaster.metrics.get("mae", float("nan")), 2),
    }


def quote_container_job(**kwargs: Any) -> dict[str, Any]:
    req, route = with_road_route(QuoteRequest(**kwargs))
    out = build_quote(req).model_dump(mode="json")
    if route is not None:
        out["road_route"] = {
            "distance_km": route.distance_km,
            "drive_hours": route.drive_hours,
            "main_roads": main_roads(route),
        }
    return out


def compare_transport_modes(**kwargs: Any) -> dict[str, Any]:
    req, _ = with_road_route(QuoteRequest(**kwargs))
    return compare_modes(req).model_dump(mode="json")


def estimate_haulage_spend(
    commodity: str,
    horizon_days: int = 30,
    destination: str = "Shah Alam",
    container_size: str = "20",
) -> dict[str, Any]:
    """Forecast TEU x per-container quote = projected haulage revenue/cost for the period."""
    volume = forecast_container_volume(commodity, horizon_days)
    teu_per_box = 2 if str(container_size).replace("ft", "") == "40" else 1
    containers = max(math.ceil(volume["total_teu"] / teu_per_box), 1)
    unit = build_quote(
        QuoteRequest(
            destination=destination,
            container_size=container_size,
            containers=1,
            cargo_type=volume["commodity"],
        )
    )
    return {
        "commodity": volume["commodity"],
        "period": f"{volume['start_date']} to {volume['end_date']}",
        "forecast_teu": volume["total_teu"],
        "containers": containers,
        "destination": unit.destination_label,
        "cost_per_container_rm": unit.cost_rm,
        "price_per_container_rm": unit.price_rm,
        "projected_cost_rm": round(unit.cost_rm * containers, 2),
        "projected_revenue_rm": round(unit.price_rm * containers, 2),
        "projected_diesel_litres": round(unit.diesel_litres * containers, 1),
        "projected_co2e_kg": round(unit.co2e_kg * containers, 1),
    }


def recommend_route_and_carrier(**kwargs: Any) -> dict[str, Any]:
    weights = {k: kwargs.pop(f"priority_{k}") for k in ("cost", "speed", "co2", "reliability")}
    rec = recommend(QuoteRequest(**kwargs), Priorities(**weights))
    out = rec.model_dump(mode="json")
    out["options"] = out["options"][:5]
    return out


def predict_delay_risk(**kwargs: Any) -> dict[str, Any]:
    return get_delay_model().predict(DelayInput(**kwargs)).model_dump(mode="json")


def get_haulier_scorecard(haulier: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    from src.hauliers.scorecard import build_scorecard  # noqa: PLC0415

    scores = build_scorecard()
    if haulier:
        needle = haulier.strip().lower()
        match = [
            s for s in scores if needle in (s.haulier, s.label.lower()) or needle in s.label.lower()
        ]
        if not match:
            raise KeyError(
                f"Unknown haulier '{haulier}'. Known: {', '.join(s.label for s in scores)}"
            )
        return match[0].model_dump(mode="json")
    return [
        s.model_dump(mode="json", include={"rank", "haulier", "label", "grade", "score", "summary"})
        for s in scores
    ]


def calculate_shipping_emissions(
    distance_km: float, weight_kg: float, mode: str = "truck"
) -> dict[str, Any]:
    mode = mode.value if isinstance(mode, TransportMode) else mode
    return calculate_emissions(distance_km, weight_kg, mode).model_dump(mode="json")


def search_operations_documents(query: str, k: int = 4) -> list[dict[str, Any]]:
    results = get_retriever().search(query, k=k)
    return [
        Source(
            source=r.chunk.source, page=r.chunk.page, snippet=r.chunk.text, score=r.score
        ).model_dump()
        for r in results
    ]


def get_volume_summary(
    commodity: str | None = None, start_date: date | None = None, end_date: date | None = None
) -> dict[str, Any]:
    return get_volume_repository().summary(commodity, start_date, end_date).model_dump()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    args_schema: type[BaseModel]


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        "quote_container_job",
        "Price a container job out of Port Klang in Malaysian ringgit (RM): itemised fuel, "
        "truck, driver, tolls, port handling, customs, cargo surcharges, warehousing and margin, "
        "plus diesel litres and CO2e.",
        quote_container_job,
        QuoteRequest,
    ),
    ToolSpec(
        "compare_transport_modes",
        "Quote every transport mode (truck, rail, sea) offered to a destination side by side, "
        "with RM price, CO2e, and the RM cost per tonne of CO2e avoided.",
        compare_transport_modes,
        QuoteRequest,
    ),
    ToolSpec(
        "recommend_route_and_carrier",
        "Recommend the best transport mode AND carrier for a job by balancing price (RM), "
        "transit days, CO2e and historical on-time rate with priority weights; explains the "
        "trade-off versus the fastest/cheapest alternative.",
        recommend_route_and_carrier,
        RouteArgs,
    ),
    ToolSpec(
        "predict_delay_risk",
        "Predict delay risk (Low/Medium/High) for a shipment from destination, carrier, "
        "commodity, ETA date, port congestion and customs inspection; returns probabilities, "
        "the main risk drivers and expected demurrage exposure in RM.",
        predict_delay_risk,
        DelayInput,
    ),
    ToolSpec(
        "get_haulier_scorecard",
        "Haulier / carrier performance and ESG scorecard: on-time rate, damage rate, claims, "
        "ESG evidence from their policy documents, composite grade (A-D) and ranking.",
        get_haulier_scorecard,
        ScorecardArgs,
    ),
    ToolSpec(
        "forecast_container_volume",
        "Forecast daily container volume (TEU) for a commodity over the next N days using the "
        "trained LightGBM model. Use for fleet and warehouse capacity planning.",
        forecast_container_volume,
        ForecastArgs,
    ),
    ToolSpec(
        "estimate_haulage_spend",
        "Project haulage cost, revenue, diesel and CO2e in RM for a commodity over the next N "
        "days by combining the volume forecast with the rate card.",
        estimate_haulage_spend,
        SpendArgs,
    ),
    ToolSpec(
        "get_volume_summary",
        "Summarise historical container volumes (total TEU, daily and weekly averages, peak "
        "day), optionally filtered by commodity and date range.",
        get_volume_summary,
        VolumeSummaryArgs,
    ),
    ToolSpec(
        "search_operations_documents",
        "Search the company operations manual: customs forms (K1/K2/K8), dangerous-goods "
        "chemicals, palm oil, plastics, haulage rules, demurrage/detention, warehousing, "
        "pricing policy and ESG commitments.",
        search_operations_documents,
        DocumentSearchArgs,
    ),
    ToolSpec(
        "calculate_shipping_emissions",
        "Generic CO2e estimate (kg) for any shipment from distance, weight and mode "
        "(truck, rail, sea, air) using tonne-km factors.",
        calculate_shipping_emissions,
        EmissionArgs,
    ),
]
TOOLS_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


@dataclass
class ToolRecorder:
    """Collects every tool call (and retrieved sources) made while answering one query."""

    calls: list[ToolCall] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)

    def run(self, name: str, **kwargs: Any) -> Any:
        spec = TOOLS_BY_NAME[name]
        args: dict[str, Any] = dict(kwargs)
        try:  # pydantic ValidationError is a ValueError: bad arguments come back as an error
            args = spec.args_schema(**kwargs).model_dump()
            output = spec.fn(**args)
        except (ValueError, KeyError, LookupError, RuntimeError) as exc:
            output = {"error": str(exc).strip("'\"")}
        self.calls.append(
            ToolCall(name=name, arguments=json.loads(json.dumps(args, default=str)), output=output)
        )
        if name == "search_operations_documents" and isinstance(output, list):
            self.sources.extend(Source(**s) for s in output)
        return output


def build_langchain_tools(recorder: ToolRecorder) -> list:
    """Wrap the tools as LangChain ``StructuredTool``s that report into ``recorder``."""
    from langchain_core.tools import StructuredTool  # noqa: PLC0415

    def make(spec: ToolSpec):
        def _call(**kwargs: Any) -> str:
            return json.dumps(recorder.run(spec.name, **kwargs), default=str)

        return StructuredTool.from_function(
            func=_call, name=spec.name, description=spec.description, args_schema=spec.args_schema
        )

    return [make(spec) for spec in TOOL_SPECS]
