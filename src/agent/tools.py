"""Tools the copilot agent can call.

Each tool is a plain, typed Python function with a Pydantic argument schema, so the
same function serves the LangGraph agent (via ``build_langchain_tools``), the
offline rule-based agent, the FastAPI endpoints and the unit tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from src.forecasting.esg_calculator import TransportMode, calculate_emissions
from src.forecasting.predict import MAX_HORIZON_DAYS, get_forecaster
from src.forecasting.sales_repository import get_sales_repository
from src.guardrails.schemas import Source, ToolCall
from src.rag.retriever import get_retriever


class ForecastArgs(BaseModel):
    sku: str = Field(description="Product SKU code, e.g. 'SKU-101'.")
    horizon_days: int = Field(
        default=14, ge=1, le=MAX_HORIZON_DAYS, description="Number of days to forecast."
    )


class EmissionArgs(BaseModel):
    distance_km: float = Field(gt=0, description="Shipping distance in kilometres.")
    weight_kg: float = Field(gt=0, description="Shipment weight in kilograms.")
    mode: TransportMode = Field(default=TransportMode.truck, description="truck, rail, sea or air.")


class DocumentSearchArgs(BaseModel):
    query: str = Field(description="What to look up in ESG / sustainability / policy documents.")
    k: int = Field(default=4, ge=1, le=10)


class SalesSummaryArgs(BaseModel):
    sku: str | None = Field(default=None, description="Optional SKU filter.")
    start_date: date | None = Field(default=None, description="Optional start date (YYYY-MM-DD).")
    end_date: date | None = Field(default=None, description="Optional end date (YYYY-MM-DD).")


def forecast_demand(sku: str, horizon_days: int = 14) -> dict[str, Any]:
    forecaster = get_forecaster()
    df = forecaster.forecast(sku, horizon_days)
    return {
        "sku": forecaster.resolve_sku(sku),
        "horizon_days": horizon_days,
        "start_date": df["date"].iloc[0],
        "end_date": df["date"].iloc[-1],
        "total_units": round(float(df["forecast"].sum()), 1),
        "avg_daily_units": round(float(df["forecast"].mean()), 1),
        "daily": df.to_dict(orient="records"),
        "model_validation_mae": round(forecaster.metrics.get("mae", float("nan")), 2),
    }


def calculate_shipping_emissions(
    distance_km: float, weight_kg: float, mode: str = "truck"
) -> dict[str, Any]:
    mode = mode.value if isinstance(mode, TransportMode) else mode
    return calculate_emissions(distance_km, weight_kg, mode).model_dump(mode="json")


def search_esg_documents(query: str, k: int = 4) -> list[dict[str, Any]]:
    results = get_retriever().search(query, k=k)
    return [
        Source(
            source=r.chunk.source, page=r.chunk.page, snippet=r.chunk.text, score=r.score
        ).model_dump()
        for r in results
    ]


def get_sales_summary(
    sku: str | None = None, start_date: date | None = None, end_date: date | None = None
) -> dict[str, Any]:
    return get_sales_repository().summary(sku, start_date, end_date).model_dump()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    args_schema: type[BaseModel]


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        "forecast_demand",
        "Forecast daily product demand (units) for a SKU over the next N days using the "
        "trained LightGBM model.",
        forecast_demand,
        ForecastArgs,
    ),
    ToolSpec(
        "calculate_shipping_emissions",
        "Calculate shipping carbon emissions (kg CO2e) for a shipment given distance (km), "
        "weight (kg) and transport mode; also compares all modes.",
        calculate_shipping_emissions,
        EmissionArgs,
    ),
    ToolSpec(
        "search_esg_documents",
        "Search company ESG, sustainability and supply chain policy documents (PDFs). Use for "
        "targets, rules, supplier requirements, compliance and policy questions.",
        search_esg_documents,
        DocumentSearchArgs,
    ),
    ToolSpec(
        "get_sales_summary",
        "Summarise historical sales (total, daily average, peak day) optionally filtered by "
        "SKU and date range.",
        get_sales_summary,
        SalesSummaryArgs,
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
        args = spec.args_schema(**kwargs).model_dump()
        try:
            output = spec.fn(**args)
        except (ValueError, KeyError, LookupError, RuntimeError) as exc:
            output = {"error": str(exc).strip("'\"")}
        self.calls.append(
            ToolCall(name=name, arguments=json.loads(json.dumps(args, default=str)), output=output)
        )
        if name == "search_esg_documents" and isinstance(output, list):
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
