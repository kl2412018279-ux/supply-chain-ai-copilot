"""FastAPI service: freight ops copilot, RM quotes, TEU forecasts and document search.

Run locally::

    uvicorn src.api.main:app --reload

Interactive docs at http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from src.agent.copilot import SupplyChainCopilot, get_copilot
from src.config import get_settings
from src.costing.quote import Quote, QuoteComparison, QuoteRequest, build_quote, compare_modes
from src.costing.rates import get_rate_card
from src.costing.road_routing import with_road_route
from src.costing.routing import Priorities, RouteRecommendation, recommend
from src.delays.model import (
    DelayInput,
    DelayPrediction,
    get_delay_model,
    upcoming_shipments,
)
from src.delays.model import ModelNotTrainedError as DelayModelNotTrainedError
from src.extraction.extract import ExtractionResult, ShippingDocument, extract
from src.extraction.store import get_document_store
from src.forecasting.esg_calculator import EmissionResult, TransportMode, calculate_emissions
from src.forecasting.predict import (
    MAX_HORIZON_DAYS,
    DemandForecaster,
    ModelNotTrainedError,
    get_forecaster,
)
from src.forecasting.volume_repository import VolumeSummary, get_volume_repository
from src.guardrails.schemas import CopilotResponse, Source
from src.hauliers.scorecard import HaulierScore, build_scorecard, monthly_on_time
from src.observability import metrics
from src.rag.ingest_pdf import SUPPORTED_SUFFIXES, ingest_bytes, ingest_directory
from src.rag.retriever import HybridRetriever, get_retriever
from src.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def bootstrap() -> None:
    """Train the model / ingest sample documents on first start (one-command demo)."""
    settings = get_settings()
    if (
        settings.auto_train
        and not settings.model_path.exists()
        and settings.volumes_csv_path.exists()
    ):
        from src.forecasting.data import load_volumes  # noqa: PLC0415
        from src.forecasting.train import train  # noqa: PLC0415

        logger.info("No model found - training on %s", settings.volumes_csv_path)
        train(load_volumes(settings.volumes_csv_path), settings.model_path)
    if settings.auto_train and not settings.delay_model_path.exists():
        from src.delays.model import train as train_delay_model  # noqa: PLC0415

        logger.info("No delay model found - training on the shipment history")
        train_delay_model()
    if settings.auto_ingest and get_vector_store().count() == 0 and settings.documents_dir.exists():
        logger.info("Vector store empty - ingesting %s", settings.documents_dir)
        ingest_directory(settings.documents_dir)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await run_in_threadpool(bootstrap)
    except Exception:
        logger.exception("Bootstrap failed; the API will start without the demo data.")
    yield


app = FastAPI(
    title="Freight Ops Copilot API",
    version="0.2.0",
    description="Port Klang 3PL copilot: RM job quotes with CO2e, container-volume forecasts, "
    "operations-manual search and an AI agent.",
    lifespan=lifespan,
)


# ------------------------------------------------------------------------- schemas


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    use_cache: bool = True


class ForecastRequest(BaseModel):
    commodity: str = Field(min_length=1, max_length=64)
    horizon_days: int = Field(default=14, ge=1, le=MAX_HORIZON_DAYS)


class ForecastPoint(BaseModel):
    date: str
    forecast: float


class ForecastResponse(BaseModel):
    commodity: str
    horizon_days: int
    total_teu: float
    forecast: list[ForecastPoint]
    model_metrics: dict[str, float]


class EmissionRequest(BaseModel):
    distance_km: float = Field(gt=0, le=50_000)
    weight_kg: float = Field(gt=0, le=10_000_000)
    mode: TransportMode = TransportMode.truck


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=4, ge=1, le=20)
    rerank: bool = True


class IngestResponse(BaseModel):
    source: str
    chunks: int


# ------------------------------------------------------------------- dependencies


def _invalidate_cache() -> None:
    """Cached answers may cite documents that just changed."""
    cache = get_copilot().cache
    if cache is not None:
        cache.clear()


def forecaster_dep() -> DemandForecaster:
    try:
        return get_forecaster()
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ---------------------------------------------------------------------- endpoints


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "model_trained": settings.model_path.exists(),
        "llm_provider": settings.llm_provider if settings.llm_api_key else "rule-based",
        "vector_store": type(get_vector_store()).__name__,
    }


@app.get("/metrics")
def get_metrics(copilot: SupplyChainCopilot = Depends(get_copilot)) -> dict[str, Any]:
    snapshot = metrics.snapshot()
    if copilot.cache is not None:
        snapshot["semantic_cache"] = copilot.cache.stats()
    return snapshot


@app.post("/chat", response_model=CopilotResponse)
def chat(req: ChatRequest, copilot: SupplyChainCopilot = Depends(get_copilot)) -> CopilotResponse:
    return copilot.ask(req.query, use_cache=req.use_cache)


@app.post("/forecast", response_model=ForecastResponse)
def forecast(
    req: ForecastRequest, forecaster: DemandForecaster = Depends(forecaster_dep)
) -> ForecastResponse:
    try:
        df = forecaster.forecast(req.commodity, req.horizon_days)
        commodity = forecaster.resolve_commodity(req.commodity)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\"")) from exc
    return ForecastResponse(
        commodity=commodity,
        horizon_days=req.horizon_days,
        total_teu=round(float(df["forecast"].sum()), 1),
        forecast=df.to_dict(orient="records"),
        model_metrics=forecaster.metrics,
    )


@app.get("/commodities")
def list_commodities(forecaster: DemandForecaster = Depends(forecaster_dep)) -> list[str]:
    return forecaster.commodities


@app.post("/quote", response_model=Quote)
def quote(req: QuoteRequest) -> Quote:
    """Itemised RM quote (fuel, truck, driver, tolls, port, customs, surcharges, margin) + CO2e."""
    try:
        return build_quote(with_road_route(req)[0])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\"")) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/quote/compare", response_model=QuoteComparison)
def quote_compare(req: QuoteRequest) -> QuoteComparison:
    """Every transport mode offered to the destination, with the RM per tonne CO2e trade-off."""
    try:
        return compare_modes(with_road_route(req)[0])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\"")) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class RouteRequest(BaseModel):
    job: QuoteRequest
    priorities: Priorities = Priorities()


@app.post("/routing/recommend", response_model=RouteRecommendation)
def route_recommend(req: RouteRequest) -> RouteRecommendation:
    """Rank every mode x carrier option by price, transit days, CO2e and on-time history."""
    try:
        return recommend(req.job, req.priorities)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\"")) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def delay_model_dep():
    try:
        return get_delay_model()
    except DelayModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/delays/predict", response_model=DelayPrediction)
def delay_predict(req: DelayInput, model=Depends(delay_model_dep)) -> DelayPrediction:
    """Low / Medium / High delay risk with drivers and demurrage exposure."""
    try:
        return model.predict(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\"")) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/delays/upcoming")
def delays_upcoming(
    days: int = Query(default=14, ge=1, le=60), model=Depends(delay_model_dep)
) -> list[dict[str, Any]]:
    """Synthetic manifest of upcoming arrivals, scored and sorted by risk."""
    scored = model.score_frame(upcoming_shipments(days=days))
    return scored.sort_values("risk_score", ascending=False).to_dict(orient="records")


@app.get("/hauliers/scorecard", response_model=list[HaulierScore])
def haulier_scorecard() -> list[HaulierScore]:
    return build_scorecard()


@app.get("/hauliers/{haulier}/trend")
def haulier_trend(haulier: str) -> list[dict[str, Any]]:
    trend = monthly_on_time(haulier)
    if trend.empty:
        raise HTTPException(status_code=404, detail=f"No deliveries for '{haulier}'.")
    return trend.to_dict(orient="records")


@app.post("/documents/extract", response_model=ExtractionResult)
async def documents_extract(file: UploadFile = File(...)) -> ExtractionResult:
    """Extract and validate fields from a Bill of Lading or commercial invoice PDF."""
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Upload a PDF.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB).")
    from src.agent.workflow import build_llm  # noqa: PLC0415

    try:
        return await run_in_threadpool(
            extract, data, Path(filename).name, build_llm(get_settings())
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Extraction failed for %s", filename)
        raise HTTPException(status_code=422, detail=f"Could not read {filename}.") from exc


class SaveDocumentRequest(BaseModel):
    file_name: str = Field(min_length=1, max_length=255)
    document: ShippingDocument


@app.post("/documents/records")
def documents_save(req: SaveDocumentRequest) -> dict[str, Any]:
    """Save a (reviewed) extraction to the database; re-saving replaces it."""
    from src.extraction.extract import validate  # noqa: PLC0415

    issues, _ = validate(req.document)
    try:
        return get_document_store().save(
            req.document, req.file_name, [i.model_dump() for i in issues]
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/documents/records")
def documents_records(limit: int = Query(default=50, ge=1, le=500)) -> list[dict[str, Any]]:
    return get_document_store().list(limit)


@app.get("/destinations")
def destinations() -> dict[str, dict[str, Any]]:
    return get_rate_card().destinations


@app.get("/rates")
def rates() -> dict[str, Any]:
    """The full rate card (example rates; edit data/reference/rates.json)."""
    return get_rate_card().data


@app.post("/emissions", response_model=EmissionResult)
def emissions(req: EmissionRequest) -> EmissionResult:
    return calculate_emissions(req.distance_km, req.weight_kg, req.mode.value)


@app.get("/volumes/summary", response_model=VolumeSummary)
def volume_summary(
    commodity: str | None = Query(default=None, max_length=64),
    start_date: date | None = None,
    end_date: date | None = None,
) -> VolumeSummary:
    try:
        return get_volume_repository().summary(commodity, start_date, end_date)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/documents/search", response_model=list[Source])
def search_documents(
    req: SearchRequest, retriever: HybridRetriever = Depends(get_retriever)
) -> list[Source]:
    results = retriever.search(req.query, k=req.k, rerank=req.rerank)
    return [
        Source(source=r.chunk.source, page=r.chunk.page, snippet=r.chunk.text, score=r.score)
        for r in results
    ]


@app.get("/documents")
def list_documents() -> dict[str, int]:
    return get_vector_store().sources()


@app.post("/documents/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """Upload a PDF / TXT / MD file. Used by the n8n Google Drive workflow."""
    filename = file.filename or "upload.pdf"
    if not any(filename.lower().endswith(s) for s in SUPPORTED_SUFFIXES):
        raise HTTPException(
            status_code=415, detail=f"Supported types: {sorted(SUPPORTED_SUFFIXES)}"
        )
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB).")
    try:
        chunks = await run_in_threadpool(ingest_bytes, data, filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # e.g. corrupt PDF
        logger.exception("Ingestion failed for %s", filename)
        raise HTTPException(status_code=422, detail=f"Could not read {filename}.") from exc
    _invalidate_cache()
    return IngestResponse(source=Path(filename).name, chunks=chunks)


@app.delete("/documents/{source}")
def delete_document(source: str) -> dict[str, int]:
    removed = get_vector_store().delete_source(source)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No document named '{source}'.")
    _invalidate_cache()
    return {"removed_chunks": removed}
