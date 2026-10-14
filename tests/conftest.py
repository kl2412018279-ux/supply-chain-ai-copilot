"""Test configuration: everything runs offline and in-process.

Environment is set *before* ``src`` is imported because settings are cached.
Integration tests use TEST_DATABASE_URL / TEST_REDIS_URL so unit tests never touch
real services even when DATABASE_URL is set in the shell.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="copilot-tests-"))
os.environ.update(
    {
        "LLM_PROVIDER": "none",
        "DATABASE_URL": "",
        "REDIS_URL": "",
        "EMBEDDING_BACKEND": "hashing",
        "RERANKER_BACKEND": "lexical",
        "MODEL_PATH": str(_TMP / "demand_model.joblib"),
        "LANGSMITH_API_KEY": "",
        "MLFLOW_TRACKING_URI": "",
        "AUTO_TRAIN": "false",
        "AUTO_INGEST": "false",
    }
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = ROOT / "data" / "raw" / "esg_policy.pdf"


@pytest.fixture(scope="session")
def sales_df():
    from src.forecasting.data import generate_sample_data

    return generate_sample_data(skus=("SKU-101", "SKU-102", "SKU-103"), periods=300)


@pytest.fixture(scope="session")
def trained_model(sales_df):
    from src.config import get_settings
    from src.forecasting.train import train

    return train(sales_df, model_path=get_settings().model_path, holdout_days=28)


@pytest.fixture(scope="session")
def ingested_store():
    from src.rag.ingest_pdf import ingest_file
    from src.rag.vector_store import get_vector_store

    store = get_vector_store()
    ingest_file(SAMPLE_PDF, store)
    return store


@pytest.fixture
def embedder():
    from src.rag.embeddings import HashingEmbedder

    return HashingEmbedder(384)
