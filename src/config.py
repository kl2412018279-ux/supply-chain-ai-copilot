"""Central configuration, read once from environment variables.

Every external dependency (Postgres, Redis, LLM API, embedding model) is optional:
when it is not configured the app falls back to an in-process implementation, so
the whole project runs offline for $0 and the test-suite needs no services.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    return default if not raw else raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq").lower())
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # OCR for scanned / photographed documents: auto | tesseract | vision | none.
    # "auto" uses the vision model when VISION_MODEL is set, else Tesseract when installed.
    ocr_engine: str = field(default_factory=lambda: _env("OCR_ENGINE", "auto").lower())
    vision_model: str = field(default_factory=lambda: _env("VISION_MODEL"))

    # External data (both optional; the app falls back to offline estimates)
    ors_api_key: str = field(default_factory=lambda: _env("ORS_API_KEY"))
    weather_enabled: bool = field(default_factory=lambda: _env_bool("WEATHER_ENABLED", True))

    # Storage
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL"))
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL"))

    # Retrieval
    embedding_backend: str = field(
        default_factory=lambda: _env("EMBEDDING_BACKEND", "sentence-transformers").lower()
    )
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 384))
    reranker_backend: str = field(
        default_factory=lambda: _env("RERANKER_BACKEND", "flashrank").lower()
    )
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 600))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 100))

    # Cache
    cache_enabled: bool = field(default_factory=lambda: _env_bool("CACHE_ENABLED", True))
    cache_similarity_threshold: float = field(
        default_factory=lambda: _env_float("CACHE_SIMILARITY_THRESHOLD", 0.92)
    )
    cache_ttl_seconds: int = field(default_factory=lambda: _env_int("CACHE_TTL_SECONDS", 86400))

    # Paths
    model_path: Path = field(
        default_factory=lambda: Path(
            _env("MODEL_PATH", str(PROJECT_ROOT / "models" / "volume_model.joblib"))
        )
    )
    delay_model_path: Path = field(
        default_factory=lambda: Path(
            _env("DELAY_MODEL_PATH", str(PROJECT_ROOT / "models" / "delay_model.joblib"))
        )
    )
    volumes_csv_path: Path = field(
        default_factory=lambda: Path(
            _env("VOLUMES_CSV_PATH", str(PROJECT_ROOT / "data" / "raw" / "container_volumes.csv"))
        )
    )
    rates_path: Path = field(
        default_factory=lambda: Path(
            _env("RATES_PATH", str(PROJECT_ROOT / "data" / "reference" / "rates.json"))
        )
    )
    documents_dir: Path = field(
        default_factory=lambda: Path(_env("DOCUMENTS_DIR", str(PROJECT_ROOT / "data" / "raw")))
    )

    # Startup behaviour (handy for one-command Docker demos)
    auto_train: bool = field(default_factory=lambda: _env_bool("AUTO_TRAIN", True))
    auto_ingest: bool = field(default_factory=lambda: _env_bool("AUTO_INGEST", True))

    # Admin sign-in (see src/auth/users.py); visitors use the public demo roles
    admin_username: str = field(default_factory=lambda: _env("ADMIN_USERNAME", "admin").lower())
    admin_password: str = field(default_factory=lambda: _env("ADMIN_PASSWORD"))

    # Observability
    langsmith_api_key: str = field(default_factory=lambda: _env("LANGSMITH_API_KEY"))
    langsmith_project: str = field(
        default_factory=lambda: _env("LANGSMITH_PROJECT", "supply-chain-ai-copilot")
    )
    mlflow_tracking_uri: str = field(default_factory=lambda: _env("MLFLOW_TRACKING_URI"))

    @property
    def llm_api_key(self) -> str:
        return {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "ollama": "local",
        }.get(self.llm_provider, "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
