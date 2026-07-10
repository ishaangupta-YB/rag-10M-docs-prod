"""Pydantic Settings configuration for rag-zero."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings, loaded from environment and .env files.

    Variables are prefixed with ``RAG_`` by default.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=Path(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_base_url: str = Field(default="http://vllm:8000/v1")
    llm_model: str = Field(default="Qwen3-32B")
    llm_api_key: str | None = Field(default=None)
    llm_timeout: float = Field(default=120.0)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_max_tokens: int = Field(default=1024)

    # Embeddings
    embedder_base_url: str = Field(default="http://vllm:8000/v1")
    embedder_model: str = Field(default="BAAI/bge-large-en-v1.5")
    embedder_dim: int = Field(default=1024)
    embedder_max_retries: int = Field(default=3, ge=0)
    embedder_query_instruction: str = Field(
        default="Represent this sentence for searching relevant passages: "
    )

    # Reranker
    reranker_base_url: str = Field(default="http://vllm:8000/v1")
    reranker_model: str = Field(default="Qwen3-32B")

    # Verifier (optional; falls back to LLM judge when unset)
    verifier_base_url: str | None = Field(default=None)
    verifier_model: str | None = Field(default=None)

    # Pipeline knobs
    chunk_tokens: int = Field(default=256, ge=32)
    chunk_overlap: int = Field(default=32, ge=0)
    retrieve_k: int = Field(default=20, ge=1)
    rerank_top_n: int = Field(default=20, ge=1)
    fusion_kk: int = Field(default=60, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    max_hops: int = Field(default=3, ge=0)
    crag_ok: float = Field(default=0.7, ge=0.0, le=1.0)
    crag_bad: float = Field(default=0.4, ge=0.0, le=1.0)
    tau_claim: float = Field(default=0.7, ge=0.0, le=1.0)
    tau_abstain: float = Field(default=0.5, ge=0.0, le=1.0)
    seed: int = Field(default=42)

    # Paths
    artifact_dir: Path = Field(default=Path("./artifacts"))
    data_dir: Path = Field(default=Path("./data"))
    checkpoint_dir: Path = Field(default=Path("./checkpoints"))
    lancedb_uri: Path = Field(default=Path("./artifacts/lancedb"))
    bm25_path: Path = Field(default=Path("./artifacts/bm25s"))

    # Serving
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    log_level: str = Field(default="INFO")

    # Observability
    otel_endpoint: str | None = Field(default=None)
    metrics_port: int = Field(default=9090, ge=1, le=65535)
    enable_tracing: bool = Field(default=False)

    # Optional memory layer (mem0)
    enable_memory: bool = Field(default=False)
    mem0_api_key: str | None = Field(default=None)
    mem0_base_url: str | None = Field(default=None)
    mem0_user_id: str | None = Field(default=None)
    memory_top_k: int = Field(default=3, ge=0, le=20)

    @field_validator("artifact_dir", "data_dir", "checkpoint_dir", "lancedb_uri", "bm25_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: object) -> Path:
        return Path(value)  # type: ignore[arg-type]

    @field_validator("llm_base_url", "embedder_base_url", "reranker_base_url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid URL: {value}")
        return value

    @model_validator(mode="after")
    def _validate_thresholds(self) -> Settings:
        if self.crag_bad >= self.crag_ok:
            raise ValueError("crag_bad must be strictly less than crag_ok")
        if self.tau_claim > self.crag_ok:
            raise ValueError("tau_claim should not exceed crag_ok")
        return self

    @property
    def lance_table(self) -> str:
        return "passages"

    @property
    def normalized_artifact_dir(self) -> Path:
        return self.artifact_dir.expanduser().resolve()

    @property
    def normalized_data_dir(self) -> Path:
        return self.data_dir.expanduser().resolve()

    @property
    def normalized_checkpoint_dir(self) -> Path:
        return self.checkpoint_dir.expanduser().resolve()

    @property
    def normalized_lancedb_uri(self) -> Path:
        return self.lancedb_uri.expanduser().resolve()

    @property
    def normalized_bm25_path(self) -> Path:
        return self.bm25_path.expanduser().resolve()

    def ensure_dirs(self) -> None:
        """Create all configured artifact directories idempotently."""
        for path in (
            self.normalized_artifact_dir,
            self.normalized_data_dir,
            self.normalized_checkpoint_dir,
            self.normalized_lancedb_uri,
            self.normalized_bm25_path,
        ):
            path.mkdir(parents=True, exist_ok=True)
