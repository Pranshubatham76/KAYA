"""
SentinelSite — Central Configuration
All environment variables flow through here. Never import os.getenv anywhere else.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "SentinelSite"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "RS256"
    JWT_EXPIRY_HOURS: int = 24

    # ── Database (PostgreSQL) ────────────────────────────────────────────────
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "sentinelsite"
    DB_USER: str = "sentinel"
    DB_PASSWORD: str = "sentinel"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # ── Redis ────────────────────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ── Celery ───────────────────────────────────────────────────────────────
    @property
    def CELERY_BROKER_URL(self) -> str:
        return self.REDIS_URL

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return self.REDIS_URL

    # ── Qdrant (Vector DB) ───────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_PREFIX: str = "sentinel"  # {prefix}_{site_id}
    QDRANT_VECTOR_SIZE: int = 1536              # text-embedding-3-small
    QDRANT_ON_DISK: bool = True                  # persist vectors

    # ── S3 / Object Storage ──────────────────────────────────────────────────
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_AUDIO: str = "sentinelsite-audio"
    S3_BUCKET_FRAMES: str = "sentinelsite-frames"
    S3_BUCKET_MODELS: str = "sentinelsite-models"
    S3_BUCKET_ADMIN_IMAGES: str = "sentinelsite-admin"
    S3_PRESIGNED_URL_EXPIRY: int = 3600         # 1 hour for OTA model downloads
    USE_MINIO: bool = False                      # local dev override
    MINIO_ENDPOINT: str = "http://localhost:9000"

    # ── OpenAI ───────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_VISION_MODEL: str = "gpt-4o"
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"

    # ── Anthropic ────────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

    # ── LLM Provider ─────────────────────────────────────────────────────────
    LLM_PROVIDER: Literal["anthropic", "openai"] = "anthropic"
    LLM_MAX_TOKENS: int = 256          # voice answers must be short

    # ── RAG ──────────────────────────────────────────────────────────────────
    RAG_CHUNK_SIZE: int = 512          # tokens per chunk
    RAG_CHUNK_OVERLAP: int = 64
    RAG_TOP_K: int = 5                 # retrieve top-5 chunks
    RAG_RERANK_TOP_K: int = 3          # rerank → keep top-3
    RAG_SCORE_THRESHOLD: float = 0.35  # minimum similarity score

    # ── ML / Training ────────────────────────────────────────────────────────
    TRAINING_MIN_NEW_SAMPLES: int = 20          # FR-L02a
    TRAINING_MIN_INTERVAL_DAYS: int = 7         # FR-L02b
    TRAINING_MAX_GPU_UTILIZATION: float = 0.50  # FR-L02c
    TRAINING_REPLAY_RATIO_HISTORICAL: float = 0.70  # FR-L03
    TRAINING_CHECK_INTERVAL_HOURS: int = 6
    TFLITE_MODEL_DIR: str = "/tmp/sentinelsite/models"

    # ── Detection Thresholds ─────────────────────────────────────────────────
    FUSION_WINDOW_MS: int = 2000       # ±2s window for AND gate
    FUSION_COOLDOWN_S: int = 30        # prevent re-triggering
    ANOMALY_SCORE_THRESHOLD: float = 0.65  # θ₁ default (overridden per site)
    IMU_JERK_THRESHOLD: float = 3.0    # θ₂ default rad/s² (overridden per site)

    # ── CORS ─────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]


@lru_cache()
def get_settings() -> Settings:
    """Cached settings — import this everywhere instead of Settings()"""
    return Settings()


settings = get_settings()
