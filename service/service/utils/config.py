import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # ======= API ========
    APP_NAME: str = "AI RAG Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    PORT: int = int(os.getenv("PORT", "8000"))

    # ====== JWT ======
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-secret-key-2026")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

    # ====== QDRANT (вектора) ======
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "knowledge")
    QDRANT_PREFER_GRPC : bool = os.getenv("QDRANT_PREFER_GRPC", "False").lower() == "true"
    # ====== MINIO (S3/файлы) ======
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")  # без http://
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "False").lower() == "true"

    # ====== REDIS (кэш) ======
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")

    # ====== DEEPSEEK (LLM) ======
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_TIMEOUT : int = os.getenv("DEEPSEEK_TIMEOUT", "30")

    # ====== EMBEDDING MODEL (локальный или API) ======
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sergeyzh/LaBSE-ru-sts")
    # Если используете OpenAI:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # ====== RAG ======
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "768"))  # LaBSE-ru-sts

    # ====== CORS ======
    ALLOWED_ORIGINS: list[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    LOCAL_CHUNKER_MODEL : str = "mistralai/Mistral-7B-Instruct-v0.2"

    @property
    def redis_url(self) -> str:
        """Redis URL для подключения"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def minio_endpoint_url(self) -> str:
        """MinIO endpoint с протоколом"""
        protocol = "https" if self.MINIO_SECURE else "http"
        return f"{protocol}://{self.MINIO_ENDPOINT}"


settings = Settings()
