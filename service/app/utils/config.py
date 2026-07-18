import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    #API
    APP_NAME : str = "AI RAG Service"
    APP_VERSION : str = "1.0.0"
    DEBUG : bool = os.getenv("DEBUG", "False").lower() == "true"

    #JWT
    JWT_SECRET_KEY : str = os.getenv("JWT_SECRET_KEY", "dev-secret-key-2026")
    JWT_ALGORITHM : str = "HS256"

    #Qdarnt
    QDRANT_HOST : str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_POST : str = os.getenv("QDRANT_POST", "6333")
    QDRANT_COLLECTION : str = os.getenv("QDRANT_COLLECTION", "knowledge")

    # DeepSeek
    DEEPSEEK_API_KEY : str = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    #Model
    EMBEDDING_MODEL : str = os.getenv("EMBEDDING_MODEL", "sergeyzh/LaBSE-ru-sts")
