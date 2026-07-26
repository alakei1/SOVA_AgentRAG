from functools import lru_cache

from service.core.embedder import Embedder
from service.core.llm_client import LLMClient
from service.core.vector_store import VectorStore


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()

@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()

@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()
