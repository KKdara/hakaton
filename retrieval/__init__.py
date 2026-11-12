"""RAG-ретривер (участник №2): поиск по ChromaDB + опциональный реранкер + оценка достаточности."""
from .types import RetrievalResult, RetrievedChunk, Status
from .retriever import Retriever, retrieve
from .config import Config

__all__ = ["retrieve", "Retriever", "RetrievalResult", "RetrievedChunk", "Status", "Config"]
