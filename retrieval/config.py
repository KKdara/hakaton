"""Настройки только через переменные окружения. Ключей в коде нет."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


class ConfigError(RuntimeError):
    pass


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Thresholds:
    high: float   # лучший чанк >= high  -> SUFFICIENT
    low: float    # лучший чанк >= low   -> PARTIAL; ниже -> INSUFFICIENT


@dataclass
class Config:
    # --- хранилище (как в build_vector_store.py) ---
    chroma_dir: str = field(default_factory=lambda: os.environ.get("CHROMA_DIR", "data/chroma_db"))
    collection: str = field(default_factory=lambda: os.environ.get("CHROMA_COLLECTION", "legal_docs"))

    # --- эмбеддинги (как в build_embeddings.py) ---
    base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "")
    api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    emb_model: str = field(default_factory=lambda: os.environ.get("EMB_MODEL", "BAAI/bge-m3"))

    # --- поиск ---
    top_k: int = field(default_factory=lambda: _i("RETRIEVAL_TOP_K", 6))               # сколько чанков отдаём дальше
    n_candidates: int = field(default_factory=lambda: _i("RETRIEVAL_CANDIDATES", 30))  # сколько берём из Chroma
    exclude_outdated: bool = field(default_factory=lambda: _b("RETRIEVAL_EXCLUDE_OUTDATED", True))

    # --- реранкер (опционально, по умолчанию выключен) ---
    rerank_enabled: bool = field(default_factory=lambda: _b("RERANK_ENABLED", False))
    rerank_model: str = field(default_factory=lambda: os.environ.get("RERANK_MODEL", "Qwen/Qwen3-Reranker-8B"))
    rerank_base_url: str = field(default_factory=lambda: os.environ.get("RERANK_BASE_URL", ""))  # пусто -> base_url
    rerank_api_key: str = field(default_factory=lambda: os.environ.get("RERANK_API_KEY", ""))    # пусто -> api_key
    rerank_candidates: int = field(default_factory=lambda: _i("RERANK_CANDIDATES", 20))
    rerank_timeout: float = field(default_factory=lambda: _f("RERANK_TIMEOUT", 60))

    # --- пороги релевантности (ТРЕБУЮТ КАЛИБРОВКИ на реальных вопросах: python -m retrieval.cli) ---
    vector_thresholds: Thresholds = field(default_factory=lambda: Thresholds(
        high=_f("VEC_HIGH", 0.45), low=_f("VEC_LOW",  0.30)))
    rerank_thresholds: Thresholds = field(default_factory=lambda: Thresholds(
        high=_f("RERANK_HIGH", 0.70), low=_f("RERANK_LOW", 0.30)))

    def require_embedding_env(self) -> None:
        if not self.base_url or not self.api_key:
            raise ConfigError("Не заданы LLM_BASE_URL / LLM_API_KEY (или OPENAI_*): нельзя векторизовать запрос.")
