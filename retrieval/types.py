"""Единый формат выхода ретривера — контракт для участников 3 и 4."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class Status(str, Enum):
    SUFFICIENT = "sufficient"      # данных достаточно, можно отвечать
    PARTIAL = "partial"            # данные есть, но неполные/слабые: отвечать с оговоркой
    INSUFFICIENT = "insufficient"  # в базе нет подходящего: обоснованный отказ


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str                       # дословный текст чанка (для цитат)
    score: float                    # итоговая оценка 0..1 (rerank, если был; иначе cosine)
    vector_score: float             # cosine similarity (1 - distance)
    rerank_score: Optional[float]
    cite: str                       # готовая ссылка: «Название, ст. X, п. Y»
    doc_id: str
    doc_title: str
    doc_type: str
    doc_number: Optional[str]
    doc_date: Optional[str]
    revision_date: Optional[str]
    article: Optional[str]
    clause_from: Optional[str]
    is_outdated: bool
    parent_doc_id: Optional[str]
    future_changes: list = field(default_factory=list)  # изменения, которые вступят в силу позже

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    query: str
    status: Status
    reason: str                                   # человекочитаемое обоснование статуса
    chunks: list[RetrievedChunk]                  # актуальные чанки выше порога, по убыванию score
    outdated_chunks: list[RetrievedChunk] = field(default_factory=list)  # найдены, но документ утратил силу
    reranked: bool = False
    best_score: float = 0.0
    debug_top: list[dict] = field(default_factory=list)  # топ сырых кандидатов (для отладки порогов)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d
