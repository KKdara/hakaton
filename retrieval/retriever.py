"""Основной модуль поиска: запрос -> эмбеддинг -> Chroma -> (реранкер) -> оценка достаточности."""
from __future__ import annotations

import logging
from typing import Callable, Optional

from .config import Config
from .embedder import Embedder
from .relevance import assess, thresholds_for
from .reranker import Reranker, RerankError
from .types import RetrievalResult, RetrievedChunk, Status

log = logging.getLogger("retrieval")

_NULLS = (None, "", "None", "none", "null")


def _clean(v) -> Optional[str]:
    """build_vector_store.py делает str(None) -> 'None'. Возвращаем настоящий None."""
    return None if v in _NULLS else str(v)


def _build_where(filters: Optional[dict]) -> Optional[dict]:
    if not filters:
        return None
    items = [{k: v} for k, v in filters.items()]
    return items[0] if len(items) == 1 else {"$and": items}


def _make_cite(meta: dict) -> str:
    """Запасная сборка ссылки (cite в Chroma не сохраняется). Формат как в chunking.make_cite."""
    if meta.get("cite"):
        return meta["cite"]
    title = meta.get("doc_title") or meta.get("doc_id") or "Документ"
    if len(title) > 90:
        title = title[:87].rstrip() + "…"
    parts = [title]
    if _clean(meta.get("article")):
        parts.append(f"ст. {meta['article']}")
    if _clean(meta.get("clause_from")):
        parts.append(f"п. {meta['clause_from']}")
    return ", ".join(parts)


class Retriever:
    def __init__(self, cfg: Optional[Config] = None,
                 embed_fn: Optional[Callable[[str], list[float]]] = None,
                 collection=None,
                 reranker: Optional[Reranker] = None):
        self.cfg = cfg or Config()
        self._embed = embed_fn or Embedder(self.cfg)
        self._collection = collection
        self._reranker = reranker

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=self.cfg.chroma_dir)
            self._collection = client.get_collection(self.cfg.collection)
        return self._collection

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self.cfg)
        return self._reranker

    # ------------------------------------------------------------------
    def _search(self, query: str, filters: Optional[dict]) -> list[RetrievedChunk]:
        vec = self._embed(query)
        res = self.collection.query(
            query_embeddings=[vec],
            n_results=self.cfg.n_candidates,
            where=_build_where(filters),
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            sim = max(0.0, min(1.0, 1.0 - float(dist)))  # cosine distance -> similarity
            fc = _clean(meta.get("future_changes"))
            out.append(RetrievedChunk(
                chunk_id=cid, text=doc, score=sim, vector_score=sim, rerank_score=None,
                cite=_make_cite(meta),
                doc_id=meta.get("doc_id", ""), doc_title=meta.get("doc_title", ""),
                doc_type=meta.get("doc_type", ""),
                doc_number=_clean(meta.get("doc_number")), doc_date=_clean(meta.get("doc_date")),
                revision_date=_clean(meta.get("revision_date")),
                article=_clean(meta.get("article")), clause_from=_clean(meta.get("clause_from")),
                is_outdated=bool(meta.get("is_outdated", False)),
                parent_doc_id=_clean(meta.get("parent_doc_id")),
                future_changes=[x.strip() for x in fc.split(",")] if fc else [],
            ))
        out.sort(key=lambda c: c.vector_score, reverse=True)
        return out

    def _rerank(self, query: str, chunks: list[RetrievedChunk]) -> bool:
        """Переранжирует на месте. True, если реранкер отработал."""
        pool = chunks[: self.cfg.rerank_candidates]
        docs = [f"{c.cite}\n{c.text}" for c in pool]
        try:
            scores = self.reranker.score(query, docs)
        except RerankError as e:
            log.warning("Реранкер недоступен, остаёмся на векторном поиске: %s", e)
            return False
        for c, s in zip(pool, scores):
            c.rerank_score = s
            c.score = s
        pool.sort(key=lambda c: c.score, reverse=True)
        chunks[: len(pool)] = pool
        return True

    # ------------------------------------------------------------------
    def retrieve(self, query: str, filters: Optional[dict] = None,
                 use_reranker: Optional[bool] = None) -> RetrievalResult:
        query = (query or "").strip()
        if not query:
            return RetrievalResult(query=query, status=Status.INSUFFICIENT,
                                   reason="Пустой запрос.", chunks=[])

        cands = self._search(query, filters)
        debug = [{"chunk_id": c.chunk_id, "cite": c.cite, "cosine": round(c.vector_score, 3),
                  "outdated": c.is_outdated} for c in cands[:5]]

        outdated = [c for c in cands if c.is_outdated]
        active = [c for c in cands if not c.is_outdated] if self.cfg.exclude_outdated else list(cands)

        want_rerank = self.cfg.rerank_enabled if use_reranker is None else use_reranker
        reranked = bool(want_rerank and active and self._rerank(query, active))
        if reranked:
            debug = [{"chunk_id": c.chunk_id, "cite": c.cite, "cosine": round(c.vector_score, 3),
                      "rerank": round(c.rerank_score or 0, 3)} for c in active[:5]]

        scores = [c.score for c in active[: self.cfg.top_k]]
        status, reason = assess(scores, self.cfg, reranked)

        low = thresholds_for(self.cfg, reranked).low
        kept = [c for c in active[: self.cfg.top_k] if c.score >= low]

        if outdated and self.cfg.exclude_outdated:
            reason += f" Найдено фрагментов из утративших силу документов: {len(outdated)} (не использованы)."
        return RetrievalResult(
            query=query, status=status, reason=reason, chunks=kept,
            outdated_chunks=outdated[:3] if self.cfg.exclude_outdated else [],
            reranked=reranked, best_score=scores[0] if scores else 0.0, debug_top=debug,
        )


_default: Optional[Retriever] = None


def retrieve(query: str, filters: Optional[dict] = None,
             use_reranker: Optional[bool] = None) -> RetrievalResult:
    """Главная точка входа для остальных участников."""
    global _default
    if _default is None:
        _default = Retriever()
    return _default.retrieve(query, filters=filters, use_reranker=use_reranker)
