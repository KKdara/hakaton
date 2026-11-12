"""Опциональный реранкер Qwen3-Reranker-8B через HTTP (формат /rerank, как у Cohere/Jina/vLLM).

Ожидаемый запрос:  POST {base}/rerank  {"model", "query", "documents": [...], "top_n"}
Ожидаемый ответ:   {"results": [{"index": i, "relevance_score": x}, ...]}
Если у платформы другой формат, правится только метод _parse()."""
from __future__ import annotations

import httpx

from .config import Config


class RerankError(RuntimeError):
    pass


class Reranker:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _url(self) -> str:
        base = (self.cfg.rerank_base_url or self.cfg.base_url).rstrip("/")
        if not base:
            raise RerankError("Не задан RERANK_BASE_URL / LLM_BASE_URL")
        return base + "/rerank"

    @staticmethod
    def _parse(data: dict, n: int) -> list[float]:
        items = data.get("results") or data.get("data") or []
        scores: list[float | None] = [None] * n
        for it in items:
            idx = it.get("index")
            sc = it.get("relevance_score", it.get("score"))
            if idx is not None and sc is not None and 0 <= idx < n:
                scores[idx] = float(sc)
        if any(s is None for s in scores):
            raise RerankError("Ответ реранкера неполный или в неожиданном формате")
        return scores  # type: ignore[return-value]

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Возвращает оценки в том же порядке, что documents."""
        key = self.cfg.rerank_api_key or self.cfg.api_key
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        payload = {"model": self.cfg.rerank_model, "query": query,
                   "documents": documents, "top_n": len(documents)}
        try:
            r = httpx.post(self._url(), json=payload, headers=headers, timeout=self.cfg.rerank_timeout)
            r.raise_for_status()
            return self._parse(r.json(), len(documents))
        except httpx.HTTPError as e:
            raise RerankError(f"Ошибка запроса к реранкеру: {e}") from e
