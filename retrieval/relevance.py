"""Порог релевантности: когда найденных чанков достаточно / недостаточно."""
from __future__ import annotations

from .config import Config, Thresholds
from .types import Status


def thresholds_for(cfg: Config, reranked: bool) -> Thresholds:
    return cfg.rerank_thresholds if reranked else cfg.vector_thresholds


def assess(scores: list[float], cfg: Config, reranked: bool) -> tuple[Status, str]:
    """scores — оценки отсортированы по убыванию.
    Правило:
      лучший >= high            -> SUFFICIENT
      low <= лучший < high      -> PARTIAL (есть связанные нормы, но уверенности мало)
      лучший < low / пусто      -> INSUFFICIENT
    """
    th = thresholds_for(cfg, reranked)
    kind = "rerank" if reranked else "cosine"
    if not scores:
        return Status.INSUFFICIENT, "Поиск не вернул ни одного актуального фрагмента."
    best = scores[0]
    n_good = sum(1 for s in scores if s >= th.low)
    if best >= th.high:
        return Status.SUFFICIENT, f"Лучший фрагмент {kind}={best:.2f} >= {th.high:.2f}; релевантных: {n_good}."
    if best >= th.low:
        return Status.PARTIAL, (f"Лучший фрагмент {kind}={best:.2f} в зоне [{th.low:.2f}; {th.high:.2f}): "
                                f"связанные положения есть, но прямого ответа может не быть.")
    return Status.INSUFFICIENT, f"Лучший фрагмент {kind}={best:.2f} < {th.low:.2f}: в базе нет подходящих положений."
