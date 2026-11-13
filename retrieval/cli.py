"""Ручная проверка и калибровка порогов:
    python -m retrieval.cli "Какой срок испытательного срока по ТК?" [--rerank] [-k 6]
Прогоните 15-20 вопросов (и 5 внепрофильных) и посмотрите на разброс cosine/rerank,
затем задайте VEC_HIGH/VEC_LOW (RERANK_HIGH/RERANK_LOW)."""
import argparse
import json

from .config import Config
from .retriever import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("-k", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    cfg = Config()
    if a.k:
        cfg.top_k = a.k
    res = Retriever(cfg).retrieve(a.query, use_reranker=a.rerank or None)

    if a.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return
    print(f"СТАТУС: {res.status.value}  | best={res.best_score:.3f} | reranked={res.reranked}")
    print(f"ПРИЧИНА: {res.reason}\n")
    for i, c in enumerate(res.chunks, 1):
        rr = f" rerank={c.rerank_score:.3f}" if c.rerank_score is not None else ""
        print(f"{i}. [{c.score:.3f}] cos={c.vector_score:.3f}{rr}  {c.cite}")
        print(f"   {c.text[:160].replace(chr(10), ' ')}...")
    print("\nТоп сырых кандидатов:", json.dumps(res.debug_top, ensure_ascii=False))


if __name__ == "__main__":
    main()
