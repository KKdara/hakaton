"""Тесты без сети: Chroma в памяти + детерминированный фейковый эмбеддер."""
import hashlib
import math
import re
import uuid

import chromadb
import pytest

from retrieval import Retriever, Status
from retrieval.config import Config, Thresholds
from retrieval.reranker import RerankError

DIM = 512


def fake_embed(text: str) -> list[float]:
    v = [0.0] * DIM
    for w in re.findall(r"\w+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


CHUNKS = [
    ("c1", "Испытательный срок при приеме на работу не может превышать трех месяцев",
     dict(doc_title="Трудовой кодекс", article="70", clause_from="", is_outdated=False)),
    ("c2", "Работник вправе расторгнуть трудовой договор предупредив работодателя за две недели",
     dict(doc_title="Трудовой кодекс", article="80", clause_from="1", is_outdated=False)),
    ("c3", "Испытательный срок составляет шесть месяцев старая редакция положения",
     dict(doc_title="Старое положение", article="", clause_from="", is_outdated=True)),
    ("c4", "Порядок согласования договоров поставки с юридическим департаментом",
     dict(doc_title="Регламент договоров", article="", clause_from="2.1", is_outdated=False,
          future_changes="01.01.2030, 01.07.2030")),
]


def make_collection():
    col = chromadb.EphemeralClient().create_collection(f"t{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
    metas = []
    for _, _, m in CHUNKS:
        m = dict(m)
        m.setdefault("doc_id", "d")
        m.update(doc_type="Кодекс", doc_number="None", doc_date="None", revision_date="None", parent_doc_id="None")
        metas.append(m)
    col.add(ids=[c[0] for c in CHUNKS], documents=[c[1] for c in CHUNKS],
            metadatas=metas, embeddings=[fake_embed(c[1]) for c in CHUNKS])
    return col


def make_cfg(**kw):
    cfg = Config(base_url="x", api_key="x", **kw)
    cfg.vector_thresholds = Thresholds(high=0.5, low=0.25)
    return cfg


class FakeReranker:
    def __init__(self, scores=None, fail=False):
        self.scores, self.fail = scores, fail

    def score(self, query, docs):
        if self.fail:
            raise RerankError("down")
        return self.scores or [0.9] + [0.1] * (len(docs) - 1)


def R(**kw):
    return Retriever(make_cfg(**kw.pop("cfg", {})), embed_fn=fake_embed, collection=make_collection(), **kw)


def test_sufficient_and_cite():
    res = R().retrieve("Какой испытательный срок при приеме на работу")
    assert res.status == Status.SUFFICIENT
    assert res.chunks[0].chunk_id == "c1"
    assert res.chunks[0].cite == "Трудовой кодекс, ст. 70"


def test_off_topic_is_insufficient():
    res = R().retrieve("рецепт борща со сметаной")
    assert res.status == Status.INSUFFICIENT
    assert res.chunks == []


def test_partial_zone():
    r = R()
    r.cfg.vector_thresholds = Thresholds(high=0.99, low=0.2)
    res = r.retrieve("испытательный срок")
    assert res.status == Status.PARTIAL and res.chunks


def test_outdated_excluded_but_reported():
    res = R().retrieve("испытательный срок составляет шесть месяцев старая редакция")
    assert all(not c.is_outdated for c in res.chunks)
    assert any(c.chunk_id == "c3" for c in res.outdated_chunks)


def test_none_strings_cleaned_and_future_changes_parsed():
    res = R().retrieve("порядок согласования договоров поставки юридическим департаментом")
    c = res.chunks[0]
    assert c.chunk_id == "c4" and c.doc_number is None and c.revision_date is None
    assert c.future_changes == ["01.01.2030", "01.07.2030"]
    assert c.cite == "Регламент договоров, п. 2.1"


def test_empty_query():
    assert R().retrieve("   ").status == Status.INSUFFICIENT


def test_reranker_reorders_and_uses_rerank_thresholds():
    r = R(reranker=FakeReranker(scores=[0.1, 0.95, 0.05]))
    res = r.retrieve("работник расторгнуть трудовой договор испытательный срок", use_reranker=True)
    assert res.reranked and res.best_score == 0.95 and res.status == Status.SUFFICIENT
    assert res.chunks[0].rerank_score == 0.95


def test_reranker_failure_falls_back_to_vector():
    r = R(reranker=FakeReranker(fail=True))
    res = r.retrieve("Какой испытательный срок при приеме на работу", use_reranker=True)
    assert not res.reranked and res.status == Status.SUFFICIENT


def test_filters():
    res = R().retrieve("испытательный срок", filters={"doc_title": "Регламент договоров", "doc_type": "Кодекс"})
    assert all(c.doc_title == "Регламент договоров" for c in res.chunks)
