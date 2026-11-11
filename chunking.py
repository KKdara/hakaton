"""Нарезка документов на чанки с учётом юридической структуры."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from parser import Doc  # замени на kb_parse, если файл называется иначе


@dataclass
class ChunkParams:
    target_chars: int = 1000
    max_chars: int = 1600
    min_chars: int = 250
    overlap_chars: int = 150
    hard_break_level: int = 3

    # профили для разных типов
    law_target_chars: int = 1400
    law_max_chars: int = 2200
    letter_target_chars: int = 800
    letter_max_chars: int = 1400
    letter_min_chars: int = 200


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_file: str
    doc_type: str
    doc_number: Optional[str]
    doc_date: Optional[str]
    revision_date: Optional[str]
    future_changes: list
    is_outdated: bool
    parent_doc_id: Optional[str]       # НОВОЕ: для допсоглашений
    section_path: list
    article: str
    clause_from: str
    clause_to: str
    start: int
    end: int
    text: str
    n_chars: int
    cite: str = ""
    idx: int = 0

    def to_dict(self):
        return asdict(self)


_SENT = re.compile(r"(?<=[.;:!?»)])\s+(?=[А-ЯЁA-Z\d«\"(—–\-а-яё)\]?])|\n")
_RE_SUBITEM = re.compile(r"^\s*(?:[а-яё]\)|\d+\)|[-–—•])\s")
_RE_LIST_ITEM = re.compile(
    r"^\s*(?:на\s+время|с\s+лицами|для\s+|в\s+случаях|в\s+целях|при\s+условии|"
    r"если\s+|когда\s+|кроме\s+|в\s+том\s+числе)", re.I
)
_RE_SERVICE_LINE = re.compile(r"^(источник:|url:|дата\s+выгрузки:|=====|page\s+\d)", re.I)


def _is_law_type(doc_type: str) -> bool:
    return doc_type in ("Федеральный закон", "Кодекс", "Закон")

def _is_unstructured_type(doc_type: str) -> bool:
    return doc_type in (
        "Письмо комплаенс", "Письмо", "Служебная записка",
        "Докладная записка", "Пояснительная записка",
        "Уведомление", "Разъяснение", "Меморандум", "Заключение",
        "Ответ на запрос", "Неизвестно",
    )


def _split_span(text: str, start: int, end: int, p: ChunkParams,
                target: Optional[int] = None, max_: Optional[int] = None
                ) -> list[tuple[int, int]]:
    target = target or p.target_chars
    max_ = max_ or p.max_chars

    seg = text[start:end]
    cuts = [0] + [m.end() for m in _SENT.finditer(seg)] + [len(seg)]
    cuts = sorted(set(cuts))
    spans, s_idx = [], 0
    while s_idx < len(cuts) - 1:
        a = cuts[s_idx]
        j = s_idx + 1
        while j < len(cuts) - 1 and cuts[j + 1] - a <= target:
            j += 1
        if cuts[j] - a > max_:
            hard_end = a + target
            sp = seg.rfind(" ", a, hard_end)
            e = sp if sp > a + target // 2 else hard_end
            spans.append((start + a, start + e))
            cuts = sorted(set(cuts + [e]))
            s_idx = cuts.index(e)
            continue
        b = cuts[j]
        spans.append((start + a, start + b))
        if j >= len(cuts) - 1:
            break
        nxt = j
        if j - 1 > s_idx and (cuts[j] - cuts[j - 1]) <= p.overlap_chars:
            nxt = j - 1
        s_idx = nxt if nxt > s_idx else j

    out = []
    for a, b in spans:
        while a < b and text[a].isspace(): a += 1
        while b > a and text[b - 1].isspace(): b -= 1
        if b > a:
            out.append((a, b))
    return out


def _clause_range(clauses: list) -> tuple[str, str]:
    return (clauses[0], clauses[-1]) if clauses else ("", "")


def make_cite(c: Chunk, short_title: str) -> str:
    parts = [short_title]
    if c.article:
        parts.append(f"ст. {c.article}")
    elif c.section_path:
        parts.append(c.section_path[-1])
    if c.clause_from:
        parts.append(
            f"п. {c.clause_from}"
            + (f"–{c.clause_to}" if c.clause_to != c.clause_from else "")
        )
    return ", ".join(parts)


def chunk_document(doc: Doc, p: Optional[ChunkParams] = None) -> list[Chunk]:
    p = p or ChunkParams()

    # --- ПРОФИЛЬ ---
    if _is_unstructured_type(doc.doc_type):
        target = p.letter_target_chars
        max_ = p.letter_max_chars
        min_ = p.letter_min_chars
        hard_break = 0
    elif _is_law_type(doc.doc_type):
        target = p.law_target_chars
        max_ = p.law_max_chars
        min_ = p.min_chars
        hard_break = p.hard_break_level
    else:
        target = p.target_chars
        max_ = p.max_chars
        min_ = p.min_chars
        hard_break = p.hard_break_level

    text = doc.text
    stack: list[tuple[int, str, str]] = []
    chunks, cur = [], None

    def snapshot():
        path = [lbl for _, lbl, _ in stack]
        art = next((a for _, _, a in reversed(stack) if a), "")
        return path, art

    def new_cur(start):
        path, art = snapshot()
        return {"start": start, "end": start, "path": path, "article": art,
                "body": False, "clauses": []}

    def flush():
        nonlocal cur
        if cur and cur["end"] > cur["start"]:
            chunks.append(cur)
        cur = None

    for b in doc.blocks:
        if b.kind == "heading":
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.label or b.text, b.article))

            if (cur and cur["body"]
                    and ((hard_break and b.level <= hard_break)
                         or cur["end"] - cur["start"] >= min_)):
                flush()

            if cur is None:
                cur = new_cur(b.start)
            elif not cur["body"]:
                cur["path"], cur["article"] = snapshot()
            cur["end"] = max(cur["end"], b.end)
            continue

        spans = [(b.start, b.end)]
        if b.end - b.start > max_:
            if _is_law_type(doc.doc_type) and _RE_LIST_ITEM.match(b.text):
                spans = _split_span(text, b.start, b.end, p,
                                    target=p.law_target_chars, max_=p.law_max_chars)
            else:
                spans = _split_span(text, b.start, b.end, p, target=target, max_=max_)

        for si, (s, e) in enumerate(spans):
            ptxt = text[s:e]
            if cur is not None and cur["body"]:
                union_len = e - cur["start"]
                overflow = union_len > max_
                full = (cur["end"] - cur["start"]) >= target and not _RE_SUBITEM.match(ptxt)
                if overflow or full:
                    flush()
            if cur is None:
                cur = new_cur(s)
            cur["end"] = max(cur["end"], e)
            cur["body"] = True
            if b.clause and si == 0:
                cur["clauses"].append(b.clause)
    flush()

    # --- СКЛЕЙКА ХВОСТОВ ---
    merged = []
    for c in chunks:
        if (merged
                and (c["end"] - c["start"]) < min_
                and merged[-1]["path"] == c["path"]
                and c["end"] - merged[-1]["start"] <= max_):
            pv = merged[-1]
            pv["end"] = c["end"]
            pv["clauses"] += c["clauses"]
        else:
            merged.append(c)

    # --- ФИНАЛЬНЫЕ ЧАНКИ + ФИЛЬТР МУСОРА ---
    out: list[Chunk] = []
    short_title = doc.title if len(doc.title) <= 90 else doc.title[:87].rstrip() + "…"
    
    # Ищем parent_doc_id для допсоглашений
    parent_doc_id = None
    if doc.doc_type == "Доп. соглашение":
        m = re.search(r"([А-ЯA-Z]-\d+[/-]\d+)", doc.file)
        if m:
            parent_doc_id = m.group(1).replace("/", "-")

    for i, c in enumerate(merged):
        t = text[c["start"]:c["end"]].strip()
        if len(re.findall(r"\w", t)) < 15:
            continue

        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        if not c["article"] and lines and all(_RE_SERVICE_LINE.match(ln) for ln in lines):
            continue

        cf, ct = _clause_range(c["clauses"])
        ch = Chunk(
            chunk_id=f"{doc.doc_id}:{i:04d}",
            doc_id=doc.doc_id,
            doc_title=doc.title,
            doc_file=doc.file,
            doc_type=doc.doc_type,
            doc_number=doc.number,
            doc_date=doc.date,
            revision_date=doc.revision_date,
            future_changes=list(doc.future_changes),
            is_outdated=doc.is_outdated,
            parent_doc_id=parent_doc_id,
            section_path=c["path"],
            article=c["article"],
            clause_from=cf, clause_to=ct,
            start=c["start"], end=c["end"],
            text=t, n_chars=len(t), idx=i,
        )
        ch.cite = make_cite(ch, short_title)
        out.append(ch)
    return out


def make_embed_text(c) -> str:
    """Текст для эмбеддинга: контекст документа + чистый текст чанка."""
    d = c if isinstance(c, dict) else c.to_dict()
    head = [d["doc_title"]]
    if d.get("doc_number"):
        head.append(f"№ {d['doc_number']}")
    if d.get("doc_date"):
        head.append(f"от {d['doc_date']}")
    if d.get("revision_date"):
        head.append(f"(ред. от {d['revision_date']})")
    if d.get("future_changes"):
        head.append(f"(изменения с {', '.join(d['future_changes'])})")
    if d.get("is_outdated"):
        head.append("[УТРАТИЛ СИЛУ]")  # НОВОЕ: пометка для эмбеддинга
    if d.get("section_path"):
        head.append(" › ".join(d["section_path"]))
    return "\n".join(head) + "\n\n" + d["text"]