"""UI adapter. Extractive results deliberately do not impersonate an LLM answer."""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from chunking import chunk_document
from parser import load_docs_from_zip

ROOT = Path(__file__).resolve().parents[1]
STOP = set('какой какая какие как что это для при или чтобы нужно можно ли по на в с и а о от до из об у не во со к за чем сколько будет есть мне мне пожалуйста расскажи покажи'.split())


def tokens(text):
    # A small prefix normalizer, not a morphological analyzer.
    return [w[:6] for w in re.findall(r'[а-яёa-z0-9]+', text.lower().replace('ё', 'е'))
            if len(w) > 2 and w not in STOP]


@lru_cache(maxsize=1)
def corpus():
    path = Path(os.getenv('KNOWLEDGE_ZIP', str(ROOT / 'data/knowledge-base.zip')))
    docs = load_docs_from_zip(path)
    chunks = [c.to_dict() for d in docs for c in chunk_document(d)]
    counts = [Counter(tokens(c['doc_title'] + ' ' + c['text'])) for c in chunks]
    df = Counter(t for count in counts for t in count)
    avg = sum(map(lambda c: sum(c.values()), counts)) / max(len(counts), 1)
    return docs, chunks, counts, df, avg


def mode():
    configured = bool((os.getenv('LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL')) and
                      (os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')))
    return os.getenv('SEARCH_MODE', 'auto') if os.getenv('SEARCH_MODE', 'auto') != 'auto' else ('vector' if configured else 'local')


def local_search(query, doc_id=None):
    _, chunks, counts, df, avg = corpus()
    terms = set(tokens(query))
    found = []
    for c, count in zip(chunks, counts):
        if c['is_outdated'] or (doc_id and c['doc_id'] != doc_id):
            continue
        matched = terms & count.keys()
        if not matched or len(matched) / max(len(terms), 1) < .25:
            continue
        score = sum(math.log(1 + (len(chunks) - df[t] + .5) / (df[t] + .5)) *
                    count[t] * 2.5 / (count[t] + 1.5 * (.25 + .75 * sum(count.values()) / max(avg, 1)))
                    for t in matched)
        found.append(dict(c, score=score, score_kind='bm25'))
    found.sort(key=lambda c: c['score'], reverse=True)
    # Drop weak incidental matches (e.g. "работа" in an unrelated tax article).
    cutoff = found[0]['score'] * .4 if found else 0
    return [c for c in found[:6] if c['score'] >= cutoff]


def search(query, doc_id=None, event_date=None):
    selected_mode = mode()
    if selected_mode not in ('local', 'vector'):
        raise ValueError('SEARCH_MODE must be auto, local or vector')
    if selected_mode == 'vector':
        from retrieval import retrieve
        result = retrieve(query, filters={'doc_id': doc_id} if doc_id else None)
        chunks = [dict(c.to_dict(), score_kind='similarity') for c in result.chunks]
        status, reason = result.status.value, result.reason
    else:
        chunks = local_search(query, doc_id)
        status = 'partial' if chunks else 'insufficient'
        reason = ('Найдены совпадения по словам запроса. Проверьте контекст и редакцию по ссылке на источник.'
                  if chunks else 'В базе не найдено достаточно близких фрагментов. Уточните название документа, номер договора или формулировку вопроса.')
    warnings = []
    if event_date:
        warnings.append(f'Дата события: {event_date}. Автоматический подбор редакции на эту дату пока не реализован. Проверьте редакцию и дополнительные соглашения в источниках; старой редакции закона в базе может не быть.')
    if any(c.get('future_changes') for c in chunks):
        warnings.append('В метаданных источников отмечены изменения. Проверьте даты их вступления в силу в тексте документа.')
    return dict(status=status, reason=reason, mode=selected_mode, sources=chunks,
                answer=('Найденные положения'
                        if chunks else 'Подходящие положения не найдены.'), warnings=warnings)
