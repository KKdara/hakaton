"""Run: uvicorn web.app:app --host 127.0.0.1 --port 8000."""
from datetime import date
import logging
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[1] / '.env')
from .service import corpus, mode, search

app = FastAPI(title='Юридический ассистент', version='1.0.0')
log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / 'static'


class Question(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    doc_id: str | None = Field(default=None, max_length=500)
    event_date: date | None = None


@app.get('/api/health')
def health():
    try:
        docs, chunks, *_ = corpus()
        return dict(ok=True, mode=mode(), documents=len(docs), chunks=len(chunks), generation=False)
    except Exception:
        log.exception('Cannot load knowledge base')
        raise HTTPException(503, 'База знаний недоступна. Проверьте KNOWLEDGE_ZIP на сервере.')


def summary(d):
    return dict(doc_id=d.doc_id, title=d.title, filename=d.file, doc_type=d.doc_type,
                revision_date=d.revision_date, date=d.date, is_outdated=d.is_outdated)


@app.get('/api/documents')
def documents():
    return [summary(d) for d in corpus()[0]]


def document(doc_id):
    found = next((d for d in corpus()[0] if d.doc_id == doc_id), None)
    if found is None:
        raise HTTPException(404, 'Документ не найден в базе знаний.')
    return found


@app.get('/api/document')
def get_document(doc_id: str):
    d = document(doc_id)
    return dict(summary(d), text=d.text)


@app.get('/api/document/download')
def download(doc_id: str):
    d = document(doc_id)
    return Response(d.text, media_type='text/plain; charset=utf-8',
                    headers={'Content-Disposition': f"attachment; filename*=UTF-8''{quote(d.doc_id + '.txt')}"})


@app.post('/api/chat')
def chat(question: Question):
    if not question.message.strip():
        raise HTTPException(422, 'Введите вопрос.')
    if question.doc_id:
        document(question.doc_id)
    try:
        return search(question.message.strip(), question.doc_id, question.event_date.isoformat() if question.event_date else None)
    except Exception:
        log.exception('Search failed')
        raise HTTPException(503, 'Поиск временно недоступен. Проверьте подключение к API и индекс на сервере. Вопрос можно отправить повторно.')


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html')


app.mount('/static', StaticFiles(directory=STATIC), name='static')
