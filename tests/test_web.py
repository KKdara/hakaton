"""Meaningful integration checks against the supplied corpus; no network or API keys."""
import pytest
from fastapi.testclient import TestClient
from web.app import app
from web import service


@pytest.fixture(autouse=True)
def local_mode(monkeypatch):
    monkeypatch.setenv('SEARCH_MODE', 'local')


client = TestClient(app)


def test_citations_are_exact_and_documents_exist():
    r = client.post('/api/chat', json={'message': 'Какой испытательный срок при приеме на работу?'})
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'partial'
    assert data['sources'][0]['article'] == '70'
    for source in data['sources']:
        doc = client.get('/api/document', params={'doc_id': source['doc_id']})
        assert doc.status_code == 200
        assert source['text'] in doc.json()['text']
        assert not source['is_outdated']


def test_unknown_question_has_no_invented_sources():
    data = client.post('/api/chat', json={'message': 'рецепт борща со сметаной'}).json()
    assert data['status'] == 'insufficient'
    assert not data['sources']


def test_document_scope_and_date_warning():
    doc_id = service.corpus()[0][0].doc_id
    data = client.post('/api/chat', json={'message': 'договор аренды', 'doc_id': doc_id, 'event_date': '2024-06-01'}).json()
    assert data['sources']
    assert all(s['doc_id'] == doc_id for s in data['sources'])
    assert any('2024-06-01' in w and 'не реализован' in w for w in data['warnings'])


@pytest.mark.parametrize('body', [{'message': ' '}, {'message': 'x' * 4001}, {'message': 'договор', 'event_date': 'invalid'}])
def test_invalid_input(body):
    assert client.post('/api/chat', json=body).status_code == 422


def test_no_arbitrary_file_access():
    assert client.get('/api/document', params={'doc_id': '../../.env'}).status_code == 404
    assert client.get('/api/document/download', params={'doc_id': '../../.env'}).status_code == 404


def test_errors_do_not_leak_backend_details(monkeypatch):
    import web.app as module
    def fail(*a, **k):
        raise RuntimeError('secret-key-example')
    monkeypatch.setattr(module, 'search', fail)
    r = client.post('/api/chat', json={'message': 'договор'})
    assert r.status_code == 503
    assert 'secret-key-example' not in r.text


def test_health_and_document_download():
    h = client.get('/api/health').json()
    assert h['documents'] == 21 and h['chunks'] == 230 and h['generation'] is False
    doc_id = service.corpus()[0][0].doc_id
    r = client.get('/api/document/download', params={'doc_id': doc_id})
    assert r.status_code == 200 and 'attachment' in r.headers['content-disposition']
    assert r.text == client.get('/api/document', params={'doc_id': doc_id}).json()['text']


@pytest.mark.parametrize('status', ['sufficient', 'partial', 'insufficient'])
def test_vector_adapter_preserves_retrieval_contract(monkeypatch, status):
    from types import SimpleNamespace
    import retrieval
    monkeypatch.setenv('SEARCH_MODE', 'vector')
    calls = []
    def retrieve(query, filters=None):
        calls.append((query, filters))
        return SimpleNamespace(status=SimpleNamespace(value=status), reason='retrieval reason', chunks=[])
    monkeypatch.setattr(retrieval, 'retrieve', retrieve)
    result = service.search('вопрос', doc_id='doc')
    assert result['mode'] == 'vector' and result['status'] == status
    assert result['reason'] == 'retrieval reason'
    assert calls == [('вопрос', {'doc_id': 'doc'})]
