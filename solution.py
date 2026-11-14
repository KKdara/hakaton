"""Единая точка входа для платформы хакатона.
Читает DATA_DIR/questions.csv, отвечает на каждый вопрос, пишет submission.csv.
"""
import csv
import os
import sys
import json
import time
from pathlib import Path
from openai import OpenAI


# ======================== Переменные окружения ========================

DATA_DIR     = Path(os.environ.get("DATA_DIR", "data"))
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
LLM_API_KEY  = os.environ.get("LLM_API_KEY")  or os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL    = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or os.environ.get("GEN_MODEL", "deepseek-ai/DeepSeek-V3")
EMB_MODEL    = os.environ.get("EMB_MODEL", "BAAI/bge-m3")
MAX_TOKENS   = int(os.environ.get("MAX_TOKENS", 1024))
TEMPERATURE  = float(os.environ.get("TEMPERATURE", 0.1))

QUESTIONS_CSV  = DATA_DIR / "questions.csv"
KB_ZIP         = DATA_DIR / "knowledge-base.zip"
SUBMISSION_CSV = Path("submission.csv")
CHROMA_DIR     = str(DATA_DIR / "chroma_db")

MAX_ANSWER_LEN = 10000


# ======================== Подготовка базы знаний ========================

def prepare_knowledge_base():
    """Строим индекс из knowledge-base.zip если ещё не построен."""
    import chromadb
    
    # Если индекс уже есть — не перестраиваем
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists():
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            col = client.get_collection("legal_docs")
            if col.count() > 0:
                print(f"✅ Индекс уже есть: {col.count()} чанков")
                return
        except Exception:
            pass

    print("🔨 Строим индекс из knowledge-base.zip...")
    
    # Импортируем pipeline участника 1
    sys.path.insert(0, str(Path(__file__).parent))
    from parser import load_docs_from_zip
    from chunking import chunk_document, ChunkParams, make_embed_text

    docs = load_docs_from_zip(KB_ZIP)
    print(f"   Документов: {len(docs)}")

    params = ChunkParams()
    all_chunks = []
    for doc in docs:
        try:
            chunks = chunk_document(doc, params)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"   ⚠️ {doc.file}: {e}")

    print(f"   Чанков: {len(all_chunks)}")

    # Эмбеддинги
    openai_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    vectors = []
    batch_size = 32
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        texts = [make_embed_text(ch) for ch in batch]
        for attempt in range(3):
            try:
                resp = openai_client.embeddings.create(model=EMB_MODEL, input=texts)
                vectors.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2)
        if (i // batch_size) % 5 == 0:
            print(f"   Эмбеддинги: {len(vectors)}/{len(all_chunks)}")

    # Записываем в ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection("legal_docs")
    except Exception:
        pass
    collection = client.create_collection("legal_docs", metadata={"hnsw:space": "cosine"})

    ids, documents, metadatas = [], [], []
    for ch, vec in zip(all_chunks, vectors):
        d = ch.to_dict()
        ids.append(d["chunk_id"])
        documents.append(d["text"])
        meta = {
            "doc_id": d.get("doc_id", ""),
            "doc_title": d.get("doc_title", ""),
            "doc_type": d.get("doc_type", ""),
            "doc_number": str(d.get("doc_number", "")),
            "doc_date": str(d.get("doc_date", "")),
            "revision_date": str(d.get("revision_date", "")),
            "article": str(d.get("article", "")),
            "clause_from": str(d.get("clause_from", "")),
            "is_outdated": bool(d.get("is_outdated", False)),
            "parent_doc_id": str(d.get("parent_doc_id", "")),
        }
        if d.get("future_changes"):
            meta["future_changes"] = ", ".join(d["future_changes"])
        metadatas.append(meta)

    for i in range(0, len(ids), 500):
        collection.add(
            ids=ids[i:i+500],
            documents=documents[i:i+500],
            metadatas=metadatas[i:i+500],
            embeddings=vectors[i:i+500],
        )
    print(f"✅ Индекс построен: {collection.count()} чанков")


# ======================== Поиск ========================

_collection = None

def get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection("legal_docs")
    return _collection


def search(query: str, top_k: int = 6) -> list[dict]:
    openai_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    resp = openai_client.embeddings.create(model=EMB_MODEL, input=[query])
    vec = resp.data[0].embedding

    results = get_collection().query(
        query_embeddings=[vec],
        n_results=top_k,
        where={"is_outdated": False},
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for cid, doc, meta, dist in zip(
        results["ids"][0], results["documents"][0],
        results["metadatas"][0], results["distances"][0]
    ):
        score = max(0.0, min(1.0, 1.0 - float(dist) / 2))
        chunks.append({"chunk_id": cid, "text": doc, "meta": meta, "score": score})

    return sorted(chunks, key=lambda x: x["score"], reverse=True)


# ======================== Генерация ответа ========================

SYSTEM_PROMPT = """Ты — юридический ассистент банка. Отвечай строго на основе \
предоставленных фрагментов нормативных документов.

Правила:
1. Используй ТОЛЬКО информацию из предоставленных фрагментов.
2. Каждый вывод подтверждай ссылкой [N], где N — номер фрагмента.
3. Если фрагментов недостаточно — прямо скажи об этом. Не придумывай нормы и факты.
4. Если вопрос не юридический — вежливо объясни границы своей компетенции.
5. Рекомендуемая длина ответа — до 150 слов, если вопрос не требует большего."""


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, ch in enumerate(chunks, 1):
        meta = ch["meta"]
        ref_parts = [meta.get("doc_title", "")]
        if meta.get("doc_number") not in (None, "", "None"):
            ref_parts.append(f"№ {meta['doc_number']}")
        if meta.get("doc_date") not in (None, "", "None"):
            ref_parts.append(f"от {meta['doc_date']}")
        if meta.get("article") not in (None, "", "None"):
            ref_parts.append(f"ст. {meta['article']}")
        if meta.get("clause_from") not in (None, "", "None"):
            ref_parts.append(f"п. {meta['clause_from']}")
        ref = ", ".join(p for p in ref_parts if p)
        parts.append(f"[{i}] {ref}\n{ch['text'][:1200]}")
    return "\n\n".join(parts)


def answer_question(question: str) -> str:
    """Поиск + генерация ответа для одного вопроса."""
    try:
        chunks = search(question, top_k=6)
    except Exception as e:
        return f"Ошибка поиска: {e}"

    # Нет релевантных чанков
    if not chunks or chunks[0]["score"] < 0.25:
        return (
            "В базе знаний не найдено документов, относящихся к данному вопросу. "
            "Дать обоснованный ответ невозможно."
        )

    context = _format_context(chunks)
    openai_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    try:
        resp = openai_client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Вопрос: {question}\n\n"
                    f"Фрагменты из базы знаний:\n{context}\n\n"
                    "Дай ответ строго на основе приведённых фрагментов. "
                    "Каждый вывод подтверди ссылкой [N]."
                )},
            ],
        )
        result = resp.choices[0].message.content.strip()
        return result[:MAX_ANSWER_LEN]
    except Exception as e:
        return f"Ошибка генерации ответа: {e}"


# ======================== Главный цикл ========================

def main():
    print(f"📂 DATA_DIR: {DATA_DIR}")
    print(f"🤖 LLM_MODEL: {LLM_MODEL}")
    print(f"📐 EMB_MODEL: {EMB_MODEL}")

    # Проверяем входные файлы
    if not QUESTIONS_CSV.exists():
        print(f"❌ Не найден {QUESTIONS_CSV}", file=sys.stderr)
        sys.exit(1)
    if not KB_ZIP.exists():
        print(f"❌ Не найден {KB_ZIP}", file=sys.stderr)
        sys.exit(1)

    # Строим индекс (или используем готовый)
    prepare_knowledge_base()

    # Читаем вопросы
    with open(QUESTIONS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = list(reader)
    print(f"📋 Вопросов: {len(questions)}")

    # Отвечаем и пишем submission.csv
    with open(SUBMISSION_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "answer"])

        for i, row in enumerate(questions, 1):
            qid      = row["id"]
            question = row["question"]
            print(f"[{i}/{len(questions)}] ID={qid}: {question[:60]}...")

            answer = answer_question(question)
            writer.writerow([qid, answer])
            print(f"   → {answer[:80]}...")

    print(f"\n✅ Готово! Ответы записаны в {SUBMISSION_CSV}")


if __name__ == "__main__":
    main()