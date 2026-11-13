"""Шаг 3: Сборка векторного индекса ChromaDB из chunks.json и embeddings.json."""
import json
import sys
from pathlib import Path
import chromadb


def main():
    chunks_path = Path("data/chunks.json")
    emb_path = Path("data/embeddings.json")
    chroma_dir = "data/chroma_db"

    if not chunks_path.exists() or not emb_path.exists():
        print("❌ Не найдены data/chunks.json или data/embeddings.json.", file=sys.stderr)
        sys.exit(1)

    print("📥 Загружаем чанки и эмбеддинги...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(emb_path, "r", encoding="utf-8") as f:
        embeddings_data = json.load(f)

    # Создаем словарь {chunk_id: embedding} для быстрого поиска
    emb_dict = {item["chunk_id"]: item["embedding"] for item in embeddings_data}

    # Проверка на нулевые заглушки
    if embeddings_data:
        sample = embeddings_data[0]["embedding"]
        if all(v == 0.0 for v in sample):
            print("⚠️ ВНИМАНИЕ: эмбеддинги нулевые (заглушки). Индекс для поиска непригоден.")
            print("   На платформе хакатона будут реальные векторы — продолжаем сборку.")

    print(f"🗄️ Инициализируем ChromaDB в папке {chroma_dir}...")
    client = chromadb.PersistentClient(path=chroma_dir)

    # Удаляем старую коллекцию, если была (чтобы не дублировать данные)
    try:
        client.delete_collection("legal_docs")
    except Exception:
        pass

    collection = client.create_collection(
        name="legal_docs",
        metadata={"hnsw:space": "cosine"}
    )

    # Подготавливаем данные для вставки
    ids = []
    documents = []
    metadatas = []
    vectors = []

    for ch in chunks:
        cid = ch["chunk_id"]
        if cid not in emb_dict:
            print(f"⚠️ Пропущен чанк {cid}: нет вектора.")
            continue

        ids.append(cid)
        documents.append(ch["text"])
        vectors.append(emb_dict[cid])

        meta = {
            "doc_id":        ch.get("doc_id", ""),
            "doc_title":     ch.get("doc_title", ""),
            "doc_type":      ch.get("doc_type", ""),
            "doc_number":    str(ch.get("doc_number", "")),
            "doc_date":      str(ch.get("doc_date", "")),
            "revision_date": str(ch.get("revision_date", "")),
            "article":       str(ch.get("article", "")),
            "clause_from":   str(ch.get("clause_from", "")),
            "is_outdated":   bool(ch.get("is_outdated", False)),
            "parent_doc_id": str(ch.get("parent_doc_id", "")),
        }
        if ch.get("future_changes"):
            meta["future_changes"] = ", ".join(ch["future_changes"])

        metadatas.append(meta)

    print(f"📦 Вставляем {len(ids)} чанков в базу...")
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            embeddings=vectors[i:i+batch_size],
        )

    print(f"✅ Готово! Коллекция 'legal_docs' создана.")
    print(f"   Всего чанков в индексе: {collection.count()}")
    print(f"📁 База сохранена в папку: {chroma_dir}")

    # Smoke-тест: ищем первый чанк сам по себе (только с реальными эмбеддингами)
    if vectors and any(v != 0.0 for v in vectors[0]):
        results = collection.query(query_embeddings=[vectors[0]], n_results=1)
        assert results["ids"][0][0] == ids[0], "❌ Smoke-тест: индекс работает некорректно"
        print("✅ Smoke-тест пройден")
    else:
        print("⏭️  Smoke-тест пропущен (заглушки)")


if __name__ == "__main__":
    main()