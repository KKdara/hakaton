"""Шаг 3: Сборка векторного индекса ChromaDB из chunks.json и embeddings.json."""
import json
import os
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

    print(f"🗄️ Инициализируем ChromaDB в папке {chroma_dir}...")
    # Используем PersistentClient, чтобы база сохранилась на диск
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Удаляем старую коллекцию, если она была (чтобы не дублировать данные)
    try:
        client.delete_collection("legal_docs")
    except Exception:
        pass
    
    # Создаем новую коллекцию
    collection = client.create_collection(
        name="legal_docs",
        metadata={"hnsw:space": "cosine"} # Используем косинусное сходство
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
        documents.append(ch["text"])           # Сам текст (для цитат)
        vectors.append(emb_dict[cid])          # Вектор (для поиска)

        # Метаданные для фильтрации (Участник №3 будет их юзать)
        meta = {
            "doc_id": ch.get("doc_id", ""),
            "doc_title": ch.get("doc_title", ""),
            "doc_type": ch.get("doc_type", ""),
            "doc_number": str(ch.get("doc_number", "")),
            "doc_date": str(ch.get("doc_date", "")),
            "revision_date": str(ch.get("revision_date", "")),
            "article": str(ch.get("article", "")),
            "clause_from": str(ch.get("clause_from", "")),
            "is_outdated": bool(ch.get("is_outdated", False)),
            "parent_doc_id": str(ch.get("parent_doc_id", ""))
        }
        # Chroma не принимает списки в метаданные, превращаем future_changes в строку
        if ch.get("future_changes"):
            meta["future_changes"] = ", ".join(ch["future_changes"])
        
        metadatas.append(meta)

    # Вставляем батчами по 500 штук (Chroma может падать на больших объемах)
    print(f"📦 Вставляем {len(ids)} чанков в базу...")
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            embeddings=vectors[i:i+batch_size]
        )

    print(f"✅ Готово! Коллекция 'legal_docs' создана.")
    print(f"Всего документов в индексе: {collection.count()}")
    print(f"📁 База сохранена в папку: {chroma_dir}")

if __name__ == "__main__":
    main()