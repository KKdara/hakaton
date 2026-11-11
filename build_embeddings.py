"""Шаг 2: Генерация эмбеддингов для чанков через OpenAI-совместимый прокси."""
import json
import os
import sys
import time
from pathlib import Path
from openai import OpenAI

# Импортируем функцию для формирования текста эмбеддинга из твоего chunking.py
# (Убедись, что chunking.py лежит в той же папке)
from chunking import make_embed_text


def main(chunks_path: str = "data/chunks.json",
         out_path: str = "data/embeddings.json"):
    
    chunks_path = Path(chunks_path)
    out_path = Path(out_path)
    
    if not chunks_path.exists():
        print(f"❌ Файл {chunks_path} не найден. Сначала запусти build_index.py", file=sys.stderr)
        sys.exit(1)

    # 1. Загружаем чанки
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"📚 Загружено чанков: {len(chunks)}")

    # 2. Берем настройки из переменных окружения (как требует регламент)
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    # По умолчанию используем bge-m3, если платформа не передала другое
    model_name = os.environ.get("EMB_MODEL", "BAAI/bge-m3")

    if not base_url or not api_key:
        print("\n⚠️ ВНИМАНИЕ: Переменные окружения LLM_BASE_URL / LLM_API_KEY не найдены.")
        print("Это нормально при локальном запуске.")
        print("Скрипт сгенерирует векторы, когда будет запущен на платформе хакатона.")
        print("Локально мы создадим заглушки (нулевые векторы), чтобы проверить логику сохранения.")
        
        # Создаем заглушки, чтобы локально можно было протестировать сохранение
        results = []
        for ch in chunks:
            results.append({
                "chunk_id": ch["chunk_id"],
                "embedding": [0.0] * 1024  # bge-m3 выдает 1024-мерные векторы
            })
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f)
        print(f"✅ Созданы локальные заглушки эмбеддингов в {out_path}")
        return

    print(f"🤖 Модель: {model_name}")
    print(f"🌐 Прокси: {base_url}")
    
    client = OpenAI(base_url=base_url, api_key=api_key)
    results = []
    
    # 3. Обрабатываем батчами по 32 чанка
    batch_size = 32
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        # Формируем текст для эмбеддинга (с контекстом документа)
        texts = [make_embed_text(ch) for ch in batch]
        
        # Пытаемся получить векторы (с 3 попытками при ошибке)
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=model_name, input=texts)
                
                # Сохраняем векторы
                for j, data in enumerate(resp.data):
                    ch = batch[j]
                    results.append({
                        "chunk_id": ch["chunk_id"],
                        "embedding": data.embedding
                    })
                
                print(f"   [{batch_num}/{total_batches}] ✅ Обработано: {len(results)} / {len(chunks)}")
                break  # Успех, выходим из цикла попыток
                
            except Exception as e:
                if attempt < 2:
                    print(f"   [{batch_num}/{total_batches}] ⚠️ Ошибка: {e}. Повтор через 2 сек...")
                    time.sleep(2)
                else:
                    print(f"   [{batch_num}/{total_batches}] ❌ Не удалось после 3 попыток: {e}")
                    print("Прерываем работу.")
                    return

    # 4. Сохраняем результат
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    
    print(f"\n🎉 Готово! Векторы сохранены в {out_path}")
    print(f"Всего векторов: {len(results)}")


if __name__ == "__main__":
    main()