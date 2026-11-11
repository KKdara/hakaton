from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from parser import load_docs_from_zip
from chunking import chunk_document, ChunkParams


def main(zip_path: str = "data/knowledge-base.zip",
         out_path: str = "data/chunks.json"):
    zip_path = Path(zip_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"Не найден {zip_path}", file=sys.stderr)
        sys.exit(1)

    docs = load_docs_from_zip(zip_path)
    print(f"Документов: {len(docs)}")

    params = ChunkParams()
    all_chunks: list[dict] = []
    stats = {"docs": 0, "chunks": 0, "chars": 0, "warnings": 0, "failed": 0}
    broken_dost = 0
    outdated_docs = 0
    with_future = 0

    for doc in docs:
        try:
            chunks = chunk_document(doc, params)
        except Exception as e:
            stats["failed"] += 1
            print(f"   {doc.file}: {type(e).__name__}: {e}")
            continue

        stats["docs"] += 1
        stats["chunks"] += len(chunks)
        stats["chars"] += sum(c.n_chars for c in chunks)

        if doc.is_outdated:
            outdated_docs += 1
        if doc.future_changes:
            with_future += 1

        if doc.warnings:
            stats["warnings"] += len(doc.warnings)
            for w in doc.warnings:
                print(f"   !!!  {doc.file}: {w}")

        # Расширенный лог: теперь видно дату, устаревание и будущие изменения
        future_str = f", будущ.={doc.future_changes}" if doc.future_changes else ""
        outdated_str = ", УСТАРЕЛ" if doc.is_outdated else ""
        print(f"    {doc.file}: тип={doc.doc_type}, "
              f"номер={doc.number}, дата={doc.date}, ред.={doc.revision_date}"
              f"{outdated_str}{future_str}, "
              f"блоков={len(doc.blocks)}, чанков={len(chunks)}")

        # проверка дословности
        for ch in chunks:
            expected = doc.text[ch.start:ch.end].strip()
            if ch.text != expected:
                broken_dost += 1
                if broken_dost <= 3:
                    print(f"   !!!  дословность: {ch.chunk_id}")

        for ch in chunks:
            all_chunks.append(ch.to_dict())

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # ---------- итоги ----------
    avg = stats["chars"] // max(stats["chunks"], 1)
    print(f"\n Готово: {stats['chunks']} чанков из {stats['docs']} документов")
    print(f"   Средний размер чанка: {avg} символов")
    if stats["failed"]:
        print(f"   !!! Упало документов: {stats['failed']}")
    if stats["warnings"]:
        print(f"   !!!  Предупреждений: {stats['warnings']}")
    if broken_dost:
        print(f"   !!! Нарушений дословности: {broken_dost}")

    print(f"\nДокументов с is_outdated=True: {outdated_docs}")
    print(f"Документов с future_changes:   {with_future}")

    # --- метрики качества (те, что нужны для шага 2) ---
    no_date = [d for d in docs if not d.date]
    no_rev = [d for d in docs if not d.revision_date]
    print(f"Документов без doc_date:      {len(no_date)} / {len(docs)}")
    print(f"Документов без revision_date: {len(no_rev)} / {len(docs)}")

    if no_date:
        print("\nПримеры без doc_date:")
        for d in no_date[:5]:
            print(f"   {d.file} ({d.doc_type})")

    print("\nТипы документов:")
    for t, n in Counter(d.doc_type for d in docs).most_common():
        print(f"   {t}: {n}")

    print("\nИсточник типа:")
    for s, n in Counter(d.type_source for d in docs).most_common():
        print(f"   {s}: {n}")

    # в конце build_index.py, перед print(f"   → {out_path}")
    print("\n--- ПРОВЕРКА EMBED_TEXT ---")
    if all_chunks:
        from chunking import make_embed_text
        for i in [0, len(all_chunks)//2, -1]:  # первый, средний, последний
            d = all_chunks[i]
            print(f"\n[CHUNK {d['chunk_id']}]")
            print(make_embed_text(d)[:500])

    print(f"\n   → {out_path}")


if __name__ == "__main__":
    main()