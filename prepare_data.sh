#!/bin/bash
set -e
python build_index.py          # → data/chunks.json
python build_embeddings.py     # → data/embeddings.json
python build_vector_store.py   # → data/chroma_db/
echo "✅ База знаний готова"