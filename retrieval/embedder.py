"""Векторизация запроса. Модель и прокси те же, что в build_embeddings.py.
У bge-m3 для запроса префикс не нужен, поэтому текст отправляем как есть."""
from __future__ import annotations

import time

from .config import Config


class Embedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self.cfg.require_embedding_env()
            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
        return self._client

    def __call__(self, text: str) -> list[float]:
        last = None
        for attempt in range(3):
            try:
                resp = self._get_client().embeddings.create(model=self.cfg.emb_model, input=[text])
                return list(resp.data[0].embedding)
            except Exception as e:  # сеть/лимиты прокси
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Не удалось получить эмбеддинг запроса: {last}")
