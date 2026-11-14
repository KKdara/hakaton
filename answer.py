"""Участник 3: Генерация ответа через LLM на основе найденных чанков."""
import os
from openai import OpenAI
from retrieval import retrieve, Status


LLM_MODEL  = os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-V3")
MAX_TOKENS = 1500
CHUNK_TEXT_LIMIT = 1200

SYSTEM_PROMPT = """Ты — юридический ассистент банка. Отвечай строго на основе \
предоставленных фрагментов нормативных документов.

Правила:
1. Используй ТОЛЬКО информацию из предоставленных фрагментов.
2. Каждый вывод подтверждай ссылкой [N], где N — номер фрагмента.
3. Если фрагментов недостаточно — прямо скажи об этом. Не придумывай нормы.
4. Если вопрос не юридический — вежливо объясни границы компетенции.
5. Отвечай чётко и структурированно."""


def _get_client():
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key  = os.environ.get("LLM_API_KEY")  or os.environ.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise EnvironmentError("Не заданы LLM_BASE_URL / LLM_API_KEY.")
    return OpenAI(base_url=base_url, api_key=api_key)


def _format_context(chunks) -> str:
    parts = []
    for i, ch in enumerate(chunks, 1):
        text = ch.text[:CHUNK_TEXT_LIMIT]
        if len(ch.text) > CHUNK_TEXT_LIMIT:
            text += "..."
        parts.append(f"[{i}] {ch.cite}\n{text}")
    return "\n\n".join(parts)


def _format_source_list(chunks) -> str:
    lines = []
    for ch in chunks:
        line = f"• {ch.doc_title}"
        if ch.doc_number:
            line += f" № {ch.doc_number}"
        if ch.doc_date:
            line += f" от {ch.doc_date}"
        lines.append(line)
    return "\n".join(lines)


def ask(query: str) -> dict:
    """
    Полный pipeline: поиск → генерация ответа.
    Возвращает:
    {
        "answer":  str,
        "sources": list,
        "refused": bool,
        "status":  str,
        "query":   str
    }
    """
    result = retrieve(query)

    # Недостаточно данных — отказ
    if result.status == Status.INSUFFICIENT:
        return {
            "query":   query,
            "answer":  (
                "В базе знаний не найдено документов по данному вопросу. "
                "Дать обоснованный ответ невозможно.\n\n"
                f"Причина: {result.reason}"
            ),
            "sources": [],
            "refused": True,
            "status":  result.status.value,
        }

    # Частичные данные — отвечаем с оговоркой
    if result.status == Status.PARTIAL:
        context = _format_context(result.chunks)
        answer  = _generate(query, context, partial=True)
        return {
            "query":   query,
            "answer":  answer,
            "sources": [ch.to_dict() for ch in result.chunks],
            "refused": False,
            "status":  result.status.value,
        }

    # Данных достаточно — полный ответ
    context = _format_context(result.chunks)
    answer  = _generate(query, context, partial=False)
    return {
        "query":   query,
        "answer":  answer,
        "sources": [ch.to_dict() for ch in result.chunks],
        "refused": False,
        "status":  result.status.value,
    }


def _generate(query: str, context: str, partial: bool) -> str:
    caveat = (
        "\n\nВАЖНО: найденные фрагменты частично относятся к теме. "
        "Если данных не хватает для полного ответа — явно укажи это."
        if partial else ""
    )
    client = _get_client()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Вопрос: {query}{caveat}\n\n"
                f"Фрагменты из базы знаний:\n{context}\n\n"
                "Дай ответ строго на основе приведённых фрагментов. "
                "Каждый вывод подтверди ссылкой [N]."
            )},
        ],
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Какова ставка НДС при реализации товаров?"
    print(f"Вопрос: {query}\n")
    try:
        r = ask(query)
        print(f"Статус: {r['status']} | Отказ: {r['refused']}")
        print(f"\nОтвет:\n{r['answer']}")
        print(f"\nИсточников: {len(r['sources'])}")
    except EnvironmentError as e:
        print(f"⚠️  {e}")