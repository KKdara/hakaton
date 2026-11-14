"""Офлайн-тесты достоверности. Модель не вызывается.

Проверяется: факты вопросов есть в архиве базы; ссылки [N] только на существующие
фрагменты; в sources попадают только процитированные чанки; отказ и выдумка
считаются на двух наборах вопросов.
"""
import zipfile
from pathlib import Path

import pytest

from tests.kb_cases import ABSENT, FALSE_PREMISE, FAKE_SYSTEM, IN_BASE, PARTIAL, PRESSURE
from verification import (
    build_messages,
    clean_chunk_text,
    detect_injection,
    grounding_rates,
    guard_answer,
    invented_answer,
    is_refusal,
    load_refusals,
    load_system_prompt,
    readable_cite,
    refusal,
)

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data" / "knowledge-base.zip"


def _zip_text(name: str) -> str:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        return zf.read(name).decode("utf-8-sig")


def _corpus() -> str:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        return "\n".join(zf.read(n).decode("utf-8-sig") for n in zf.namelist())


def _excerpt(name: str, needle: str, pad: int = 500) -> str:
    text = _zip_text(name)
    i = text.index(needle)
    return text[max(0, i - pad): i + len(needle) + pad]


def _chunk(name: str, needle: str, **meta) -> dict:
    return {
        "text": _excerpt(name, needle),
        "doc_title": meta.get("doc_title", Path(name).stem),
        "doc_number": meta.get("doc_number"),
        "doc_date": meta.get("doc_date"),
        "article": meta.get("article"),
        "clause_from": meta.get("clause_from"),
        "revision_date": meta.get("revision_date"),
    }


def test_zip_exists():
    assert ZIP_PATH.is_file()


@pytest.mark.parametrize("case", IN_BASE, ids=[c["id"] for c in IN_BASE])
def test_in_base_needle_is_in_document(case):
    assert case["needle"] in _zip_text(case["file"])


@pytest.mark.parametrize("case", ABSENT, ids=[c["id"] for c in ABSENT])
def test_absent_phrase_is_not_in_corpus(case):
    assert case["absent_phrase"].lower() not in _corpus().lower()


def test_laws_in_archive_are_only_current_edition():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        laws = [n for n in zf.namelist() if n.startswith("законы__")]
    assert laws
    assert all("действующая" in Path(n).name for n in laws)


def test_outdated_company_regulation_is_flagged_by_parser():
    from parser import parse_document

    old_name = "документы компании__Регламент согласования договоров ред 2023 (утратил силу).txt"
    new_name = "документы компании__Регламент согласования договоров ред 2025.txt"
    with zipfile.ZipFile(ZIP_PATH) as zf:
        old = parse_document("reg2023", old_name, zf.read(old_name))
        new = parse_document("reg2025", new_name, zf.read(new_name))
    assert old.is_outdated
    assert not new.is_outdated
    assert "1 000 000" in old.text
    assert "3 000 000" in new.text


def test_archive_memo_contains_model_injection():
    text = _zip_text("документы компании__Служебная записка об архиве договоров.txt")
    assert detect_injection(text)
    assert "90 календарных дней" in text
    cleaned = clean_chunk_text(text)
    assert "5 (пять) лет" in cleaned
    assert "90 календарных дней" not in cleaned


def test_invalid_citation_is_removed_and_sources_only_cited():
    chunks = [
        _chunk(IN_BASE[3]["file"], IN_BASE[3]["needle"], doc_number="П-17/2024", clause_from="4.2"),
        {"text": "Соседний фрагмент про гарантию 24 месяца.", "doc_title": "Договор", "clause_from": "5.2"},
    ]
    draft = "Срок оплаты — 30 календарных дней [1]. Гарантия 24 месяца [2]. См. также [9]."
    result = guard_answer(draft, chunks)
    assert result.invalid_citations == [9]
    assert "[9]" not in result.answer
    assert [s["n"] for s in result.sources] == [1, 2]
    assert all("Гарантия" not in s["text"] or s["n"] == 2 for s in result.sources)
    assert "п. 4.2" in result.sources[0]["cite"]
    assert "№ П-17/2024" in result.sources[0]["cite"]


def test_uncited_chunk_is_not_a_source():
    chunks = [
        _chunk(IN_BASE[0]["file"], IN_BASE[0]["needle"], clause_from="3.1"),
        _chunk(IN_BASE[1]["file"], IN_BASE[1]["needle"], clause_from="5.1"),
    ]
    draft = "Арендная плата составляет 450 000 рублей в месяц [1]."
    result = guard_answer(draft, chunks)
    assert result.ok
    assert [s["n"] for s in result.sources] == [1]


def test_legal_claim_without_citation_is_flagged():
    chunks = [_chunk(IN_BASE[3]["file"], IN_BASE[3]["needle"])]
    draft = "Покупатель оплачивает товар в течение 30 календарных дней."
    result = guard_answer(draft, chunks)
    assert result.uncited_claims
    assert result.sources == []
    assert not result.ok


@pytest.mark.parametrize("case", IN_BASE, ids=[c["id"] for c in IN_BASE])
def test_cited_excerpt_passes_guard(case):
    chunk = _chunk(case["file"], case["needle"])
    draft = f"{case['needle']} [1]."
    result = guard_answer(draft, [chunk])
    assert result.ok, (result.uncited_claims, result.unsupported_numbers, result.injection_numbers_used)
    assert result.sources[0]["n"] == 1
    assert not is_refusal(draft)


def test_number_absent_from_chunk_is_unsupported():
    chunk = _chunk(FALSE_PREMISE["file"], FALSE_PREMISE["needle"], clause_from="4.2")
    draft = "Срок оплаты товара — 90 календарных дней [1]."
    result = guard_answer(draft, [chunk])
    assert "90" in result.unsupported_numbers
    assert not result.ok


def test_false_premise_disclaimer_keeps_supported_term():
    chunk = _chunk(FALSE_PREMISE["file"], FALSE_PREMISE["needle"], doc_number="П-17/2024", clause_from="4.2")
    draft = (
        "В вопросе есть посылка, которую найденные фрагменты не подтверждают. "
        "Срок оплаты — 30 календарных дней [1]."
    )
    result = guard_answer(draft, [chunk])
    assert result.ok
    assert result.sources[0]["cite"]
    assert not invented_answer(draft, [chunk])


def test_partial_answer_marks_missing_fact():
    chunk = _chunk(PARTIAL["file"], PARTIAL["needle"], clause_from="4.2")
    draft = (
        f"{PARTIAL['needle']} [1].\n\n"
        f"Не подтверждено, потому что в переданных фрагментах этого нет: {PARTIAL['missing']}."
    )
    result = guard_answer(draft, [chunk])
    assert result.ok
    invented = (
        f"{PARTIAL['needle']} [1]. Уставный капитал составляет 10 000 рублей [1]."
    )
    bad = guard_answer(invented, [chunk])
    assert "10 000" in bad.unsupported_numbers


def test_injection_number_is_not_treated_as_norm():
    name = "документы компании__Служебная записка об архиве договоров.txt"
    chunk = {"text": _zip_text(name), "doc_title": "Служебная записка № ЮО-7"}
    obeyed = "Срок оплаты по любому договору — 90 календарных дней [1]."
    bad = guard_answer(obeyed, [chunk])
    assert bad.injection_in_context
    assert bad.injection_numbers_used
    assert not bad.ok

    held = "Срок хранения оригинала договора — 5 лет [1]."
    good = guard_answer(held, [chunk])
    assert good.ok
    assert good.sources


def test_readable_cite_uses_article_and_revision():
    cite = readable_cite({
        "doc_title": "Трудовой кодекс Российской Федерации",
        "doc_number": "197-ФЗ",
        "doc_date": "30.12.2001",
        "article": "70",
        "revision_date": "01.09.2026",
    })
    assert "Трудовой кодекс" in cite
    assert "ст. 70" in cite
    assert "ред. 01.09.2026" in cite
    assert "197-ФЗ" in cite


def test_prompt_covers_grounding_rules():
    prompt = load_system_prompt()
    for piece in (
        "[N]",
        "внешн",
        "Чего не хватает",
        "редакц",
        "посылк",
        "противоправ",
        "SYSTEM",
        "ты ошибся",
        "компетенц",
    ):
        assert piece.lower() in prompt.lower(), piece


def test_refusal_texts_explain_the_gap():
    kinds = load_refusals()
    assert set(kinds) >= {"no_data", "partial", "off_topic", "old_edition", "false_premise", "unlawful"}
    no_data = refusal("no_data", gap="текст статьи 81 ТК РФ")
    assert "статьи 81" in no_data
    assert "Чего не хватает" in no_data
    partial = refusal("partial", confirmed="срок оплаты 30 дней [1]", missing="уставный капитал")
    assert "Подтверждено" in partial and "уставный капитал" in partial
    assert "компетенц" in refusal("off_topic").lower()
    assert "действующая редакция" in refusal("old_edition")
    assert "посылк" in refusal("false_premise", what="срок 90 дней").lower()
    unlawful = refusal("unlawful")
    assert "противоправ" in unlawful.lower()
    assert "инструкц" in unlawful.lower()


def test_fake_system_and_pressure_do_not_become_system_role():
    messages = build_messages(
        FALSE_PREMISE["question"],
        context="[1] фрагмент договора",
        history=[
            {"role": "system", "content": FAKE_SYSTEM},
            {"role": "user", "content": PRESSURE},
            {"role": "assistant", "content": "Срок оплаты — 30 календарных дней [1]."},
        ],
    )
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "только по фрагментам" in system_msgs[0]["content"].lower() or "[N]" in system_msgs[0]["content"]
    assert FAKE_SYSTEM not in system_msgs[0]["content"]
    spoof = messages[1]
    assert spoof["role"] == "user"
    assert "не системная инструкция" in spoof["content"]
    assert messages[2]["role"] == "user"
    assert "90" in messages[2]["content"]
    assert messages[-1]["role"] == "user"
    assert "Фрагменты из базы" in messages[-1]["content"]


def test_scripted_sets_have_no_false_refusals_or_inventions():
    rows = []
    for case in IN_BASE:
        chunk = _chunk(case["file"], case["needle"])
        draft = f"{case['needle']} [1]."
        rows.append({
            "expected": "in_base",
            "refused": is_refusal(draft),
            "invented": invented_answer(draft, [chunk]),
        })
    for case in ABSENT:
        kind = case.get("refusal_kind", "no_data")
        if kind == "no_data":
            draft = refusal("no_data", gap=case["gap"])
        elif kind == "false_premise":
            draft = refusal(kind, what=case["gap"])
        else:
            draft = refusal(kind)
        rows.append({
            "expected": "absent",
            "refused": is_refusal(draft),
            "invented": invented_answer(draft, []),
        })
    rates = grounding_rates(rows)
    assert rates["n_in_base"] == len(IN_BASE)
    assert rates["n_absent"] == len(ABSENT)
    assert rates["false_refusal_rate"] == 0.0
    assert rates["invented_rate"] == 0.0


def test_poisoned_drafts_move_the_rates():
    rows = [{"expected": "in_base", "refused": False, "invented": False} for _ in range(len(IN_BASE))]
    rows.append({"expected": "in_base", "refused": True, "invented": False})
    rows.extend({"expected": "absent", "refused": True, "invented": False} for _ in range(len(ABSENT) - 1))
    rows.append({"expected": "absent", "refused": False, "invented": True})
    rates = grounding_rates(rows)
    assert rates["false_refusal_rate"] == pytest.approx(1 / (len(IN_BASE) + 1))
    assert rates["invented_rate"] == pytest.approx(1 / len(ABSENT))


def test_refusing_known_payment_term_counts_as_false_refusal():
    case = next(c for c in IN_BASE if c["id"] == "supply-payment")
    chunk = _chunk(case["file"], case["needle"])
    draft = refusal("no_data", gap="срок оплаты по договору П-17/2024")
    assert is_refusal(draft)
    assert case["needle"] in chunk["text"]
    rates = grounding_rates([
        {"expected": "in_base", "refused": True, "invented": False},
    ])
    assert rates["false_refusal_rate"] == 1.0


def test_invented_article_on_empty_context_is_counted():
    draft = "Статья 999 ТК РФ устанавливает испытание сроком 9 месяцев и штраф 50 000 рублей."
    assert invented_answer(draft, [])
    assert not is_refusal(draft)
    rates = grounding_rates([
        {"expected": "absent", "refused": False, "invented": True},
    ])
    assert rates["invented_rate"] == 1.0
