"""Парсинг .txt базы знаний: чтение, очистка, метаданные, разметка структуры.

Правки по итогам прогона на реальной базе:
  * doc_date для законов берётся из первой строки (шапки), а не из «изменена с...»;
  * future_changes: только даты в будущем (отсекаются уже вступившие в силу);
  * Служебная записка имеет наивысший приоритет в _SPECIFIC_TYPES (не становится Доп. соглашением);
  * номер извлекается из doc_title, если его нет в шапке (для кодексов).
"""
from __future__ import annotations

import re
import zipfile
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================ Типы данных ============================

@dataclass
class Block:
    kind: str
    text: str
    start: int
    end: int
    level: int = 0
    label: str = ""
    article: str = ""
    clause: str = ""
    page: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class Doc:
    doc_id: str
    file: str
    title: str
    doc_type: str
    text: str
    blocks: list[Block]
    number: Optional[str] = None
    date: Optional[str] = None
    revision_date: Optional[str] = None
    future_changes: list = field(default_factory=list)
    is_outdated: bool = False
    dates: list = field(default_factory=list)
    type_source: str = "none"
    encoding: str = ""
    pages: int = 0
    warnings: list = field(default_factory=list)


# ============================ Чтение ============================

def read_txt_bytes(data: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "windows-1251"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("iso-8859-1", errors="replace"), "iso-8859-1(!)"


# ============================ Очистка ============================

_RE_PAGE_MARK  = re.compile(r"^\s*=+\s*Page\s+\d+\s*=+\s*$", re.I)
_RE_PAGE_NUM   = re.compile(r"^\s*(?:стр\.?\s*)?\d{1,4}\s*(?:из\s*\d+)?\s*$", re.I)
_RE_MULTISPACE = re.compile(r"[ \t]{2,}")
_RE_SOFT_HYPH  = re.compile(r"(\w)[-\u00ad]\n(\w)")
_RE_GARBAGE    = re.compile(r"[\u2500-\u257f\u25a0-\u25ff]+")

_SENT_TERMINATOR = re.compile(r"[.!?:;»)]\s*$")

def clean_text(raw: str) -> str:
    t = raw.replace("\r\n", "\n").replace("\r", "\n")
    t = _RE_SOFT_HYPH.sub(r"\1\2", t)
    t = "\n".join(l for l in t.split("\n") if not _RE_PAGE_MARK.match(l))

    lines = t.split("\n")
    out: list[str] = []
    for ln in lines:
        if (out and ln
                and len(out[-1]) > 40
                and not _SENT_TERMINATOR.search(out[-1])
                and re.match(r"^[а-яёa-z]", ln)
                and not _RE_PAGE_NUM.match(ln)):
            out[-1] = out[-1].rstrip() + " " + ln.lstrip()
        else:
            out.append(ln)
    t = "\n".join(out)
    t = "\n".join(l for l in t.split("\n") if not _RE_PAGE_NUM.match(l))
    t = _RE_GARBAGE.sub("", t)
    t = _RE_MULTISPACE.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ============================ Нормализация дат ============================

_MONTH_MAP = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "мая": 5,
    "июн": 6, "июл": 7, "август": 8, "сентябр": 9, "октябр": 10,
    "ноябр": 11, "декабр": 12,
}

def _normalize_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip().rstrip("г.").strip()
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", s)
    if m:
        return f"{int(m.group(1)):02d}.{int(m.group(2)):02d}.{m.group(3)}"
    m = re.match(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", s, re.I)
    if m:
        mon_word = m.group(2).lower()
        for prefix, num in _MONTH_MAP.items():
            if mon_word.startswith(prefix):
                return f"{int(m.group(1)):02d}.{num:02d}.{m.group(3)}"
    return s


# ============================ Метаданные: правила ============================

_RE_TYPE_RULES = [
    ("Кодекс",            re.compile(r"\bкодекс\w*\b", re.I)),
    ("Федеральный закон", re.compile(r"федеральн\w+\s+закон", re.I)),
    ("Письмо комплаенс",  re.compile(r"письм\w*\s+компл[аи]енс|компл[аи]енс[- ]?служб", re.I)),
    ("Служебная записка", re.compile(r"\bслужебн\w+\s+записк", re.I)),
    ("Доп. соглашение",   re.compile(r"дополнительн\w+\s+соглашени", re.I)),
    ("Регламент",         re.compile(r"\bрегламент\w*\b", re.I)),
    ("Положение",         re.compile(r"\bположени\w*\b", re.I)),
    ("Инструкция",        re.compile(r"\bинструкци\w*\b", re.I)),
    ("Политика",          re.compile(r"\bполитик\w*\b", re.I)),
    ("Порядок",           re.compile(r"\bпорядок\b", re.I)),
    ("Методика",          re.compile(r"\bметодик\w*\b", re.I)),
    ("Закон",             re.compile(r"\bзакон\w*\b", re.I)),
    ("Договор",           re.compile(r"\bдоговор\w*\b", re.I)),
    ("Соглашение",        re.compile(r"\bсоглашени\w*\b", re.I)),
    ("Приказ",            re.compile(r"\bприказ\w*\b", re.I)),
    ("Распоряжение",      re.compile(r"\bраспоряжени\w*\b", re.I)),
]

# Служебная записка идёт ДО Доп. соглашения, чтобы "записка об архиве договоров" не стала "Доп. соглашением"
_SPECIFIC_TYPES = {
    "Кодекс", "Федеральный закон", "Письмо комплаенс", 
    "Служебная записка", "Доп. соглашение",
}

_RE_NUMBER = re.compile(r"(?:№|N|номер)\s*([A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9\-/]*)", re.I)

_MONTHS = (r"январ\w+|феврал\w+|март\w+|апрел\w+|ма[йя]|июн\w+|июл\w+|"
           r"август\w+|сентябр\w+|октябр\w+|ноябр\w+|декабр\w+")

# Ищем дату в шапке, но игнорируем "изменена с..." и "ред. от..."
_RE_DATE_DOC_HEADER = re.compile(
    rf"(?:^|[,.]\s+)(\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}\s*(?:г\.?)?)|"
    r"от\s+(\d{1,2}\.\d{1,2}\.\d{4})"
)

_RE_DATE_WORD = re.compile(
    rf"от\s+(\d{{1,2}}[.\s]+(?:{_MONTHS})\.?\s+\d{{4}}\s*(?:г\.?)?)", re.I
)
_RE_DATE_DOT = re.compile(r"от\s+(\d{1,2}\.\d{1,2}\.\d{4})")
_RE_DATE_DASH = re.compile(
    rf"—\s*(\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}\s*(?:г\.?)?)", re.I
)
_RE_DATE_IN_FILENAME = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{4})")

_RE_REVISION_ACTUAL = re.compile(
    r"(?:в\s+редакции|ред\.)\s*(?:от\s+)?"
    r"(\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}\s+(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\s+\d{4})",
    re.I,
)
_RE_FUTURE_CHANGE = re.compile(
    r"изменен[а-я]*\s+с\s+"
    r"(\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}\s+(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\s+\d{4})",
    re.I,
)
_RE_OUTDATED = re.compile(
    r"(?:документ|закон|норма|статья|пункт|положение|редакция)\s+"
    r"(?:утратил[а-я]*\s+силу|не\s+действует|отменен[а-я]*|прекратил[а-я]*\s+действие)|"
    r"\(утратил[а-я]*\s+силу\)|"
    r"\(недействующ[а-я]*\s+редакция\)|"
    r"недействующ[а-я]*\s+редакция|"
    r"утратил[а-я]*\s+силу\s+с\s+\d",
    re.I,
)

_RE_ANY_DATE = re.compile(
    r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b|"
    rf"\b(\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}})\b",
    re.I,
)

_BAD_TITLE = re.compile(
    r"^(утвержден|утверждён|приказом|распоряжением|от\s+\d|=====|page\s+\d|"
    r"оглавление|содержание|полное\s+наименование|"
    r"источник|url|дата\s+выгрузки|"
    r"раздел\s|глава\s|параграф\s|п\.\s*\d|"
    r"служебн\w+\s+записк|письм\w*\s+компл|^письмо\b|^записка\b)",
    re.I,
)


# ============================ Метаданные: извлечение ============================

def _extract_title(text: str, filename: str) -> str:
    lines = text.split("\n")

    for ln in lines[:50]:
        m = re.match(r"^\s*(Статья\s+[\dIVXLCDMА-ЯA-Z]+\.\s+[^\n]{3,200})", ln, re.I)
        if m:
            return m.group(1).strip()

    for ln in lines[:50]:
        m = re.match(r"^\s*Тема\s*:\s*(.+)$", ln, re.I)
        if m and 5 <= len(m.group(1).strip()) <= 200:
            return m.group(1).strip()

    for ln in lines[:50]:
        s = ln.strip()
        if not s or len(s) < 8 or len(s) > 200:
            continue
        if _BAD_TITLE.match(s):
            continue
        if sum(ch.isalpha() for ch in s) < len(s) * 0.5:
            continue
        return s

    return Path(filename).stem.replace("_", " ").strip()


def _detect_type(text: str, filename: str, debug: bool = False) -> tuple[str, str]:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").lower()
    head_lines = [l for l in text.split("\n") if l.strip()][:5]
    head = "\n".join(head_lines)[:1000]

    if debug:
        print(f"   [detect_type] stem={stem!r}")
        print(f"   [detect_type] head={head[:120]!r}")

    for name, rx in _RE_TYPE_RULES:
        if name in _SPECIFIC_TYPES and rx.search(head):
            if debug: print(f"   [detect_type] → {name} (header)")
            return name, "header"

    for name, rx in _RE_TYPE_RULES:
        if rx.search(stem):
            if debug: print(f"   [detect_type] → {name} (filename)")
            return name, "filename"

    for name, rx in _RE_TYPE_RULES:
        if rx.search(head):
            if debug: print(f"   [detect_type] → {name} (header fallback)")
            return name, "header"

    if debug: print("   [detect_type] → Неизвестно (none)")
    return "Неизвестно", "none"


def extract_metadata(text: str, filename: str) -> dict:
    header     = text[:3000]
    head_short  = "\n".join(text.split("\n")[:3])
    today = datetime.now().strftime("%d.%m.%Y")

    doc_type, type_source = _detect_type(text, filename)

    m = _RE_NUMBER.search(header)
    number = m.group(1).strip(" .,;") if m else None
    if number and len(number) < 2:
        number = None

    # --- дата документа ---
    date = None
    # 1. Для законов сначала ищем в первой строке (шапка), игнорируя поправки
    first_line = text.split("\n")[0]
    if doc_type in ("Кодекс", "Федеральный закон", "Закон") and not filename.startswith("ТК_ст"):
        # Ищем в первой строке, но не в "изменена с"
        clean_first = re.sub(r"изменен[а-я]*\s+с\s+\d+.*?$", "", first_line, flags=re.I)
        dm = re.search(r"от\s+(\d{1,2}\.\d{1,2}\.\d{4})", clean_first)
        if dm:
            date = _normalize_date(dm.group(1))
    
    if date is None:
        for rx in (_RE_DATE_WORD, _RE_DATE_DOT, _RE_DATE_DASH):
            dm = rx.search(text[:5000])
            if dm:
                # Проверяем, что это не "изменена с"
                line = text[:dm.start()].split("\n")[-1]
                if "изменен" in line.lower() or "ред." in line.lower():
                    continue
                date = _normalize_date(dm.group(1))
                break

    if date is None:
        fm = _RE_DATE_IN_FILENAME.search(filename)
        if fm:
            date = _normalize_date(fm.group(1))

    # --- фактическая редакция ---
    rm = _RE_REVISION_ACTUAL.search(header)
    revision = _normalize_date(rm.group(1)) if rm else None

    if revision and doc_type in ("Письмо комплаенс", "Письмо", "Служебная записка", "Уведомление"):
        if date is None:
            date = revision
        revision = None

    # --- будущие изменения (ТОЛЬКО БУДУЩИЕ) ---
    future_changes = set()
    for x in _RE_FUTURE_CHANGE.findall(header):
        d = _normalize_date(x)
        if d and d > today:
            future_changes.add(d)
    
    # --- признак «утратил силу» ---
    is_outdated = bool(_RE_OUTDATED.search(head_short) or _RE_OUTDATED.search(filename))

    # --- номер из doc_title ---
    if number is None:
        t = _extract_title(text, filename)
        tm = re.search(r"\bN\s*(\d+[-/]\w+)", t, re.I)
        if tm:
            number = tm.group(1)

    raw_dates = [a or b for a, b in _RE_ANY_DATE.findall(header) if a or b]
    dates = sorted({d for d in (_normalize_date(x) for x in raw_dates) if d})

    return {
        "doc_title":      _extract_title(text, filename),
        "doc_type":       doc_type,
        "type_source":    type_source,
        "doc_number":     number,
        "doc_date":       date,
        "revision_date":  revision,
        "future_changes": sorted(list(future_changes)),
        "is_outdated":    is_outdated,
        "dates":          dates,
    }


# ============================ Структура ============================

_RE_HEADING = [
    (1, re.compile(r"^\s*(Раздел\s+[IVXLCDM0-9А-ЯA-Z]+(?:\.[^\n]{0,120})?)\s*$", re.I)),
    (2, re.compile(r"^\s*(Глава\s+[IVXLCDM0-9А-ЯA-Z]+(?:\.[^\n]{0,120})?)\s*$", re.I)),
    (2, re.compile(r"^\s*(\d{1,2}\.\s+[А-ЯЁ][а-яё][^\n]{3,120})\s*$")),
    (3, re.compile(r"^\s*((?:Статья|Параграф)\s+[\dIVXLCDMА-ЯA-Z]+[.\s][^\n]{0,160})\s*$", re.I)),
    (4, re.compile(r"^\s*(п\.?\s*\d+(?:\.\d+)*(?:\.[^\n]{0,120})?)\s*$", re.I)),
    (4, re.compile(r"^\s*(\d{1,2}\.\d{1,3}(?:\.\d{1,3})*(?:\.[^\n]{0,120})?)\s*$")),
]

_RE_ARTICLE_NUM = re.compile(r"(?:Статья|Параграф)\s+([\dIVXLCDMА-ЯA-Z]+)", re.I)
_RE_CLAUSE_NUM_P       = re.compile(r"п\.?\s*(\d+(?:\.\d+)*)", re.I)
_RE_CLAUSE_NUM_DOT     = re.compile(r"^(\d{1,2}\.\d{1,3}(?:\.\d{1,3})*)")
_RE_CLAUSE_NUM_BRACKET = re.compile(r"^(\d{1,2})\)\s+")

_RE_LOOKS_LIKE_DATE        = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}")
_RE_LOOKS_LIKE_NUMBER_ONLY = re.compile(r"^\d{1,2}\.\d{1,3}(?:\.\d{1,3})*\.?\s*(?:[а-яё]*\.?)?\s*$", re.I)
_RE_LOOKS_LIKE_MONEY       = re.compile(r"^\d{1,3}(?:[ .]\d{3})+(?:[.,]\d+)?\s*(?:руб|₽|млн|тыс|%)", re.I)


def extract_clause_number(label: str) -> str:
    for rx in (_RE_CLAUSE_NUM_P, _RE_CLAUSE_NUM_DOT, _RE_CLAUSE_NUM_BRACKET):
        m = rx.search(label)
        if m:
            return m.group(1)
    return ""


def _classify_line(line: str) -> tuple[int, str, str, str]:
    s = line.strip()
    if not s or len(s) > 250:
        return (0, "", "", "")

    if _RE_LOOKS_LIKE_DATE.match(s) or _RE_LOOKS_LIKE_MONEY.match(s):
        return (0, "", "", "")

    for level, rx in _RE_HEADING:
        m = rx.match(s)
        if not m:
            continue
        label = m.group(1).strip()
        if not label:
            return (0, "", "", "")

        if level == 4 and _RE_LOOKS_LIKE_NUMBER_ONLY.match(label):
            continue

        art = ""
        cl = ""
        if level == 3:
            am = _RE_ARTICLE_NUM.search(label)
            if am: art = am.group(1)
        elif level == 4:
            cl = extract_clause_number(label)

        return (level, label, art, cl)

    return (0, "", "", "")


def parse_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    pos = 0
    current_article = ""
    for raw_line in text.split("\n"):
        line_start = pos
        line_end = pos + len(raw_line)
        pos = line_end + 1

        stripped = raw_line.strip()
        if not stripped:
            continue

        level, label, art, cl = _classify_line(raw_line)
        if level:
            if level <= 2:
                current_article = ""
            elif level == 3 and art:
                current_article = art

            blocks.append(Block(
                kind="heading", text=stripped,
                start=line_start, end=line_end,
                level=level, label=label,
                article=current_article if level >= 3 else "",
                clause=cl,
            ))
        else:
            blocks.append(Block(
                kind="paragraph", text=stripped,
                start=line_start, end=line_end,
                article=current_article,
            ))
    return blocks


# ============================ Сборка Doc ============================

def parse_document(doc_id: str, filename: str, raw_bytes: bytes) -> Doc:
    text_raw, enc = read_txt_bytes(raw_bytes)
    text = clean_text(text_raw)
    meta = extract_metadata(text, filename)
    blocks = parse_blocks(text)

    warnings: list[str] = []
    if "iso-8859-1" in enc:
        warnings.append(f"fallback кодировка: {enc} — возможны кракозябры")
    if meta["doc_type"] == "Неизвестно":
        warnings.append("тип не определён — добавьте правило в _RE_TYPE_RULES")
    if not meta["doc_title"]:
        warnings.append("title пустой — проверьте шапку документа")
    if meta["is_outdated"]:
        warnings.append("документ помечен как утративший силу")

    return Doc(
        doc_id=doc_id,
        file=filename,
        title=meta["doc_title"],
        doc_type=meta["doc_type"],
        text=text,
        blocks=blocks,
        number=meta["doc_number"],
        date=meta["doc_date"],
        revision_date=meta["revision_date"],
        future_changes=meta["future_changes"],
        is_outdated=meta["is_outdated"],
        dates=meta["dates"],
        type_source=meta["type_source"],
        encoding=enc,
        warnings=warnings,
    )


def load_docs_from_zip(zip_path: str | Path) -> list[Doc]:
    docs: list[Doc] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".txt"):
                continue
            if name.startswith("__MACOSX/") or "/." in name:
                continue
            data = zf.read(name)
            doc_id = Path(name).stem
            docs.append(parse_document(doc_id, name, data))
    return docs