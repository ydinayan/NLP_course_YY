"""
regex_detector.py — rule-based детектор структурированных PII.
"""
import re


# =========================================================================== #
#  Validation helpers                                                          #
# =========================================================================== #

def normalize_digits(text: str) -> str:
    """Убирает пробелы и дефисы — для проверки длины и контрольных сумм."""
    return re.sub(r'[\s\-]', '', text)


def validate_ogrn(text: str) -> bool:
    """ОГРН: 13 цифр. Контрольная цифра = (первые 12) % 11, если ≥10 то % 10."""
    d = normalize_digits(text)
    if len(d) != 13 or not d.isdigit():
        return False
    remainder = int(d[:-1]) % 11
    if remainder >= 10:
        remainder %= 10
    return int(d[-1]) == remainder


def validate_ogrnip(text: str) -> bool:
    """ОГРНИП: 15 цифр. Контрольная цифра = (первые 14) % 13, если ≥10 то % 10."""
    d = normalize_digits(text)
    if len(d) != 15 or not d.isdigit():
        return False
    remainder = int(d[:-1]) % 13
    if remainder >= 10:
        remainder %= 10
    return int(d[-1]) == remainder


def validate_card_luhn(text: str) -> bool:
    """Алгоритм Луна для номера банковской карты."""
    d = normalize_digits(text)
    if not d.isdigit() or len(d) != 16:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# =========================================================================== #
#  Patterns                                                                    #
# =========================================================================== #

# Простые паттерны: вся группа совпадения = искомый текст
REGEX_PATTERNS: dict[str, str] = {
    "EMAIL":
        r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}',

    "PHONE_NUMBER":
        r'(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}',

    "BANK_CARD_NUMBER":
        r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',

    "SNILS":
        r'\b\d{3}[\-\s]?\d{3}[\-\s]?\d{3}[\-\s]?\d{2}\b',

    "INN":
        r'\b(?:\d{12}|\d{10})\b',

    "KPP":
        r'\b\d{4}[0-9A-Z]{2}\d{3}\b',

    "PASSPORT_NUMBER":
        r'\b\d{2}\s\d{2}\s\d{6}\b|\b\d{4}\s\d{6}\b',

    "TOKEN":
        r'(?<![A-Za-z0-9_\-\.])[A-Za-z0-9_\-\.]{20,}(?![A-Za-z0-9_\-\.])',
}

# Контекстные паттерны для OGRN / OGRNIP.
# Группа 1 — сам код; группа 0 — полное совпадение включая ключевое слово.
# CVC определяется ML-моделью, здесь не детектируется.
_CTX_PATTERNS: dict[str, re.Pattern] = {
    # ОГРНИП перед ОГРН — чтобы 15-значный не попал под 13-значный
    "OGRNIP": re.compile(
        r'(?:ОГРНИП|OGRNIP)[:\s]*([\d\s\-]{15,20})',
        re.IGNORECASE,
    ),
    "OGRN": re.compile(
        r'(?:ОГРН|OGRN)[:\s]*([\d\s\-]{13,18})',
        re.IGNORECASE,
    ),
}

# Standalone паттерны (без ключевых слов), как fallback
_STANDALONE: dict[str, re.Pattern] = {
    "OGRNIP": re.compile(r'\b\d{15}\b'),
    "OGRN":   re.compile(r'\b\d{13}\b'),
}

_VALIDATORS = {
    "BANK_CARD_NUMBER": lambda t: len(normalize_digits(t)) == 16,
    "SNILS":            lambda t: len(normalize_digits(t)) == 11,
    "INN":              lambda t: len(normalize_digits(t)) in (10, 12),
    "OGRN":             validate_ogrn,
    "OGRNIP":           validate_ogrnip,
    "TOKEN":            lambda t: bool(re.search(r'[A-Za-z]', t) and re.search(r'\d', t)),
}

_COMPILED: dict[str, re.Pattern] = {
    label: re.compile(pat) for label, pat in REGEX_PATTERNS.items()
}


# =========================================================================== #
#  Detection                                                                   #
# =========================================================================== #

def _hit(text: str, label: str, start: int, end: int) -> dict:
    return {"start": start, "end": end, "text": text[start:end],
            "label": label, "source": "regex"}


def _simple_hits(text: str) -> list[dict]:
    """Паттерны без контекста."""
    hits: list[dict] = []
    for label, pattern in _COMPILED.items():
        validator = _VALIDATORS.get(label)
        for m in pattern.finditer(text):
            raw = m.group()
            if validator and not validator(raw):
                continue
            hits.append(_hit(text, label, m.start(), m.end()))
    return hits


def _contextual_hits(text: str) -> list[dict]:
    """Паттерны с ключевыми словами; позиция span — только сам код (group 1)."""
    hits: list[dict] = []

    for label, pattern in _CTX_PATTERNS.items():
        validator = _VALIDATORS.get(label)
        for m in pattern.finditer(text):
            raw = m.group(1)
            stripped = raw.strip()
            if not stripped:
                continue
            if validator and not validator(stripped):
                continue
            # Находим точную позицию stripped внутри group(1)
            offset = m.start(1) + raw.index(stripped)
            hits.append(_hit(text, label, offset, offset + len(stripped)))

    return hits


def _standalone_hits(text: str, occupied: set[int]) -> list[dict]:
    """Fallback standalone для OGRN/OGRNIP там, где нет контекстного совпадения."""
    hits: list[dict] = []
    for label, pattern in _STANDALONE.items():
        validator = _VALIDATORS.get(label)
        for m in pattern.finditer(text):
            if validator and not validator(m.group()):
                continue
            # Пропускаем, если позиция уже занята контекстным совпадением
            if set(range(m.start(), m.end())) & occupied:
                continue
            hits.append(_hit(text, label, m.start(), m.end()))
    return hits


def detect_regex_pii(text: str) -> list[dict]:
    """
    Детектирует структурированные PII через regex + правила.

    Возвращает список словарей:
        { start, end, text, label, source="regex" }

    Приоритеты при пересечении:
      1. CVC/OGRN/OGRNIP с контекстом
      2. Простые паттерны
      3. Standalone OGRN/OGRNIP (fallback)
    """
    ctx  = _contextual_hits(text)
    simp = _simple_hits(text)

    # Объединяем контекстные + простые, сортируем по позиции
    all_hits = sorted(ctx + simp, key=lambda h: h["start"])

    # Жадное удаление перекрытий
    result:   list[dict] = []
    occupied: set[int]   = set()
    for h in all_hits:
        chars = set(range(h["start"], h["end"]))
        if not chars & occupied:
            result.append(h)
            occupied |= chars

    # Standalone OGRN/OGRNIP туда, где нет совпадений
    for h in sorted(_standalone_hits(text, occupied), key=lambda h: h["start"]):
        chars = set(range(h["start"], h["end"]))
        if not chars & occupied:
            result.append(h)
            occupied |= chars

    return sorted(result, key=lambda h: h["start"])
