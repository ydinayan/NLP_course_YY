"""
inference.py — ML-инференс и гибридный пайплайн.
"""
import torch
from regex_detector import detect_regex_pii


# --------------------------------------------------------------------------- #
#  ML-детектор (NAME / ADDRESS / CVC)                                         #
# --------------------------------------------------------------------------- #

def detect_ml_pii(text: str, model, tokenizer) -> list[dict]:
    """
    Прогоняет текст через обученную модель NER.

    Возвращает список словарей:
        { start, end, text, label, source="ml", score }
    """
    enc     = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
    offsets = enc.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        logits = model(**enc).logits

    probs   = torch.softmax(logits[0], dim=-1)
    pred_ids = logits.argmax(-1)[0].tolist()

    spans: list[dict] = []
    cur_label: str | None = None
    cur_start = cur_end = 0
    cur_score = 0.0

    for idx, ((tok_start, tok_end), pid) in enumerate(zip(offsets, pred_ids)):
        if tok_start == tok_end:          # special token
            if cur_label is not None:
                spans.append(_span(text, cur_start, cur_end, cur_label, cur_score))
                cur_label = None
            continue

        tag   = model.config.id2label[pid]
        score = float(probs[idx][pid])

        if tag.startswith("B-"):
            if cur_label is not None:
                spans.append(_span(text, cur_start, cur_end, cur_label, cur_score))
            cur_label = tag[2:]
            cur_start = tok_start
            cur_end   = tok_end
            cur_score = score
        elif tag.startswith("I-") and cur_label == tag[2:]:
            cur_end   = tok_end
            cur_score = min(cur_score, score)   # консервативная оценка
        else:
            if cur_label is not None:
                spans.append(_span(text, cur_start, cur_end, cur_label, cur_score))
                cur_label = None

    if cur_label is not None:
        spans.append(_span(text, cur_start, cur_end, cur_label, cur_score))

    return spans


def _span(text: str, start: int, end: int, label: str, score: float) -> dict:
    return {
        "start":  start,
        "end":    end,
        "text":   text[start:end],
        "label":  label,
        "source": "ml",
        "score":  round(score, 4),
    }


# --------------------------------------------------------------------------- #
#  Гибридный пайплайн                                                         #
# --------------------------------------------------------------------------- #

def detect_pii(text: str, model, tokenizer) -> list[dict]:
    """
    Гибридный детектор:
      - regex → структурированные PII (EMAIL, PHONE, BANK_CARD, INN, …)
      - ML    → контекстные PII (NAME, ADDRESS, CVC)

    Приоритет: regex > ML при пересечении спанов.
    """
    regex_spans = detect_regex_pii(text)
    ml_spans    = detect_ml_pii(text, model, tokenizer)

    occupied: set[int] = set()
    result: list[dict] = []

    for s in regex_spans:
        result.append(s)
        occupied |= set(range(s["start"], s["end"]))

    for s in ml_spans:
        chars = set(range(s["start"], s["end"]))
        if not chars & occupied:
            result.append(s)
            occupied |= chars

    return sorted(result, key=lambda x: x["start"])
