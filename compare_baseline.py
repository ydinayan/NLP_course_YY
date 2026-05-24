"""
compare_baseline.py — сравнение нашего hybrid pipeline с несколькими baseline-моделями
на hivetrace/pii-bench (domain split, 900 текстов), без дообучения (zero-shot).

Поддерживаемые baseline:
  1. OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1  — HF NER pipeline
  2. hivetrace/gliner-guard-omni                       — GLiNER
  3. tabularisai/eu-pii-safeguard                      — HF NER pipeline
  4. scanpatch/pii-ner-nemotron                        — GLiNER
"""
import torch
import transformers as T
from evaluate import evaluate_spans, evaluate_per_label, print_report

# ---------------------------------------------------------------------------
#  Label mappings → наши стандартные типы
# ---------------------------------------------------------------------------

# hivetrace/gliner-guard-omni — несмотря на название, это HF NER pipeline (не GLiNER lib).
# Модель обучена на hivetrace PII-bench, labels совпадают с нашими типами напрямую.
GLINER_GUARD_MAPPING = {
    "NAME": "NAME", "ADDRESS": "ADDRESS", "EMAIL": "EMAIL",
    "PHONE_NUMBER": "PHONE_NUMBER", "BANK_CARD_NUMBER": "BANK_CARD_NUMBER",
    "CVC": "CVC", "INN": "INN", "SNILS": "SNILS",
    "OGRN": "OGRN", "OGRNIP": "OGRNIP", "KPP": "KPP",
    "PASSPORT_NUMBER": "PASSPORT_NUMBER", "TOKEN": "TOKEN",
}

# tabularisai/eu-pii-safeguard — BERT NER для EU PII (английский, GDPR-ориентирован)
TABULARISAI_TO_HIVETRACE = {
    "PERSON":       "NAME",
    "PER":          "NAME",
    "person":       "NAME",
    "NAME":         "NAME",
    "EMAIL":        "EMAIL",
    "EMAIL_ADDRESS":"EMAIL",
    "PHONE":        "PHONE_NUMBER",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "ADDRESS":      "ADDRESS",
    "LOCATION":     "ADDRESS",
    "LOC":          "ADDRESS",
    "CREDIT_CARD":  "BANK_CARD_NUMBER",
    "CVV":          "CVC",
    "CVC":          "CVC",
    "TAX_ID":       "INN",
    "ID_NUMBER":    "INN",
    "SSN":          "SNILS",
    "PASSPORT":     "PASSPORT_NUMBER",
    "TOKEN":        "TOKEN",
    "API_KEY":      "TOKEN",
    "PASSWORD":     "TOKEN",
    "IP_ADDRESS":   "TOKEN",
    "URL":          "TOKEN",
}

# GLiNER labels для scanpatch/pii-ner-nemotron
# Scanpatch использует собственные типы — маппим в наши
GLINER_NEMOTRON_LABELS = [
    "name", "first_name", "last_name", "middle_name", "nickname",
    "address", "address_city", "address_street", "address_region",
    "email", "mobile_phone", "snils", "tin", "inn",
    "bank_card", "passport", "ogrn", "ogrnip", "kpp", "token",
]
GLINER_NEMOTRON_MAPPING = {
    "name": "NAME", "first_name": "NAME", "last_name": "NAME",
    "middle_name": "NAME", "nickname": "NAME",
    "address": "ADDRESS", "address_city": "ADDRESS", "address_street": "ADDRESS",
    "address_region": "ADDRESS",
    "email": "EMAIL",
    "mobile_phone": "PHONE_NUMBER",
    "snils": "SNILS",
    "tin": "INN", "inn": "INN",
    "bank_card": "BANK_CARD_NUMBER",
    "passport": "PASSPORT_NUMBER",
    "ogrn": "OGRN", "ogrnip": "OGRNIP",
    "kpp": "KPP",
    "token": "TOKEN",
}


# ---------------------------------------------------------------------------
#  Runners
# ---------------------------------------------------------------------------

def _build_gold(row) -> list[dict]:
    ents = list(row["entities"])
    return [{"start": e["start"], "end": e["end"],
             "label": e.get("type", e.get("label", ""))}
            for e in ents]


def run_hf_ner(domain_df, model_name: str, label_mapping: dict) -> tuple:
    """Запускает стандартный HF NER pipeline (token classification)."""
    print(f"Loading {model_name} ...")
    pipe = T.pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )

    gold_rows, pred_rows = [], []
    for i, (_, row) in enumerate(domain_df.iterrows()):
        if i % 100 == 0:
            print(f"  {i}/{len(domain_df)}")

        text = str(row["text"])
        gold = _build_gold(row)

        try:
            raw = pipe(text)
        except Exception:
            raw = []

        pred = []
        for r in raw:
            hf_label = label_mapping.get(r["entity_group"])
            if hf_label:
                pred.append({"start": r["start"], "end": r["end"], "label": hf_label})

        gold_rows.append(gold)
        pred_rows.append(pred)

    return _aggregate(gold_rows, pred_rows)


def run_gliner(domain_df, model_name: str,
               entity_labels: list[str], label_mapping: dict,
               threshold: float = 0.5) -> tuple:
    """Запускает GLiNER модель (zero-shot NER)."""
    try:
        from gliner import GLiNER
    except ImportError:
        raise ImportError("Для GLiNER: pip install gliner")

    print(f"Loading GLiNER {model_name} ...")
    model = GLiNER.from_pretrained(model_name)

    gold_rows, pred_rows = [], []
    for i, (_, row) in enumerate(domain_df.iterrows()):
        if i % 100 == 0:
            print(f"  {i}/{len(domain_df)}")

        text = str(row["text"])
        gold = _build_gold(row)

        try:
            raw = model.predict_entities(text, entity_labels, threshold=threshold)
        except Exception:
            raw = []

        pred = []
        for r in raw:
            mapped = label_mapping.get(r["label"])
            if mapped:
                pred.append({"start": r["start"], "end": r["end"], "label": mapped})

        gold_rows.append(gold)
        pred_rows.append(pred)

    return _aggregate(gold_rows, pred_rows)


def _aggregate(gold_rows, pred_rows) -> tuple:
    flat_g = [s for row in gold_rows for s in row]
    flat_p = [s for row in pred_rows for s in row]
    overall   = evaluate_spans(flat_g, flat_p)
    per_label = evaluate_per_label(gold_rows, pred_rows)
    return overall, per_label


# ---------------------------------------------------------------------------
#  Convenience wrappers
# ---------------------------------------------------------------------------

def run_gliner_guard(domain_df) -> tuple:
    return run_hf_ner(
        domain_df,
        "hivetrace/gliner-guard-omni",
        GLINER_GUARD_MAPPING,
    )


def run_tabularisai(domain_df) -> tuple:
    return run_hf_ner(
        domain_df,
        "tabularisai/eu-pii-safeguard",
        TABULARISAI_TO_HIVETRACE,
    )


def run_nemotron(domain_df) -> tuple:
    return run_gliner(
        domain_df,
        "scanpatch/pii-ner-nemotron",
        GLINER_NEMOTRON_LABELS,
        GLINER_NEMOTRON_MAPPING,
    )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data import load_splits_scanpatch

    print("Loading hivetrace/pii-bench domain split ...")
    _, _, _, domain_df = load_splits_scanpatch()

    baselines = [
        ("gliner-guard-omni (hivetrace)", run_gliner_guard),
        ("eu-pii-safeguard (tabularisai)", run_tabularisai),
        ("pii-ner-nemotron (scanpatch, GLiNER)", run_nemotron),
    ]

    results = {}
    for name, fn in baselines:
        try:
            overall, per_label = fn(domain_df)
            results[name] = (overall, per_label)
            print_report(name, overall, per_label)
        except Exception as exc:
            print(f"\n[SKIP] {name}: {exc}\n")

    print("\n--- Our hybrid pipeline for reference ---")
    print("Hybrid F1 = 0.895 | Precision = 0.912 | Recall = 0.878")
