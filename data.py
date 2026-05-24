"""
data.py — загрузка датасетов, конвертация спанов в BIO-метки, токенизация.

Поддерживает два источника:
  - hivetrace/pii-bench      (load_splits)
  - scanpatch/pii-ner-corpus-synthetic-controlled  (load_splits_scanpatch)
"""
import json
import pathlib
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

ML_LABELS  = {"NAME", "ADDRESS", "CVC"}
LABEL_LIST = ["O", "B-ADDRESS", "I-ADDRESS", "B-CVC", "I-CVC", "B-NAME", "I-NAME"]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for l, i in LABEL2ID.items()}

# Маппинг тонких лейблов scanpatch → наши ML-лейблы
SCANPATCH_TO_ML: dict[str, str] = {
    # NAME-группа
    "name":          "NAME",
    "first_name":    "NAME",
    "last_name":     "NAME",
    "middle_name":   "NAME",
    "name_initials": "NAME",
    "nickname":      "NAME",
    # ADDRESS-группа (родительский span и все подтипы)
    "address":              "ADDRESS",
    "address_city":         "ADDRESS",
    "address_street":       "ADDRESS",
    "address_house":        "ADDRESS",
    "address_district":     "ADDRESS",
    "address_region":       "ADDRESS",
    "address_country":      "ADDRESS",
    "address_postal_code":  "ADDRESS",
    "address_apartment":    "ADDRESS",
    "address_building":     "ADDRESS",
    "address_geolocation":  "ADDRESS",
    # Остальные типы игнорируем: regex их покрывает или они не нужны
    # (email, mobile_phone, snils, tin, ip, date, document_number,
    #  organization, vehicle_number, military_individual_number)
}


def _char_labels(text: str, entities: list, ml_only: bool = True) -> list:
    """
    Создаёт массив BIO-меток для каждого символа текста.
    Сортирует сущности от большей к меньшей, чтобы вложенные подтипы
    (например address_city внутри address) не затирали родительский лейбл.
    """
    labels = ["O"] * len(text)
    # Сортируем по убыванию длины — сначала крупные спаны
    for ent in sorted(entities, key=lambda e: -(e["end"] - e["start"])):
        etype = ent.get("type", ent.get("label", ""))
        if ml_only and etype not in ML_LABELS:
            continue
        s, e = ent["start"], ent["end"]
        if s >= len(text) or e > len(text):
            continue
        for i in range(s, e):
            # Не перезаписываем уже размеченные символы
            if labels[i] == "O":
                labels[i] = f"B-{etype}" if i == s else f"I-{etype}"
    return labels


def _make_tokenize_fn(tokenizer):
    """Возвращает функцию токенизации, которую можно передать в Dataset.map."""
    def fn(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding=False,
            return_offsets_mapping=True,
        )
        all_labels = []
        for text, ents, offsets in zip(batch["text"], batch["entities"], enc["offset_mapping"]):
            clabels = _char_labels(str(text), list(ents))
            token_labels = []
            for start, end in offsets:
                if start == end:          # special token ([CLS], [SEP], <pad>)
                    token_labels.append(-100)
                else:
                    token_labels.append(LABEL2ID.get(clabels[start], LABEL2ID["O"]))
            all_labels.append(token_labels)
        enc["labels"] = all_labels
        del enc["offset_mapping"]
        return enc
    return fn


def load_splits(tokenizer_name: str = "xlm-roberta-base"):
    """
    Загружает датасет, разбивает entity split на train/val (80/20),
    возвращает токенизированные датасеты и сырой domain DataFrame для теста.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset("hivetrace/pii-bench")

    entity_df = ds["entity"].to_pandas()
    domain_df = ds["domain"].to_pandas()

    train_df, val_df = train_test_split(entity_df, test_size=0.2, random_state=42)

    def to_hf(df):
        return Dataset.from_list([
            {"text": str(r["text"]), "entities": list(r["entities"])}
            for _, r in df.iterrows()
        ])

    tok_fn = _make_tokenize_fn(tokenizer)
    cols   = ["text", "entities"]

    train_tok = to_hf(train_df).map(tok_fn, batched=True, remove_columns=cols)
    val_tok   = to_hf(val_df).map(tok_fn,   batched=True, remove_columns=cols)

    return tokenizer, train_tok, val_tok, domain_df


def load_splits_scanpatch(tokenizer_name: str = "xlm-roberta-base"):
    """
    Обучение: scanpatch (NAME/ADDRESS) + hivetrace entity split (CVC).
    Тест: hivetrace/pii-bench domain split (900 текстов).

    scanpatch: 4786 train / 532 test, параллельные массивы.
    hivetrace entity split: добавляем только примеры с CVC.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    sp    = load_dataset("scanpatch/pii-ner-corpus-synthetic-controlled")
    bench = load_dataset("hivetrace/pii-bench")
    domain_df = bench["domain"].to_pandas()
    entity_df = bench["entity"].to_pandas()

    def scanpatch_to_entities(row: dict) -> list[dict]:
        """Конвертирует параллельные массивы → список спанов с ML-лейблами."""
        entities = []
        for s, e, lbl in zip(row["entity_starts"], row["entity_ends"], row["entity_labels"]):
            ml_label = SCANPATCH_TO_ML.get(lbl)
            if ml_label:
                entities.append({"start": s, "end": e, "type": ml_label})
        return entities

    def hivetrace_ml_entities(ents: list) -> list[dict]:
        """Оставляет только ML_LABELS сущности из hivetrace."""
        result = []
        for e in ents:
            lbl = e.get("type", e.get("label", ""))
            if lbl in ML_LABELS:
                result.append({"start": e["start"], "end": e["end"], "type": lbl})
        return result

    # Scanpatch train/val (NAME/ADDRESS)
    sp_train = [
        {"text": str(row["text"]), "entities": scanpatch_to_entities(row)}
        for row in sp["train"]
    ]
    sp_val = [
        {"text": str(row["text"]), "entities": scanpatch_to_entities(row)}
        for row in sp["test"]
    ]

    # Hivetrace entity split: только строки с CVC (77 примеров)
    hive_cvc = []
    for _, row in entity_df.iterrows():
        ents = hivetrace_ml_entities(list(row["entities"]))
        if any(e["type"] == "CVC" for e in ents):
            hive_cvc.append({"text": str(row["text"]), "entities": ents})

    # Синтетические CVC примеры из ner_cvc_1000.json
    cvc_path = pathlib.Path("ner_cvc_1000.json")
    if cvc_path.exists():
        with open(cvc_path, encoding="utf-8") as f:
            raw_cvc = json.load(f)
        synthetic_cvc = [
            {"text": row["text"],
             "entities": [{"start": e["start"], "end": e["end"], "type": e["type"]}
                          for e in row["entities"]]}
            for row in raw_cvc
        ]
    else:
        synthetic_cvc = []

    # hivetrace CVC → в валидацию (не в train), synthetic CVC → в train
    combined_train = sp_train + synthetic_cvc
    combined_val   = sp_val   + hive_cvc

    tok_fn = _make_tokenize_fn(tokenizer)
    cols   = ["text", "entities"]

    train_tok = Dataset.from_list(combined_train).map(tok_fn, batched=True, remove_columns=cols)
    val_tok   = Dataset.from_list(combined_val).map(tok_fn,   batched=True, remove_columns=cols)

    print(f"Train: {len(train_tok)} (scanpatch: {len(sp_train)}, synthetic CVC: {len(synthetic_cvc)})")
    print(f"Val: {len(val_tok)} (scanpatch test: {len(sp_val)}, hivetrace CVC: {len(hive_cvc)})")
    print(f"Benchmark (domain): {len(domain_df)}")
    return tokenizer, train_tok, val_tok, domain_df
