# Hybrid PII Detection Pipeline

Детектор персональных данных (PII) на русском языке — учебный проект курса NLP, Spring 2026.

## Задача

Обнаружение персональных данных в русскоязычных текстах. PII делится на два типа:

- **Контекстно-зависимые** (имена, адреса, CVC) — их значение определяется контекстом
- **Структурированные** (телефоны, email, ИНН, СНИЛС и т.д.) — имеют фиксированный формат

## Подход: Hybrid Pipeline

```
Текст
  ├── ML-модель (DeepPavlov/rubert-base-cased) → NAME, ADDRESS, CVC
  └── Regex-детектор              → EMAIL, PHONE, INN, SNILS, OGRN, ...
           ↓
     Объединение (ML приоритетнее при перекрытии)
           ↓
     Список PII-спанов { start, end, label, text, source }
```

ML-модель решает задачу классификации токенов (Token Classification):

- `AutoModelForTokenClassification`, 7 меток
- BIO-схема: `O`, `B-NAME`, `I-NAME`, `B-ADDRESS`, `I-ADDRESS`, `B-CVC`, `I-CVC`
- Выравнивание токенов по символьным смещениям (`return_offsets_mapping=True`)

## Данные

### Обучающий датасет


| Датасет                                                                                                                        | Роль                     | Размер                |
| ------------------------------------------------------------------------------------------------------------------------------ | ------------------------ | --------------------- |
| [scanpatch/pii-ner-corpus-synthetic-controlled](https://huggingface.co/datasets/scanpatch/pii-ner-corpus-synthetic-controlled) | Обучение (NAME, ADDRESS) | 4 786 train / 532 val |
| Синтетические CVC (`ner_cvc_1000.json`)                                                                                        | Обучение (CVC)           | 1 000 примеров        |
| [hivetrace/pii-bench](https://huggingface.co/datasets/hivetrace/pii-bench) entity split                                        | Валидация (CVC)          | 77 примеров с CVC     |
| [hivetrace/pii-bench](https://huggingface.co/datasets/hivetrace/pii-bench) domain split                                        | Финальный бенчмарк       | 900 текстов           |


**Итоговые сплиты:**


| Сплит | Источник                       | Примеров | Типы сущностей     |
| ----- | ------------------------------ | -------- | ------------------ |
| Train | scanpatch + synthetic CVC      | 5 786    | NAME, ADDRESS, CVC |
| Val   | scanpatch test + hivetrace CVC | 609      | NAME, ADDRESS, CVC |
| Test  | hivetrace domain split         | 900      | 13 типов           |


**Маппинг scanpatch → ML-лейблы:**


| Scanpatch                                                                                                                                                                                                    | ML-лейбл                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| `name`, `first_name`, `last_name`, `middle_name`, `name_initials`, `nickname`                                                                                                                                | **NAME**                   |
| `address`, `address_city`, `address_street`, `address_house`, `address_district`, `address_region`, `address_country`, `address_postal_code`, `address_apartment`, `address_building`, `address_geolocation` | **ADDRESS**                |
| Остальные (email, mobile_phone, snils, tin, ...)                                                                                                                                                             | игнор — regex их покрывает |


CVC — только из hivetrace entity split и синтетических данных (в scanpatch отсутствует).

### Параметры обучения


| Параметр       | Значение                       |
| -------------- | ------------------------------ |
| Модель         | `DeepPavlov/rubert-base-cased` |
| Эпохи          | 5                              |
| Batch size     | 16                             |
| Learning rate  | 2e-5                           |
| Weight decay   | 0.01                           |
| Warmup ratio   | 0.1                            |
| Max length     | 512                            |
| Метрика выбора | F1 (seqeval strict IOB2)       |
| Seed           | 42                             |


Обучение производилось на Google Colab T4 GPU (~40 минут).

## Результаты

Оценка на `hivetrace/pii-bench` domain split (900 текстов, 13 типов сущностей).  
Метрика: **строгое совпадение спанов** — предсказание засчитывается только если start, end и label совпадают точно.

### Наша модель vs baseline


| Метод                        | F1        |
| ---------------------------- | --------- |
| **Hybrid Pipeline (ours)**   | **0.919** |
| Regex-only FULL *(baseline)* | 0.835     |


> Regex-only FULL — regex с приближёнными паттернами для NAME и ADDRESS (эвристики по заглавным буквам и ключевым словам). Показывает вклад ML-модуля: +8.4 п.п.

### Сравнение с конкурентами (zero-shot, без дообучения)

Конкурирующие модели не покрывают часть российских типов (ОГРН, КПП и т.д.).  
Для честного сравнения используется **augmented F1**: там, где модель не предсказывает ни одного спана данного типа, подставляется F1 regex для этого типа.


| Модель                                                                              | Параметров | Языки      | F1 (aug.) |
| ----------------------------------------------------------------------------------- | ---------- | ---------- | --------- |
| [hivetrace/gliner-guard-omni](https://huggingface.co/hivetrace/gliner-guard-omni)   | 307 M      | EN, RU     | 0.859     |
| [scanpatch/pii-ner-nemotron](https://huggingface.co/scanpatch/pii-ner-nemotron)     | 600 M      | EN, RU, UK | 0.610     |
| [tabularisai/eu-pii-safeguard](https://huggingface.co/tabularisai/eu-pii-safeguard) | 600 M      | 26 EU      | 0.600     |


**Выводы:**

- Наш гибридный пайплайн на RuBERT (~180 M параметров) превосходит все три модели, включая gliner-guard-omni, специально обученный на русских PII.
- Ключевое преимущество: дообучение на in-domain данных + детерминированные regex для структурированных типов.

### Детализация по типам сущностей — все модели


| Entity           | Sup | Hybrid    | Regex FULL | gliner-guard | nemotron  | eu-pii-safe |
| ---------------- | --- | --------- | ---------- | ------------ | --------- | ----------- |
| NAME             | 158 | **0.965** | 0.861      | 0.954        | 0.029     | 0.000       |
| PHONE_NUMBER     | 147 | **0.993** | **0.993**  | 0.957        | 0.865     | 0.841       |
| ADDRESS          | 106 | **0.758** | 0.210      | 0.648        | 0.423     | 0.012       |
| EMAIL            | 103 | 0.995     | 0.995      | 0.990        | **0.995** | 0.152       |
| PASSPORT_NUMBER  | 50  | **0.926** | **0.926**  | 0.922        | 0.000     | 0.000       |
| INN              | 48  | **0.950** | **0.950**  | **0.950**    | 0.040     | 0.036       |
| SNILS            | 27  | **1.000** | **1.000**  | 0.833        | 0.000     | 0.000       |
| TOKEN            | 25  | **0.762** | **0.762**  | 0.571        | 0.000     | 0.571       |
| KPP              | 24  | **1.000** | **1.000**  | 0.941        | 0.000     | 0.000       |
| OGRN             | 23  | **1.000** | **1.000**  | 0.868        | 0.000     | 0.000       |
| BANK_CARD_NUMBER | 22  | **1.000** | **1.000**  | 0.657        | 0.000     | 0.000       |
| OGRNIP           | 17  | **0.938** | **0.938**  | 0.100        | 0.000     | 0.000       |
| CVC*             | 7   | 0.343     | **0.444**  | 0.105        | 0.000     | 0.000       |
| **OVERALL**      | 623 | **0.919** | 0.835      | 0.845†       | 0.610†    | 0.600†      |


>  CVC: только ~7 примеров в тест-сете — оценка ненадёжна. Hybrid: P=0.214, R=0.857.  
> † Augmented F1 конкурентов: для типов без предсказаний подставляется Regex F1. Raw: gliner 0.845, nemotron 0.463, eu-pii 0.299.  
> ADDRESS проседает из-за domain shift: scanpatch — короткие предложения, hivetrace domain — многоходовые тексты с длинными адресными блоками.

## Структура проекта

```
├── data.py             # Загрузка датасетов, конвертация в BIO, токенизация
├── train.py            # Обучение rubert-base-cased (NAME/ADDRESS/CVC)
├── regex_detector.py   # Regex-детектор структурированных PII + валидация
├── inference.py        # ML-инференс, гибридный пайплайн detect_pii()
├── evaluate.py         # Строгое span matching, per-label метрики, вывод таблиц
├── compare_baseline.py # Сравнение с конкурентами (zero-shot)
├── main.ipynb          # Точка входа: загрузка модели → демо → оценка → сравнение
└── report.tex          # Отчёт в формате курса
```

## Быстрый старт

```bash
pip install torch transformers datasets seqeval scikit-learn sentencepiece matplotlib accelerate gliner gliner2
```

```python
# Загрузка обученной модели
from transformers import AutoModelForTokenClassification, AutoTokenizer
model     = AutoModelForTokenClassification.from_pretrained("./pii-ner-model")
tokenizer = AutoTokenizer.from_pretrained("./pii-ner-model")

# Инференс
from inference import detect_pii
spans = detect_pii("Иван Петров, тел. +79261234567, email ivan@example.com", model, tokenizer)
for s in spans:
    print(f"[{s['source']:5}] {s['label']:<20} '{s['text']}'")

# Оценка на domain split
from datasets import load_dataset
from evaluate import run_evaluation, display_results
domain_df = load_dataset("hivetrace/pii-bench")["domain"].to_pandas()
results = run_evaluation(domain_df, model, tokenizer)
display_results(results)
```

## Regex-детектор


| Тип              | Паттерн                    | Валидация                    |
| ---------------- | -------------------------- | ---------------------------- |
| EMAIL            | стандартный                | —                            |
| PHONE_NUMBER     | +7/8 + 10 цифр             | —                            |
| BANK_CARD_NUMBER | 4×4 цифры                  | алгоритм Луна                |
| SNILS            | XXX-XXX-XXX XX             | длина 11                     |
| INN              | —                          | длина 10 или 12              |
| KPP              | XXXXAAZZZ                  | формат                       |
| OGRN             | —                          | длина 13 + контрольная сумма |
| OGRNIP           | —                          | длина 15 + контрольная сумма |
| PASSPORT_NUMBER  | XX XX XXXXXX               | —                            |
| TOKEN            | alphanumeric ≥ 20 символов | есть буквы и цифры           |


ОГРН и ОГРНИП детектируются как с ключевым словом (`ОГРН: 1234567890123`), так и standalone (fallback).