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
     Объединение с разрешением конфликтов (regex приоритетнее)
           ↓
     Список PII-спанов { start, end, label, text, source }
```

ML-модель решает задачу классификации токенов (Token Classification Head):
- `AutoModelForTokenClassification` с `num_labels=7`, `id2label`, `label2id`
- BIO-метки: `O`, `B-NAME`, `I-NAME`, `B-ADDRESS`, `I-ADDRESS`, `B-CVC`, `I-CVC`
- Выравнивание токенов по символьным смещениям (`return_offsets_mapping=True`)

## Данные

### Обучающий датасет: scanpatch + hivetrace (CVC)

| Датасет | Роль | Размер |
|---|---|---|
| [scanpatch/pii-ner-corpus-synthetic-controlled](https://huggingface.co/datasets/scanpatch/pii-ner-corpus-synthetic-controlled) | Обучение (NAME, ADDRESS) | 4786 train / 532 test |
| [hivetrace/pii-bench](https://huggingface.co/datasets/hivetrace/pii-bench) entity split | Обучение (CVC) | ~77 примеров с CVC |
| [hivetrace/pii-bench](https://huggingface.co/datasets/hivetrace/pii-bench) domain split | Финальный бенчмарк | 900 текстов |

**Стратегия обучения:**

1. Scanpatch (4786 примеров) — обучение на NAME и ADDRESS. Формат: параллельные массивы `entity_starts`, `entity_ends`, `entity_labels` (27 типов сущностей → маппинг в NAME/ADDRESS).
2. Hivetrace entity split — дополнительно берём только строки с CVC (77 примеров).
3. Объединяем: `combined_train = scanpatch_train + hivetrace_cvc`.
4. Валидация: scanpatch test split (532 примера).
5. Финальный тест: hivetrace domain split (900 текстов) — не используется при обучении.

Scanpatch содержит 27 типов сущностей. Маппинг в ML-лейблы:

| Scanpatch | ML-лейбл |
|---|---|
| `name`, `first_name`, `last_name`, `middle_name`, `name_initials`, `nickname` | **NAME** |
| `address`, `address_city`, `address_street`, `address_house`, `address_district`, `address_region`, `address_country`, `address_postal_code`, `address_apartment`, `address_building`, `address_geolocation` | **ADDRESS** |
| Остальные (email, mobile_phone, snils, tin, ...) | игнор — regex их покрывает |

CVC — только из hivetrace entity split (в scanpatch отсутствует).

### Параметры обучения

| Параметр | Значение |
|---|---|
| Модель | `DeepPavlov/rubert-base-cased` |
| Эпохи | 5 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Метрика выбора | F1 (seqeval strict IOB2) |

## Результаты на hivetrace/pii-bench (domain split, 900 текстов)

### Наша модель

| Режим | Precision | Recall | F1 |
|---|---|---|---|
| ML-only (NAME + ADDRESS + CVC) | 0.789 | 0.848 | 0.818 |
| Regex-only (структурированные) | 0.977 | 0.908 | 0.941 |
| **Hybrid pipeline** | **0.912** | **0.878** | **0.895** |

Детализация по типам (hybrid):

| Entity | F1 | Source |
|---|---|---|
| NAME | 0.946 | ML |
| PHONE_NUMBER | 0.983 | Regex |
| ADDRESS | 0.688 | ML |
| EMAIL | 0.995 | Regex |
| PASSPORT_NUMBER | 0.926 | Regex |
| INN | 0.950 | Regex |
| SNILS | 1.000 | Regex |
| KPP | 1.000 | Regex |
| OGRN | 1.000 | Regex |
| OGRNIP | 0.938 | Regex |
| BANK_CARD_NUMBER | 1.000 | Regex |
| CVC | — | ML |
| TOKEN | 0.762 | Regex |

> ADDRESS проседает из-за domain shift: scanpatch — короткие предложения, hivetrace domain — многоходовые JSON-диалоги.

## Сравнение с baseline-моделями

Все baseline-модели прогонялись **без дообучения (zero-shot)** на hivetrace/pii-bench domain split.

| Модель | Подход | Precision | Recall | F1 |
|---|---|---|---|---|
| **Наш Hybrid Pipeline** | Fine-tuned rubert-base-cased + Regex | **0.912** | **0.878** | **0.895** |
| Regex-only | Только правила (без ML) | 0.977 | 0.908 | 0.941 |
| [hivetrace/gliner-guard-omni](https://huggingface.co/hivetrace/gliner-guard-omni) | Zero-shot NER | TBD | TBD | TBD |
| [tabularisai/eu-pii-safeguard](https://huggingface.co/tabularisai/eu-pii-safeguard) | Zero-shot NER (EU PII) | TBD | TBD | TBD |
| [scanpatch/pii-ner-nemotron](https://huggingface.co/scanpatch/pii-ner-nemotron) | Zero-shot GLiNER | TBD | TBD | TBD |

**Ключевые наблюдения:**
- Regex-only даёт F1=0.941 на структурированных PII, но не находит NAME/ADDRESS/CVC.
- Наш hybrid превосходит чистый regex за счёт ML-детекции контекстных сущностей.

## Структура проекта

```
├── data.py            # Загрузка датасетов, конвертация в BIO, токенизация
├── train.py           # Обучение rubert-base-cased (NAME/ADDRESS/CVC)
├── regex_detector.py  # Regex-детектор структурированных PII + валидация
├── inference.py       # ML-инференс, гибридный пайплайн detect_pii()
├── evaluate.py        # Строгое span matching, per-label метрики, вывод таблиц
├── compare_baseline.py # Сравнение с baseline-моделями (zero-shot)
└── main.ipynb         # Точка входа: обучение → демо → оценка → baselines
```

## Быстрый старт

```bash
pip install torch transformers datasets seqeval scikit-learn sentencepiece matplotlib accelerate gliner
```

```python
# Обучение
from train import train_model
model, tokenizer, domain_df = train_model()

# Инференс
from inference import detect_pii
spans = detect_pii("Иван Петров, тел. +79261234567, email ivan@example.com", model, tokenizer)

# Оценка
from evaluate import run_evaluation, display_results
results = run_evaluation(domain_df, model, tokenizer)
display_results(results)
```

## Regex-детектор

CVC определяется ML-моделью. Regex покрывает только структурированные PII:

| Тип | Паттерн | Валидация |
|---|---|---|
| EMAIL | стандартный | — |
| PHONE_NUMBER | +7/8 + 10 цифр | — |
| BANK_CARD_NUMBER | 4×4 цифры | длина 16 |
| SNILS | XXX-XXX-XXX XX | длина 11 |
| INN | — | длина 10 или 12 |
| KPP | XXXXAAZZZ | формат |
| OGRN | — | длина 13 + контрольная сумма |
| OGRNIP | — | длина 15 + контрольная сумма |
| PASSPORT_NUMBER | XX XX XXXXXX | — |
| TOKEN | длинная alphanumeric строка ≥20 символов | есть буквы и цифры |

ОГРН и ОГРНИП детектируются как с ключевым словом (`ОГРН: 1234567890123`), так и standalone.
