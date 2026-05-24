"""
evaluate.py — строгое сравнение спанов (exact span matching).
"""
from collections import defaultdict
import pandas as pd


def _key(s: dict) -> tuple:
    return (s["start"], s["end"], s["label"])


# --------------------------------------------------------------------------- #
#  Базовые метрики                                                             #
# --------------------------------------------------------------------------- #

def evaluate_spans(gold: list[dict], pred: list[dict]) -> dict:
    """Строгое совпадение: правильно только если start, end и label совпадают."""
    g = {_key(s) for s in gold}
    p = {_key(s) for s in pred}

    tp = len(g & p)
    fp = len(p - g)
    fn = len(g - p)

    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# --------------------------------------------------------------------------- #
#  Per-label метрики                                                           #
# --------------------------------------------------------------------------- #

def evaluate_per_label(gold_rows: list[list[dict]], pred_rows: list[list[dict]]) -> dict:
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for gold, pred in zip(gold_rows, pred_rows):
        g = {_key(s) for s in gold}
        p = {_key(s) for s in pred}
        for k in p - g:
            counts[k[2]]["fp"] += 1
        for k in g - p:
            counts[k[2]]["fn"] += 1
        for k in g & p:
            counts[k[2]]["tp"] += 1

    result = {}
    for label, c in sorted(counts.items()):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        result[label] = {"precision": prec, "recall": rec, "f1": f1,
                         "support": tp + fn}
    return result


# --------------------------------------------------------------------------- #
#  Вывод отчёта                                                               #
# --------------------------------------------------------------------------- #

def print_report(title: str, overall: dict, per_label: dict) -> None:
    w = 22
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")
    print(f"{'Label':<{w}} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}")
    print(f"{'-' * 62}")
    for label, m in sorted(per_label.items(), key=lambda x: -x[1]["support"]):
        print(f"{label:<{w}} {m['precision']:>8.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>8.3f} {m['support']:>8}")
    print(f"{'-' * 62}")
    total = overall["tp"] + overall["fn"]
    print(f"{'OVERALL':<{w}} {overall['precision']:>8.3f} {overall['recall']:>8.3f} "
          f"{overall['f1']:>8.3f} {total:>8}")
    print(f"{'=' * 62}\n")


# --------------------------------------------------------------------------- #
#  Полный цикл оценки                                                         #
# --------------------------------------------------------------------------- #

def run_evaluation(domain_df, model, tokenizer,
                   ml_labels=("NAME", "ADDRESS", "CVC")):
    """
    Оценивает три режима на domain split:
      1. ML-only  (NAME / ADDRESS / CVC)
      2. Regex-only (структурированные PII)
      3. Hybrid   (полный пайплайн)
    """
    from regex_detector import detect_regex_pii
    from inference import detect_ml_pii, detect_pii

    REGEX_LABELS = {"EMAIL", "PHONE_NUMBER", "BANK_CARD_NUMBER",
                    "INN", "KPP", "OGRN", "OGRNIP", "SNILS",
                    "PASSPORT_NUMBER", "TOKEN"}

    gold_ml, pred_ml         = [], []
    gold_regex, pred_regex   = [], []
    gold_hybrid, pred_hybrid = [], []

    for _, row in domain_df.iterrows():
        text  = str(row["text"])
        ents  = list(row["entities"])

        gold_all = [{"start": e["start"], "end": e["end"],
                     "label": e.get("type", e.get("label", ""))}
                    for e in ents]

        # ── ML evaluation ──────────────────────────────────────────────────
        g_ml = [s for s in gold_all if s["label"] in ml_labels]
        p_ml = detect_ml_pii(text, model, tokenizer)
        p_ml = [{"start": s["start"], "end": s["end"], "label": s["label"]}
                for s in p_ml]
        gold_ml.append(g_ml)
        pred_ml.append(p_ml)

        # ── Regex evaluation ────────────────────────────────────────────────
        g_rx = [s for s in gold_all if s["label"] in REGEX_LABELS]
        p_rx = detect_regex_pii(text)
        p_rx = [{"start": s["start"], "end": s["end"], "label": s["label"]}
                for s in p_rx]
        gold_regex.append(g_rx)
        pred_regex.append(p_rx)

        # ── Hybrid evaluation ───────────────────────────────────────────────
        p_hyb = detect_pii(text, model, tokenizer)
        p_hyb = [{"start": s["start"], "end": s["end"], "label": s["label"]}
                 for s in p_hyb]
        gold_hybrid.append(gold_all)
        pred_hybrid.append(p_hyb)

    def _agg(golds, preds):
        flat_g = [s for row in golds for s in row]
        flat_p = [s for row in preds for s in row]
        return evaluate_spans(flat_g, flat_p), evaluate_per_label(golds, preds)

    ov_ml,    pl_ml    = _agg(gold_ml,    pred_ml)
    ov_rx,    pl_rx    = _agg(gold_regex, pred_regex)
    ov_hyb,   pl_hyb   = _agg(gold_hybrid, pred_hybrid)

    return {
        "ml":     (ov_ml,  pl_ml),
        "regex":  (ov_rx,  pl_rx),
        "hybrid": (ov_hyb, pl_hyb),
    }


def unified_dataframe(pl_ml: dict, pl_rx: dict, pl_hyb: dict) -> pd.DataFrame:
    """Возвращает единый DataFrame со всеми метками и тремя режимами."""
    all_labels = sorted(
        set(pl_ml) | set(pl_rx) | set(pl_hyb),
        key=lambda l: -(pl_hyb.get(l, {}).get("support", 0)),
    )

    def _get(d, label, key):
        return d.get(label, {}).get(key, None)

    rows = []
    for label in all_labels:
        support = (pl_hyb.get(label) or pl_ml.get(label) or pl_rx.get(label) or {}).get("support", 0)
        source = "ML" if label in pl_ml else "Regex"
        rows.append({
            "Label":        label,
            "Source":       source,
            "ML  F1":       _get(pl_ml,  label, "f1"),
            "ML  Rec":      _get(pl_ml,  label, "recall"),
            "Regex  F1":    _get(pl_rx,  label, "f1"),
            "Regex  Rec":   _get(pl_rx,  label, "recall"),
            "Hybrid  F1":   _get(pl_hyb, label, "f1"),
            "Hybrid  Rec":  _get(pl_hyb, label, "recall"),
            "Support":      int(support),
        })

    df = pd.DataFrame(rows).set_index("Label")
    return df


def display_results(results: dict) -> None:
    """Красивый вывод в Jupyter: три отдельных таблицы + единая сводная."""
    from IPython.display import display, HTML

    (ov_ml, pl_ml), (ov_rx, pl_rx), (ov_hyb, pl_hyb) = (
        results["ml"], results["regex"], results["hybrid"]
    )

    def _section_df(pl: dict, ov: dict, title: str) -> None:
        rows = []
        for label, m in sorted(pl.items(), key=lambda x: -x[1]["support"]):
            rows.append({"Label": label,
                         "Precision": m["precision"],
                         "Recall":    m["recall"],
                         "F1":        m["f1"],
                         "Support":   m["support"]})
        rows.append({"Label": "OVERALL",
                     "Precision": ov["precision"],
                     "Recall":    ov["recall"],
                     "F1":        ov["f1"],
                     "Support":   ov["tp"] + ov["fn"]})
        df = pd.DataFrame(rows).set_index("Label")
        styled = (df.style
                    .format({"Precision": "{:.3f}", "Recall": "{:.3f}",
                             "F1": "{:.3f}", "Support": "{:.0f}"}, na_rep="-")
                    .background_gradient(subset=["F1"], cmap="RdYlGn", vmin=0, vmax=1)
                    .set_caption(f"<b>{title}</b>")
                    .set_table_styles([
                        {"selector": "caption",
                         "props": [("font-size", "14px"), ("text-align", "left"),
                                   ("padding-bottom", "6px")]},
                        {"selector": "th",
                         "props": [("background-color", "#2d2d2d"),
                                   ("color", "white"), ("padding", "6px 12px")]},
                        {"selector": "td",
                         "props": [("padding", "5px 12px"), ("text-align", "right")]},
                        {"selector": "tr:last-child td",
                         "props": [("font-weight", "bold"),
                                   ("border-top", "2px solid #555")]},
                    ]))
        display(styled)
        display(HTML("<br>"))

    _section_df(pl_ml,  ov_ml,  "ML-only — NAME / ADDRESS")
    _section_df(pl_rx,  ov_rx,  "Regex-only — structured PII")
    _section_df(pl_hyb, ov_hyb, "Hybrid pipeline — all PII")

    # Единая сводная таблица
    df_uni = unified_dataframe(pl_ml, pl_rx, pl_hyb)
    float_cols = [c for c in df_uni.columns if c not in ("Source", "Support")]
    styled_uni = (df_uni.style
                  .format({c: "{:.3f}" for c in float_cols}, na_rep="-")
                  .background_gradient(subset=["Hybrid  F1"], cmap="RdYlGn", vmin=0, vmax=1)
                  .set_caption("<b>Unified results — all labels x all modes</b>")
                  .set_table_styles([
                      {"selector": "caption",
                       "props": [("font-size", "14px"), ("text-align", "left"),
                                 ("padding-bottom", "6px")]},
                      {"selector": "th",
                       "props": [("background-color", "#2d2d2d"),
                                 ("color", "white"), ("padding", "6px 14px")]},
                      {"selector": "td",
                       "props": [("padding", "5px 12px"), ("text-align", "right")]},
                  ]))
    display(styled_uni)


def print_unified_table(pl_ml: dict, pl_rx: dict, pl_hyb: dict) -> None:
    """Единая таблица: все теги x три режима (ML / Regex / Hybrid)."""
    all_labels = sorted(set(pl_ml) | set(pl_rx) | set(pl_hyb),
                        key=lambda l: -(pl_hyb.get(l, {}).get("support", 0)))

    W = 20
    col = 14

    header = (f"{'Label':<{W}}"
              f"{'ML F1':>{col}}{'ML Rec':>{col}}"
              f"{'Rx F1':>{col}}{'Rx Rec':>{col}}"
              f"{'Hyb F1':>{col}}{'Hyb Rec':>{col}}"
              f"{'Support':>{col}}")
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("  Unified results (all labels, all modes)")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    def _get(d, label, key):
        return d.get(label, {}).get(key, 0.0)

    for label in all_labels:
        support = int(_get(pl_hyb, label, "support") or
                      _get(pl_ml,  label, "support") or
                      _get(pl_rx,  label, "support"))
        ml_f1  = _get(pl_ml,  label, "f1")
        ml_rec = _get(pl_ml,  label, "recall")
        rx_f1  = _get(pl_rx,  label, "f1")
        rx_rec = _get(pl_rx,  label, "recall")
        hy_f1  = _get(pl_hyb, label, "f1")
        hy_rec = _get(pl_hyb, label, "recall")

        def fmt(v): return f"{v:.3f}" if v > 0 else "  -  "

        print(f"{label:<{W}}"
              f"{fmt(ml_f1):>{col}}{fmt(ml_rec):>{col}}"
              f"{fmt(rx_f1):>{col}}{fmt(rx_rec):>{col}}"
              f"{fmt(hy_f1):>{col}}{fmt(hy_rec):>{col}}"
              f"{support:>{col}}")

    print(f"{'=' * len(header)}\n")
