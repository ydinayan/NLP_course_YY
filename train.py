"""
train.py — обучение XLM-RoBERTa для NER (NAME / ADDRESS).
"""
import numpy as np
import transformers as T
from seqeval.metrics import precision_score, recall_score, f1_score
from seqeval.scheme import IOB2

from data import LABEL_LIST, LABEL2ID, ID2LABEL, load_splits, load_splits_scanpatch

MODEL_NAME = "DeepPavlov/rubert-base-cased"


def _compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    true_labels, true_preds = [], []
    for pred_row, label_row in zip(preds, labels):
        true_labels.append([ID2LABEL[l] for l in label_row if l != -100])
        true_preds.append([ID2LABEL[p] for p, l in zip(pred_row, label_row) if l != -100])
    return {
        "precision": precision_score(true_labels, true_preds, mode="strict", scheme=IOB2),
        "recall":    recall_score(true_labels,    true_preds, mode="strict", scheme=IOB2),
        "f1":        f1_score(true_labels,        true_preds, mode="strict", scheme=IOB2),
    }


def train_model(model_name: str = MODEL_NAME,
                dataset: str = "scanpatch") -> tuple:
    """
    Обучает модель и сохраняет в ./pii-ner-model.

    dataset: "scanpatch" (default) или "hivetrace"
    Возвращает (model, tokenizer, domain_df).
    """
    T.set_seed(42)

    if dataset == "scanpatch":
        tokenizer, train_tok, val_tok, domain_df = load_splits_scanpatch(model_name)
    else:
        tokenizer, train_tok, val_tok, domain_df = load_splits(model_name)

    model = T.AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = T.TrainingArguments(
        output_dir="./pii-ner-model",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        seed=42,
    )

    trainer = T.Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=T.DataCollatorForTokenClassification(tokenizer),
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    trainer.save_model("./pii-ner-model")
    tokenizer.save_pretrained("./pii-ner-model")
    print("Model saved to ./pii-ner-model")

    return model, tokenizer, domain_df


if __name__ == "__main__":
    train_model()
