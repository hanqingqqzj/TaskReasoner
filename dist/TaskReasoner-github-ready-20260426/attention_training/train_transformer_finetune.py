"""Fine-tune a transformer classifier or regressor on prepared TAWOS labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "attention_training" / "data" / "clean30"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "attention_training" / "models" / "finetuned"
NO_RELATION_LABELS = {"no_relation", "not_related"}


TASK_COLUMNS = {
    "task_type": ("task_supervised.csv", ("task_model_text", "text"), ("task_type_model", "task_type_clean"), "classification", False),
    "priority": ("task_supervised.csv", ("priority_model_text", "text"), ("priority_coarse",), "classification", False),
    "difficulty_label": ("task_supervised.csv", ("difficulty_model_text", "text"), ("difficulty_label_derived",), "classification", False),
    "story_point_bin5": ("task_supervised.csv", ("difficulty_model_text", "text"), ("story_point_bin5",), "classification", False),
    "story_point_regression": ("task_supervised.csv", ("difficulty_model_text", "text"), ("story_point_log",), "regression", False),
    "task_relation": ("link_supervised.csv", ("pair_model_text", "pair_text"), ("relation_label",), "classification", False),
    "task_relation_binary": ("link_supervised.csv", ("pair_model_text", "pair_text"), ("relation_binary",), "classification", False),
    "task_relation_type": ("link_supervised.csv", ("pair_model_text", "pair_text"), ("relation_label",), "classification", True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune one transformer head on a prepared TAWOS task.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task", choices=sorted(TASK_COLUMNS), required=True)
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--max-rows", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--max-classes", type=int, default=20)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    return parser.parse_args()


class TextDataset:
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: object, max_length: int) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def compute_metrics_factory(mode: str):
    def compute_metrics(eval_pred: object) -> dict:
        if hasattr(eval_pred, "predictions"):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            logits, labels = eval_pred
        if mode == "regression":
            pred = np.squeeze(logits)
            return {
                "mae": float(mean_absolute_error(labels, pred)),
                "rmse": float(mean_squared_error(labels, pred) ** 0.5),
                "r2": float(r2_score(labels, pred)),
            }
        pred = np.argmax(logits, axis=-1)
        report = classification_report(labels, pred, output_dict=True, zero_division=0)
        return {
            "accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
        }

    return compute_metrics


def first_existing(columns: Tuple[str, ...], df: pd.DataFrame) -> str:
    for column in columns:
        if column in df.columns:
            return column
    raise KeyError(f"None of these columns exist: {columns}")


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "Missing transformer dependencies. Install with: "
            "/opt/anaconda3/bin/python -m pip install -r attention_training/requirements-attention.txt"
        ) from exc

    filename, text_columns, label_columns, mode, positive_only = TASK_COLUMNS[args.task]
    df = pd.read_csv(args.data_dir / filename)
    text_column = first_existing(text_columns, df)
    label_column = first_existing(label_columns, df)
    if positive_only:
        df = df[~df[label_column].astype(str).isin(NO_RELATION_LABELS)].copy()
    df = df[[text_column, label_column]].dropna()
    df[text_column] = df[text_column].astype(str)
    if mode == "classification":
        df[label_column] = df[label_column].astype(str)
        df = df[df[label_column].str.len() > 0]
        keep = df[label_column].value_counts()
        keep = set(keep[keep >= args.min_class_count].head(args.max_classes).index)
        df = df[df[label_column].isin(keep)].copy()
    else:
        df[label_column] = pd.to_numeric(df[label_column], errors="coerce")
        df = df.dropna()
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    texts = df[text_column].tolist()
    if mode == "classification":
        encoder = LabelEncoder()
        labels = encoder.fit_transform(df[label_column]).astype(np.int64)
        num_labels = len(encoder.classes_)
        problem_type = "single_label_classification"
        stratify = labels if np.bincount(labels).min() >= 2 else None
        class_weights = None
        if args.class_weight == "balanced":
            counts = np.bincount(labels)
            weights = len(labels) / (len(counts) * np.maximum(counts, 1))
            class_weights = torch.tensor(weights, dtype=torch.float32)
    else:
        encoder = None
        labels = df[label_column].astype(float).to_numpy(dtype=np.float32)
        num_labels = 1
        problem_type = "regression"
        stratify = None
        class_weights = None

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=args.random_state,
        stratify=stratify,
    )
    train_dataset = TextDataset([texts[i] for i in train_idx], labels[train_idx], tokenizer, args.max_length)
    test_dataset = TextDataset([texts[i] for i in test_idx], labels[test_idx], tokenizer, args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        problem_type=problem_type,
    )
    output_dir = args.output_dir / args.task
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
    )
    trainer_class = Trainer
    if class_weights is not None:
        class WeightedTrainer(Trainer):
            def __init__(self, *inner_args: object, class_weights: object = None, **inner_kwargs: object) -> None:
                super().__init__(*inner_args, **inner_kwargs)
                self.class_weights = class_weights

            def compute_loss(self, model: object, inputs: dict, return_outputs: bool = False, **kwargs: object):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer_class = WeightedTrainer

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_factory(mode),
        **({"class_weights": class_weights} if class_weights is not None else {}),
    )
    trainer.train()
    metrics = trainer.evaluate()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if encoder is not None:
        (output_dir / "label_classes.json").write_text(json.dumps(encoder.classes_.tolist(), indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
