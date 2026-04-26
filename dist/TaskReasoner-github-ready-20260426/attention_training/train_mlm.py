"""Continue pretraining a transformer with masked language modeling."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT_DIR / "attention_training" / "data" / "clean30" / "unlabeled_corpus.txt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "attention_training" / "models" / "tawos_mlm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Domain-adaptive MLM pretraining on TAWOS text.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--max-lines", type=int, default=50000, help="Use 0 for all lines.")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=1000)
    return parser.parse_args()


class LineDataset:
    def __init__(self, path: Path, tokenizer: object, max_length: int, max_lines: int) -> None:
        self.examples = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if max_lines and line_number >= max_lines:
                    break
                text = line.strip()
                if not text:
                    continue
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )
                self.examples.append(encoded)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def main() -> None:
    args = parse_args()
    try:
        from transformers import (
            AutoModelForMaskedLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing transformer dependencies. Install with: "
            "/opt/anaconda3/bin/python -m pip install -r attention_training/requirements-attention.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)
    dataset = LineDataset(args.corpus, tokenizer, args.max_length, args.max_lines)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
