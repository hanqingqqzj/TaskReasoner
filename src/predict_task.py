"""Run inference with the saved task attribute and difficulty models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from text_utils import make_task_text, top_predictions


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT_DIR / "models" / "task_model" / "task_model.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict task attributes and difficulty from text.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    text = make_task_text(args.title, args.description)

    output = {}
    for target, entry in bundle["models"].items():
        model = entry["classifier"]
        label_encoder = entry["label_encoder"]
        # New task models store one train-only TF-IDF vectorizer per target. The
        # fallback keeps older saved models usable.
        vectorizer = entry.get("vectorizer", bundle.get("vectorizer"))
        if vectorizer is None:
            raise KeyError(f"No vectorizer found for target {target}. Retrain the task model.")
        x = vectorizer.transform([text])
        # predict_proba returns softmax-like probabilities over the encoded
        # classes, which are converted back to the original text labels.
        probabilities = model.predict_proba(x)[0]
        labels = label_encoder.inverse_transform(model.classes_.astype(int))
        output[target] = top_predictions(labels, probabilities, top_k=args.top_k)

    for target, entry in bundle.get("regressors", {}).items():
        vectorizer = entry["vectorizer"]
        model = entry["model"]
        x = vectorizer.transform([text])
        predicted_log = float(model.predict(x)[0])
        predicted_value = max(0.0, float(np.expm1(predicted_log)))
        bins = entry.get("bins")
        labels = entry.get("labels")
        difficulty_band = None
        if bins is not None and labels is not None:
            for index, label in enumerate(labels):
                if predicted_value <= bins[index + 1]:
                    difficulty_band = label
                    break
            if difficulty_band is None and labels:
                difficulty_band = labels[-1]
        output[target] = {
            "predicted_story_point": predicted_value,
            "difficulty_band": difficulty_band,
        }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
