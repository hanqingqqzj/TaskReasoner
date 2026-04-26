"""Run inference with the saved pairwise task relation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from text_utils import make_pair_text, top_predictions


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT_DIR / "models" / "link_model" / "link_model.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict relation/dependency between two task texts.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-title", required=True)
    parser.add_argument("--source-description", default="")
    parser.add_argument("--target-title", required=True)
    parser.add_argument("--target-description", default="")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    text = make_pair_text(
        args.source_title,
        args.source_description,
        args.target_title,
        args.target_description,
    )
    # Use the training-time TF-IDF vocabulary so inference lives in the same
    # feature space as the fitted classifier.
    x = bundle["vectorizer"].transform([text])
    # The classifier outputs one probability per relation class.
    probabilities = bundle["model"].predict_proba(x)[0]
    labels = bundle["label_encoder"].inverse_transform(bundle["model"].classes_.astype(int))
    output = {
        "relation": top_predictions(labels, probabilities, top_k=args.top_k)
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
