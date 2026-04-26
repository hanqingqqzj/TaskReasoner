"""Compare candidate classifiers on the current TAWOS training pipeline."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from text_utils import make_pair_text, make_task_text, normalize_relation_label  # noqa: E402
from train_link_model import (  # noqa: E402
    DATASET_COLUMNS,
    build_negative_pairs,
    deduplicate_positive_rows,
    read_links,
    read_tasks as read_link_tasks,
    split_by_pair_identity,
)
from train_task_model import prepare_target, read_tasks as read_task_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare algorithms on the current task and link data flow.")
    parser.add_argument("--tasks", type=Path, default=ROOT_DIR / "data" / "processed" / "tasks.tsv")
    parser.add_argument("--links", type=Path, default=ROOT_DIR / "data" / "processed" / "task_links.tsv")
    parser.add_argument("--task-rows", type=int, default=20000)
    parser.add_argument("--link-tasks", type=int, default=30000)
    parser.add_argument("--link-positive-links", type=int, default=12000)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--max-classes-per-target", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--mlp-iter", type=int, default=20)
    parser.add_argument("--sgd-iter", type=int, default=80)
    parser.add_argument("--logreg-iter", type=int, default=200)
    parser.add_argument(
        "--algorithms",
        default="mlp,sgd_log_loss,sgd_log_loss_balanced,complement_nb",
        help="Comma-separated algorithms to compare. Add logreg_balanced for the slower balanced logistic regression.",
    )
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "models" / "algorithm_comparison.json")
    return parser.parse_args()


def build_classifiers(args: argparse.Namespace) -> Dict[str, object]:
    available = {
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.1,
            max_iter=args.mlp_iter,
            random_state=args.random_state,
        ),
        "sgd_log_loss": SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=args.sgd_iter,
            tol=1e-3,
            random_state=args.random_state,
        ),
        "sgd_log_loss_balanced": SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=args.sgd_iter,
            tol=1e-3,
            class_weight="balanced",
            random_state=args.random_state,
        ),
        "logreg_balanced": LogisticRegression(
            class_weight="balanced",
            max_iter=args.logreg_iter,
            n_jobs=-1,
            solver="saga",
            random_state=args.random_state,
        ),
        "complement_nb": ComplementNB(alpha=0.5),
    }
    selected = [name.strip() for name in args.algorithms.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown algorithm(s): {', '.join(unknown)}")
    return {name: available[name] for name in selected}


def score_classifier(
    name: str,
    classifier: object,
    x_train: object,
    y_train: Iterable[str],
    x_test: object,
    y_test: Iterable[str],
) -> Dict[str, object]:
    print(f"  fitting {name}...", flush=True)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(list(y_train))
    y_test_encoded = label_encoder.transform(list(y_test))
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(x_train, y_train_encoded)
    predictions = classifier.predict(x_test)
    elapsed = time.perf_counter() - started
    report = classification_report(y_test_encoded, predictions, output_dict=True, zero_division=0)
    return {
        "algorithm": name,
        "accuracy": float(accuracy_score(y_test_encoded, predictions)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "seconds": round(elapsed, 3),
        "warnings": [str(item.message) for item in caught],
    }


def vectorize_train_test(
    texts: List[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    max_features: int,
) -> Tuple[object, object, int]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform([texts[i] for i in train_idx])
    x_test = vectorizer.transform([texts[i] for i in test_idx])
    return x_train, x_test, len(vectorizer.vocabulary_)


def compare_task_target(args: argparse.Namespace, target: str) -> Dict[str, object]:
    print(f"Comparing task target: {target}", flush=True)
    df = read_task_rows(args.tasks, args.task_rows)
    texts = [make_task_text(row.title, row.description_text) for row in df.itertuples(index=False)]
    y = prepare_target(df, target, args.min_class_count, args.max_classes_per_target)
    mask = y.notna().to_numpy()
    indices = np.flatnonzero(mask)
    labels = y.iloc[indices].astype(str)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.random_state,
        stratify=labels if labels.value_counts().min() >= 2 else None,
    )
    x_train, x_test, vocabulary_size = vectorize_train_test(texts, train_idx, test_idx, args.max_features)
    y_train = y.iloc[train_idx].astype(str)
    y_test = y.iloc[test_idx].astype(str)

    results = [
        score_classifier(name, classifier, x_train, y_train, x_test, y_test)
        for name, classifier in build_classifiers(args).items()
    ]
    results.sort(key=lambda row: row["macro_f1"], reverse=True)
    return {
        "target": target,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "tfidf_vocabulary_size": int(vocabulary_size),
        "classes": labels.value_counts().to_dict(),
        "majority_baseline_accuracy": float(y_test.value_counts(normalize=True).max()),
        "results": results,
    }


def build_link_dataset(args: argparse.Namespace) -> pd.DataFrame:
    tasks = read_link_tasks(args.tasks, args.link_tasks)
    links = read_links(args.links, args.link_positive_links)
    task_ids = set(tasks["task_id"].astype(str))
    links = links[
        links["source_task_id"].astype(str).isin(task_ids)
        & links["target_task_id"].astype(str).isin(task_ids)
    ].copy()
    links["label"] = [
        normalize_relation_label(row.relation_name, row.relation_description)
        for row in links.itertuples(index=False)
    ]
    counts = links["label"].value_counts()
    keep = counts[counts >= args.min_class_count].index
    positives = links[links["label"].isin(keep)].copy()[DATASET_COLUMNS].copy()
    positives, _ = deduplicate_positive_rows(positives, "drop")
    positive_pairs = set(zip(positives["source_task_id"].astype(str), positives["target_task_id"].astype(str)))
    negatives, _ = build_negative_pairs(tasks, positive_pairs, len(positives), args.random_state)
    dataset = pd.concat([positives, negatives], ignore_index=True)
    dataset = dataset.drop_duplicates(subset=["source_task_id", "target_task_id", "label"]).copy()
    return dataset.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)


def compare_link(args: argparse.Namespace) -> Dict[str, object]:
    print("Comparing link target: task_relation", flush=True)
    dataset = build_link_dataset(args)
    texts = [
        make_pair_text(
            row.source_title,
            row.source_description_text,
            row.target_title,
            row.target_description_text,
        )
        for row in dataset.itertuples(index=False)
    ]
    y = dataset["label"].astype(str)
    train_idx, test_idx, split_stats = split_by_pair_identity(dataset, args.random_state)
    x_train, x_test, vocabulary_size = vectorize_train_test(texts, train_idx, test_idx, args.max_features)
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    results = [
        score_classifier(name, classifier, x_train, y_train, x_test, y_test)
        for name, classifier in build_classifiers(args).items()
    ]
    results.sort(key=lambda row: row["macro_f1"], reverse=True)
    return {
        "target": "task_relation",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "tfidf_vocabulary_size": int(vocabulary_size),
        "classes": y.value_counts().to_dict(),
        "majority_baseline_accuracy": float(y_test.value_counts(normalize=True).max()),
        "split": split_stats,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    report = {
        "settings": settings,
        "task": [compare_task_target(args, target) for target in ["task_type", "priority", "difficulty_label"]],
        "link": compare_link(args),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
