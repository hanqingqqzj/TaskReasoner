"""Validate the proposed MLP-heavy training plan on controlled samples."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

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


STORY_POINT_BINS = [-np.inf, 1, 2, 3, 5, 8, 13, 21, 34, 55, np.inf]
STORY_POINT_LABELS = [f"difficulty_{index:02d}" for index in range(1, 11)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate proposed MLP layer sizes and two-stage relation modeling.")
    parser.add_argument("--tasks", type=Path, default=ROOT_DIR / "data" / "processed" / "tasks.tsv")
    parser.add_argument("--links", type=Path, default=ROOT_DIR / "data" / "processed" / "task_links.tsv")
    parser.add_argument("--task-rows", type=int, default=12000)
    parser.add_argument("--link-tasks", type=int, default=15000)
    parser.add_argument("--link-positive-links", type=int, default=6000)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "models" / "mlp_plan_validation.json")
    return parser.parse_args()


def vectorize(texts: List[str], train_idx: np.ndarray, test_idx: np.ndarray, max_features: int) -> Tuple[object, object, int]:
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


def fit_predict(classifier: object, x_train: object, y_train: Iterable[str], x_test: object) -> Tuple[np.ndarray, Dict[str, object]]:
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(list(y_train))
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(x_train, y_train_encoded)
    elapsed = time.perf_counter() - started
    predictions = label_encoder.inverse_transform(classifier.predict(x_test).astype(int))
    details = {
        "seconds": round(elapsed, 3),
        "n_iter": int(getattr(classifier, "n_iter_", 0)),
        "warnings": [str(item.message) for item in caught],
    }
    return predictions, details


def metrics(name: str, predictions: Iterable[str], truth: Iterable[str], extra: Dict[str, object]) -> Dict[str, object]:
    report = classification_report(truth, predictions, output_dict=True, zero_division=0)
    return {
        "name": name,
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            label: {
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1-score"]),
                "support": int(values["support"]),
            }
            for label, values in report.items()
            if isinstance(values, dict) and label not in {"macro avg", "weighted avg"}
        },
        **extra,
    }


def mlp(hidden: Tuple[int, ...], max_iter: int, random_state: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        early_stopping=True,
        validation_fraction=0.1,
        max_iter=max_iter,
        random_state=random_state,
    )


def validate_task_target(df: pd.DataFrame, texts: List[str], y: pd.Series, target: str, args: argparse.Namespace) -> Dict[str, object]:
    mask = y.notna().to_numpy()
    indices = np.flatnonzero(mask)
    labels = y.iloc[indices].astype(str)
    if labels.value_counts().min() < 2:
        raise ValueError(f"{target} has a class with fewer than two examples.")
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.random_state,
        stratify=labels,
    )
    x_train, x_test, vocabulary_size = vectorize(texts, train_idx, test_idx, args.max_features)
    y_train = y.iloc[train_idx].astype(str)
    y_test = y.iloc[test_idx].astype(str)

    candidates = [
        ("mlp_128_64", mlp((128, 64), args.max_iter, args.random_state)),
        ("mlp_256_128", mlp((256, 128), args.max_iter, args.random_state)),
    ]
    if target == "priority":
        candidates = [
            (
                "balanced_sgd_log_loss",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-5,
                    max_iter=args.max_iter,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=args.random_state,
                ),
            )
        ]

    results = []
    for name, classifier in candidates:
        print(f"  fitting {target}: {name}", flush=True)
        predictions, details = fit_predict(classifier, x_train, y_train, x_test)
        results.append(metrics(name, predictions, y_test, details))
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


def story_point_10_labels(df: pd.DataFrame) -> pd.Series:
    story_point = pd.to_numeric(df["story_point"].astype(str).str.strip(), errors="coerce")
    labels = pd.cut(story_point, bins=STORY_POINT_BINS, labels=STORY_POINT_LABELS, right=True)
    return labels.astype(object).where(story_point.gt(0))


def read_task_with_story_points(path: Path, max_rows: int) -> pd.DataFrame:
    columns = ["title", "description_text", "task_type", "priority", "difficulty_label", "story_point"]
    nrows = None if max_rows == 0 else max_rows
    return pd.read_csv(path, sep="\t", usecols=columns, dtype=str, keep_default_na=False, na_values=[], nrows=nrows)


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
    keep = counts[counts >= 20].index
    positives = links[links["label"].isin(keep)].copy()[DATASET_COLUMNS].copy()
    positives, _ = deduplicate_positive_rows(positives, "drop")
    positive_pairs = set(zip(positives["source_task_id"].astype(str), positives["target_task_id"].astype(str)))
    negatives, _ = build_negative_pairs(tasks, positive_pairs, len(positives), args.random_state)
    dataset = pd.concat([positives, negatives], ignore_index=True)
    dataset = dataset.drop_duplicates(subset=["source_task_id", "target_task_id", "label"]).copy()
    return dataset.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)


def validate_link_two_stage(args: argparse.Namespace) -> Dict[str, object]:
    dataset = build_link_dataset(args)
    texts = [
        make_pair_text(row.source_title, row.source_description_text, row.target_title, row.target_description_text)
        for row in dataset.itertuples(index=False)
    ]
    y = dataset["label"].astype(str)
    train_idx, test_idx, split_stats = split_by_pair_identity(dataset, args.random_state)
    x_train, x_test, vocabulary_size = vectorize(texts, train_idx, test_idx, args.max_features)
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("  fitting link: direct_mlp_256_128", flush=True)
    direct_predictions, direct_details = fit_predict(
        mlp((256, 128), args.max_iter, args.random_state),
        x_train,
        y_train,
        x_test,
    )

    binary_train = np.where(y_train.eq("no_relation"), "no_relation", "has_relation")
    print("  fitting link: two_stage_binary_mlp_256_128", flush=True)
    binary_predictions, binary_details = fit_predict(
        mlp((256, 128), args.max_iter, args.random_state),
        x_train,
        binary_train,
        x_test,
    )

    positive_train_mask = y_train.ne("no_relation").to_numpy()
    positive_test_mask = binary_predictions == "has_relation"
    stage2_predictions = np.array(["no_relation"] * len(y_test), dtype=object)
    stage2_details = {"seconds": 0.0, "n_iter": 0, "warnings": []}
    if positive_train_mask.sum() > 0 and positive_test_mask.sum() > 0:
        print("  fitting link: two_stage_positive_mlp_256_128", flush=True)
        positive_predictions, stage2_details = fit_predict(
            mlp((256, 128), args.max_iter, args.random_state),
            x_train[positive_train_mask],
            y_train.iloc[np.flatnonzero(positive_train_mask)],
            x_test[positive_test_mask],
        )
        stage2_predictions[positive_test_mask] = positive_predictions

    two_stage_details = {
        "seconds": round(float(binary_details["seconds"]) + float(stage2_details["seconds"]), 3),
        "stage1_n_iter": binary_details["n_iter"],
        "stage2_n_iter": stage2_details["n_iter"],
        "warnings": binary_details["warnings"] + stage2_details["warnings"],
    }

    return {
        "target": "task_relation",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "tfidf_vocabulary_size": int(vocabulary_size),
        "classes": y.value_counts().to_dict(),
        "majority_baseline_accuracy": float(y_test.value_counts(normalize=True).max()),
        "split": split_stats,
        "results": [
            metrics("direct_mlp_256_128", direct_predictions, y_test, direct_details),
            metrics("two_stage_mlp_256_128", stage2_predictions, y_test, two_stage_details),
        ],
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = read_task_with_story_points(args.tasks, args.task_rows)
    texts = [make_task_text(row.title, row.description_text) for row in df.itertuples(index=False)]

    print("Validating task targets", flush=True)
    task_type = validate_task_target(df, texts, prepare_target(df, "task_type", 20, 12), "task_type", args)
    priority = validate_task_target(df, texts, prepare_target(df, "priority", 20, 12), "priority", args)
    difficulty_3 = validate_task_target(df, texts, prepare_target(df, "difficulty_label", 20, 12), "difficulty_label_3", args)
    difficulty_10 = validate_task_target(df, texts, story_point_10_labels(df), "story_point_difficulty_10", args)

    print("Validating link relation", flush=True)
    link = validate_link_two_stage(args)

    report = {
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "story_point_bins": {
            "bins": [str(value) for value in STORY_POINT_BINS],
            "labels": STORY_POINT_LABELS,
        },
        "task": [task_type, priority, difficulty_3, difficulty_10],
        "link": link,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
