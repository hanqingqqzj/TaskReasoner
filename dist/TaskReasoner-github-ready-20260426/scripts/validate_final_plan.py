"""Validate the proposed final modeling plan before changing production training."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
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
from train_task_model import prepare_target  # noqa: E402


STORY_POINT_BINS = [-np.inf, 1, 2, 3, 5, 8, 13, 21, 34, 55, np.inf]
STORY_POINT_LABELS = [f"difficulty_{index:02d}" for index in range(1, 11)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final candidate task, priority, relation, and effort models.")
    parser.add_argument("--tasks", type=Path, default=ROOT_DIR / "data" / "processed" / "tasks.tsv")
    parser.add_argument("--links", type=Path, default=ROOT_DIR / "data" / "processed" / "task_links.tsv")
    parser.add_argument("--task-rows", type=int, default=12000)
    parser.add_argument("--link-tasks", type=int, default=15000)
    parser.add_argument("--link-positive-links", type=int, default=6000)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "models" / "final_plan_validation.json")
    return parser.parse_args()


def read_task_rows(path: Path, max_rows: int) -> pd.DataFrame:
    columns = ["title", "description_text", "task_type", "priority", "difficulty_label", "story_point"]
    nrows = None if max_rows == 0 else max_rows
    return pd.read_csv(path, sep="\t", usecols=columns, dtype=str, keep_default_na=False, na_values=[], nrows=nrows)


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


def mlp_classifier(hidden: Tuple[int, ...], max_iter: int, random_state: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        early_stopping=True,
        validation_fraction=0.1,
        max_iter=max_iter,
        random_state=random_state,
    )


def fit_classification(classifier: object, x_train: object, y_train: Iterable[str], x_test: object) -> Tuple[np.ndarray, Dict[str, object]]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(list(y_train))
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(x_train, y_encoded)
    seconds = time.perf_counter() - started
    predictions = encoder.inverse_transform(classifier.predict(x_test).astype(int))
    n_iter = getattr(classifier, "n_iter_", 0)
    if isinstance(n_iter, np.ndarray):
        n_iter = int(n_iter.max()) if n_iter.size else 0
    return predictions, {
        "seconds": round(seconds, 3),
        "n_iter": int(n_iter),
        "warnings": [str(item.message) for item in caught],
    }


def classification_metrics(name: str, predictions: Iterable[str], truth: Iterable[str], extra: Dict[str, object]) -> Dict[str, object]:
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


def validate_single_classifier_target(
    texts: List[str],
    y: pd.Series,
    target: str,
    classifier_name: str,
    classifier: object,
    args: argparse.Namespace,
) -> Dict[str, object]:
    mask = y.notna().to_numpy()
    indices = np.flatnonzero(mask)
    labels = y.iloc[indices].astype(str)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.random_state,
        stratify=labels if labels.value_counts().min() >= 2 else None,
    )
    x_train, x_test, vocabulary_size = vectorize(texts, train_idx, test_idx, args.max_features)
    y_train = y.iloc[train_idx].astype(str)
    y_test = y.iloc[test_idx].astype(str)
    print(f"  fitting {target}: {classifier_name}", flush=True)
    predictions, details = fit_classification(classifier, x_train, y_train, x_test)
    return {
        "target": target,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "tfidf_vocabulary_size": int(vocabulary_size),
        "classes": labels.value_counts().to_dict(),
        "majority_baseline_accuracy": float(y_test.value_counts(normalize=True).max()),
        "result": classification_metrics(classifier_name, predictions, y_test, details),
    }


def validate_priority_algorithms(texts: List[str], y: pd.Series, args: argparse.Namespace) -> Dict[str, object]:
    mask = y.notna().to_numpy()
    indices = np.flatnonzero(mask)
    labels = y.iloc[indices].astype(str)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.random_state,
        stratify=labels if labels.value_counts().min() >= 2 else None,
    )
    x_train, x_test, vocabulary_size = vectorize(texts, train_idx, test_idx, args.max_features)
    y_train = y.iloc[train_idx].astype(str)
    y_test = y.iloc[test_idx].astype(str)

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
        ),
        (
            "sgd_log_loss",
            SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                max_iter=args.max_iter,
                tol=1e-3,
                random_state=args.random_state,
            ),
        ),
        ("complement_nb", ComplementNB(alpha=0.5)),
        ("mlp_128_64", mlp_classifier((128, 64), args.max_iter, args.random_state)),
        (
            "balanced_logistic_regression",
            LogisticRegression(
                class_weight="balanced",
                max_iter=max(100, args.max_iter),
                solver="liblinear",
                random_state=args.random_state,
            ),
        ),
    ]

    results = []
    for name, classifier in candidates:
        print(f"  fitting priority: {name}", flush=True)
        predictions, details = fit_classification(classifier, x_train, y_train, x_test)
        results.append(classification_metrics(name, predictions, y_test, details))
    results.sort(key=lambda row: row["macro_f1"], reverse=True)
    return {
        "target": "priority",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "tfidf_vocabulary_size": int(vocabulary_size),
        "classes": labels.value_counts().to_dict(),
        "majority_baseline_accuracy": float(y_test.value_counts(normalize=True).max()),
        "results": results,
    }


def story_point_target(df: pd.DataFrame) -> pd.Series:
    story_point = pd.to_numeric(df["story_point"].astype(str).str.strip(), errors="coerce")
    return story_point.where(story_point.gt(0))


def story_point_bin_labels(values: Iterable[float]) -> pd.Series:
    return pd.cut(pd.Series(values), bins=STORY_POINT_BINS, labels=STORY_POINT_LABELS, right=True).astype(str)


def validate_story_point_regression(df: pd.DataFrame, texts: List[str], args: argparse.Namespace) -> Dict[str, object]:
    y = story_point_target(df)
    mask = y.notna().to_numpy()
    indices = np.flatnonzero(mask)
    y_values = y.iloc[indices].astype(float)
    y_bins = story_point_bin_labels(y_values)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y_bins if y_bins.value_counts().min() >= 2 else None,
    )
    x_train, x_test, vocabulary_size = vectorize(texts, train_idx, test_idx, args.max_features)
    y_train_raw = y.iloc[train_idx].astype(float).to_numpy()
    y_test_raw = y.iloc[test_idx].astype(float).to_numpy()
    y_train_log = np.log1p(y_train_raw)
    y_test_log = np.log1p(y_test_raw)

    regressors = [
        ("median_baseline", DummyRegressor(strategy="median")),
        ("ridge_log_story_point", Ridge(alpha=1.0, random_state=args.random_state)),
        (
            "sgd_huber_log_story_point",
            SGDRegressor(
                loss="huber",
                penalty="l2",
                alpha=1e-5,
                max_iter=max(100, args.max_iter),
                tol=1e-3,
                random_state=args.random_state,
            ),
        ),
    ]

    results = []
    for name, regressor in regressors:
        print(f"  fitting story_point regression: {name}", flush=True)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            regressor.fit(x_train, y_train_log)
        predicted_log = regressor.predict(x_test)
        predicted_raw = np.expm1(predicted_log).clip(min=0)
        predicted_bins = story_point_bin_labels(predicted_raw)
        truth_bins = story_point_bin_labels(y_test_raw)
        bin_report = classification_report(truth_bins, predicted_bins, output_dict=True, zero_division=0)
        results.append(
            {
                "name": name,
                "log_mae": float(mean_absolute_error(y_test_log, predicted_log)),
                "log_rmse": float(mean_squared_error(y_test_log, predicted_log) ** 0.5),
                "log_r2": float(r2_score(y_test_log, predicted_log)),
                "raw_mae": float(mean_absolute_error(y_test_raw, predicted_raw)),
                "bin_accuracy": float(accuracy_score(truth_bins, predicted_bins)),
                "bin_macro_f1": float(bin_report["macro avg"]["f1-score"]),
                "bin_weighted_f1": float(bin_report["weighted avg"]["f1-score"]),
                "seconds": round(time.perf_counter() - started, 3),
                "warnings": [str(item.message) for item in caught],
            }
        )
    results.sort(key=lambda row: row["log_mae"])
    return {
        "target": "story_point_regression",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "tfidf_vocabulary_size": int(vocabulary_size),
        "story_point_positive_rows": int(mask.sum()),
        "story_point_bins": y_bins.value_counts().sort_index().to_dict(),
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
    keep = counts[counts >= 20].index
    positives = links[links["label"].isin(keep)].copy()[DATASET_COLUMNS].copy()
    positives, _ = deduplicate_positive_rows(positives, "drop")
    positive_pairs = set(zip(positives["source_task_id"].astype(str), positives["target_task_id"].astype(str)))
    negatives, _ = build_negative_pairs(tasks, positive_pairs, len(positives), args.random_state)
    dataset = pd.concat([positives, negatives], ignore_index=True)
    dataset = dataset.drop_duplicates(subset=["source_task_id", "target_task_id", "label"]).copy()
    return dataset.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)


def validate_relation_direct_mlp(args: argparse.Namespace) -> Dict[str, object]:
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
    print("  fitting task_relation: direct_mlp_256_128", flush=True)
    predictions, details = fit_classification(
        mlp_classifier((256, 128), args.max_iter, args.random_state),
        x_train,
        y_train,
        x_test,
    )
    return {
        "target": "task_relation",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "tfidf_vocabulary_size": int(vocabulary_size),
        "classes": y.value_counts().to_dict(),
        "majority_baseline_accuracy": float(y_test.value_counts(normalize=True).max()),
        "split": split_stats,
        "result": classification_metrics("direct_mlp_256_128", predictions, y_test, details),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = read_task_rows(args.tasks, args.task_rows)
    texts = [make_task_text(row.title, row.description_text) for row in df.itertuples(index=False)]

    print("Validating final candidate plan", flush=True)
    task_type = validate_single_classifier_target(
        texts,
        prepare_target(df, "task_type", 20, 12),
        "task_type",
        "mlp_128_64",
        mlp_classifier((128, 64), args.max_iter, args.random_state),
        args,
    )
    priority = validate_priority_algorithms(texts, prepare_target(df, "priority", 20, 12), args)
    difficulty = validate_single_classifier_target(
        texts,
        prepare_target(df, "difficulty_label", 20, 12),
        "difficulty_label",
        "mlp_128_64",
        mlp_classifier((128, 64), args.max_iter, args.random_state),
        args,
    )
    story_point = validate_story_point_regression(df, texts, args)
    relation = validate_relation_direct_mlp(args)

    report = {
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "story_point_bins": {
            "bins": [str(value) for value in STORY_POINT_BINS],
            "labels": STORY_POINT_LABELS,
        },
        "task_type": task_type,
        "priority": priority,
        "difficulty_label": difficulty,
        "story_point_regression": story_point,
        "task_relation": relation,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
