"""Train a pairwise two-hidden-layer MLP for task relation prediction."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder

from text_utils import make_pair_text, normalize_relation_label


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT_DIR / "data" / "processed" / "tasks.tsv"
DEFAULT_LINKS = ROOT_DIR / "data" / "processed" / "task_links.tsv"
DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "link_model"
PAIR_COLUMNS = ["source_task_id", "target_task_id"]
DATASET_COLUMNS = [
    "source_task_id",
    "target_task_id",
    "source_title",
    "source_description_text",
    "target_title",
    "target_description_text",
    "label",
]
MIXED_PAIR_LABEL = "__mixed_pair_labels__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pairwise task dependency/relation MLP model.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-tasks", type=int, default=0, help="Use 0 for all task rows.")
    parser.add_argument("--max-positive-links", type=int, default=0, help="Use 0 for all links.")
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--max-features", type=int, default=80000)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--hidden-size-1", type=int, default=128)
    parser.add_argument("--hidden-size-2", type=int, default=64)
    parser.add_argument("--min-class-count", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--algorithm",
        choices=["mlp", "sgd_log_loss", "sgd_log_loss_balanced", "complement_nb", "auto"],
        default="mlp",
        help="Classifier to train. auto uses the fastest validated relation classifier.",
    )
    parser.add_argument(
        "--conflict-policy",
        choices=["drop", "keep"],
        default="drop",
        help="How to handle the same source->target pair appearing with multiple labels.",
    )
    return parser.parse_args()


def read_tasks(path: Path, max_tasks: int) -> pd.DataFrame:
    nrows = None if max_tasks == 0 else max_tasks
    return pd.read_csv(
        path,
        sep="\t",
        usecols=["task_id", "project_key", "title", "description_text"],
        dtype=str,
        keep_default_na=False,
        na_values=[],
        nrows=nrows,
    )


def read_links(path: Path, max_links: int) -> pd.DataFrame:
    nrows = None if max_links == 0 else max_links
    return pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "source_task_id",
            "target_task_id",
            "relation_name",
            "relation_description",
            "source_title",
            "source_description_text",
            "target_title",
            "target_description_text",
        ],
        dtype=str,
        keep_default_na=False,
        na_values=[],
        nrows=nrows,
    )


def make_pair_keys(df: pd.DataFrame) -> pd.Series:
    return df["source_task_id"].astype(str) + "->" + df["target_task_id"].astype(str)


def deduplicate_positive_rows(positives: pd.DataFrame, conflict_policy: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    deduped = positives.drop_duplicates(subset=PAIR_COLUMNS + ["label"]).copy()
    exact_duplicate_rows_removed = len(positives) - len(deduped)
    label_counts = deduped.groupby(PAIR_COLUMNS)["label"].nunique()
    conflicting_pairs = label_counts[label_counts > 1]
    conflict_keys = set("->".join(pair) for pair in conflicting_pairs.index)
    conflict_row_mask = make_pair_keys(deduped).isin(conflict_keys)
    conflict_rows_removed = 0
    if conflict_policy == "drop" and conflict_keys:
        conflict_rows_removed = int(conflict_row_mask.sum())
        deduped = deduped.loc[~conflict_row_mask].copy()

    stats = {
        "positive_rows_before_dedup": int(len(positives)),
        "positive_rows_after_dedup": int(len(deduped)),
        "positive_duplicate_rows_removed": int(exact_duplicate_rows_removed),
        "conflicting_positive_pairs": int(len(conflicting_pairs)),
        "positive_conflict_rows_removed": int(conflict_rows_removed),
        "conflict_policy": conflict_policy,
    }
    return deduped, stats


def choose_algorithm(requested: str) -> str:
    return "sgd_log_loss" if requested == "auto" else requested


def build_classifier(algorithm: str, args: argparse.Namespace) -> object:
    if algorithm == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(args.hidden_size_1, args.hidden_size_2),
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.1,
            max_iter=args.max_iter,
            random_state=args.random_state,
            verbose=False,
        )
    if algorithm == "sgd_log_loss":
        return SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=args.max_iter,
            tol=1e-3,
            random_state=args.random_state,
        )
    if algorithm == "sgd_log_loss_balanced":
        return SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=args.max_iter,
            tol=1e-3,
            class_weight="balanced",
            random_state=args.random_state,
        )
    if algorithm == "complement_nb":
        return ComplementNB(alpha=0.5)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def show_and_collect_fit_warnings(model: object, x_train: object, y_train: np.ndarray) -> List[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)

    messages = [str(item.message) for item in caught]
    for item in caught:
        warnings.showwarning(item.message, item.category, item.filename, item.lineno, file=sys.stderr)

    if any("Training interrupted by user" in message for message in messages):
        raise KeyboardInterrupt("Training interrupted by user; partial model was not saved.")
    return messages


def extract_class_metrics(report: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    class_metrics: Dict[str, Dict[str, float]] = {}
    for label, values in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"} or not isinstance(values, dict):
            continue
        class_metrics[label] = {
            "precision": float(values["precision"]),
            "recall": float(values["recall"]),
            "f1": float(values["f1-score"]),
            "support": int(values["support"]),
        }
    return class_metrics


def build_negative_pairs(
    tasks: pd.DataFrame,
    positive_pairs: Set[Tuple[str, str]],
    count: int,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    # TAWOS only stores positive links. Random same-project pairs with no known
    # edge become "no_relation" examples so the model can learn when no
    # dependency should be predicted.
    rng = np.random.default_rng(random_state)
    task_records = tasks.to_dict("records")
    by_project: Dict[str, List[dict]] = {}
    for row in task_records:
        by_project.setdefault(row["project_key"], []).append(row)

    projects = [project for project, rows in by_project.items() if len(rows) >= 2]
    negatives = []
    seen_negative_pairs: Set[Tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(count * 20, 1000)

    if count == 0 or not projects:
        return pd.DataFrame(columns=DATASET_COLUMNS), {
            "requested_negative_rows": int(count),
            "generated_negative_rows": 0,
            "negative_rows_shortfall": int(count),
            "negative_sampling_attempts": 0,
        }

    while len(negatives) < count and attempts < max_attempts:
        attempts += 1
        project = rng.choice(projects)
        rows = by_project[project]
        source, target = rng.choice(rows, size=2, replace=False)
        pair = (source["task_id"], target["task_id"])
        if pair in positive_pairs or pair in seen_negative_pairs:
            continue
        seen_negative_pairs.add(pair)
        negatives.append(
            {
                "source_task_id": source["task_id"],
                "target_task_id": target["task_id"],
                "source_title": source["title"],
                "source_description_text": source["description_text"],
                "target_title": target["title"],
                "target_description_text": target["description_text"],
                "label": "no_relation",
            }
        )

    negative_frame = pd.DataFrame(negatives, columns=DATASET_COLUMNS)
    stats = {
        "requested_negative_rows": int(count),
        "generated_negative_rows": int(len(negative_frame)),
        "negative_rows_shortfall": int(count - len(negative_frame)),
        "negative_sampling_attempts": int(attempts),
    }
    return negative_frame, stats


def split_by_pair_identity(dataset: pd.DataFrame, random_state: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    pair_keys = make_pair_keys(dataset)
    grouped = dataset.assign(pair_key=pair_keys).groupby("pair_key")["label"]
    pair_labels = grouped.first()
    mixed_pairs = grouped.nunique().gt(1)
    pair_labels = pair_labels.mask(mixed_pairs, MIXED_PAIR_LABEL)

    unique_pair_keys = pair_labels.index.to_numpy()
    stratify = pair_labels.to_numpy() if pair_labels.value_counts().min() >= 2 else None
    train_pairs, test_pairs = train_test_split(
        unique_pair_keys,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    train_pair_set = set(train_pairs)
    test_pair_set = set(test_pairs)
    train_idx = np.flatnonzero(pair_keys.isin(train_pair_set).to_numpy())
    test_idx = np.flatnonzero(pair_keys.isin(test_pair_set).to_numpy())

    stats = {
        "unique_pair_count": int(len(unique_pair_keys)),
        "train_unique_pair_count": int(len(train_pair_set)),
        "test_unique_pair_count": int(len(test_pair_set)),
        "mixed_label_pair_count": int(mixed_pairs.sum()),
        "train_test_pair_overlap": int(len(train_pair_set & test_pair_set)),
    }
    return train_idx, test_idx, stats


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    tasks = read_tasks(args.tasks, args.max_tasks)
    links = read_links(args.links, args.max_positive_links)

    task_ids = set(tasks["task_id"].astype(str))
    links = links[
        links["source_task_id"].astype(str).isin(task_ids)
        & links["target_task_id"].astype(str).isin(task_ids)
    ].copy()
    # Collapse many Jira-specific relation names into a smaller set of labels
    # such as depends, blocks, duplicate, related, and no_relation.
    links["label"] = [
        normalize_relation_label(row.relation_name, row.relation_description)
        for row in links.itertuples(index=False)
    ]

    counts = links["label"].value_counts()
    keep = counts[counts >= args.min_class_count].index
    positives = links[links["label"].isin(keep)].copy()
    positives = positives[DATASET_COLUMNS].copy()
    positives, positive_stats = deduplicate_positive_rows(positives, args.conflict_policy)
    positive_pairs = set(zip(positives["source_task_id"].astype(str), positives["target_task_id"].astype(str)))

    negative_count = int(len(positives) * args.negative_ratio)
    negatives, negative_stats = build_negative_pairs(tasks, positive_pairs, negative_count, args.random_state)
    if positive_stats["conflicting_positive_pairs"] > 0:
        if args.conflict_policy == "drop":
            print(
                f"Warning: dropped {positive_stats['positive_conflict_rows_removed']} row(s) from "
                f"{positive_stats['conflicting_positive_pairs']} pair(s) with multiple labels."
            )
        else:
            print(
                f"Warning: found {positive_stats['conflicting_positive_pairs']} pair(s) with multiple labels; "
                "they will stay within the same train/test split."
            )
    if negative_stats["negative_rows_shortfall"] > 0:
        print(
            f"Warning: requested {negative_stats['requested_negative_rows']} no_relation negatives but generated "
            f"{negative_stats['generated_negative_rows']}."
        )

    # The final training set mixes true positive relations with sampled negative
    # pairs. Shuffling prevents the model from seeing examples grouped by label.
    dataset = pd.concat(
        [
            positives[DATASET_COLUMNS],
            negatives,
        ],
        ignore_index=True,
    )
    dataset_rows_before_final_dedup = len(dataset)
    dataset = dataset.drop_duplicates(subset=PAIR_COLUMNS + ["label"]).copy()
    dataset = dataset.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

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
    # Convert relation labels to integer ids before neural-network training.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split by pair identity so the exact same source->target pair cannot appear
    # in both train and test, even if the raw data repeated that pair.
    train_idx, test_idx, split_stats = split_by_pair_identity(dataset, args.random_state)

    # Fit TF-IDF only on training pairs. The held-out relation pairs must not
    # influence the vocabulary or IDF statistics used for evaluation.
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform([texts[i] for i in train_idx])
    x_test = vectorizer.transform([texts[i] for i in test_idx])

    algorithm = choose_algorithm(args.algorithm)
    # The classifier maps TF-IDF pair vectors to relation probabilities. The
    # default MLP remains available, while auto mode uses the validated fast
    # linear relation classifier.
    model = build_classifier(algorithm, args)
    fit_warnings = show_and_collect_fit_warnings(model, x_train, y_encoded[train_idx])

    # Evaluate on held-out pairs that were not used for fitting.
    predictions = label_encoder.inverse_transform(model.predict(x_test))
    report = classification_report(y.iloc[test_idx], predictions, output_dict=True, zero_division=0)
    metrics = {
        "train_rows": int(len(train_idx)),
        "algorithm": algorithm,
        "test_rows": int(len(test_idx)),
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "dataset_rows_before_final_dedup": int(dataset_rows_before_final_dedup),
        "dataset_rows_after_final_dedup": int(len(dataset)),
        "dataset_duplicate_rows_removed": int(dataset_rows_before_final_dedup - len(dataset)),
        "classes": y.value_counts().to_dict(),
        "accuracy": float(accuracy_score(y.iloc[test_idx], predictions)),
        "majority_baseline_accuracy": float(y.iloc[test_idx].value_counts(normalize=True).max()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": extract_class_metrics(report),
        "n_iter": int(getattr(model, "n_iter_", 0)),
        "max_iter": int(args.max_iter),
        "fit_warnings": fit_warnings,
    }
    metrics.update(positive_stats)
    metrics.update(negative_stats)
    metrics.update(split_stats)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    bundle = {
        "kind": "task_relation_model",
        "vectorizer": vectorizer,
        "model": model,
        "algorithm": algorithm,
        "label_encoder": label_encoder,
        "metrics": metrics,
        "evaluation_protocol": "deduplicate exact pair-label rows, drop conflicting pair labels by default, split by unique source->target pair identity, then fit TF-IDF on train pairs only",
        "args": vars(args),
    }
    joblib.dump(bundle, args.model_dir / "link_model.joblib")
    (args.model_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved model to {args.model_dir / 'link_model.joblib'}")


if __name__ == "__main__":
    main()
