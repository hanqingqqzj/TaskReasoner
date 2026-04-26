"""Train two-hidden-layer MLP classifiers for task attributes and difficulty."""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder

from text_utils import make_task_text


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT_DIR / "data" / "processed" / "tasks.tsv"
DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "task_model"
TARGETS = ["task_type", "priority", "difficulty_label"]
STORY_POINT_BINS_10 = [-np.inf, 1, 2, 3, 5, 8, 13, 21, 34, 55, np.inf]
STORY_POINT_BINS_5 = [-np.inf, 1, 2, 5, 13, np.inf]
STORY_POINT_LABELS_10 = [f"difficulty_{index:02d}" for index in range(1, 11)]
STORY_POINT_LABELS_5 = [f"difficulty_{index:02d}" for index in range(1, 6)]
COARSE_PRIORITY_LABELS = {
    "Blocker": "high",
    "Critical": "high",
    "Highest": "high",
    "High": "high",
    "Major": "medium",
    "Medium": "medium",
    "Minor": "low",
    "Low": "low",
    "Lowest": "low",
    "Trivial": "low",
}
VALID_PRIORITY_LABELS = {
    "Blocker",
    "Critical",
    "Highest",
    "High",
    "Major",
    "Medium",
    "Minor",
    "Low",
    "Lowest",
    "Trivial",
    "To be reviewed",
}
MISSING_LABELS = {"", "NULL", "None", "nan"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train task attribute and difficulty MLP models.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-rows", type=int, default=0, help="Use 0 for all rows.")
    parser.add_argument("--max-features", type=int, default=80000)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--hidden-size-1", type=int, default=128)
    parser.add_argument("--hidden-size-2", type=int, default=64)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--max-classes-per-target", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--algorithm",
        choices=["mlp", "sgd_log_loss", "sgd_log_loss_balanced", "logreg_balanced", "complement_nb", "auto", "final"],
        default="mlp",
        help="Classifier to train. final uses the selected target-specific model plan.",
    )
    parser.add_argument("--train-story-point-regressor", action="store_true")
    parser.add_argument("--coarse-priority", action="store_true")
    parser.add_argument("--derive-difficulty-from-story-point", action="store_true")
    parser.add_argument("--story-point-max", type=float, default=0, help="Use 0 to keep all positive story points.")
    parser.add_argument("--story-point-bins", choices=["ten", "five"], default="ten")
    return parser.parse_args()


def read_tasks(path: Path, max_rows: int) -> pd.DataFrame:
    columns = ["title", "description_text", *TARGETS, "story_point"]
    nrows = None if max_rows == 0 else max_rows
    return pd.read_csv(
        path,
        sep="\t",
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        nrows=nrows,
    )


def normalize_priority_label(value: object) -> object:
    label = str(value).strip()
    if label in MISSING_LABELS:
        return np.nan
    label = re.sub(r"\s+-\s+P[0-9]+$", "", label)
    return label if label in VALID_PRIORITY_LABELS else np.nan


def clean_story_point(df: pd.DataFrame, max_value: float = 0) -> pd.Series:
    story_point = pd.to_numeric(df["story_point"].astype(str).str.strip(), errors="coerce")
    story_point = story_point.where(story_point.gt(0))
    if max_value > 0:
        story_point = story_point.where(story_point.le(max_value))
    return story_point


def story_point_bins(args: argparse.Namespace) -> tuple[List[float], List[str]]:
    if args.story_point_bins == "five":
        return STORY_POINT_BINS_5, STORY_POINT_LABELS_5
    return STORY_POINT_BINS_10, STORY_POINT_LABELS_10


def derive_difficulty_label(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    story_point = clean_story_point(df, args.story_point_max)
    labels = pd.Series(np.nan, index=df.index, dtype=object)
    labels = labels.mask(story_point.le(2), "easy")
    labels = labels.mask(story_point.gt(2) & story_point.le(5), "medium")
    labels = labels.mask(story_point.gt(5), "hard")
    return labels


def prepare_target(df: pd.DataFrame, target: str, args: argparse.Namespace) -> pd.Series:
    # Keep only frequent, usable classes so the classifier is not dominated by
    # one-off labels that cannot be learned or fairly evaluated.
    y = df[target].astype(str).str.strip()
    if target == "priority":
        # Priority labels in TAWOS include aliases such as "Major - P3" and some
        # malformed non-priority values. Normalize aliases and drop invalid labels.
        y = y.map(normalize_priority_label)
        if args.coarse_priority:
            y = y.map(COARSE_PRIORITY_LABELS)
    elif target == "difficulty_label" and args.derive_difficulty_from_story_point:
        y = derive_difficulty_label(df, args)
    else:
        y = y.replace({label: np.nan for label in MISSING_LABELS})
    counts = y.value_counts(dropna=True)
    keep = counts[counts >= args.min_class_count].head(args.max_classes_per_target).index
    return y.where(y.isin(keep))


def choose_algorithm(requested: str, target: str) -> str:
    if requested == "final":
        if target == "priority":
            return "logreg_balanced"
        return "mlp"
    if requested != "auto":
        return requested
    if target == "priority":
        return "sgd_log_loss_balanced"
    if target == "difficulty_label":
        return "complement_nb"
    return "sgd_log_loss"


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
    if algorithm == "logreg_balanced":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=max(100, args.max_iter),
            solver="liblinear",
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


def scalar_n_iter(model: object) -> int:
    n_iter = getattr(model, "n_iter_", 0)
    if isinstance(n_iter, np.ndarray):
        return int(n_iter.max()) if n_iter.size else 0
    return int(n_iter)


def story_point_bin_labels(values: object, args: argparse.Namespace) -> pd.Series:
    bins, labels = story_point_bins(args)
    return pd.cut(pd.Series(values), bins=bins, labels=labels, right=True).astype(str)


def train_one_target(
    texts: List[str],
    y: pd.Series,
    target: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    mask = y.notna().to_numpy()
    classes = y[mask].value_counts()
    if len(classes) < 2:
        raise ValueError(f"{target} has fewer than 2 usable classes after filtering.")

    # Neural classifiers train on integer class ids. The label encoder maps
    # original labels such as "Bug" or "hard" to 0, 1, 2, ...
    indices = np.flatnonzero(mask)
    label_encoder = LabelEncoder()
    encoded = np.full(len(y), -1, dtype=int)
    encoded[indices] = label_encoder.fit_transform(y.iloc[indices].astype(str))

    # Stratification preserves class proportions in the train/test split, which
    # is important because TAWOS labels are highly imbalanced.
    stratify = encoded[indices] if classes.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.random_state,
        stratify=stratify,
    )

    # Fit TF-IDF only on training text. Fitting on all texts would leak test-set
    # vocabulary and IDF statistics into evaluation.
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

    algorithm = choose_algorithm(args.algorithm, target)
    # The classifier maps TF-IDF vectors to class probabilities. The default MLP
    # remains available, while auto mode uses faster validated text classifiers.
    model = build_classifier(algorithm, args)
    fit_warnings = show_and_collect_fit_warnings(model, x_train, encoded[train_idx])

    # Convert predicted class ids back to human-readable labels before scoring.
    predictions = label_encoder.inverse_transform(model.predict(x_test))
    truth = y.iloc[test_idx].astype(str)
    accuracy = accuracy_score(truth, predictions)
    report = classification_report(truth, predictions, output_dict=True, zero_division=0)

    return {
        "model": model,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "algorithm": algorithm,
        "metrics": {
            "target": target,
            "algorithm": algorithm,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
            "classes": classes.to_dict(),
            "accuracy": float(accuracy),
            "majority_baseline_accuracy": float(truth.value_counts(normalize=True).max()),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "per_class": extract_class_metrics(report),
            "n_iter": scalar_n_iter(model),
            "max_iter": int(getattr(model, "max_iter", args.max_iter)),
            "fit_warnings": fit_warnings,
        },
    }


def train_story_point_regressor(texts: List[str], df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, object]:
    y = clean_story_point(df, args.story_point_max)
    mask = y.notna().to_numpy()
    indices = np.flatnonzero(mask)
    if len(indices) < 20:
        raise ValueError("story_point has fewer than 20 positive numeric rows.")

    y_values = y.iloc[indices].astype(float)
    y_bins = story_point_bin_labels(y_values, args)
    stratify = y_bins if y_bins.value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.random_state,
        stratify=stratify,
    )

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

    y_train_raw = y.iloc[train_idx].astype(float).to_numpy()
    y_test_raw = y.iloc[test_idx].astype(float).to_numpy()
    y_train_log = np.log1p(y_train_raw)
    y_test_log = np.log1p(y_test_raw)

    model = Ridge(alpha=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, y_train_log)

    predicted_log = model.predict(x_test)
    predicted_raw = np.expm1(predicted_log).clip(min=0)
    truth_bins = story_point_bin_labels(y_test_raw, args)
    predicted_bins = story_point_bin_labels(predicted_raw, args)
    bin_report = classification_report(truth_bins, predicted_bins, output_dict=True, zero_division=0)
    bins, labels = story_point_bins(args)
    metrics = {
        "target": "story_point",
        "algorithm": "ridge_log_story_point",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "story_point_positive_rows": int(len(indices)),
        "story_point_max": float(args.story_point_max),
        "story_point_bins": y_bins.value_counts().sort_index().to_dict(),
        "log_mae": float(mean_absolute_error(y_test_log, predicted_log)),
        "log_rmse": float(mean_squared_error(y_test_log, predicted_log) ** 0.5),
        "log_r2": float(r2_score(y_test_log, predicted_log)),
        "raw_mae": float(mean_absolute_error(y_test_raw, predicted_raw)),
        "bin_accuracy": float(accuracy_score(truth_bins, predicted_bins)),
        "bin_macro_f1": float(bin_report["macro avg"]["f1-score"]),
        "bin_weighted_f1": float(bin_report["weighted avg"]["f1-score"]),
        "fit_warnings": [str(item.message) for item in caught],
    }
    return {
        "model": model,
        "vectorizer": vectorizer,
        "metrics": metrics,
        "bins": bins,
        "labels": labels,
    }


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    df = read_tasks(args.tasks, args.max_rows)
    texts: List[str] = [
        make_task_text(row.title, row.description_text)
        for row in df.itertuples(index=False)
    ]

    # Train one classifier per output target. Each target gets its own
    # train-only TF-IDF vectorizer because each target has a different valid
    # label mask and therefore a different train/test split.
    models: Dict[str, Dict[str, Any]] = {}
    metrics = []
    for target in TARGETS:
        y = prepare_target(df, target, args)
        result = train_one_target(texts, y, target, args)
        models[target] = {
            "classifier": result["model"],
            "label_encoder": result["label_encoder"],
            "vectorizer": result["vectorizer"],
            "algorithm": result["algorithm"],
        }
        metrics.append(result["metrics"])
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))

    regressors: Dict[str, Dict[str, Any]] = {}
    regressor_metrics = []
    if args.train_story_point_regressor:
        result = train_story_point_regressor(texts, df, args)
        regressors["story_point"] = {
            "model": result["model"],
            "vectorizer": result["vectorizer"],
            "bins": result["bins"],
            "labels": result["labels"],
        }
        regressor_metrics.append(result["metrics"])
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))

    bundle = {
        "kind": "task_attribute_and_difficulty_model",
        "models": models,
        "regressors": regressors,
        "targets": TARGETS,
        "metrics": metrics,
        "regressor_metrics": regressor_metrics,
        "evaluation_protocol": "split target indices first; fit each TF-IDF vectorizer on train text only",
        "args": vars(args),
    }
    joblib.dump(bundle, args.model_dir / "task_model.joblib")
    (args.model_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    if regressor_metrics:
        (args.model_dir / "regressor_metrics.json").write_text(json.dumps(regressor_metrics, ensure_ascii=False, indent=2))
    print(f"Saved model to {args.model_dir / 'task_model.joblib'}")


if __name__ == "__main__":
    main()
