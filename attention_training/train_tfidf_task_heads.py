"""Train fast TF-IDF task heads on prepared TAWOS task data.

This script is intentionally separate from the MiniLM embedding-head trainer so
we can compare lightweight text models without re-encoding transformer
embeddings for every candidate.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "attention_training" / "data" / "strategy_v3_random50_task_optimized"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "attention_training" / "models" / "tfidf_task_heads_strategy_v3_random50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF task heads on prepared 50% task data.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-features", type=int, default=180000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--targets",
        default="task_type,priority,difficulty_label",
        help="Comma-separated targets to train.",
    )
    parser.add_argument(
        "--candidate-profile",
        choices=["compact", "full"],
        default="compact",
        help="compact is faster; full tries more regularization settings.",
    )
    parser.add_argument(
        "--difficulty-label-source",
        choices=["story_point_threshold", "project_quantile"],
        default="story_point_threshold",
        help="Use global story-point thresholds or project-local story-point percentile labels.",
    )
    parser.add_argument("--difficulty-quantile-low", type=float, default=0.30)
    parser.add_argument("--difficulty-quantile-high", type=float, default=0.70)
    parser.add_argument(
        "--difficulty-project-min-count",
        type=int,
        default=30,
        help="Minimum story-point rows required for project-local difficulty labels.",
    )
    return parser.parse_args()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


def story_point_difficulty_labels(df: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(df["story_point_clean"], errors="coerce")
    labels = pd.Series("", index=df.index, dtype=object)
    labels.loc[values.le(2)] = "easy"
    labels.loc[values.gt(2) & values.le(5)] = "medium"
    labels.loc[values.gt(5)] = "hard"
    return labels.astype(str)


def project_quantile_difficulty_labels(
    df: pd.DataFrame,
    low_quantile: float,
    high_quantile: float,
    min_project_count: int,
) -> pd.Series:
    values = pd.to_numeric(df["story_point_clean"], errors="coerce")
    projects = df["project_key"].fillna("").astype(str)
    valid = values.notna() & projects.str.len().gt(0)
    project_counts = projects.loc[valid].value_counts()
    eligible_project = projects.map(project_counts).fillna(0).ge(min_project_count)
    valid = valid & eligible_project

    ranks = pd.Series(np.nan, index=df.index, dtype=float)
    ranks.loc[valid] = values.loc[valid].groupby(projects.loc[valid]).rank(method="average", pct=True)

    labels = pd.Series("", index=df.index, dtype=object)
    labels.loc[valid & ranks.le(low_quantile)] = "easy"
    labels.loc[valid & ranks.gt(low_quantile) & ranks.le(high_quantile)] = "medium"
    labels.loc[valid & ranks.gt(high_quantile)] = "hard"
    return labels.astype(str)


def difficulty_labels_for_args(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.Series, str]:
    if args.difficulty_label_source == "project_quantile":
        labels = project_quantile_difficulty_labels(
            df,
            args.difficulty_quantile_low,
            args.difficulty_quantile_high,
            args.difficulty_project_min_count,
        )
        source = (
            f"story_point_project_quantile_"
            f"{args.difficulty_quantile_low:.2f}_{args.difficulty_quantile_high:.2f}_"
            f"min_project_count_{args.difficulty_project_min_count}"
        )
        return labels, source
    return story_point_difficulty_labels(df), "story_point_clean_3class"


def target_specs(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    task_type_target = "task_type_model" if "task_type_model" in df.columns else "task_type_clean"
    difficulty_labels, difficulty_label_source = difficulty_labels_for_args(df, args)
    return {
        "task_type": {
            "label_source": task_type_target,
            "labels": df[task_type_target].fillna("").astype(str),
            "text_column": "task_model_text",
            "categorical_columns": ["project_key", "priority_original_coarse", "status_clean", "resolution_clean"],
        },
        "priority": {
            "label_source": "priority_original_coarse",
            "labels": df["priority_original_coarse"].fillna("").astype(str),
            "text_column": "priority_model_text",
            "categorical_columns": [task_type_target, "project_key", "status_clean", "resolution_clean"],
        },
        "difficulty_label": {
            "label_source": difficulty_label_source,
            "labels": difficulty_labels,
            "text_column": "difficulty_model_text",
            "categorical_columns": [task_type_target, "project_key", "priority_original_coarse"],
        },
    }


def candidate_models(target: str, profile: str, random_state: int) -> Dict[str, object]:
    base = {
        "linear_svc_balanced_c1": LinearSVC(class_weight="balanced", C=1.0, dual=False, max_iter=6000, random_state=random_state),
        "logreg_balanced_liblinear": LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=400,
            solver="liblinear",
            random_state=random_state,
        ),
    }
    if profile == "compact":
        return base
    full = dict(base)
    full.update(
        {
            "linear_svc_balanced_c05": LinearSVC(class_weight="balanced", C=0.5, dual=False, max_iter=6000, random_state=random_state),
            "linear_svc_balanced_c2": LinearSVC(class_weight="balanced", C=2.0, dual=False, max_iter=6000, random_state=random_state),
            "sgd_log_loss_balanced": SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                alpha=1e-5,
                max_iter=80,
                tol=1e-3,
                random_state=random_state,
            ),
        }
    )
    return full


def class_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> Dict[str, object]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            key: {
                "precision": float(value["precision"]),
                "recall": float(value["recall"]),
                "f1": float(value["f1-score"]),
                "support": int(value["support"]),
            }
            for key, value in report.items()
            if isinstance(value, dict) and key not in {"macro avg", "weighted avg"}
        },
    }


def filtered_target_frame(
    df: pd.DataFrame,
    labels: pd.Series,
    min_class_count: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    clean_labels = labels.fillna("").astype(str)
    mask = clean_labels.str.len().gt(0)
    filtered_df = df.loc[mask].reset_index(drop=True)
    filtered_y = clean_labels.loc[mask].reset_index(drop=True)
    keep = filtered_y.value_counts()
    keep = set(keep[keep >= min_class_count].index)
    class_mask = filtered_y.isin(keep)
    return filtered_df.loc[class_mask].reset_index(drop=True), filtered_y.loc[class_mask].reset_index(drop=True)


def build_pipeline(
    text_column: str,
    categorical_columns: List[str],
    classifier: object,
    max_features: int,
    min_df: int,
    max_df: float,
) -> Pipeline:
    transformer = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=max_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
                text_column,
            ),
            ("categorical", make_one_hot_encoder(), categorical_columns),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline([("features", transformer), ("classifier", classifier)])


def train_target(df: pd.DataFrame, target: str, spec: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    target_df, y = filtered_target_frame(df, spec["labels"], args.min_class_count)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        random_state=args.random_state,
        stratify=y if y.value_counts().min() >= 2 else None,
    )
    train_df = target_df.iloc[train_idx].copy()
    test_df = target_df.iloc[test_idx].copy()
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    candidate_results = []
    best_result = None
    best_pipeline = None
    for name, classifier in candidate_models(target, args.candidate_profile, args.random_state).items():
        pipeline = build_pipeline(
            str(spec["text_column"]),
            list(spec["categorical_columns"]),
            classifier,
            args.max_features,
            args.min_df,
            args.max_df,
        )
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(train_df, y_train)
        pred = pipeline.predict(test_df)
        elapsed = time.perf_counter() - started
        metrics = class_metrics(y_test, pred)
        metrics.update(
            {
                "candidate": name,
                "algorithm": classifier.__class__.__name__,
                "seconds": round(elapsed, 3),
                "warnings": [str(item.message) for item in caught],
            }
        )
        candidate_results.append(metrics)
        if best_result is None or metrics["macro_f1"] > best_result["macro_f1"]:
            best_result = metrics
            best_pipeline = pipeline
        print(json.dumps({target: metrics}, ensure_ascii=False, indent=2), flush=True)

    assert best_result is not None and best_pipeline is not None
    best_result.update(
        {
            "target": target,
            "label_source": spec["label_source"],
            "text_column": spec["text_column"],
            "categorical_columns": spec["categorical_columns"],
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "classes": y.value_counts().to_dict(),
            "candidate_profile": args.candidate_profile,
            "selection_metric": "macro_f1",
        }
    )
    return {
        "best_model": best_pipeline,
        "best_metrics": best_result,
        "candidate_metrics": candidate_results,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data_dir / "task_supervised.csv")
    specs = target_specs(df, args)
    selected_targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    unknown = sorted(set(selected_targets) - set(specs))
    if unknown:
        raise ValueError(f"Unknown target(s): {', '.join(unknown)}")

    models: Dict[str, object] = {}
    metrics: Dict[str, object] = {
        "settings": {
            "data_dir": str(args.data_dir),
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "min_class_count": args.min_class_count,
            "candidate_profile": args.candidate_profile,
            "random_state": args.random_state,
            "difficulty_label_source": args.difficulty_label_source,
            "difficulty_quantile_low": args.difficulty_quantile_low,
            "difficulty_quantile_high": args.difficulty_quantile_high,
            "difficulty_project_min_count": args.difficulty_project_min_count,
        },
        "targets": {},
    }
    for target in selected_targets:
        result = train_target(df, target, specs[target], args)
        models[target] = result["best_model"]
        metrics["targets"][target] = {
            "best": result["best_metrics"],
            "candidates": result["candidate_metrics"],
        }

    joblib.dump({"models": models, "settings": metrics["settings"]}, args.output_dir / "tfidf_task_heads.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved TF-IDF task heads to {args.output_dir / 'tfidf_task_heads.joblib'}")


if __name__ == "__main__":
    main()
