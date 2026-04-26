"""Train downstream heads on Sentence-BERT embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "attention_training" / "data" / "clean30"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "attention_training" / "models" / "embedding_heads_clean30"
NO_RELATION_LABELS = {"no_relation", "not_related"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train task heads on transformer sentence embeddings.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-task-rows", type=int, default=0, help="Use 0 for all prepared task rows.")
    parser.add_argument("--max-link-rows", type=int, default=0, help="Use 0 for all prepared link rows.")
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--max-classes", type=int, default=20)
    parser.add_argument("--max-text-chars", type=int, default=1800)
    parser.add_argument("--max-seq-length", type=int, default=192)
    parser.add_argument("--only-link", action="store_true", help="Train only the task relation head.")
    parser.add_argument("--skip-link", action="store_true", help="Skip the task relation head.")
    parser.add_argument("--use-structured-features", action="store_true", help="Append project/source/graph features to embedding features.")
    parser.add_argument("--two-stage-link", action="store_true", help="Train no_relation-vs-relation first, then positive relation type.")
    parser.add_argument("--oversample-minority", action="store_true", help="Oversample minority classes in MLP training folds.")
    parser.add_argument("--oversample-cap-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--task-head-profile",
        choices=["default", "optimized_linear"],
        default="default",
        help="Use the original task heads or optimized linear heads with cleaner priority/difficulty labels.",
    )
    return parser.parse_args()


def load_embedder(model_name: str, max_seq_length: int) -> object:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing sentence-transformers. Install with: "
            "/opt/anaconda3/bin/python -m pip install -r attention_training/requirements-attention.txt"
        ) from exc
    embedder = SentenceTransformer(model_name)
    embedder.max_seq_length = max_seq_length
    return embedder


def truncate_texts(texts: List[str], max_chars: int) -> List[str]:
    if max_chars <= 0:
        return texts
    return [text[:max_chars] for text in texts]


def encode_texts(embedder: object, texts: List[str], batch_size: int) -> np.ndarray:
    return np.asarray(
        embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    )


def class_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, object]:
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


def split_indices(labels: pd.Series, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    stratify = labels if labels.value_counts().min() >= 2 else None
    return train_test_split(indices, test_size=0.2, random_state=random_state, stratify=stratify)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def fit_feature_preprocessor(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    categorical_cols: Iterable[str],
    numeric_cols: Iterable[str],
) -> Dict[str, object]:
    categorical = [col for col in categorical_cols if col in df.columns]
    numeric = [col for col in numeric_cols if col in df.columns]
    preprocessor: Dict[str, object] = {"categorical_cols": categorical, "numeric_cols": numeric}
    if categorical:
        encoder = make_one_hot_encoder()
        encoder.fit(df.iloc[train_idx][categorical].fillna("").astype(str))
        preprocessor["one_hot"] = encoder
    if numeric:
        scaler = StandardScaler()
        values = df.iloc[train_idx][numeric].apply(pd.to_numeric, errors="coerce").fillna(-1.0)
        scaler.fit(values)
        preprocessor["scaler"] = scaler
    return preprocessor


def transform_features(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    indices: np.ndarray,
    preprocessor: Optional[Dict[str, object]],
) -> object:
    base = sparse.csr_matrix(embeddings[indices])
    if not preprocessor:
        return embeddings[indices]
    parts = [base]
    categorical = preprocessor.get("categorical_cols", [])
    numeric = preprocessor.get("numeric_cols", [])
    if categorical:
        encoder = preprocessor["one_hot"]
        parts.append(encoder.transform(df.iloc[indices][categorical].fillna("").astype(str)))
    if numeric:
        scaler = preprocessor["scaler"]
        values = df.iloc[indices][numeric].apply(pd.to_numeric, errors="coerce").fillna(-1.0)
        parts.append(sparse.csr_matrix(scaler.transform(values)))
    return sparse.hstack(parts, format="csr")


def balanced_fit_positions(y_encoded: np.ndarray, random_state: int, cap_multiplier: float) -> np.ndarray:
    counts = pd.Series(y_encoded).value_counts()
    if counts.empty or counts.min() == counts.max():
        return np.arange(len(y_encoded))
    rng = np.random.default_rng(random_state)
    target = int(min(counts.max(), max(counts.median() * cap_multiplier, counts.min())))
    positions: List[np.ndarray] = []
    for label, count in counts.items():
        class_positions = np.flatnonzero(y_encoded == label)
        if count >= target:
            positions.append(class_positions)
            continue
        extra = rng.choice(class_positions, size=target - count, replace=True)
        positions.append(np.concatenate([class_positions, extra]))
    fit_positions = np.concatenate(positions)
    rng.shuffle(fit_positions)
    return fit_positions


def story_point_difficulty_labels(df: pd.DataFrame) -> pd.Series:
    if "story_point_clean" not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    values = pd.to_numeric(df["story_point_clean"], errors="coerce")
    labels = pd.Series("", index=df.index, dtype=object)
    labels.loc[values.le(2)] = "easy"
    labels.loc[values.gt(2) & values.le(5)] = "medium"
    labels.loc[values.gt(5)] = "hard"
    return labels.astype(str)


def algorithm_name(model: object) -> str:
    return model.__class__.__name__


def train_classification_head(
    embeddings: np.ndarray,
    labels: pd.Series,
    feature_df: pd.DataFrame,
    model: object,
    random_state: int,
    min_class_count: int,
    max_classes: int,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    oversample_minority: bool = False,
    oversample_cap_multiplier: float = 3.0,
) -> Dict[str, object]:
    mask = labels.astype(str).str.len().gt(0).to_numpy()
    filtered_x = embeddings[mask]
    filtered_y = labels[mask].astype(str).reset_index(drop=True)
    filtered_features = feature_df.loc[mask].reset_index(drop=True)
    keep = filtered_y.value_counts()
    keep = set(keep[keep >= min_class_count].head(max_classes).index)
    class_mask = filtered_y.isin(keep).to_numpy()
    filtered_x = filtered_x[class_mask]
    filtered_y = filtered_y[class_mask].reset_index(drop=True)
    filtered_features = filtered_features.loc[class_mask].reset_index(drop=True)
    train_idx, test_idx = split_indices(filtered_y, random_state)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(filtered_y)
    preprocessor = None
    if categorical_cols or numeric_cols:
        preprocessor = fit_feature_preprocessor(
            filtered_features,
            train_idx,
            categorical_cols or [],
            numeric_cols or [],
        )
    x_train = transform_features(filtered_x, filtered_features, train_idx, preprocessor)
    x_test = transform_features(filtered_x, filtered_features, test_idx, preprocessor)
    y_train = y_encoded[train_idx]
    fit_positions = np.arange(len(y_train))
    if oversample_minority:
        fit_positions = balanced_fit_positions(y_train, random_state, oversample_cap_multiplier)
    model.fit(x_train[fit_positions], y_train[fit_positions])
    pred = encoder.inverse_transform(model.predict(x_test))
    metrics = class_metrics(filtered_y.iloc[test_idx], pred)
    metrics.update(
        {
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "classes": filtered_y.value_counts().to_dict(),
            "structured_categorical_features": list(categorical_cols or []),
            "structured_numeric_features": list(numeric_cols or []),
            "oversampled_train_rows": int(len(fit_positions)),
        }
    )
    return {
        "model": model,
        "label_encoder": encoder,
        "feature_preprocessor": preprocessor,
        "metrics": metrics,
    }


def train_story_point_head(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    random_state: int,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
) -> Dict[str, object]:
    mask = df["story_point_log"].notna().to_numpy()
    filtered_x = embeddings[mask]
    story_log = df.loc[mask, "story_point_log"].astype(float).reset_index(drop=True)
    story_raw = df.loc[mask, "story_point_clean"].astype(float).reset_index(drop=True)
    bin_labels = df.loc[mask, "story_point_bin5"].astype(str).reset_index(drop=True)
    train_idx, test_idx = split_indices(bin_labels, random_state)
    preprocessor = None
    if categorical_cols or numeric_cols:
        feature_df = df.loc[mask].reset_index(drop=True)
        preprocessor = fit_feature_preprocessor(feature_df, train_idx, categorical_cols or [], numeric_cols or [])
        x_train = transform_features(filtered_x, feature_df, train_idx, preprocessor)
        x_test = transform_features(filtered_x, feature_df, test_idx, preprocessor)
    else:
        x_train = filtered_x[train_idx]
        x_test = filtered_x[test_idx]
    model = Ridge(alpha=1.0)
    model.fit(x_train, story_log.iloc[train_idx])
    pred_log = model.predict(x_test)
    pred_raw = np.expm1(pred_log).clip(min=0)
    pred_bins = pd.cut(
        pd.Series(pred_raw),
        bins=[-np.inf, 1, 2, 5, 13, np.inf],
        labels=[f"difficulty_{i:02d}" for i in range(1, 6)],
        right=True,
    ).astype(str)
    bin_report = classification_report(bin_labels.iloc[test_idx], pred_bins, output_dict=True, zero_division=0)
    metrics = {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "story_point_rows": int(len(story_log)),
        "log_mae": float(mean_absolute_error(story_log.iloc[test_idx], pred_log)),
        "log_rmse": float(mean_squared_error(story_log.iloc[test_idx], pred_log) ** 0.5),
        "log_r2": float(r2_score(story_log.iloc[test_idx], pred_log)),
        "raw_mae": float(mean_absolute_error(story_raw.iloc[test_idx], pred_raw)),
        "bin_accuracy": float(accuracy_score(bin_labels.iloc[test_idx], pred_bins)),
        "bin_macro_f1": float(bin_report["macro avg"]["f1-score"]),
        "bin_weighted_f1": float(bin_report["weighted avg"]["f1-score"]),
        "structured_categorical_features": list(categorical_cols or []),
        "structured_numeric_features": list(numeric_cols or []),
    }
    return {"model": model, "feature_preprocessor": preprocessor, "metrics": metrics}


def train_two_stage_relation_head(
    embeddings: np.ndarray,
    links: pd.DataFrame,
    random_state: int,
    min_class_count: int,
    max_classes: int,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    oversample_minority: bool = False,
    oversample_cap_multiplier: float = 3.0,
) -> Dict[str, object]:
    relation = links["relation_label"].fillna("").astype(str).reset_index(drop=True)
    keep = relation.value_counts()
    keep = set(keep[keep >= min_class_count].head(max_classes).index)
    mask = relation.isin(keep).to_numpy()
    filtered_x = embeddings[mask]
    filtered_links = links.loc[mask].reset_index(drop=True)
    filtered_relation = relation[mask].reset_index(drop=True)
    train_idx, test_idx = split_indices(filtered_relation, random_state)
    no_relation_output_label = "not_related" if filtered_relation.isin(["not_related"]).any() else "no_relation"
    preprocessor = None
    if categorical_cols or numeric_cols:
        preprocessor = fit_feature_preprocessor(
            filtered_links,
            train_idx,
            categorical_cols or [],
            numeric_cols or [],
        )
    x_train = transform_features(filtered_x, filtered_links, train_idx, preprocessor)
    x_test = transform_features(filtered_x, filtered_links, test_idx, preprocessor)

    train_no_relation = filtered_relation.iloc[train_idx].isin(NO_RELATION_LABELS).to_numpy()
    test_no_relation = filtered_relation.iloc[test_idx].isin(NO_RELATION_LABELS).to_numpy()
    binary_train = np.where(train_no_relation, no_relation_output_label, "has_relation")
    binary_test = np.where(test_no_relation, no_relation_output_label, "has_relation")
    binary_encoder = LabelEncoder()
    y_binary_train = binary_encoder.fit_transform(binary_train)
    stage1 = LogisticRegression(class_weight="balanced", max_iter=300, solver="liblinear", random_state=random_state)
    stage1.fit(x_train, y_binary_train)
    stage1_pred = binary_encoder.inverse_transform(stage1.predict(x_test))
    stage1_metrics = class_metrics(pd.Series(binary_test), stage1_pred)

    positive_train_positions = np.flatnonzero(~train_no_relation)
    positive_test_positions = np.flatnonzero(~test_no_relation)
    relation_encoder = LabelEncoder()
    y_relation_train = relation_encoder.fit_transform(filtered_relation.iloc[train_idx].iloc[positive_train_positions])
    stage2 = MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, random_state=random_state)
    fit_positions = np.arange(len(y_relation_train))
    if oversample_minority:
        fit_positions = balanced_fit_positions(y_relation_train, random_state, oversample_cap_multiplier)
    stage2.fit(x_train[positive_train_positions][fit_positions], y_relation_train[fit_positions])

    cascade_pred = np.full(len(test_idx), no_relation_output_label, dtype=object)
    predicted_positive_positions = np.flatnonzero(stage1_pred == "has_relation")
    if len(predicted_positive_positions):
        cascade_pred[predicted_positive_positions] = relation_encoder.inverse_transform(
            stage2.predict(x_test[predicted_positive_positions])
        )
    cascade_metrics = class_metrics(filtered_relation.iloc[test_idx], cascade_pred)

    if len(positive_test_positions):
        stage2_pred = relation_encoder.inverse_transform(stage2.predict(x_test[positive_test_positions]))
        stage2_metrics = class_metrics(filtered_relation.iloc[test_idx].iloc[positive_test_positions], stage2_pred)
    else:
        stage2_metrics = {}

    cascade_metrics.update(
        {
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "classes": filtered_relation.value_counts().to_dict(),
            "structured_categorical_features": list(categorical_cols or []),
            "structured_numeric_features": list(numeric_cols or []),
            "oversampled_stage2_train_rows": int(len(fit_positions)),
            "stage1_binary": stage1_metrics,
            "stage2_positive_type": stage2_metrics,
        }
    )
    return {
        "stage1_model": stage1,
        "stage1_label_encoder": binary_encoder,
        "stage2_model": stage2,
        "stage2_label_encoder": relation_encoder,
        "feature_preprocessor": preprocessor,
        "metrics": cascade_metrics,
    }


def main() -> None:
    args = parse_args()
    if args.only_link and args.skip_link:
        raise SystemExit("--only-link and --skip-link cannot be used together.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedder = load_embedder(args.model_name, args.max_seq_length)

    heads: Dict[str, object] = {"model_name": args.model_name, "task_heads": {}, "link_head": None}
    metrics: Dict[str, object] = {}

    if not args.only_link:
        tasks = pd.read_csv(args.data_dir / "task_supervised.csv")
        if args.max_task_rows:
            tasks = tasks.head(args.max_task_rows).copy()
        task_texts = truncate_texts(tasks["text"].fillna("").astype(str).tolist(), args.max_text_chars)
        task_embeddings = encode_texts(embedder, task_texts, args.batch_size)
        task_type_target = "task_type_model" if "task_type_model" in tasks.columns else "task_type_clean"
        task_feature_specs = {
            "task_type": {
                "categorical": ["project_key", "priority_original_coarse", "status_clean", "resolution_clean"],
                "numeric": [],
            },
            "priority": {
                "categorical": [task_type_target, "project_key", "status_clean", "resolution_clean"],
                "numeric": [],
            },
            "difficulty_label": {
                "categorical": [task_type_target, "project_key", "priority_original_coarse"],
                "numeric": [],
            },
            "story_point": {
                "categorical": [task_type_target, "project_key", "priority_original_coarse"],
                "numeric": [],
            },
        }

        head_specs = {
            "task_type": {
                "labels": tasks[task_type_target].fillna("").astype(str),
                "model": MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, random_state=args.random_state),
                "label_source": task_type_target,
            },
            "priority": {
                "labels": tasks["priority_coarse"].fillna("").astype(str),
                "model": LogisticRegression(class_weight="balanced", max_iter=200, solver="liblinear", random_state=args.random_state),
                "label_source": "priority_coarse",
            },
            "difficulty_label": {
                "labels": tasks["difficulty_label_derived"].fillna("").astype(str),
                "model": MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, random_state=args.random_state),
                "label_source": "difficulty_label_derived",
            },
        }
        if args.task_head_profile == "optimized_linear":
            head_specs = {
                "task_type": {
                    "labels": tasks[task_type_target].fillna("").astype(str),
                    "model": LinearSVC(class_weight="balanced", dual=False, max_iter=5000, random_state=args.random_state),
                    "label_source": task_type_target,
                },
                "priority": {
                    "labels": tasks["priority_original_coarse"].fillna("").astype(str),
                    "model": LogisticRegression(class_weight="balanced", max_iter=300, solver="liblinear", random_state=args.random_state),
                    "label_source": "priority_original_coarse",
                },
                "difficulty_label": {
                    "labels": story_point_difficulty_labels(tasks),
                    "model": LinearSVC(class_weight="balanced", dual=False, max_iter=5000, random_state=args.random_state),
                    "label_source": "story_point_clean_3class",
                },
            }
        for target, spec in head_specs.items():
            labels = spec["labels"]
            model = spec["model"]
            feature_spec = task_feature_specs[target] if args.use_structured_features else {"categorical": [], "numeric": []}
            result = train_classification_head(
                task_embeddings,
                labels,
                tasks,
                model,
                args.random_state,
                args.min_class_count,
                args.max_classes,
                categorical_cols=feature_spec["categorical"],
                numeric_cols=feature_spec["numeric"],
                oversample_minority=args.oversample_minority and isinstance(model, MLPClassifier),
                oversample_cap_multiplier=args.oversample_cap_multiplier,
            )
            result["metrics"].update(
                {
                    "algorithm": algorithm_name(model),
                    "task_head_profile": args.task_head_profile,
                    "target_label_source": spec["label_source"],
                }
            )
            heads["task_heads"][target] = {
                "model": result["model"],
                "label_encoder": result["label_encoder"],
                "feature_preprocessor": result["feature_preprocessor"],
            }
            metrics[target] = result["metrics"]
            print(json.dumps({target: result["metrics"]}, indent=2, ensure_ascii=False))

        story_feature_spec = task_feature_specs["story_point"] if args.use_structured_features else {"categorical": [], "numeric": []}
        story = train_story_point_head(
            task_embeddings,
            tasks,
            args.random_state,
            categorical_cols=story_feature_spec["categorical"],
            numeric_cols=story_feature_spec["numeric"],
        )
        heads["task_heads"]["story_point"] = {
            "model": story["model"],
            "feature_preprocessor": story["feature_preprocessor"],
        }
        metrics["story_point"] = story["metrics"]
        print(json.dumps({"story_point": story["metrics"]}, indent=2, ensure_ascii=False))

    if not args.skip_link:
        links = pd.read_csv(args.data_dir / "link_supervised.csv")
        if args.max_link_rows:
            links = links.head(args.max_link_rows).copy()
        link_text_column = "pair_model_text" if "pair_model_text" in links.columns else "pair_text"
        link_texts = truncate_texts(links[link_text_column].fillna("").astype(str).tolist(), args.max_text_chars)
        link_embeddings = encode_texts(embedder, link_texts, args.batch_size)
        link_feature_spec = {
            "categorical": ["source_project_key", "target_project_key", "source_task_type", "target_task_type", "source_priority", "target_priority"],
            "numeric": ["same_project", "same_task_type", "same_priority", "component_overlap", "component_overlap_count", "story_point_abs_diff"],
        } if args.use_structured_features else {"categorical": [], "numeric": []}
        if args.two_stage_link:
            link_result = train_two_stage_relation_head(
                link_embeddings,
                links,
                args.random_state,
                args.min_class_count,
                args.max_classes,
                categorical_cols=link_feature_spec["categorical"],
                numeric_cols=link_feature_spec["numeric"],
                oversample_minority=args.oversample_minority,
                oversample_cap_multiplier=args.oversample_cap_multiplier,
            )
            heads["link_head"] = {
                "mode": "two_stage",
                "stage1_model": link_result["stage1_model"],
                "stage1_label_encoder": link_result["stage1_label_encoder"],
                "stage2_model": link_result["stage2_model"],
                "stage2_label_encoder": link_result["stage2_label_encoder"],
                "feature_preprocessor": link_result["feature_preprocessor"],
            }
        else:
            link_result = train_classification_head(
                link_embeddings,
                links["relation_label"].fillna("").astype(str),
                links,
                MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, random_state=args.random_state),
                args.random_state,
                args.min_class_count,
                args.max_classes,
                categorical_cols=link_feature_spec["categorical"],
                numeric_cols=link_feature_spec["numeric"],
                oversample_minority=args.oversample_minority,
                oversample_cap_multiplier=args.oversample_cap_multiplier,
            )
            heads["link_head"] = {
                "mode": "direct",
                "model": link_result["model"],
                "label_encoder": link_result["label_encoder"],
                "feature_preprocessor": link_result["feature_preprocessor"],
            }
        metrics["task_relation"] = link_result["metrics"]
        print(json.dumps({"task_relation": link_result["metrics"]}, indent=2, ensure_ascii=False))

    joblib.dump(heads, args.output_dir / "embedding_heads.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved embedding heads to {args.output_dir / 'embedding_heads.joblib'}")


if __name__ == "__main__":
    main()
