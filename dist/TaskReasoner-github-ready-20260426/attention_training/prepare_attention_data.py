"""Prepare cleaned TAWOS data for transformer pretraining and fine-tuning.

Outputs:
- unlabeled_corpus.txt: cleaned task text for masked language-model pretraining.
- task_supervised.csv: labels for task_type, coarse priority, weak difficulty, and story point.
- link_supervised.csv: relation labels for task-pair classification.
- label_report.json: counts and cleaning statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text_utils import make_pair_text, make_task_text, normalize_relation_label  # noqa: E402


DEFAULT_TASKS = ROOT_DIR / "data" / "processed" / "tasks.tsv"
DEFAULT_LINKS = ROOT_DIR / "data" / "processed" / "task_links.tsv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "attention_training" / "data" / "clean30"

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
VALID_DIFFICULTY_LABELS = {"easy", "medium", "hard"}
COARSE_PRIORITY = {
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
MISSING = {"", "NULL", "None", "none", "nan", "NaN"}
TASK_COLUMNS = [
    "task_id",
    "project_key",
    "title",
    "description_text",
    "task_type",
    "priority",
    "status",
    "resolution",
    "difficulty_label",
    "story_point",
    "timespent",
    "in_progress_minutes",
    "total_effort_minutes",
    "resolution_time_minutes",
    "components",
]
LINK_COLUMNS = [
    "source_task_id",
    "target_task_id",
    "relation_name",
    "relation_description",
    "source_title",
    "source_description_text",
    "target_title",
    "target_description_text",
    "source_project_id",
    "target_project_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cleaned attention-model datasets.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-task-rows", type=int, default=120000, help="Use 0 for all task rows.")
    parser.add_argument("--max-link-rows", type=int, default=60000, help="Use 0 for all link rows.")
    parser.add_argument("--task-sample-frac", type=float, default=0.0, help="Randomly sample this fraction of raw task rows. Overrides --max-task-rows when > 0.")
    parser.add_argument("--link-sample-frac", type=float, default=0.0, help="Randomly sample this fraction of raw link rows. Overrides --max-link-rows when > 0.")
    parser.add_argument("--task-type-top-n", type=int, default=0, help="If > 0, keep the top N task types and map the rest to Other.")
    parser.add_argument(
        "--difficulty-label-mode",
        choices=["broad", "source_aware", "consensus"],
        default="broad",
        help="Use broad weak labels, stricter source-aware labels, or multi-source consensus labels.",
    )
    parser.add_argument("--difficulty-min-agree", type=int, default=2, help="Minimum agreeing sources for consensus difficulty labels.")
    parser.add_argument(
        "--difficulty-balance-ratio",
        type=float,
        default=0.0,
        help="If > 0, cap each difficulty class at min_class_count * ratio by dropping lower-trust weak labels.",
    )
    parser.add_argument("--story-point-max", type=float, default=100.0)
    parser.add_argument("--max-effort-minutes", type=float, default=10080.0)
    parser.add_argument("--min-text-chars", type=int, default=8)
    parser.add_argument("--min-link-class-count", type=int, default=30)
    parser.add_argument("--link-negative-ratio", type=float, default=1.0)
    parser.add_argument(
        "--relation-label-mode",
        choices=["fine", "coarse4", "polarity3"],
        default="fine",
        help=(
            "Use fine-grained relation labels, collapse them to depends/blocks/related/no_relation, "
            "or collapse them to not_related/positive_related/negative_related."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def clean_text(value: object) -> str:
    text = str(value or "").strip()
    if text in MISSING:
        return ""
    text = text.replace("\t", " ").replace("\n", " ").replace("\\t", " ").replace("\\n", " ")
    text = re.sub(r"<USER>", " user ", text)
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_priority(value: object) -> str:
    label = str(value or "").strip()
    if label in MISSING:
        return ""
    label = re.sub(r"\s+-\s+P[0-9]+$", "", label)
    return label if label in VALID_PRIORITY_LABELS else ""


def normalize_category(value: object) -> str:
    label = clean_text(value)
    return "" if label in MISSING else label


def normalize_difficulty_label(value: object) -> str:
    label = str(value or "").strip().lower()
    if label in MISSING:
        return ""
    return label if label in VALID_DIFFICULTY_LABELS else ""


def weak_priority_from_text(text: str) -> Tuple[str, str]:
    lowered = text.lower()
    if re.search(
        r"\b(blocker|critical|crash|security|data loss|production down|outage|urgent|"
        r"regression|broken|cannot start|fails to start|memory leak)\b",
        lowered,
    ):
        return "high", "keyword"
    if re.search(
        r"\b(doc|documentation|typo|cleanup|minor|trivial|wording|cosmetic|"
        r"spelling|readme|comment|translation)\b",
        lowered,
    ):
        return "low", "keyword"
    return "", ""


def weak_priority_from_task_type(task_type: str) -> Tuple[str, str]:
    if task_type in {"Documentation", "Wish", "Test"}:
        return "low", "task_type"
    if task_type in {
        "Bug",
        "Improvement",
        "Story",
        "Task",
        "Sub-task",
        "Suggestion",
        "New Feature",
        "Epic",
        "Enhancement Request",
        "Support Request",
        "Technical task",
        "Technical Debt",
    }:
        return "medium", "task_type"
    return "", ""


def weak_priority_from_story_point(story_point: float) -> Tuple[str, str]:
    if pd.isna(story_point):
        return "", ""
    if story_point >= 8:
        return "high", "story_point"
    if story_point <= 2:
        return "low", "story_point"
    return "medium", "story_point"


def clean_minutes(series: pd.Series, max_value: float) -> pd.Series:
    values = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
    values = values.where(values.gt(0))
    if max_value > 0:
        values = values.where(values.le(max_value))
    return values


def clean_story_point(series: pd.Series, max_value: float) -> pd.Series:
    values = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
    values = values.where(values.gt(0))
    if max_value > 0:
        values = values.where(values.le(max_value))
    return values


def first_label(candidates: Iterable[Tuple[str, str]]) -> Tuple[str, str]:
    for label, source in candidates:
        if label:
            return label, source
    return "", ""


def derive_difficulty(story_point: float) -> str:
    if pd.isna(story_point):
        return ""
    if story_point <= 2:
        return "easy"
    if story_point <= 5:
        return "medium"
    return "hard"


def weak_difficulty_from_effort(*minute_values: float) -> Tuple[str, str]:
    effort = next((value for value in minute_values if not pd.isna(value)), np.nan)
    if pd.isna(effort):
        return "", ""
    if effort <= 120:
        return "easy", "effort"
    if effort <= 960:
        return "medium", "effort"
    return "hard", "effort"


def weak_difficulty_from_resolution_time(minutes: float) -> Tuple[str, str]:
    if pd.isna(minutes):
        return "", ""
    if minutes <= 1440:
        return "easy", "resolution_time"
    if minutes <= 10080:
        return "medium", "resolution_time"
    return "hard", "resolution_time"


def weak_difficulty_from_text(text: str) -> Tuple[str, str]:
    lowered = text.lower()
    if re.search(
        r"\b(migration|refactor|architecture|scalability|performance|security|cluster|"
        r"distributed|concurrency|deadlock|race condition|data loss|memory leak|"
        r"breaking change|compatibility|upgrade|database|authentication|authorization)\b",
        lowered,
    ):
        return "hard", "text_keyword"
    if re.search(
        r"\b(typo|spelling|wording|readme|documentation|comment|cleanup|cosmetic|"
        r"rename|translation|minor|trivial)\b",
        lowered,
    ):
        return "easy", "text_keyword"
    word_count = len(text.split())
    if word_count <= 16:
        return "easy", "text_length"
    if word_count >= 140:
        return "hard", "text_length"
    return "medium", "text_length"


def weak_difficulty_from_task_type(task_type: str, allow_generic_medium: bool = True) -> Tuple[str, str]:
    if task_type in {"Documentation", "Wish", "Test", "Test Task", "Question", "Sub-task"}:
        return "easy", "task_type"
    if task_type in {"Epic", "Technical task", "Technical Debt", "Build Failure", "Problem Ticket"}:
        return "hard", "task_type"
    if allow_generic_medium and task_type in {
        "Bug",
        "Improvement",
        "Story",
        "Task",
        "Suggestion",
        "New Feature",
        "Support Request",
        "Enhancement Request",
    }:
        return "medium", "task_type"
    return "", ""


DIFFICULTY_SOURCE_RANK = {
    "consensus": 0,
    "original": 0,
    "story_point": 1,
    "effort": 2,
    "text_keyword": 3,
    "task_type": 4,
    "text_length": 5,
    "balance_drop": 99,
    "": 99,
}


def difficulty_source_rank(source: object) -> int:
    text = str(source or "")
    if text.startswith("consensus:"):
        return DIFFICULTY_SOURCE_RANK["consensus"]
    return DIFFICULTY_SOURCE_RANK.get(text, 99)


def difficulty_candidates(row: pd.Series, original_difficulty: str, mode: str) -> List[Tuple[str, str]]:
    candidates = [
        (original_difficulty, "original" if original_difficulty else ""),
        (derive_difficulty(row["story_point_clean"]), "story_point" if not pd.isna(row["story_point_clean"]) else ""),
        weak_difficulty_from_effort(
            row["timespent_clean"],
            row["total_effort_minutes_clean"],
            row["in_progress_minutes_clean"],
        ),
        weak_difficulty_from_task_type(row["task_type_clean"], allow_generic_medium=mode == "broad"),
        weak_difficulty_from_text(row["text"]),
    ]
    if mode == "source_aware":
        return [
            (label, source)
            for label, source in candidates
            if source and source != "text_length"
        ]
    return candidates


def consensus_difficulty_label(
    row: pd.Series,
    original_difficulty: str,
    min_agree: int,
) -> Tuple[str, str]:
    candidates = [
        (original_difficulty, "original" if original_difficulty else ""),
        (derive_difficulty(row["story_point_clean"]), "story_point" if not pd.isna(row["story_point_clean"]) else ""),
        weak_difficulty_from_effort(
            row["timespent_clean"],
            row["total_effort_minutes_clean"],
            row["in_progress_minutes_clean"],
        ),
        weak_difficulty_from_resolution_time(row["resolution_time_minutes_clean"]),
        weak_difficulty_from_task_type(row["task_type_clean"], allow_generic_medium=False),
    ]
    votes: Dict[str, List[str]] = {}
    for label, source in candidates:
        if label and source:
            votes.setdefault(label, []).append(source)
    if not votes:
        return "", ""
    label, sources = max(votes.items(), key=lambda item: (len(item[1]), -DIFFICULTY_SOURCE_RANK.get(item[1][0], 99)))
    if len(sources) < max(1, min_agree):
        return "", ""
    return label, "consensus:" + "+".join(sources)


def balance_difficulty_labels(
    output: pd.DataFrame,
    ratio: float,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if ratio <= 0:
        return output, {
            "difficulty_balance_ratio": float(ratio),
            "difficulty_counts_before_balance": output["difficulty_label_derived"].value_counts().to_dict(),
            "difficulty_counts_after_balance": output["difficulty_label_derived"].value_counts().to_dict(),
            "difficulty_balance_rows_removed": 0,
        }

    balanced = output.copy()
    labels = balanced["difficulty_label_derived"].astype(str)
    non_empty_counts = labels[labels.str.len().gt(0)].value_counts()
    if non_empty_counts.empty:
        return balanced, {
            "difficulty_balance_ratio": float(ratio),
            "difficulty_counts_before_balance": {},
            "difficulty_counts_after_balance": {},
            "difficulty_balance_rows_removed": 0,
        }

    cap = int(np.ceil(non_empty_counts.min() * ratio))
    drop_indices: List[int] = []
    rng = np.random.default_rng(random_state)
    random_order = pd.Series(rng.random(len(balanced)), index=balanced.index)

    for label, count in non_empty_counts.items():
        if count <= cap:
            continue
        label_rows = balanced[balanced["difficulty_label_derived"].eq(label)].copy()
        label_rows["_source_rank"] = label_rows["difficulty_label_source"].map(difficulty_source_rank)
        label_rows["_random_order"] = random_order.loc[label_rows.index]
        keep_indices = set(
            label_rows.sort_values(["_source_rank", "_random_order"])
            .head(cap)
            .index
        )
        drop_indices.extend(index for index in label_rows.index if index not in keep_indices)

    if drop_indices:
        balanced.loc[drop_indices, "difficulty_label_derived"] = ""
        balanced.loc[drop_indices, "difficulty_label_source"] = "balance_drop"

    after_counts = balanced["difficulty_label_derived"].value_counts().to_dict()
    return balanced, {
        "difficulty_balance_ratio": float(ratio),
        "difficulty_balance_class_cap": int(cap),
        "difficulty_counts_before_balance": non_empty_counts.to_dict(),
        "difficulty_counts_after_balance": after_counts,
        "difficulty_balance_rows_removed": int(len(drop_indices)),
    }


def story_point_bin5(story_point: float) -> str:
    if pd.isna(story_point):
        return ""
    if story_point <= 1:
        return "difficulty_01"
    if story_point <= 2:
        return "difficulty_02"
    if story_point <= 5:
        return "difficulty_03"
    if story_point <= 13:
        return "difficulty_04"
    return "difficulty_05"


def collapse_top_labels(series: pd.Series, top_n: int, other_label: str = "Other") -> pd.Series:
    labels = series.fillna("").astype(str).str.strip()
    if top_n <= 0:
        return labels
    counts = labels[labels.str.len().gt(0)].value_counts()
    keep = set(counts.head(top_n).index)
    return labels.where(labels.isin(keep) | labels.eq(""), other_label)


NO_RELATION_LABELS = {"no_relation", "not_related"}
NEGATIVE_RELATION_LABELS = {"blocks", "depends", "causal", "supersedes"}


def no_relation_label_for_mode(mode: str) -> str:
    return "not_related" if mode == "polarity3" else "no_relation"


def collapse_relation_label(label: str, mode: str) -> str:
    if mode == "fine":
        return label
    if mode == "coarse4":
        if label in {"no_relation", "depends", "blocks"}:
            return label
        return "related"
    if label in NO_RELATION_LABELS:
        return "not_related"
    if label in NEGATIVE_RELATION_LABELS:
        return "negative_related"
    return "positive_related"


def relation_binary_label(label: str) -> str:
    if label in NO_RELATION_LABELS:
        return label
    return "has_relation"


def component_set(value: object) -> Set[str]:
    text = clean_text(value).lower()
    if not text:
        return set()
    return {part.strip() for part in re.split(r"[,;|/]+", text) if part.strip()}


def add_metadata_prefix(text: str, metadata: Iterable[Tuple[str, object]]) -> str:
    parts = []
    for key, value in metadata:
        cleaned = clean_text(value)
        if cleaned:
            parts.append(f"{key}: {cleaned}")
    if parts:
        return " | ".join(parts) + " | TEXT: " + text
    return text


def write_text_lines(path: Path, texts: Iterable[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            if text:
                handle.write(text.replace("\n", " ") + "\n")
                count += 1
    return count


def make_prepared_pair_text(source_text: str, target_text: str) -> str:
    return f"SOURCE TASK: {source_text} TARGET TASK: {target_text}"


def prepare_tasks(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, object]]:
    nrows = None if args.task_sample_frac > 0 or args.max_task_rows == 0 else args.max_task_rows
    df = pd.read_csv(
        args.tasks,
        sep="\t",
        usecols=TASK_COLUMNS,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        nrows=nrows,
    )
    df = df[df["task_id"].ne("task_id")].reset_index(drop=True)
    raw_rows = len(df)
    if args.task_sample_frac > 0:
        df = df.sample(frac=args.task_sample_frac, random_state=args.random_state).reset_index(drop=True)
    df["title_clean"] = df["title"].map(clean_text)
    df["description_clean"] = df["description_text"].map(clean_text)
    df["text"] = [
        make_task_text(row.title_clean, row.description_clean)
        for row in df.itertuples(index=False)
    ]
    text_mask = df["text"].str.len().ge(args.min_text_chars)
    task_type = df["task_type"].astype(str).str.strip()
    df["task_type_clean"] = task_type.where(~task_type.isin(MISSING), "")
    df["task_type_model"] = collapse_top_labels(df["task_type_clean"], args.task_type_top_n)
    df["priority_original_clean"] = df["priority"].map(normalize_priority)
    df["priority_original_coarse"] = df["priority_original_clean"].map(COARSE_PRIORITY).fillna("")
    df["status_clean"] = df["status"].map(normalize_category)
    df["resolution_clean"] = df["resolution"].map(normalize_category)
    df["components_clean"] = df["components"].map(normalize_category)
    df["story_point_clean"] = clean_story_point(df["story_point"], args.story_point_max)
    df["story_point_log"] = np.log1p(df["story_point_clean"])
    df["story_point_bin5"] = df["story_point_clean"].map(story_point_bin5)
    df["timespent_clean"] = clean_minutes(df["timespent"], args.max_effort_minutes)
    df["total_effort_minutes_clean"] = clean_minutes(df["total_effort_minutes"], args.max_effort_minutes)
    df["in_progress_minutes_clean"] = clean_minutes(df["in_progress_minutes"], args.max_effort_minutes)
    df["resolution_time_minutes_clean"] = clean_minutes(df["resolution_time_minutes"], args.max_effort_minutes * 12)

    df["priority_coarse"] = df["priority_original_coarse"]
    priority_labels: List[str] = []
    priority_sources: List[str] = []
    difficulty_labels: List[str] = []
    difficulty_sources: List[str] = []
    normalized_difficulty = df["difficulty_label"].map(normalize_difficulty_label)

    for _, row in df.iterrows():
        priority_label, priority_source = first_label(
            [
                (row["priority_coarse"], "original" if row["priority_coarse"] else ""),
                weak_priority_from_story_point(row["story_point_clean"]),
                weak_priority_from_text(row["text"]),
                weak_priority_from_task_type(row["task_type_clean"]),
            ]
        )
        priority_labels.append(priority_label)
        priority_sources.append(priority_source)
        original_difficulty = normalized_difficulty.loc[row.name]
        if args.difficulty_label_mode == "consensus":
            difficulty_label, difficulty_source = consensus_difficulty_label(
                row,
                original_difficulty,
                args.difficulty_min_agree,
            )
        else:
            difficulty_label, difficulty_source = first_label(
                difficulty_candidates(row, original_difficulty, args.difficulty_label_mode)
            )
        difficulty_labels.append(difficulty_label)
        difficulty_sources.append(difficulty_source)

    df["priority_coarse"] = priority_labels
    df["priority_label_source"] = priority_sources
    df["difficulty_label_derived"] = difficulty_labels
    df["difficulty_label_source"] = difficulty_sources
    df["task_model_text"] = [
        add_metadata_prefix(
            row.text,
            [
                ("PROJECT", row.project_key),
                ("PRIORITY", row.priority_original_coarse),
                ("STATUS", row.status_clean),
                ("RESOLUTION", row.resolution_clean),
                ("COMPONENTS", row.components_clean),
            ],
        )
        for row in df.itertuples(index=False)
    ]
    df["priority_model_text"] = [
        add_metadata_prefix(
            row.text,
            [
                ("PROJECT", row.project_key),
                ("ISSUE_TYPE", row.task_type_model),
                ("STATUS", row.status_clean),
                ("RESOLUTION", row.resolution_clean),
                ("COMPONENTS", row.components_clean),
            ],
        )
        for row in df.itertuples(index=False)
    ]
    df["difficulty_model_text"] = [
        add_metadata_prefix(
            row.text,
            [
                ("PROJECT", row.project_key),
                ("ISSUE_TYPE", row.task_type_model),
                ("PRIORITY", row.priority_original_coarse),
                ("COMPONENTS", row.components_clean),
            ],
        )
        for row in df.itertuples(index=False)
    ]

    output = df.loc[
        text_mask,
        [
            "task_id",
            "project_key",
            "text",
            "task_model_text",
            "priority_model_text",
            "difficulty_model_text",
            "task_type_clean",
            "task_type_model",
            "priority_original_clean",
            "priority_original_coarse",
            "priority_coarse",
            "priority_label_source",
            "difficulty_label_derived",
            "difficulty_label_source",
            "story_point_clean",
            "story_point_log",
            "story_point_bin5",
            "status_clean",
            "resolution_clean",
            "components_clean",
            "timespent_clean",
            "total_effort_minutes_clean",
            "in_progress_minutes_clean",
            "resolution_time_minutes_clean",
        ],
    ].copy()
    output, balance_report = balance_difficulty_labels(
        output,
        args.difficulty_balance_ratio,
        args.random_state,
    )
    report = {
        "task_rows_read": int(raw_rows),
        "task_rows_after_random_sample": int(len(df)),
        "task_sample_frac": float(args.task_sample_frac),
        "task_type_top_n": int(args.task_type_top_n),
        "difficulty_label_mode": args.difficulty_label_mode,
        "difficulty_min_agree": int(args.difficulty_min_agree),
        "task_rows_with_text": int(len(output)),
        "task_type_counts": output["task_type_clean"].value_counts().head(20).to_dict(),
        "task_type_model_counts": output["task_type_model"].value_counts().to_dict(),
        "priority_coarse_counts": output["priority_coarse"].value_counts().to_dict(),
        "priority_label_source_counts": output["priority_label_source"].value_counts().to_dict(),
        "difficulty_label_derived_counts": output["difficulty_label_derived"].value_counts().to_dict(),
        "difficulty_label_source_counts": output["difficulty_label_source"].value_counts().to_dict(),
        **balance_report,
        "story_point_bin5_counts": output["story_point_bin5"].value_counts().to_dict(),
        "story_point_rows": int(output["story_point_clean"].notna().sum()),
        "story_point_max": float(args.story_point_max),
        "max_effort_minutes": float(args.max_effort_minutes),
    }
    return output, report


def drop_conflicting_link_pairs(links: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    before_dedup = len(links)
    deduped = links.drop_duplicates(subset=["source_task_id", "target_task_id", "relation_label"]).copy()
    label_counts = deduped.groupby(["source_task_id", "target_task_id"])["relation_label"].nunique()
    conflict_index = label_counts[label_counts > 1].index
    conflict_keys = set(tuple(map(str, pair)) for pair in conflict_index)
    conflict_mask = [
        (str(row.source_task_id), str(row.target_task_id)) in conflict_keys
        for row in deduped.itertuples(index=False)
    ]
    cleaned = deduped.loc[~np.asarray(conflict_mask, dtype=bool)].copy()
    return cleaned, {
        "positive_rows_before_dedup": int(before_dedup),
        "positive_duplicate_rows_removed": int(before_dedup - len(deduped)),
        "conflicting_positive_pairs": int(len(conflict_keys)),
        "positive_conflict_rows_removed": int(len(deduped) - len(cleaned)),
    }


def build_negative_links(
    tasks: pd.DataFrame,
    known_relation_pairs: Set[Tuple[str, str]],
    count: int,
    random_state: int,
    negative_label: str = "no_relation",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    columns = [
        "source_task_id",
        "target_task_id",
        "source_text",
        "target_text",
        "pair_text",
        "relation_label",
        "source_project_key",
        "target_project_key",
        "source_task_type",
        "target_task_type",
        "source_priority",
        "target_priority",
        "source_components",
        "target_components",
        "source_story_point",
        "target_story_point",
    ]
    if count <= 0:
        return pd.DataFrame(columns=columns), {
            "requested_negative_rows": int(count),
            "generated_negative_rows": 0,
            "negative_rows_shortfall": int(count),
            "negative_sampling_attempts": 0,
        }

    rng = np.random.default_rng(random_state)
    optional_cols = [
        "task_id",
        "project_key",
        "text",
        "task_type_model",
        "priority_coarse",
        "components_clean",
        "story_point_clean",
    ]
    task_rows = tasks[[col for col in optional_cols if col in tasks.columns]].copy()
    task_rows["task_id"] = task_rows["task_id"].astype(str)
    task_rows["project_key"] = task_rows["project_key"].astype(str)
    task_rows["text"] = task_rows["text"].fillna("").astype(str)
    for col in ["task_type_model", "priority_coarse", "components_clean"]:
        if col not in task_rows.columns:
            task_rows[col] = ""
        task_rows[col] = task_rows[col].fillna("").astype(str)
    if "story_point_clean" not in task_rows.columns:
        task_rows["story_point_clean"] = np.nan

    by_project: Dict[str, List[dict]] = {}
    for row in task_rows.to_dict("records"):
        by_project.setdefault(row["project_key"], []).append(row)
    projects = [project for project, rows in by_project.items() if len(rows) >= 2]

    negatives: List[dict] = []
    seen_negative_pairs: Set[Tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(count * 50, 1000)

    while len(negatives) < count and attempts < max_attempts and projects:
        attempts += 1
        project = rng.choice(projects)
        source, target = rng.choice(by_project[project], size=2, replace=False)
        pair = (str(source["task_id"]), str(target["task_id"]))
        reverse_pair = (pair[1], pair[0])
        if (
            pair in known_relation_pairs
            or reverse_pair in known_relation_pairs
            or pair in seen_negative_pairs
        ):
            continue
        seen_negative_pairs.add(pair)
        source_text = str(source["text"])
        target_text = str(target["text"])
        negatives.append(
            {
                "source_task_id": pair[0],
                "target_task_id": pair[1],
                "source_text": source_text,
                "target_text": target_text,
                "pair_text": make_prepared_pair_text(source_text, target_text),
                "relation_label": negative_label,
                "source_project_key": str(source.get("project_key", "")),
                "target_project_key": str(target.get("project_key", "")),
                "source_task_type": str(source.get("task_type_model", "")),
                "target_task_type": str(target.get("task_type_model", "")),
                "source_priority": str(source.get("priority_coarse", "")),
                "target_priority": str(target.get("priority_coarse", "")),
                "source_components": str(source.get("components_clean", "")),
                "target_components": str(target.get("components_clean", "")),
                "source_story_point": source.get("story_point_clean", np.nan),
                "target_story_point": target.get("story_point_clean", np.nan),
            }
        )

    frame = pd.DataFrame(negatives, columns=columns)
    return frame, {
        "requested_negative_rows": int(count),
        "generated_negative_rows": int(len(frame)),
        "negative_rows_shortfall": int(count - len(frame)),
        "negative_sampling_attempts": int(attempts),
    }


def add_pair_graph_features(links: pd.DataFrame, tasks: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    output = links.copy()
    task_meta_cols = [
        "task_id",
        "project_key",
        "task_type_model",
        "priority_coarse",
        "components_clean",
        "story_point_clean",
    ]
    available = [col for col in task_meta_cols if col in tasks.columns]
    lookup = tasks[available].copy()
    lookup["task_id"] = lookup["task_id"].astype(str)
    lookup = lookup.drop_duplicates("task_id").set_index("task_id")
    missing_source = 0
    missing_target = 0

    def fill_from_lookup(row: pd.Series, side: str, field: str, fallback: object = "") -> object:
        task_id = str(row[f"{side}_task_id"])
        if task_id not in lookup.index:
            return fallback
        source_field = {
            "project_key": "project_key",
            "task_type": "task_type_model",
            "priority": "priority_coarse",
            "components": "components_clean",
            "story_point": "story_point_clean",
        }[field]
        return lookup.at[task_id, source_field] if source_field in lookup.columns else fallback

    for side in ["source", "target"]:
        project_col = f"{side}_project_key"
        if project_col not in output.columns:
            output[project_col] = ""
        for field in ["task_type", "priority", "components", "story_point"]:
            col = f"{side}_{field}"
            if col not in output.columns:
                output[col] = ""

    for index, row in output.iterrows():
        source_known = str(row["source_task_id"]) in lookup.index
        target_known = str(row["target_task_id"]) in lookup.index
        missing_source += int(not source_known)
        missing_target += int(not target_known)
        for side in ["source", "target"]:
            for field in ["project_key", "task_type", "priority", "components", "story_point"]:
                col = f"{side}_{field}"
                empty_fallback = np.nan if field == "story_point" else ""
                current = row[col] if col in row and str(row[col]) not in MISSING else empty_fallback
                output.at[index, col] = fill_from_lookup(row, side, field, current)

    output["source_project_key"] = output["source_project_key"].fillna("").astype(str)
    output["target_project_key"] = output["target_project_key"].fillna("").astype(str)
    output["source_task_type"] = output["source_task_type"].fillna("").astype(str)
    output["target_task_type"] = output["target_task_type"].fillna("").astype(str)
    output["source_priority"] = output["source_priority"].fillna("").astype(str)
    output["target_priority"] = output["target_priority"].fillna("").astype(str)
    output["source_components"] = output["source_components"].fillna("").astype(str)
    output["target_components"] = output["target_components"].fillna("").astype(str)
    output["source_story_point"] = pd.to_numeric(output["source_story_point"], errors="coerce")
    output["target_story_point"] = pd.to_numeric(output["target_story_point"], errors="coerce")

    source_component_sets = output["source_components"].map(component_set)
    target_component_sets = output["target_components"].map(component_set)
    overlap_counts = [
        len(source_components & target_components)
        for source_components, target_components in zip(source_component_sets, target_component_sets)
    ]
    output["same_project"] = (
        output["source_project_key"].str.len().gt(0)
        & output["source_project_key"].eq(output["target_project_key"])
    ).astype(int)
    output["same_task_type"] = (
        output["source_task_type"].str.len().gt(0)
        & output["source_task_type"].eq(output["target_task_type"])
    ).astype(int)
    output["same_priority"] = (
        output["source_priority"].str.len().gt(0)
        & output["source_priority"].eq(output["target_priority"])
    ).astype(int)
    output["component_overlap_count"] = overlap_counts
    output["component_overlap"] = (output["component_overlap_count"] > 0).astype(int)
    output["story_point_abs_diff"] = (
        output["source_story_point"] - output["target_story_point"]
    ).abs()
    output["relation_binary"] = output["relation_label"].map(relation_binary_label)
    output["pair_model_text"] = [
        add_metadata_prefix(
            row.pair_text,
            [
                ("SAME_PROJECT", row.same_project),
                ("SAME_TYPE", row.same_task_type),
                ("SAME_PRIORITY", row.same_priority),
                ("COMPONENT_OVERLAP", row.component_overlap),
                ("SOURCE_PROJECT", row.source_project_key),
                ("TARGET_PROJECT", row.target_project_key),
                ("SOURCE_TYPE", row.source_task_type),
                ("TARGET_TYPE", row.target_task_type),
            ],
        )
        for row in output.itertuples(index=False)
    ]
    return output, {
        "graph_feature_missing_source_tasks": int(missing_source),
        "graph_feature_missing_target_tasks": int(missing_target),
        "same_project_rows": int(output["same_project"].sum()),
        "component_overlap_rows": int(output["component_overlap"].sum()),
    }


def prepare_links(args: argparse.Namespace, tasks: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    nrows = None if args.link_sample_frac > 0 or args.max_link_rows == 0 else args.max_link_rows
    links = pd.read_csv(
        args.links,
        sep="\t",
        usecols=LINK_COLUMNS,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        nrows=nrows,
    )
    raw_rows = len(links)
    if args.link_sample_frac > 0:
        links = links.sample(frac=args.link_sample_frac, random_state=args.random_state).reset_index(drop=True)
    sampled_rows = len(links)
    links["source_text"] = [
        make_task_text(clean_text(row.source_title), clean_text(row.source_description_text))
        for row in links.itertuples(index=False)
    ]
    links["target_text"] = [
        make_task_text(clean_text(row.target_title), clean_text(row.target_description_text))
        for row in links.itertuples(index=False)
    ]
    links["pair_text"] = [
        make_pair_text(row.source_title, row.source_description_text, row.target_title, row.target_description_text)
        for row in links.itertuples(index=False)
    ]
    links["relation_label_fine"] = [
        normalize_relation_label(row.relation_name, row.relation_description)
        for row in links.itertuples(index=False)
    ]
    links["relation_label"] = links["relation_label_fine"].map(
        lambda label: collapse_relation_label(label, args.relation_label_mode)
    )
    links = links[
        links["source_text"].str.len().ge(args.min_text_chars)
        & links["target_text"].str.len().ge(args.min_text_chars)
    ].copy()
    links["source_task_id"] = links["source_task_id"].astype(str)
    links["target_task_id"] = links["target_task_id"].astype(str)
    links, positive_stats = drop_conflicting_link_pairs(links)
    label_counts = links["relation_label"].value_counts()
    keep = set(label_counts[label_counts >= args.min_link_class_count].index)
    links = links[links["relation_label"].isin(keep)].copy()
    positives = links[
        [
            "source_task_id",
            "target_task_id",
            "source_text",
            "target_text",
            "pair_text",
            "relation_label",
            "source_project_id",
            "target_project_id",
        ]
    ].copy()
    positives = positives.rename(
        columns={
            "source_project_id": "source_project_key",
            "target_project_id": "target_project_key",
        }
    )
    known_pairs = set(zip(positives["source_task_id"], positives["target_task_id"]))
    negative_count = int(len(positives) * args.link_negative_ratio)
    negatives, negative_stats = build_negative_links(
        tasks,
        known_pairs,
        negative_count,
        args.random_state,
        no_relation_label_for_mode(args.relation_label_mode),
    )
    output = pd.concat([positives, negatives], ignore_index=True)
    before_final_dedup = len(output)
    output = output.drop_duplicates(subset=["source_task_id", "target_task_id", "relation_label"]).copy()
    output, graph_stats = add_pair_graph_features(output, tasks)
    output = output.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
    report = {
        "link_rows_read": int(raw_rows),
        "link_rows_after_random_sample": int(sampled_rows),
        "link_sample_frac": float(args.link_sample_frac),
        **positive_stats,
        **negative_stats,
        "relation_label_mode": args.relation_label_mode,
        "link_negative_ratio": float(args.link_negative_ratio),
        "link_rows_before_final_dedup": int(before_final_dedup),
        "link_rows_after_cleaning": int(len(output)),
        "link_duplicate_rows_removed": int(before_final_dedup - len(output)),
        **graph_stats,
        "relation_label_counts": output["relation_label"].value_counts().to_dict(),
        "relation_binary_counts": output["relation_binary"].value_counts().to_dict(),
    }
    return output, report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks, task_report = prepare_tasks(args)
    links, link_report = prepare_links(args, tasks)

    tasks.to_csv(args.output_dir / "task_supervised.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    links.to_csv(args.output_dir / "link_supervised.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    unlabeled_count = write_text_lines(
        args.output_dir / "unlabeled_corpus.txt",
        list(tasks["text"]) + list(links["source_text"]) + list(links["target_text"]),
    )

    report = {
        "output_dir": str(args.output_dir),
        "unlabeled_corpus_lines": int(unlabeled_count),
        **task_report,
        **link_report,
    }
    (args.output_dir / "label_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
