"""Shared text cleaning and label utilities for training and inference."""

from __future__ import annotations

import re
from typing import Iterable


NULL_VALUES = {"", "NULL", "None", "nan", "NaN", "NaT"}


def clean_text(value: object, max_chars: int = 1200) -> str:
    """Normalize noisy Jira text while keeping enough signal for TF-IDF."""
    if value is None:
        return ""
    text = str(value)
    if text in NULL_VALUES:
        return ""
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def make_task_text(title: object, description: object, max_chars: int = 1200) -> str:
    # The model sees title and description as one document before TF-IDF.
    title_text = clean_text(title, max_chars=300)
    description_text = clean_text(description, max_chars=max_chars)
    return f"{title_text}. {description_text}".strip()


def make_pair_text(
    source_title: object,
    source_description: object,
    target_title: object,
    target_description: object,
    max_chars: int = 900,
) -> str:
    # Pairwise relation prediction needs both task texts in a stable order.
    source = make_task_text(source_title, source_description, max_chars=max_chars)
    target = make_task_text(target_title, target_description, max_chars=max_chars)
    return f"SOURCE TASK: {source} TARGET TASK: {target}"


def top_predictions(classes: Iterable[str], probabilities: Iterable[float], top_k: int = 3):
    # Sort softmax-like probabilities from largest to smallest for display.
    pairs = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    return [{"label": str(label), "probability": float(prob)} for label, prob in pairs[:top_k]]


def normalize_relation_label(name: object, description: object) -> str:
    # Jira projects use many relation names for similar ideas; this maps them
    # into fewer classes that are learnable with available data.
    text = f"{clean_text(name, 120)} {clean_text(description, 240)}".lower()

    if "duplicate" in text or "duplicated" in text:
        return "duplicate"
    if "block" in text:
        return "blocks"
    if "depend" in text or "required" in text or "requires" in text or "fs-depend" in text:
        return "depends"
    if "relate" in text or "reference" in text:
        return "related"
    if "clone" in text:
        return "clone"
    if any(word in text for word in ["parent", "child", "epic", "part", "incorpor", "contain", "detail"]):
        return "hierarchy"
    if any(word in text for word in ["regression", "cause", "caused", "breaks"]):
        return "causal"
    if "super" in text:
        return "supersedes"
    return "other_relation"
