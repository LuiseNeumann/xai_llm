"""Einfache Metriken, durch die das Beispielprojekt ohne Abhängigkeiten auskommt."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from itertools import combinations
from math import log2, sqrt
from statistics import fmean


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return fmean(materialized) if materialized else 0.0


def tokenize(text: str) -> list[str]:
    return [token.strip(".,!?;:()[]{}\"'").lower() for token in text.split() if token.strip()]


def token_f1(left: str, right: str) -> float:
    left_counts = Counter(tokenize(left))
    right_counts = Counter(tokenize(right))
    overlap = sum((left_counts & right_counts).values())
    if not left_counts or not right_counts or overlap == 0:
        return 0.0
    precision = overlap / sum(left_counts.values())
    recall = overlap / sum(right_counts.values())
    return 2 * precision * recall / (precision + recall)


def sequence_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def pairwise_mean_similarity(values: Sequence[str]) -> float:
    pairs = list(combinations(values, 2))
    return mean(sequence_similarity(left, right) for left, right in pairs) if pairs else 1.0


def pairwise_mean_jaccard(values: Sequence[Iterable[str]]) -> float:
    pairs = list(combinations(values, 2))
    return mean(jaccard(left, right) for left, right in pairs) if pairs else 1.0


def normalized_entropy(values: Sequence[str]) -> float:
    if len(values) <= 1:
        return 0.0
    counts = Counter(values)
    entropy = -sum((count / len(values)) * log2(count / len(values)) for count in counts.values())
    maximum = log2(len(counts))
    return entropy / maximum if maximum else 0.0


def expected_calibration_error(
    confidences: Sequence[float], labels: Sequence[int], bins: int = 10
) -> float:
    if not confidences or len(confidences) != len(labels):
        return 0.0
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [
            item
            for item, confidence in enumerate(confidences)
            if lower <= confidence < upper or (index == bins - 1 and confidence == 1.0)
        ]
        if not selected:
            continue
        accuracy = mean(labels[item] for item in selected)
        confidence = mean(confidences[item] for item in selected)
        error += len(selected) / total * abs(accuracy - confidence)
    return error


def brier_score(confidences: Sequence[float], labels: Sequence[int]) -> float:
    if not confidences or len(confidences) != len(labels):
        return 0.0
    return mean((confidence - label) ** 2 for confidence, label in zip(confidences, labels))


def population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = mean(values)
    return sqrt(mean((value - center) ** 2 for value in values))
