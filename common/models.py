"""Kleine abhängigkeitsfreie Schemata für Benchmarkfälle und Metrikergebnisse."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class BenchmarkCase:
    id: str
    prompt: str
    expected_answer: str
    alternative_answer: str = ""
    corrupted_prompt: str = ""
    gold_steps: list[str] = field(default_factory=list)
    gold_concepts: list[str] = field(default_factory=list)
    paraphrases: list[str] = field(default_factory=list)
    contrast_prompt: str = ""
    contrast_answer: str = ""
    interventions: list[dict[str, Any]] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BenchmarkCase":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value[key] for key in allowed if key in value})


@dataclass(slots=True)
class MetricResult:
    name: str
    score: float | None
    higher_is_better: bool
    details: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    @classmethod
    def unavailable(cls, name: str, reason: str, higher_is_better: bool = True) -> "MetricResult":
        return cls(
            name=name,
            score=None,
            higher_is_better=higher_is_better,
            details={"reason": reason},
            status="not_available",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
