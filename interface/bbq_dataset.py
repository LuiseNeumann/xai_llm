"""Lädt BBQ/Gender_identity und überführt Fälle in das gemeinsame JSONL-Schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.io import write_jsonl


DATASET_NAME = "heegyu/bbq"
CONFIG_NAME = "Gender_identity"
BBQ_PATH = Path(__file__).resolve().parents[1] / "data" / "downloads" / "bbq_gender_identity.jsonl"
CHOICES = "ABC"


def convert_bbq_row(row: dict[str, Any]) -> dict[str, Any]:
    """Bewahrt Antwortlabel, Optionen und Gruppenmetadaten eines BBQ-Testfalls."""
    label = int(row["label"])
    if label not in range(3):
        raise ValueError(f"Ungültiges BBQ-Label: {label}")
    answers = [str(row[f"ans{index}"]) for index in range(3)]
    prompt = (
        f"Context: {row['context']}\n"
        f"Question: {row['question']}\n"
        + "\n".join(f"({letter}) {answer}" for letter, answer in zip(CHOICES, answers))
        + "\nAnswer with exactly one letter (A, B or C) in the JSON field 'answer'. "
        "Base your reasoning only on the context."
    )
    return {
        "id": f"bbq-gender-identity-{int(row['example_id']):05d}",
        "prompt": prompt,
        "expected_answer": CHOICES[label],
        "metadata": {
            "dataset": DATASET_NAME,
            "config": CONFIG_NAME,
            "split": "test",
            "example_id": int(row["example_id"]),
            "label": label,
            "options": answers,
            "context_condition": str(row["context_condition"]),
            "question_polarity": str(row["question_polarity"]),
            "question_index": str(row["question_index"]),
            "answer_info": row.get("answer_info", {}),
            "additional_metadata": row.get("additional_metadata", {}),
        },
    }


def download_bbq(path: str | Path = BBQ_PATH) -> tuple[Path, int]:
    """Lädt genau die Konfiguration Gender_identity lokal herunter."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Die Abhängigkeit 'datasets<4' fehlt: uv sync --extra ui") from exc

    dataset = load_dataset(DATASET_NAME, CONFIG_NAME, trust_remote_code=True)
    records = [convert_bbq_row(row) for row in dataset["test"]]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, records)
    return output, len(records)
