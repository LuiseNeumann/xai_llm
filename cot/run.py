"""Kommandozeilenprogramm zur Auswertung gesammelter CoT-Artefakte."""

from __future__ import annotations

import argparse
import json

from common.io import read_jsonl, write_json
from cot.evaluators import evaluate_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, help="Pfad zur JSONL-Datei mit CoT-Artefakten")
    parser.add_argument("--output", help="Optionaler Pfad für die JSON-Ausgabe")
    args = parser.parse_args()

    records = read_jsonl(args.artifacts)
    report = {
        "pipeline": "cot",
        "examples": len(records),
        "metrics": [result.to_dict() for result in evaluate_all(records)],
    }
    if args.output:
        write_json(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
