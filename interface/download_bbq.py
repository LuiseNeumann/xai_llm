"""Lädt BBQ/Gender_identity für die Benchmark-Auswahl im Streamlit-Interface."""

from __future__ import annotations

import argparse

from interface.bbq_dataset import BBQ_PATH, download_bbq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(BBQ_PATH), help="Zielpfad der JSONL-Datei")
    args = parser.parse_args()
    path, count = download_bbq(args.output)
    print(f"{count} BBQ-Testfälle gespeichert: {path}")


if __name__ == "__main__":
    main()
