"""Startet einen Benchmark mit LM Studio oder Hugging Face ohne Browser."""

from __future__ import annotations

import argparse

from common.io import read_jsonl
from common.models import BenchmarkCase
from interface.benchmark import run_benchmark, save_benchmark
from interface.lm_studio import DEFAULT_MODEL, DEFAULT_URL, LMStudioService
from interface.pipeline_service import AnalysisConfig, LocalModelService


HF_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["lmstudio", "hf"], default="lmstudio")
    parser.add_argument("--model", help="Modell-ID; Standard hängt vom Backend ab")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data", default="data/sample_benchmark.jsonl")
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    cases = [BenchmarkCase.from_dict(record) for record in read_jsonl(args.data)][: args.limit]
    model_name = args.model or (HF_DEFAULT_MODEL if args.backend == "hf" else DEFAULT_MODEL)
    if args.backend == "hf":
        service = LocalModelService(model_name, args.max_new_tokens, args.device)
    else:
        service = LMStudioService(model_name, args.url, args.max_new_tokens, args.timeout)
    config = AnalysisConfig(mode="schnell", samples=args.samples, run_structure=False)

    def progress(value: float, text: str) -> None:
        print(f"{value:5.0%} {text}", flush=True)

    report = run_benchmark(service, cases, config, model_name, progress)
    report["backend"] = args.backend
    report_path, artifacts_path = save_benchmark(report, args.output_dir)
    print(f"Bericht: {report_path}")
    print(f"Artefakte: {artifacts_path}")
    print(f"Bewertet: {report['evaluated_cases']}/{report['total_cases']}")
    if report["accuracy"] is not None:
        print(f"Accuracy: {report['accuracy']:.1%}")
    if not report["evaluated_cases"]:
        raise SystemExit("Kein Fall konnte ausgewertet werden; Fehlermeldungen stehen im Bericht.")


if __name__ == "__main__":
    main()
