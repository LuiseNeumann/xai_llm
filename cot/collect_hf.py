"""Erzeugt CoT-Basisartefakte mit einem kausalen Hugging-Face-Sprachmodell.

Dieser Sammler deckt wiederholte Stichproben, Paraphrasen, Kontrastpaare und
Eingabeinterventionen ab. Schrittkorrektheit, Faktizität, Nutzerrelevanz und
Minimalitätslabels benötigen separate Prüfer oder Annotationen und fehlen daher
bewusst.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from common.io import read_jsonl, write_jsonl
from common.models import BenchmarkCase


SYSTEM_INSTRUCTION = """Löse die Aufgabe und gib ausschließlich JSON in diesem Schema zurück:
{"steps": ["kurzer Begründungsschritt"], "answer": "endgültige Antwort", "step_confidences": [0.0]}
Jeder Eintrag in steps muss ein vollständiger Satz als String sein, kein Objekt und keine Nummer.
Gib für jeden Schritt genau eine Konfidenz in [0, 1] an. Verwende keine Markdown-Codeblöcke."""


def _extract_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {"steps": [text.strip()], "answer": "", "step_confidences": []}
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"steps": [text.strip()], "answer": "", "step_confidences": []}
    steps = value.get("steps", [])
    if not isinstance(steps, list):
        steps = [str(steps)]
    cleaned_steps: list[str] = []
    for step in steps:
        if isinstance(step, str) and step.strip():
            cleaned_steps.append(step)
        elif isinstance(step, dict):
            text = next(
                (step[key] for key in ("explanation", "reasoning", "text", "content") if isinstance(step.get(key), str)),
                "",
            )
            if text.strip():
                cleaned_steps.append(text)
    confidences = value.get("step_confidences", [])
    if not isinstance(confidences, list):
        confidences = []
    return {
        "steps": cleaned_steps,
        "answer": str(value.get("answer", "")),
        "step_confidences": confidences,
    }


class HuggingFaceGenerator:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        device: str = "auto",
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Optionale Abhängigkeiten installieren mit: pip install -e '.[hf]'"
            ) from exc

        self.torch = torch
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        if model is not None:
            self.model = model
        else:
            load_arguments: dict[str, Any] = {"torch_dtype": "auto"}
            if device == "auto":
                load_arguments["device_map"] = "auto"
            elif device == "cuda":
                if not torch.cuda.is_available():
                    raise ValueError("CUDA wurde gewählt, ist aber nicht verfügbar")
                load_arguments["device_map"] = {"": "cuda"}
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_arguments)
            if device == "cpu":
                self.model.to("cpu")
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, seed: int, temperature: float) -> dict[str, Any]:
        random.seed(seed)
        self.torch.manual_seed(seed)
        messages = [
            {
                "role": "user",
                "content": f"{SYSTEM_INSTRUCTION}\n\nAufgabe: {prompt}",
            }
        ]
        if getattr(self.tokenizer, "chat_template", None):
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = f"{SYSTEM_INSTRUCTION}\n\nAufgabe: {prompt}\nAntwort:"
        inputs = self.tokenizer(rendered, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        do_sample = temperature > 0
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = 0.95
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        generated = output[0, inputs["input_ids"].shape[1] :]
        raw_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return {**_extract_json(raw_text), "raw_text": raw_text}


def collect_case(
    generator: HuggingFaceGenerator,
    case: BenchmarkCase,
    samples: int,
    seed: int,
    temperature: float,
) -> dict[str, Any]:
    original = generator.generate(case.prompt, seed, temperature=0.0)
    artifact: dict[str, Any] = {
        "id": case.id,
        **original,
        "expected_answer": case.expected_answer,
        "gold_steps": case.gold_steps,
        "gold_concepts": case.gold_concepts,
        "constraints": case.constraints,
        "sampled_traces": [
            generator.generate(case.prompt, seed + index + 1, temperature)
            for index in range(samples)
        ],
        "paraphrase_traces": [
            generator.generate(prompt, seed + 100 + index, temperature=0.0)
            for index, prompt in enumerate(case.paraphrases)
        ],
        "interventions": [],
    }
    for index, intervention in enumerate(case.interventions):
        result = generator.generate(
            str(intervention["prompt"]), seed + 200 + index, temperature=0.0
        )
        artifact["interventions"].append(
            {
                "name": intervention.get("name", f"intervention_{index}"),
                "answer": result["answer"],
                "steps": result["steps"],
                "expected_answer_change": intervention.get("expected_answer_change"),
            }
        )
    if case.contrast_prompt:
        contrast = generator.generate(case.contrast_prompt, seed + 300, temperature=0.0)
        artifact["contrast"] = {**contrast, "expected_answer": case.contrast_answer}
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    generator = HuggingFaceGenerator(args.model, args.max_new_tokens)
    cases = [BenchmarkCase.from_dict(value) for value in read_jsonl(args.data)]
    records = [
        collect_case(generator, case, args.samples, args.seed + 1000 * index, args.temperature)
        for index, case in enumerate(cases)
    ]
    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
