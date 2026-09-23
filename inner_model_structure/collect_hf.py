"""Sammelt ein grobes Aktivierungs-Patching-Artefakt auf Layer-Ebene.

Dies ist ein ausführbarer Ausgangspunkt und keine vollständige Methode zur
Circuit-Erkennung. Der Residualzustand des letzten Tokens aus einem
manipulierten Prompt wird in einen unveränderten Lauf eingesetzt. Anschließend
wird die Änderung der Präferenz für das erste Antworttoken gemessen. Für
publizierbare Arbeiten sollte die Analyse auf Head-, MLP-, Kanten- oder
SAE-Feature-Ebene erfolgen und um Vollständigkeits-, Minimalitäts-, Bootstrap-
und Off-Target-Experimente ergänzt werden.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from common.io import read_jsonl, write_jsonl
from common.models import BenchmarkCase


class LayerPatchingCollector:
    def __init__(
        self,
        model_name: str,
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
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
        )
        self.model.eval()
        self.layers = self._resolve_layers()

    def _resolve_layers(self) -> Any:
        candidates = (
            ("model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
        )
        for parent_name, layer_name in candidates:
            parent = getattr(self.model, parent_name, None)
            layers = getattr(parent, layer_name, None) if parent is not None else None
            if layers is not None:
                return layers
        raise ValueError("Für diese Architektur konnten keine Transformer-Layer gefunden werden")

    def _answer_token(self, answer: str) -> int:
        token_ids = self.tokenizer.encode(" " + answer.strip(), add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Die Antwort {answer!r} hat keine Tokens erzeugt")
        return int(token_ids[0])

    def _inputs(self, prompt: str) -> dict[str, Any]:
        values = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in values.items()}

    def preference_score(self, prompt: str, preferred: str, alternative: str) -> float:
        preferred_id = self._answer_token(preferred)
        alternative_id = self._answer_token(alternative)
        with self.torch.inference_mode():
            logits = self.model(**self._inputs(prompt)).logits[:, -1, :]
        return float((logits[0, preferred_id] - logits[0, alternative_id]).item())

    def cache_last_token(self, prompt: str) -> dict[int, Any]:
        cache: dict[int, Any] = {}
        handles = []

        def make_hook(index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                cache[index] = hidden[:, -1, :].detach()

            return hook

        for index, layer in enumerate(self.layers):
            handles.append(layer.register_forward_hook(make_hook(index)))
        try:
            with self.torch.inference_mode():
                self.model(**self._inputs(prompt))
        finally:
            for handle in handles:
                handle.remove()
        return cache

    @contextmanager
    def patch_layer(self, layer_index: int, replacement: Any) -> Iterator[None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            patched = hidden.clone()
            patched[:, -1, :] = replacement.to(device=hidden.device, dtype=hidden.dtype)
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        handle = self.layers[layer_index].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def collect(self, case: BenchmarkCase, top_k: int) -> dict[str, Any]:
        if not case.corrupted_prompt or not case.alternative_answer:
            raise ValueError(
                f"Fall {case.id} benötigt corrupted_prompt und alternative_answer für das Patching"
            )
        clean_score = self.preference_score(
            case.prompt, case.expected_answer, case.alternative_answer
        )
        corrupted_score = self.preference_score(
            case.corrupted_prompt, case.expected_answer, case.alternative_answer
        )
        corrupted_cache = self.cache_last_token(case.corrupted_prompt)
        effects: dict[str, float] = {}
        patched_scores: dict[str, float] = {}
        for index in range(len(self.layers)):
            with self.patch_layer(index, corrupted_cache[index]):
                patched = self.preference_score(
                    case.prompt, case.expected_answer, case.alternative_answer
                )
            name = f"layer_{index}"
            patched_scores[name] = patched
            effects[name] = clean_score - patched

        selected = sorted(effects, key=lambda name: abs(effects[name]), reverse=True)[:top_k]
        return {
            "id": case.id,
            "full_score": clean_score,
            "corrupted_score": corrupted_score,
            "total_nodes": len(self.layers),
            "nodes": selected,
            "node_effects": effects,
            "patched_scores": patched_scores,
            "metadata": {
                "method": "Aktivierungs-Patching des letzten Tokens auf Layer-Ebene",
                "answer_scoring": "Logit-Differenz des ersten Antworttokens",
                "limitations": [
                    "Einheiten auf Layer-Ebene sind grob und meist polysemantisch.",
                    "Antworten mit mehreren Tokens werden nur anhand ihres ersten Tokens bewertet.",
                    "Die ausgewählten Layer sind noch kein validierter kausaler Circuit.",
                ],
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    collector = LayerPatchingCollector(args.model)
    cases = [BenchmarkCase.from_dict(value) for value in read_jsonl(args.data)]
    records = [collector.collect(case, args.top_k) for case in cases]
    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
