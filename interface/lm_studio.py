"""Lokaler LM-Studio-Adapter über die OpenAI-kompatible HTTP-Schnittstelle."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

from cot.collect_hf import SYSTEM_INSTRUCTION, _extract_json


DEFAULT_MODEL = "google/gemma-4-26b-a4b"
DEFAULT_URL = "http://127.0.0.1:1234/v1"


class LMStudioService:
    """Erzeugt CoT-Antworten mit einem in LM Studio geladenen GGUF-Modell."""

    structure_collector = None

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        max_new_tokens: int = 256,
        timeout: int = 240,
    ) -> None:
        if not base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("Für LM Studio ist nur ein lokaler Server auf 127.0.0.1/localhost erlaubt.")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

    def generate(self, prompt: str, seed: int, temperature: float) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_INSTRUCTION}\n\nAufgabe: {prompt}",
                }
            ],
            "temperature": float(temperature),
            "seed": int(seed),
            "max_tokens": self.max_new_tokens,
            "reasoning_effort": "none",
            "stream": False,
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LM Studio meldet HTTP {exc.code}: {details[:500]}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"LM Studio ist nicht erreichbar ({self.base_url}). Server in LM Studio starten."
            ) from exc

        try:
            message = result["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("LM Studio hat keine auswertbare Antwort geliefert.") from exc
        if isinstance(content, list):
            content = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "LM Studio hat keinen sichtbaren Antworttext geliefert. "
                "Möglicherweise wurden alle Tokens für internes Reasoning verbraucht; "
                "maximale neue Tokens erhöhen."
            )
        parsed = _extract_json(content)
        parsed["raw_text"] = content
        return parsed


def available_models(base_url: str = DEFAULT_URL) -> list[str]:
    """Liest die Modellkennungen vom lokalen LM-Studio-Server."""
    if not base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("Nur lokale LM-Studio-Server werden unterstützt.")
    with urlopen(f"{base_url.rstrip('/')}/models", timeout=5) as response:
        result = json.load(response)
    return [str(model["id"]) for model in result.get("data", []) if "id" in model]
