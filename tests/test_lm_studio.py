"""Prüft das lokale API-Protokoll ohne Modell- oder Serverdownload."""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from interface.lm_studio import LMStudioService


class LMStudioTests(unittest.TestCase):
    def test_uses_local_api_and_parses_visible_answer(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"steps": ["17 + 25 = 42"], "answer": "42", "step_confidences": [1.0]}',
                        "reasoning_content": "",
                    }
                }
            ]
        }
        with patch("interface.lm_studio.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as urlopen:
            result = LMStudioService().generate("Was ist 17 + 25?", 42, 0)
        self.assertEqual(result["answer"], "42")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["model"], "google/gemma-4-26b-a4b")

    def test_rejects_non_local_url(self) -> None:
        with self.assertRaises(ValueError):
            LMStudioService(base_url="https://example.org/v1")


if __name__ == "__main__":
    unittest.main()
