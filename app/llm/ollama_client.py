from __future__ import annotations

import requests
from typing import Optional, Dict, Any


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        base_url = (base_url or "").strip()
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
        }

        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=180)

        if r.status_code == 404:
            raise RuntimeError(
                f"Ollama returned 404 for /api/chat. Model '{self.model}' likely not pulled. "
                f"Try: docker compose exec -T ollama ollama pull {self.model}"
            )

        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]
