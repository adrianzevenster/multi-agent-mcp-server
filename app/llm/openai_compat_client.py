from __future__ import annotations

import requests
from typing import Optional, Dict, Any, List
"""

"""
class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed"):
        """
        Initialize the client.

        :param base_url: Base URL of the API
        :param model: Model name to use for chat completions
        :param api_key: API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def chat(self, system: str, user: str) -> str:
        """
        Send a chat request with system and user messages.

        :param system: System prompt (instructions / role)
        :param user: User message
        :return: Assistant response text
        """
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=180)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
