"""Cloud model wrapper.

The cloud model is optional. If an API key is not present, the project still
runs in edge-only mode so you can develop and test the pipeline locally first.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from src.utils import word_count

load_dotenv()


@dataclass
class CloudResult:
    model: str
    response: str
    latency_sec: float
    prompt_words: int
    response_words: int


class CloudModel:
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")

        # Delayed import avoids breaking edge-only runs when openai is installed but unused.
        self._client = None
        if self.api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)

    def is_available(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str) -> CloudResult:
        if not self._client:
            raise RuntimeError("Cloud model is not available because OPENAI_API_KEY is missing.")

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Give clear, direct, concise answers.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        end = time.perf_counter()

        text = response.choices[0].message.content.strip()
        return CloudResult(
            model=f"cloud:{self.model_name}",
            response=text,
            latency_sec=round(end - start, 4),
            prompt_words=word_count(prompt),
            response_words=word_count(text),
        )
