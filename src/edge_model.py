"""Local edge model wrapper.

We keep this class small on purpose so it is easy to understand, test, and
explain in the paper. The edge model tries first because local inference is
usually cheaper and can preserve privacy better.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import word_count


@dataclass
class EdgeResult:
    model: str
    response: str
    latency_sec: float
    prompt_words: int
    response_words: int


class EdgeModel:
    def __init__(self, model_name: str = "distilgpt2") -> None:
        self.model_name = model_name

        # Apple Silicon can use MPS, CUDA works on supported GPUs, and CPU is the fallback.
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        # GPT-style models may not define a pad token. Reusing EOS keeps generation simple.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int = 60) -> EdgeResult:
        """Run one local generation and return a compact record of the result."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        end = time.perf_counter()

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Some decoder-only models echo the prompt. We keep only the new portion when possible.
        response = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()

        return EdgeResult(
            model=f"edge:{self.model_name}",
            response=response,
            latency_sec=round(end - start, 4),
            prompt_words=word_count(prompt),
            response_words=word_count(response),
        )
