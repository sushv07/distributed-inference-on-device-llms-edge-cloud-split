"""Shared helper functions for file paths, saving outputs, and simple text checks."""

from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SAMPLES_DIR = RESULTS_DIR / "sample_outputs"


def ensure_output_dirs() -> None:
    """Create result folders once so later file writes do not fail."""
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    SAMPLES_DIR.mkdir(exist_ok=True)


def word_count(text: str) -> int:
    """Count words in a simple, readable way for routing heuristics."""
    return len(text.split())


def contains_repetition(text: str) -> bool:
    """A tiny heuristic to catch obviously repetitive or broken responses.

    This is not meant to be perfect. It is only meant to flag outputs that look
    suspicious enough to justify a cloud fallback in our class project.
    """
    lowered = text.lower()

    # If the same 3+ word chunk repeats back-to-back, the output is probably weak.
    repeated_phrase = re.search(r"\b(\w+(?:\s+\w+){2,})\b(?:\s+\1\b)+", lowered)
    if repeated_phrase:
        return True

    # A basic repeated-line check also helps when generations get stuck.
    lines = [line.strip() for line in lowered.splitlines() if line.strip()]
    return len(lines) != len(set(lines)) if lines else False
