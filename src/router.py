"""Routing logic for deciding whether to stay on the edge or escalate to cloud."""

from __future__ import annotations

from dataclasses import dataclass

from src.edge_model import EdgeResult
from src.utils import contains_repetition


@dataclass
class RouteDecision:
    routed_to_cloud: bool
    reason: str


class Router:
    def __init__(
        self,
        prompt_len_threshold: int = 20,
        min_response_words: int = 10,
        slow_edge_threshold_sec: float = 4.0,
    ) -> None:
        self.prompt_len_threshold = prompt_len_threshold
        self.min_response_words = min_response_words
        self.slow_edge_threshold_sec = slow_edge_threshold_sec

    def decide(self, prompt: str, edge_result: EdgeResult, cloud_available: bool) -> RouteDecision:
        """Choose whether to keep the answer local or send the prompt to cloud.

        These rules are intentionally simple and explainable. In a larger project,
        you could replace them with a learned router or confidence estimator.
        """
        if not cloud_available:
            return RouteDecision(False, "Cloud unavailable, staying on edge")

        if edge_result.prompt_words > self.prompt_len_threshold:
            return RouteDecision(True, "Prompt is long, likely harder for the edge model")

        if edge_result.response_words < self.min_response_words:
            return RouteDecision(True, "Edge response is too short to trust")

        if contains_repetition(edge_result.response):
            return RouteDecision(True, "Edge response looks repetitive or unstable")

        if edge_result.latency_sec > self.slow_edge_threshold_sec:
            return RouteDecision(True, "Edge inference was unexpectedly slow")

        return RouteDecision(False, "Edge response looks acceptable")
