"""Batch benchmarking utilities for the hybrid edge–cloud inference pipeline.
Generates CSV results, plots, and sample outputs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt

from src.cloud_model import CloudModel
from src.edge_model import EdgeModel
from src.router import Router
from src.utils import RESULTS_DIR, PLOTS_DIR, SAMPLES_DIR, ensure_output_dirs


def run_benchmark(prompts: Iterable[str], edge_model: EdgeModel, cloud_model: CloudModel, router: Router) -> pd.DataFrame:
    """Run all prompts through edge first, then optionally route to cloud."""
    ensure_output_dirs()
    rows = []
    sample_lines = []

    for idx, prompt in enumerate(prompts, start=1):
        edge_result = edge_model.generate(prompt)
        decision = router.decide(prompt, edge_result, cloud_model.is_available())

        if decision.routed_to_cloud:
            cloud_result = cloud_model.generate(prompt)
            final_model = cloud_result.model
            final_latency = cloud_result.latency_sec
            final_response = cloud_result.response
            final_response_words = cloud_result.response_words
        else:
            final_model = edge_result.model
            final_latency = edge_result.latency_sec
            final_response = edge_result.response
            final_response_words = edge_result.response_words

        rows.append(
            {
                "prompt": prompt,
                "prompt_words": edge_result.prompt_words,
                "edge_model": edge_result.model,
                "edge_latency_sec": edge_result.latency_sec,
                "edge_response_words": edge_result.response_words,
                "routed_to_cloud": decision.routed_to_cloud,
                "route_reason": decision.reason,
                "final_model": final_model,
                "final_latency_sec": final_latency,
                "final_response_words": final_response_words,
            }
        )

        sample_lines.append(f"Prompt {idx}: {prompt}\n")
        sample_lines.append(f"Decision: {decision.reason}\n")
        sample_lines.append(f"Final model: {final_model}\n")
        sample_lines.append(f"Final response: {final_response}\n")
        sample_lines.append("-" * 80 + "\n")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)

    with open(SAMPLES_DIR / "sample_outputs.txt", "w", encoding="utf-8") as f:
        f.writelines(sample_lines)

    return df


def save_plots(df: pd.DataFrame) -> None:
    """Create simple grayscale-friendly plots for the report."""
    ensure_output_dirs()

    avg_edge = df["edge_latency_sec"].mean()
    avg_final = df["final_latency_sec"].mean()

    plt.figure(figsize=(7, 4))
    plt.bar(["Edge Only View", "Final Served Result"], [avg_edge, avg_final])
    plt.ylabel("Average Latency (sec)")
    plt.title("Latency Comparison")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "latency_comparison.png", dpi=200)
    plt.close()

    routing_counts = df["routed_to_cloud"].value_counts()
    labels = ["Stayed on Edge" if not key else "Routed to Cloud" for key in routing_counts.index]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, routing_counts.values)
    plt.ylabel("Number of Prompts")
    plt.title("Routing Breakdown")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "routing_breakdown.png", dpi=200)
    plt.close()
