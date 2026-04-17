"""Run the full benchmark suite and save plots/results."""

from __future__ import annotations

from src.benchmark import run_benchmark, save_plots
from src.cloud_model import CloudModel
from src.edge_model import EdgeModel
from src.prompts import BENCHMARK_PROMPTS
from src.router import Router


def main() -> None:
    edge_model = EdgeModel()
    cloud_model = CloudModel()
    router = Router()

    df = run_benchmark(BENCHMARK_PROMPTS, edge_model, cloud_model, router)
    save_plots(df)

    print("Benchmark finished.")
    print(df)
    print("\nSaved files:")
    print("- results/benchmark_results.csv")
    print("- results/plots/latency_comparison.png")
    print("- results/plots/routing_breakdown.png")
    print("- results/sample_outputs/sample_outputs.txt")


if __name__ == "__main__":
    main()
