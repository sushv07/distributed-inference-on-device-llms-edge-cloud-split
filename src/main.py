"""Entry point for single-prompt testing.

Run this first to confirm the pipeline works before moving to benchmark runs.
"""

from __future__ import annotations

from src.cloud_model import CloudModel
from src.edge_model import EdgeModel
from src.router import Router


def main() -> None:
    edge_model = EdgeModel()
    cloud_model = CloudModel()
    router = Router()

    prompt = input("Enter a prompt: ").strip()
    if not prompt:
        print("Please enter a non-empty prompt.")
        return

    edge_result = edge_model.generate(prompt)
    decision = router.decide(prompt, edge_result, cloud_model.is_available())

    if decision.routed_to_cloud:
        final_result = cloud_model.generate(prompt)
    else:
        final_result = edge_result

    print("\n=== EDGE RESULT ===")
    print(f"Model: {edge_result.model}")
    print(f"Latency: {edge_result.latency_sec} sec")
    print(f"Response: {edge_result.response}\n")

    print("=== ROUTER DECISION ===")
    print(f"Routed to cloud: {decision.routed_to_cloud}")
    print(f"Reason: {decision.reason}\n")

    print("=== FINAL RESULT ===")
    print(f"Model: {final_result.model}")
    print(f"Latency: {final_result.latency_sec} sec")
    print(f"Response: {final_result.response}")


if __name__ == "__main__":
    main()
