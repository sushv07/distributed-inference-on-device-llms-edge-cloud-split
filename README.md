
# Hybrid Edge-Cloud Inference for On-Device LLMs

This project simulates a small edge model handling easy prompts locally and routing harder prompts to a stronger cloud model.

## Run

```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
# or
venv\Scripts\activate      # Windows
pip install -r requirements.txt
cp .env.example .env        # Mac/Linux
# create .env manually on Windows
python -m src.main
python -m experiments.run_benchmarks
```

## Outputs
- `results/benchmark_results.csv`
- `results/plots/latency_comparison.png`
- `results/plots/routing_breakdown.png`

# distributed-inference-on-device-llms-edge-cloud-split

