# Distributed Inference for On-Device LLMs (Edge–Cloud Split Execution)

A hybrid distributed inference pipeline that enables lightweight on-device language model execution with selective cloud fallback using confidence-based routing.

This project demonstrates how intelligent assistants can balance **latency efficiency**, **response completeness**, and **compute constraints** by dynamically routing prompts between edge and cloud models.

---

## 🚀 Overview

Large Language Models (LLMs) typically require cloud infrastructure due to their computational complexity. However, cloud-only inference introduces:

- network latency
- privacy concerns
- operational cost per request
- connectivity dependency

This system implements a **hybrid edge–cloud split execution pipeline** that:

1. Executes lightweight inference locally using **DistilGPT2**
2. Evaluates response sufficiency using heuristic confidence routing
3. Escalates complex prompts to **GPT-4o-mini**
4. Returns the most efficient response path dynamically

The architecture demonstrates a practical deployment strategy for **resource-efficient intelligent assistants on edge devices**.

---

## 🎯 What This Project Demonstrates

This repository showcases practical experience with:

- distributed inference system design
- edge AI deployment strategies
- transformer latency optimization
- hybrid cloud routing pipelines
- confidence-based execution control
- real-world LLM infrastructure tradeoffs
- latency-performance benchmarking
- hybrid intelligent assistant deployment

---

## 🧠 Key Features

- Hybrid Edge–Cloud LLM inference pipeline
- Confidence-based routing mechanism
- Lightweight transformer execution using DistilGPT2
- Cloud fallback via GPT-4o-mini
- Response-length heuristic routing strategy
- Latency benchmarking utilities
- CPU-only edge execution support
- Experiment reproducibility support

---

## 🏗 System Architecture

The inference pipeline consists of three modules:

User Prompt
↓
Edge Inference (DistilGPT2)
↓
Confidence Evaluation
↓
Routing Decision
├── Return Edge Output (low latency)
└── Escalate to Cloud Model (GPT-4o-mini)


### Modules

### Edge Inference Module

Runs DistilGPT2 locally.

Optimized for:

- low latency execution
- reduced parameter footprint
- CPU-only inference environments

---

### Routing Module

Evaluates response sufficiency using:

- response-length thresholds
- repetition detection
- latency heuristics
- prompt complexity approximation

---

### Cloud Inference Module

Invoked only when necessary.

Uses:

GPT-4o-mini

Improves:

- reasoning depth
- contextual completeness
- multi-step explanation capability

---

## 📊 Routing Strategy

Instead of always invoking a large cloud model:
Short response → escalate to cloud
Sufficient response → return edge output



Benefits:

- reduced latency
- lower API usage cost
- improved scalability
- improved responsiveness

Routing decisions approximate response confidence without expensive entropy-based uncertainty estimation.

---

## 📈 Experimental Results

Benchmark evaluation across representative prompt categories showed:

| Execution Mode | Avg Latency |
|---------------|-------------|
| Edge-only | ~0.85 seconds |
| Hybrid Edge–Cloud | ~2.65 seconds |

Routing distribution:

8 prompts routed to cloud
2 prompts resolved locally


Key insight:

Edge inference handles lightweight prompts efficiently  
Cloud fallback improves reasoning-intensive responses

---

## 🧪 Prompt Categories Evaluated

Evaluation tasks simulate intelligent assistant interaction patterns:

- Concept explanation
- Summarization
- Definition queries
- Model comparison tasks

---

## ⚙️ Tech Stack

| Component | Technology |
|----------|-------------|
| Edge Model | DistilGPT2 |
| Cloud Model | GPT-4o-mini |
| Language | Python |
| Routing Logic | Heuristic confidence evaluation |
| Benchmarking | Custom latency measurement utilities |
| Execution Platform | CPU-only laptop environment |

---

## 📂 Repository Structure

distributed-llm-edge-cloud/
│
├── experiments/
│ ├── init.py
│ └── run_benchmarks.py
│
├── results/
│ ├── benchmark_results.csv
│ │
│ ├── plots/
│ │ ├── latency_comparison.png
│ │ └── routing_breakdown.png
│ │
│ └── sample_outputs/
│ └── sample_outputs.txt
│
├── src/
│
├── requirements.txt
├── .gitignore
└── README.md


---

## ▶️ Run the Project

Follow these steps to reproduce hybrid edge–cloud inference experiments locally.

### 1️⃣ Create virtual environment

python -m venv venv


Activate environment

Mac / Linux


source venv/bin/activate


Windows


venv\Scripts\activate


---

### 2️⃣ Install dependencies


pip install -r requirements.txt


---

### 3️⃣ Configure environment variables

Mac / Linux


cp .env.example .env


Windows

Create `.env` manually:


OPENAI_API_KEY=your_api_key_here


---

### 4️⃣ Run inference pipeline


python -m src.main


Runs:

- edge inference
- routing logic
- cloud fallback execution

---

### 5️⃣ Run benchmarking experiments


python -m experiments.run_benchmarks


Evaluates:

- routing behavior
- latency performance
- hybrid execution tradeoffs

---

## 📊 Outputs

Running the benchmark script generates evaluation artifacts automatically.

### Benchmark Results


results/benchmark_results.csv


Contains:

- prompt category
- routing decision
- edge latency
- hybrid latency

---

### Latency Comparison Plot


results/plots/latency_comparison.png


Shows execution-time differences between:

- edge-only inference
- hybrid edge–cloud inference

---

### Routing Distribution Plot


results/plots/routing_breakdown.png


Illustrates routing behavior across benchmark prompts.

---

## 🧪 Sample Outputs

Example inference responses:


results/sample_outputs/sample_outputs.txt


Includes routing behavior across:

- definition prompts
- explanation prompts
- summarization tasks
- comparison reasoning queries

---

## 🔁 Reproducing Paper Results

Figures included in the research paper can be regenerated using:


python -m experiments.run_benchmarks


This script produces:

- latency comparison visualization
- routing breakdown analysis
- benchmark dataset

---

## 🧩 Design Decisions

### Why DistilGPT2?

Chosen because:

- reduced parameter size
- fast inference speed
- CPU compatibility
- preserves contextual modeling ability
- suitable for realistic edge deployment simulation

---

### Why Response-Length Routing?

Selected because:

- lightweight
- fast to compute
- avoids entropy-estimation overhead
- effective proxy for response sufficiency

Future upgrades may include:

- entropy-based routing
- classifier confidence scoring
- reinforcement-learning routing policies

---

## 📉 Limitations

Current routing mechanism:

- heuristic-based confidence estimation
- not semantic-confidence aware
- routing thresholds tuned empirically
- cloud fallback required for reasoning-heavy prompts

---

## 🔮 Future Work

Planned improvements:

- transformer quantization support
- semantic uncertainty-based routing
- reinforcement-learning routing optimization
- multimodal routing (text + audio + vision)
- mobile deployment benchmarking
- edge TPU compatibility

---

## 📄 Research Paper

This repository accompanies the research paper:

**Distributed Inference for On-Device LLMs (Edge–Cloud Split Execution)**

Included with the project submission.

---

## 👩‍💻 Author

Sushmitha Vijayakumar  
Sanjai Kumar Chandra Mohan
MS Computer Science  
California State University Long Beach

Focus areas:

Distributed Systems  
Edge AI  
LLM Infrastructure  
Hybrid Inference Architectures  
Intelligent Assistant Systems
