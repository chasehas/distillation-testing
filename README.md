# Empirical AI Distillation Measurement Suite

[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-brightgreen.svg)](https://python.org)
[![PyTorch: 2.6 CUDA](https://img.shields.io/badge/PyTorch-2.6%20CUDA%2012.4-orange.svg)](https://pytorch.org)
[![Hardware: RTX 4070 Ti Super](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%204070%20Ti%20Super-green.svg)](https://nvidia.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

An empirical research codebase measuring **how much capability transfers when black-box distilling from frontier AI model APIs**, using paired datasets on consumer hardware.

## Core Question

> When a fast-follower queries a frontier API (GPT-4, Claude, etc.) and fine-tunes a small open model on the responses — *exactly how much better* is that than training on human-written data, and why?

---

## Results Summary (Domain A: Math Reasoning)

Fine-tuned **Qwen2.5-0.5B-Instruct** via LoRA on an **NVIDIA RTX 4070 Ti Super** across N=300 strictly paired GSM8K training problems, evaluated on 100 held-out problems:

| Condition | Training Data Source | Accuracy | Δ vs. Human |
|---|---|:---:|:---:|
| **0A: Base Floor** | Untrained zero-shot | **3%** | -20% |
| **0B: Human SFT** | GSM8K crowdworker step-by-step solutions | **23%** | Baseline |
| **1: Direct Answers** | Same questions → final integer only (stripped reasoning) | **0%** | -23% |
| **2: GPT-4 CoT Distill** | MetaMathQA GPT-4 synthetic reasoning traces | **40%** | **+17%** |

**Key findings:**
- **Frontier distillation beats human annotation by +17 percentage points** on identical questions
- **Stripping reasoning traces collapses accuracy to 0%** — CoT tokens are the core intellectual property
- **Total experiment runtime: ~5 minutes** on consumer hardware with $0 API spend

---

## Project Structure

```
distillation-testing/
├── distillation_benchmark/          # Main experiment code
│   ├── dataset_builder.py           # Strict 1-to-1 paired dataset loading (GSM8K ↔ MetaMathQA)
│   ├── trainer.py                   # LoRA fine-tuning with PEFT
│   ├── evaluator.py                 # Batched GPU evaluation (math exact-match, instruction F1)
│   ├── run_dual_benchmark.py        # Full dual-domain experiment runner
│   ├── run_math_scale.py            # Scaled math-only benchmark
│   └── run_base_instruction.py      # Domain B: base foundation model instruction experiment
├── distillation_economics/          # Economic modeling engine
│   ├── economics.py                 # Cost asymmetry model (frontier R&D vs. distillation cost)
│   ├── simulator.py                 # Statistical learning simulation (argmax/logprob/CoT access modes)
│   ├── plotter.py                   # Visualization
│   └── cli.py                       # CLI entry point
├── docs/                            # Interactive presentation dashboard
│   ├── index.html                   # Dashboard UI
│   ├── style.css                    # Styling
│   ├── app.js                       # Chart.js visualization engine
│   └── dual_benchmark_results.json  # Latest empirical results (auto-generated)
├── run.py                           # Economics suite entry point
└── requirements.txt                 # Python dependencies
```

## Quickstart

### 1. Requirements

Python 3.12+ with PyTorch CUDA. Install dependencies:

```bash
pip install torch transformers peft datasets accelerate numpy scikit-learn matplotlib
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 2. Run the Math Reasoning Benchmark

```bash
# Full run: N=300 training, N=100 test (~5 minutes on RTX 4070 Ti Super)
python -m distillation_benchmark.run_math_scale

# Quick sanity check: N=100 training, N=50 test (~2 minutes)
python -m distillation_benchmark.run_math_scale --n-train 100 --n-test 50
```

### 3. Run the Instruction Following Benchmark (Domain B)

```bash
python -m distillation_benchmark.run_base_instruction
```

### 4. Run the Economics Model

```bash
python run.py --quick
```

### 5. View the Interactive Dashboard

```bash
python -m http.server 8000 --directory docs
```

Navigate to `http://localhost:8000`.

---

## Methodology

### Why This Experiment Design?

The critical methodological requirement is **strict 1-to-1 prompt matching**. Every experimental condition uses the *exact same questions* — only the response used for training differs. This isolates a single independent variable: the **source and quality of the training signal**.

### Datasets Used

| Dataset | Role | Source |
|---|---|---|
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | Questions + human crowdworker solutions | Conditions 0A, 0B, 1 |
| [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | GPT-4 synthetic reasoning traces for the same questions | Condition 2 |
| [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | Paired multi-model responses (weak/medium/frontier) | Domain B |

### Known Limitations

- **Small scale**: N=300 training samples. Production distillation uses 50K–500K samples.
- **Single student model**: Only Qwen2.5-0.5B tested. Results may differ with larger students.
- **MetaMathQA matching**: Some GSM8K questions may not have exact matches in MetaMathQA (see implementation plan for the fix).
- **Domain B evaluation**: Token-level F1 against GPT-4's response structurally favors Condition 2. Needs replacement with objective metrics (code execution, LLM-as-judge).

---

## Strategic Context

This project was developed to empirically ground policy discussions about AI distillation risk. For strategic framing notes on how to present these results to different audiences (labs, policymakers, investors), see the companion strategic framing document.

### The Policy Narrative in One Paragraph

A single person on an $800 consumer GPU, using only publicly available datasets and $0 in API spend, can measure a 17-percentage-point accuracy premium from training on GPT-4's synthetic reasoning traces versus human-written solutions — on identical math problems. Scaling this to 50,000 samples with $150 in API credits would capture a substantial fraction of frontier reasoning capability. This is the empirical basis for the "distillation free-rider problem" facing frontier AI labs.

---

## License

MIT License. Developed for research, policy analysis, and open strategic evaluation.
