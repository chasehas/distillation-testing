# The Distillation Premium: An Empirical Benchmark

An empirical measurement of how much capability transfers when fine-tuning a small open model on frontier AI outputs vs. human-written solutions, built in 24 hours on consumer hardware.

[Live Dashboard](https://chasehas.github.io/distillation-testing/) | [Results Data](https://chasehas.github.io/distillation-testing/math_scaling_summary.json)

## The Question

When a competitor queries a frontier API and fine-tunes a small open model on the responses, how much better is that than training on human-written data? Is distillation actually worth worrying about, or is the capability transfer marginal?

## The Answer

I fine-tuned Qwen2.5-0.5B on math reasoning problems using LoRA on an NVIDIA RTX 4070 Ti SUPER, holding compute constant across conditions. The only variable was the source of training solutions: human crowdworkers (GSM8K reference answers) vs. GPT-3.5 chain-of-thought outputs.

| Training Data | GSM8K Accuracy | vs. Human Solutions |
| :--- | :---: | :---: |
| No training (base model) | 29% | -3 |
| Human crowdworker solutions | 32% | baseline |
| GPT-3.5 chain-of-thought solutions | 46% | **+14** |

The distillation premium persisted across training set sizes from 150 to 1,000 samples, with GPT-3.5-trained models consistently scoring 44–48% while human-trained models hovered near or below the 29% untrained baseline. Human-written solutions provided negligible improvement over the untrained baseline. Further performance gains effectively require distillation from a stronger model.

## How It Works

The experimental design isolates a single variable: training data quality. Every condition uses the same questions, the same model, the same LoRA configuration, the same compute budget. Only the responses differ.

### Conditions tested:
- **Untrained Base** — Qwen2.5-0.5B zero-shot, no fine-tuning
- **Human Solutions** — Fine-tuned on GSM8K's crowdworker step-by-step solutions
- **GPT-3.5 Solutions** — Fine-tuned on MetaMathQA's GPT-3.5-generated chain-of-thought traces for the same questions
- **Direct Answers Only** — Fine-tuned on the correct final answer (#### N) with all reasoning traces stripped

The last condition tests whether the reasoning traces themselves are the mechanism for capability transfer. They are: stripping chain-of-thought collapses accuracy to 3-14%, worse than the untrained model.

**Training setup:** LoRA (r=16, alpha=32) on attention projections, 3 epochs, FP16, prompt loss masking (-100). Total training time per condition: 30-90 seconds depending on sequence length.

Token counts are comparable across conditions (26,749 human tokens vs. 29,028 GPT-3.5 tokens at N=150), ruling out sequence length as a confound.

**Evaluation:** Exact-match on 100 held-out GSM8K problems with stop-string truncation to prevent answer extraction from hallucinated follow-up questions.

## What I Tried Along the Way

This wasn't a straight line from question to answer. The repo reflects the iteration.

- **Started with economic modeling.** The `distillation_economics/` directory is a cost asymmetry simulator I built first — modeling how much cheaper distillation is vs. in-house R&D. Useful framing, but theoretical. I wanted empirical numbers.
- **First empirical attempt was CPU-only.** The early scripts (`empirical_distillation.py`, `pytorch_distillation.py`) ran on CPU with smaller models and simpler evaluation. Moved to GPU when it became clear the experiments needed to scale.
- **Iterated through single-domain runners.** Before building `run_full_suite.py`, I wrote separate scripts for each domain (math, instruction, code). These are still in the repo history as `run_math_scale.py`, `run_base_instruction.py`, `run_code_benchmark.py` — they worked, but the unified suite replaced them.
- **Tested five domains, only math was conclusive.** I ran the full experimental design across math reasoning, instruction following (UltraFeedback), code generation (MBPP), structured JSON extraction, and multiple-choice science reasoning (ARC-Challenge). Math showed a clear distillation premium. Code generation showed none — sub-1B models can't write executable Python regardless of training data quality. Instruction following showed a small premium. JSON and MCQ were inconclusive.
- **Tested two model scales.** Qwen2.5-0.5B (the headline results) and Qwen2.5-1.5B. The 1.5B model already scores 62% zero-shot on GSM8K, compressing the distillation premium to +4 percentage points. More notably, training the 1.5B model on human solutions degraded its performance from 62% to 54%.
- **Found and fixed an evaluation bug.** The initial base model evaluation reported 3% accuracy. The real number was 29%. The base model was solving problems correctly but then hallucinating follow-up questions in its output; the answer extractor was grabbing numbers from the hallucinated text instead of the actual solution. Adding stop-string truncation fixed this and aligned all results.
- **Ran a scaling study.** Five training set sizes (N=150, 300, 500, 750, 1000) to test whether the premium was a small-N artifact. It wasn't — GPT-3.5-trained models held steady at ~46% across all sizes while human-trained models hovered near or below the 29% untrained baseline.

## Repo Structure

```
distillation-testing/
├── distillation_benchmark/       # Core experiment code
│   ├── dataset_builder.py        # Paired dataset loading (GSM8K <-> MetaMathQA, UltraFeedback, MBPP, ARC)
│   ├── trainer.py                # LoRA fine-tuning with PEFT (prompt loss masking)
│   ├── evaluator.py              # Batched GPU evaluation (exact-match, token-F1, pass@1, json.loads)
│   ├── run_full_suite.py         # Universal 5-domain benchmark runner
│   └── run_math_scaling.py       # Math-only scaling study across training set sizes
├── distillation_economics/       # Economic cost asymmetry modeling (built first, separate from empirical work)
├── docs/                         # Interactive dashboard (GitHub Pages)
│   ├── index.html / app.js / style.css
│   ├── benchmark_results_0_5b.json # Full 0.5B results across all domains
│   ├── benchmark_results_1_5b.json # Full 1.5B results (math, instruction, code)
│   └── math_scaling_summary.json   # Scaling curve data (N=150 to N=1000)
└── requirements.txt
```

## Quickstart

Requires Python 3.12+ with PyTorch CUDA.

```bash
pip install torch transformers peft datasets accelerate
```

Run the math benchmark (the headline experiment):
```bash
python -m distillation_benchmark.run_full_suite --model Qwen/Qwen2.5-0.5B --domains math --n-train 150 --n-test 100
```

Run the scaling study:
```bash
python -m distillation_benchmark.run_math_scaling
```

View the dashboard:
```bash
python -m http.server 8000 --directory docs # Open http://localhost:8000
```

## Datasets

| Dataset | Role |
| :--- | :--- |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | Math questions + human crowdworker solutions |
| [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | GPT-3.5 chain-of-thought solutions for the same questions |
| [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | Paired weak/medium/frontier responses for instruction following |
| [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | Python programming problems with test assertions |
| [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | ARC-Challenge science multiple choice |

## Hardware

All experiments ran on a single NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM). Total wall-clock time for the full suite including the scaling study: approximately 2 hours. $0 API spend — all training data comes from publicly available datasets that include frontier model outputs.

## License

MIT
