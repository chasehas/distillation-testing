"""
Domain C: Sandboxed Code Execution Distillation Benchmark
=========================================================
Demonstrates distillation transfer on Python code generation using
objective pass@1 evaluation via sandboxed subprocess execution.

Training data: UltraFeedback coding subset (paired weak/medium/frontier)
Evaluation:    MBPP sanitized test split (assert-based unit tests)

Student model: Qwen/Qwen2.5-0.5B (raw unaligned base, same as Domain B)

Conditions:
  - Condition 0A: Untrained Base Foundation Floor
  - Condition 0B: Weak Open Model Code SFT
  - Condition 1:  Medium Commercial Model Code SFT
  - Condition 2:  Frontier GPT-4 Code Distillation SFT

Updates docs/dual_benchmark_results.json under 'code_execution'.
"""

import os
import time
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .dataset_builder import load_code_dataset, load_mbpp_test
from .trainer import fine_tune_lora
from .evaluator import evaluate_code_batched


def run_code_benchmark(
    base_model_name: str = "Qwen/Qwen2.5-0.5B",
    n_train: int = 150,
    n_test: int = 100,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    eval_batch_size: int = 10,
    train_batch_size: int = 4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_start = time.time()

    print("=" * 85)
    print(" DOMAIN C: SANDBOXED CODE EXECUTION DISTILLATION BENCHMARK")
    print(f" • GPU Accelerator:       {gpu_name.upper()}")
    print(f" • Base Foundation Model: {base_model_name} (Unaligned / No Instruct)")
    print(f" • Training Data:         UltraFeedback coding subset (N={n_train} paired prompts)")
    print(f" • Test Evaluation:       MBPP sanitized (N={n_test} problems, pass@1)")
    print("=" * 85)

    # 1. Load datasets
    code_train_dict = load_code_dataset(N_train=n_train)
    code_test = load_mbpp_test(N_test=n_test)

    # 2. Condition 0A: Untrained Base Floor
    print("\n" + "#" * 60)
    print(" [1/4] EVALUATING UNTRAINED BASE FOUNDATION FLOOR (CONDITION 0A)")
    print("#" * 60)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    code_acc_c0a, code_samples_c0a, code_t_c0a = evaluate_code_batched(
        base_model, tokenizer, code_test,
        "Code: Condition 0A (Raw Base Floor)",
        batch_size=eval_batch_size,
    )
    del base_model
    torch.cuda.empty_cache()

    # Helper to fine-tune and evaluate
    def run_cond(name: str, records: list):
        model, tok, train_stats = fine_tune_lora(
            model_name=base_model_name,
            dataset_records=records,
            condition_name=f"code_{name}",
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=train_batch_size,
        )
        acc, samples, eval_time = evaluate_code_batched(
            model, tok, code_test,
            f"Code: {name}",
            batch_size=eval_batch_size,
        )
        del model
        torch.cuda.empty_cache()
        return acc, samples, train_stats, eval_time

    # 3. Condition 0B: Weak Open Model Code SFT
    print("\n" + "#" * 60)
    print(" [2/4] FINE-TUNING & EVALUATING WEAK MODEL CODE SFT (CONDITION 0B)")
    print("#" * 60)
    code_acc_c0b, code_samples_c0b, code_train_c0b, code_t_c0b = run_cond(
        "Condition 0B (Weak Code Baseline)", code_train_dict["condition_0b"]
    )

    # 4. Condition 1: Medium Commercial Model Code SFT
    print("\n" + "#" * 60)
    print(" [3/4] FINE-TUNING & EVALUATING MEDIUM MODEL CODE SFT (CONDITION 1)")
    print("#" * 60)
    code_acc_c1, code_samples_c1, code_train_c1, code_t_c1 = run_cond(
        "Condition 1 (Medium Commercial Code)", code_train_dict["condition_1"]
    )

    # 5. Condition 2: Frontier GPT-4 Code Distillation SFT
    print("\n" + "#" * 60)
    print(" [4/4] FINE-TUNING & EVALUATING FRONTIER GPT-4 CODE DISTILLATION (CONDITION 2)")
    print("#" * 60)
    code_acc_c2, code_samples_c2, code_train_c2, code_t_c2 = run_cond(
        "Condition 2 (Frontier GPT-4 Code Distill)", code_train_dict["condition_2"]
    )

    total_elapsed = round(time.time() - total_start, 2)

    # 6. Update JSON results
    json_path = os.path.join("docs", "dual_benchmark_results.json")
    existing_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = {}

    existing_data["code_execution"] = {
        "title": "Domain C: Sandboxed Code Execution (UltraFeedback Code + MBPP pass@1)",
        "metric_name": "pass@1 Code Execution Accuracy (%)",
        "scores": {
            "c0a_base_floor": round(code_acc_c0a, 4),
            "c0b_weak_baseline": round(code_acc_c0b, 4),
            "c1_medium_model": round(code_acc_c1, 4),
            "c2_frontier_distill": round(code_acc_c2, 4),
            "distill_vs_weak_premium": round(code_acc_c2 - code_acc_c0b, 4),
            "frontier_vs_medium_gain": round(code_acc_c2 - code_acc_c1, 4),
        },
        "training_stats": {
            "c0b_weak_tokens": code_train_c0b["train_tokens"],
            "c1_medium_tokens": code_train_c1["train_tokens"],
            "c2_frontier_tokens": code_train_c2["train_tokens"],
        },
        "sample_traces": {
            "c0a": code_samples_c0a,
            "c0b": code_samples_c0b,
            "c1": code_samples_c1,
            "c2": code_samples_c2,
        },
    }

    os.makedirs("docs", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    # 7. Summary
    print("\n" + "=" * 85)
    print(f" CODE EXECUTION BENCHMARK RESULTS (N_train = {n_train}, N_test = {n_test})")
    print("=" * 85)
    print(f" • Condition 0A (Untrained Raw Base Floor):   {code_acc_c0a * 100:.1f}%")
    print(f" • Condition 0B (Weak Open Model Code):       {code_acc_c0b * 100:.1f}%  (Uplift: +{(code_acc_c0b - code_acc_c0a)*100:.1f}%)")
    print(f" • Condition 1  (Medium Commercial Code):     {code_acc_c1 * 100:.1f}%  (Uplift: +{(code_acc_c1 - code_acc_c0a)*100:.1f}%)")
    print(f" • Condition 2  (Frontier GPT-4 Code):        {code_acc_c2 * 100:.1f}%  (Uplift: +{(code_acc_c2 - code_acc_c0a)*100:.1f}%)")
    print(f" ==> Code Distillation Premium over Weak:     +{(code_acc_c2 - code_acc_c0b)*100:.1f}%")
    print(f" [+] Total Runtime: {total_elapsed}s")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Domain C: Sandboxed Code Execution Benchmark"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n-train", type=int, default=150)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (N=75 train, N=50 test)"
    )
    args = parser.parse_args()

    n_train = 75 if args.quick else args.n_train
    n_test = 50 if args.quick else args.n_test

    run_code_benchmark(
        base_model_name=args.model,
        n_train=n_train,
        n_test=n_test,
        epochs=args.epochs,
    )
