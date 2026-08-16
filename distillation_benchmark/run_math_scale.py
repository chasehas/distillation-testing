"""
Scaled Domain A: Math Reasoning Benchmark Runner
================================================
Executes a larger-scale benchmark sweep on RTX 4070 Ti Super for:
  - Condition 0A: Base Model Floor
  - Condition 0B: Human Reference SFT (Crowdsourced GSM8K CoT)
  - Condition 1: Direct Answers Only (Stripped CoT)
  - Condition 2: Frontier GPT-4 CoT Distillation (MetaMathQA)

Updates docs/dual_benchmark_results.json while preserving existing Domain B metadata.
"""

import os
import sys
import time
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .dataset_builder import load_math_dataset
from .trainer import fine_tune_lora
from .evaluator import evaluate_math_batched


def run_math_scale_benchmark(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    n_train: int = 300,
    n_test: int = 100,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    eval_batch_size: int = 25,
    train_batch_size: int = 4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_start = time.time()

    print("=" * 85)
    print(" SCALED DOMAIN A: MATH REASONING EMPIRICAL BENCHMARK")
    print(f" • GPU Accelerator:       {gpu_name.upper()}")
    print(f" • Base Student Model:    {model_name}")
    print(f" • Training Budget (N):   {n_train} paired math problems")
    print(f" • Test Evaluation:       {n_test} held-out math problems")
    print("=" * 85)

    # 1. Load paired Math dataset
    math_train_dict, math_test = load_math_dataset(N_train=n_train, N_test=n_test)

    # 2. Condition 0A: Untrained Base Floor
    print("\n" + "#" * 60)
    print(" [1/4] EVALUATING UNTRAINED BASE MODEL FLOOR (CONDITION 0A)")
    print("#" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    math_acc_c0a, math_samples_c0a, math_t_c0a = evaluate_math_batched(
        base_model, tokenizer, math_test, "Math: Condition 0A (Base Floor)", batch_size=eval_batch_size
    )
    del base_model
    torch.cuda.empty_cache()

    # Helper to run fine-tuning + evaluation
    def run_cond(name: str, records: list):
        model, tok, train_stats = fine_tune_lora(
            model_name=model_name,
            dataset_records=records,
            condition_name=f"math_{name}",
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=train_batch_size,
        )
        acc, samples, eval_time = evaluate_math_batched(
            model, tok, math_test, f"Math: {name}", batch_size=eval_batch_size
        )
        del model
        torch.cuda.empty_cache()
        return acc, samples, train_stats, eval_time

    # 3. Condition 0B: Human Reference SFT
    print("\n" + "#" * 60)
    print(" [2/4] FINE-TUNING & EVALUATING HUMAN SFT BASELINE (CONDITION 0B)")
    print("#" * 60)
    math_acc_c0b, math_samples_c0b, math_train_c0b, math_t_c0b = run_cond(
        "Condition 0B (Human Reference SFT)", math_train_dict["condition_0b"]
    )

    # 4. Condition 1: Direct Answers Only
    print("\n" + "#" * 60)
    print(" [3/4] FINE-TUNING & EVALUATING DIRECT ANSWERS ONLY (CONDITION 1)")
    print("#" * 60)
    math_acc_c1, math_samples_c1, math_train_c1, math_t_c1 = run_cond(
        "Condition 1 (Direct Answers Only)", math_train_dict["condition_1"]
    )

    # 5. Condition 2: Frontier GPT-4 CoT Distillation
    print("\n" + "#" * 60)
    print(" [4/4] FINE-TUNING & EVALUATING FRONTIER GPT-4 COT DISTILLATION (CONDITION 2)")
    print("#" * 60)
    math_acc_c2, math_samples_c2, math_train_c2, math_t_c2 = run_cond(
        "Condition 2 (Frontier GPT-4 CoT Distill)", math_train_dict["condition_2"]
    )

    total_elapsed = round(time.time() - total_start, 2)

    # 6. Load existing JSON or initialize structure
    json_path = os.path.join("docs", "dual_benchmark_results.json")
    existing_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = {}

    meta = existing_data.get("meta", {
        "model_name": model_name,
        "gpu_name": gpu_name,
    })
    meta["n_train"] = n_train
    meta["n_test"] = n_test
    meta["math_elapsed_seconds"] = total_elapsed
    meta["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    existing_data["meta"] = meta
    existing_data["math_reasoning"] = {
        "title": "Domain A: Math Reasoning (GSM8K <-> MetaMathQA)",
        "metric_name": "Exact-Match Benchmark Accuracy (%)",
        "scores": {
            "c0a_base_floor": round(math_acc_c0a, 4),
            "c0b_human_sft": round(math_acc_c0b, 4),
            "c1_direct_answer": round(math_acc_c1, 4),
            "c2_frontier_distill": round(math_acc_c2, 4),
            "distill_vs_human_premium": round(math_acc_c2 - math_acc_c0b, 4),
            "cot_vs_direct_multiplier": round(math_acc_c2 - math_acc_c1, 4),
        },
        "training_stats": {
            "c0b_human_tokens": math_train_c0b["train_tokens"],
            "c1_direct_tokens": math_train_c1["train_tokens"],
            "c2_frontier_tokens": math_train_c2["train_tokens"],
        },
        "sample_traces": {
            "c0a": math_samples_c0a,
            "c0b": math_samples_c0b,
            "c1": math_samples_c1,
            "c2": math_samples_c2,
        },
    }

    os.makedirs("docs", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    print("\n" + "=" * 85)
    print(f" SCALED MATH BENCHMARK RESULTS (N_train = {n_train}, N_test = {n_test})")
    print("=" * 85)
    print(f" • Condition 0A (Untrained Base Floor):        {math_acc_c0a * 100:.1f}%")
    print(f" • Condition 0B (Human Reference SFT):       {math_acc_c0b * 100:.1f}%  (Uplift: +{(math_acc_c0b - math_acc_c0a)*100:.1f}%)")
    print(f" • Condition 1  (Direct Answers Only):        {math_acc_c1 * 100:.1f}%  (Uplift: +{(math_acc_c1 - math_acc_c0a)*100:.1f}%)")
    print(f" • Condition 2  (Frontier GPT-4 CoT Distill): {math_acc_c2 * 100:.1f}%  (Uplift: +{(math_acc_c2 - math_acc_c0a)*100:.1f}%)")
    print(f" ==> Distillation Premium over Human:        +{(math_acc_c2 - math_acc_c0b)*100:.1f}%")
    print(f" [+] Total Math Run Time: {total_elapsed}s")
    print("=" * 85)
    print(f"[+] Updated '{json_path}' with scaled empirical results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Scaled Domain A (Math Reasoning)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    run_math_scale_benchmark(
        model_name=args.model,
        n_train=args.n_train,
        n_test=args.n_test,
        epochs=args.epochs,
    )
