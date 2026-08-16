"""
Domain B: Unaligned Base Foundation Model Instruction Benchmark
==============================================================
Demonstrates the 'Stanford Alpaca / Distillation Emergence' effect:
Uses raw unaligned base foundation model (Qwen/Qwen2.5-0.5B) across
strictly 1-to-1 paired prompts from UltraFeedback:
  - Condition 0A: Untrained Base Foundation Floor (Raw Auto-completion)
  - Condition 0B: Weak Open Model SFT (Rating 1-2 completions)
  - Condition 1:  Medium Commercial Model SFT (Rating 3-4 completions)
  - Condition 2:  Frontier GPT-4 Distillation SFT (Rating 5 completions)

Updates docs/dual_benchmark_results.json under 'instruction_following'.
"""

import os
import sys
import time
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .dataset_builder import load_instruction_dataset
from .trainer import fine_tune_lora
from .evaluator import evaluate_instruction_batched


def run_base_instruction_benchmark(
    base_model_name: str = "Qwen/Qwen2.5-0.5B",
    n_train: int = 150,
    n_test: int = 50,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    eval_batch_size: int = 25,
    train_batch_size: int = 4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_start = time.time()

    print("=" * 85)
    print(" DOMAIN B: BASE FOUNDATION MODEL INSTRUCTION DISTILLATION BENCHMARK")
    print(f" • GPU Accelerator:       {gpu_name.upper()}")
    print(f" • Base Foundation Model: {base_model_name} (Unaligned / No Instruct)")
    print(f" • Training Budget (N):   {n_train} paired multi-domain instructions")
    print(f" • Test Evaluation:       {n_test} held-out paired instructions")
    print("=" * 85)

    # 1. Load paired UltraFeedback dataset
    inst_train_dict, inst_test = load_instruction_dataset(N_train=n_train, N_test=n_test)

    # 2. Evaluate Untrained Raw Base Model Floor (Condition 0A)
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
    inst_acc_c0a, inst_samples_c0a, inst_t_c0a = evaluate_instruction_batched(
        base_model, tokenizer, inst_test, "Inst: Condition 0A (Raw Base Floor)", batch_size=eval_batch_size
    )
    del base_model
    torch.cuda.empty_cache()

    # Helper to fine-tune and evaluate
    def run_cond(name: str, records: list):
        model, tok, train_stats = fine_tune_lora(
            model_name=base_model_name,
            dataset_records=records,
            condition_name=f"inst_base_{name}",
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=train_batch_size,
        )
        acc, samples, eval_time = evaluate_instruction_batched(
            model, tok, inst_test, f"Inst: {name}", batch_size=eval_batch_size
        )
        del model
        torch.cuda.empty_cache()
        return acc, samples, train_stats, eval_time

    # 3. Condition 0B: Weak Open Model SFT
    print("\n" + "#" * 60)
    print(" [2/4] FINE-TUNING & EVALUATING WEAK OPEN MODEL SFT (CONDITION 0B)")
    print("#" * 60)
    inst_acc_c0b, inst_samples_c0b, inst_train_c0b, inst_t_c0b = run_cond(
        "Condition 0B (Weak Open Baseline)", inst_train_dict["condition_0b"]
    )

    # 4. Condition 1: Medium Commercial Model SFT
    print("\n" + "#" * 60)
    print(" [3/4] FINE-TUNING & EVALUATING MEDIUM MODEL SFT (CONDITION 1)")
    print("#" * 60)
    inst_acc_c1, inst_samples_c1, inst_train_c1, inst_t_c1 = run_cond(
        "Condition 1 (Medium Commercial Model)", inst_train_dict["condition_1"]
    )

    # 5. Condition 2: Frontier GPT-4 Distillation SFT
    print("\n" + "#" * 60)
    print(" [4/4] FINE-TUNING & EVALUATING FRONTIER GPT-4 DISTILLATION (CONDITION 2)")
    print("#" * 60)
    inst_acc_c2, inst_samples_c2, inst_train_c2, inst_t_c2 = run_cond(
        "Condition 2 (Frontier GPT-4 Distill)", inst_train_dict["condition_2"]
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

    existing_data["instruction_following"] = {
        "title": "Domain B: General Instruction Emergence (Raw Base Student + UltraFeedback)",
        "metric_name": "Instruction Alignment Score (%)",
        "scores": {
            "c0a_base_floor": round(inst_acc_c0a, 4),
            "c0b_weak_baseline": round(inst_acc_c0b, 4),
            "c1_medium_model": round(inst_acc_c1, 4),
            "c2_frontier_distill": round(inst_acc_c2, 4),
            "distill_vs_weak_premium": round(inst_acc_c2 - inst_acc_c0b, 4),
            "frontier_vs_medium_gain": round(inst_acc_c2 - inst_acc_c1, 4),
        },
        "training_stats": {
            "c0b_weak_tokens": inst_train_c0b["train_tokens"],
            "c1_medium_tokens": inst_train_c1["train_tokens"],
            "c2_frontier_tokens": inst_train_c2["train_tokens"],
        },
        "sample_traces": {
            "c0a": inst_samples_c0a,
            "c0b": inst_samples_c0b,
            "c1": inst_samples_c1,
            "c2": inst_samples_c2,
        },
    }

    with open(json_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    print("\n" + "=" * 85)
    print(f" BASE STUDENT INSTRUCTION BENCHMARK RESULTS (N_train = {n_train})")
    print("=" * 85)
    print(f" • Condition 0A (Untrained Raw Base Floor):   {inst_acc_c0a * 100:.1f}%")
    print(f" • Condition 0B (Weak Open Model Baseline):   {inst_acc_c0b * 100:.1f}%  (Uplift: +{(inst_acc_c0b - inst_acc_c0a)*100:.1f}%)")
    print(f" • Condition 1  (Medium Commercial Model):    {inst_acc_c1 * 100:.1f}%  (Uplift: +{(inst_acc_c1 - inst_acc_c0a)*100:.1f}%)")
    print(f" • Condition 2  (Frontier GPT-4 Distill):     {inst_acc_c2 * 100:.1f}%  (Uplift: +{(inst_acc_c2 - inst_acc_c0a)*100:.1f}%)")
    print(f" ==> Distillation Premium over Weak Baseline: +{(inst_acc_c2 - inst_acc_c0b)*100:.1f}%")
    print(f" [+] Total Runtime: {total_elapsed}s")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Base Model Instruction Distillation Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n-train", type=int, default=150)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    run_base_instruction_benchmark(
        base_model_name=args.model,
        n_train=args.n_train,
        n_test=args.n_test,
        epochs=args.epochs,
    )
