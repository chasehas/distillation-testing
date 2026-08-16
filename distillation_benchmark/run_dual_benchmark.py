"""
Master Dual-Domain Distillation Benchmark Runner
================================================
Executes 4-tier experimental matrix on RTX 4070 Ti Super across:
  1. Domain A: Math Reasoning (GSM8K <-> MetaMathQA)
  2. Domain B: General Instruction Following (UltraFeedback)
Exports complete results and qualitative traces to docs/dual_benchmark_results.json.
"""

import os
import sys
import time
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .dataset_builder import load_math_dataset, load_instruction_dataset
from .trainer import fine_tune_lora
from .evaluator import evaluate_math_batched, evaluate_instruction_batched


def run_benchmark(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    n_train: int = 250,
    n_test: int = 100,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    eval_batch_size: int = 20,
    train_batch_size: int = 4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_start = time.time()

    print("=" * 85)
    print(" DUAL-DOMAIN EMPIRICAL LLM DISTILLATION BENCHMARK")
    print(f" • GPU Accelerator:       {gpu_name.upper()}")
    print(f" • Base Student Model:    {model_name}")
    print(f" • Training Budget (N):   {n_train} paired prompts per domain")
    print(f" • Test Evaluation:       {n_test} held-out prompts per domain")
    print("=" * 85)

    # -------------------------------------------------------------
    # 1. LOAD DATASETS WITH STRICT 1-TO-1 PROMPT MATCHING
    # -------------------------------------------------------------
    math_train_dict, math_test = load_math_dataset(N_train=n_train, N_test=n_test)
    inst_train_dict, inst_test = load_instruction_dataset(N_train=n_train, N_test=n_test)

    # -------------------------------------------------------------
    # 2. EVALUATE UNTRAINED BASE MODEL (CONDITION 0A) ON BOTH DOMAINS
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print(" [PHASE 1/3] EVALUATING UNTRAINED BASE MODEL FLOOR (CONDITION 0A)")
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
    
    inst_acc_c0a, inst_samples_c0a, inst_t_c0a = evaluate_instruction_batched(
        base_model, tokenizer, inst_test, "Inst: Condition 0A (Base Floor)", batch_size=eval_batch_size
    )
    
    del base_model
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 3. RUN DOMAIN A: MATH REASONING EXPERIMENT
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print(" [PHASE 2/3] DOMAIN A: MATHEMATICAL REASONING (GSM8K <-> METAMATH)")
    print("#" * 60)

    # Helper to run condition
    def run_math_cond(name: str, records: list):
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

    # Condition 0B (Human Reference)
    math_acc_c0b, math_samples_c0b, math_train_c0b, math_t_c0b = run_math_cond(
        "Condition 0B (Human Reference SFT)", math_train_dict["condition_0b"]
    )
    
    # Condition 1 (Direct Commercial Answer)
    math_acc_c1, math_samples_c1, math_train_c1, math_t_c1 = run_math_cond(
        "Condition 1 (Direct Answers Only)", math_train_dict["condition_1"]
    )
    
    # Condition 2 (Frontier GPT-4 CoT Distillation)
    math_acc_c2, math_samples_c2, math_train_c2, math_t_c2 = run_math_cond(
        "Condition 2 (Frontier GPT-4 CoT Distill)", math_train_dict["condition_2"]
    )

    # -------------------------------------------------------------
    # 4. RUN DOMAIN B: GENERAL INSTRUCTION & CODING EXPERIMENT
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print(" [PHASE 3/3] DOMAIN B: GENERAL INSTRUCTION FOLLOWING (ULTRAFEEDBACK)")
    print("#" * 60)

    def run_inst_cond(name: str, records: list):
        model, tok, train_stats = fine_tune_lora(
            model_name=model_name,
            dataset_records=records,
            condition_name=f"inst_{name}",
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=train_batch_size,
        )
        score, samples, eval_time = evaluate_instruction_batched(
            model, tok, inst_test, f"Inst: {name}", batch_size=eval_batch_size
        )
        del model
        torch.cuda.empty_cache()
        return score, samples, train_stats, eval_time

    # Condition 0B (Weak Open Model Baseline)
    inst_acc_c0b, inst_samples_c0b, inst_train_c0b, inst_t_c0b = run_inst_cond(
        "Condition 0B (Weak Open Model Baseline)", inst_train_dict["condition_0b"]
    )
    
    # Condition 1 (Medium Commercial Model)
    inst_acc_c1, inst_samples_c1, inst_train_c1, inst_t_c1 = run_inst_cond(
        "Condition 1 (Medium Commercial Model)", inst_train_dict["condition_1"]
    )
    
    # Condition 2 (Frontier GPT-4 Distillation)
    inst_acc_c2, inst_samples_c2, inst_train_c2, inst_t_c2 = run_inst_cond(
        "Condition 2 (Frontier GPT-4 Distill)", inst_train_dict["condition_2"]
    )

    elapsed_total = time.time() - total_start

    # -------------------------------------------------------------
    # 5. SUMMARY COMPARISON TABLES
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(f" DUAL-BENCHMARK EMPIRICAL RESULTS SUMMARY (N = {n_train} Prompts)")
    print("=" * 85)
    
    print("\n [DOMAIN A: MATH REASONING]")
    print(f"  • Condition 0A (Untrained Base Floor):        {math_acc_c0a:.1%}")
    print(f"  • Condition 0B (Human Reference SFT):       {math_acc_c0b:.1%}  (Uplift: {math_acc_c0b - math_acc_c0a:+.1%})")
    print(f"  • Condition 1  (Direct Answers Only):        {math_acc_c1:.1%}  (Uplift: {math_acc_c1 - math_acc_c0a:+.1%})")
    print(f"  • Condition 2  (Frontier GPT-4 CoT Distill): {math_acc_c2:.1%}  (Uplift: {math_acc_c2 - math_acc_c0a:+.1%})")
    print(f"  ==> Math Distillation Premium over Human:    {math_acc_c2 - math_acc_c0b:+.1%}")

    print("\n [DOMAIN B: GENERAL INSTRUCTION FOLLOWING]")
    print(f"  • Condition 0A (Untrained Base Floor):        {inst_acc_c0a:.1%}")
    print(f"  • Condition 0B (Weak Open Model Baseline):   {inst_acc_c0b:.1%}  (Uplift: {inst_acc_c0b - inst_acc_c0a:+.1%})")
    print(f"  • Condition 1  (Medium Commercial Model):    {inst_acc_c1:.1%}  (Uplift: {inst_acc_c1 - inst_acc_c0a:+.1%})")
    print(f"  • Condition 2  (Frontier GPT-4 Distill):     {inst_acc_c2:.1%}  (Uplift: {inst_acc_c2 - inst_acc_c0a:+.1%})")
    print(f"  ==> Instruction Distill Premium over Weak:   {inst_acc_c2 - inst_acc_c0b:+.1%}")
    
    print(f"\n [+] TOTAL DUAL-BENCHMARK GPU RUNTIME:        {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print("=" * 85)

    # -------------------------------------------------------------
    # 6. EXPORT STRUCTURED JSON DATA
    # -------------------------------------------------------------
    results_payload = {
        "meta": {
            "model_name": model_name,
            "gpu_name": gpu_name,
            "n_train": n_train,
            "n_test": n_test,
            "elapsed_seconds": round(elapsed_total, 2),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "math_reasoning": {
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
        },
        "instruction_following": {
            "title": "Domain B: General Instruction Following (UltraFeedback)",
            "metric_name": "Quality Alignment Score (%)",
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
        },
    }

    os.makedirs("docs", exist_ok=True)
    out_path = os.path.join("docs", "dual_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[+] Saved complete dual-benchmark results to '{out_path}'.")

    return results_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dual Empirical LLM Distillation Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-train", type=int, default=250)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Quick mode (N=100 train, N=50 test)")
    args = parser.parse_args()

    n_train = 100 if args.quick else args.n_train
    n_test = 50 if args.quick else args.n_test

    run_benchmark(
        model_name=args.model,
        n_train=n_train,
        n_test=n_test,
        epochs=args.epochs,
    )
