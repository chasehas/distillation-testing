"""
Math Scaling Benchmark Suite (Domain A)
Runs Domain A across N_train in [150, 300, 500, 750, 1000] with N_test=100.
Saves docs/math_scale_{n_train}.json for each run and builds docs/math_scaling_summary.json.
"""

import os
import sys
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from distillation_benchmark.dataset_builder import load_math_dataset
from distillation_benchmark.trainer import fine_tune_lora
from distillation_benchmark.evaluator import evaluate_math_batched


def run_math_scaling():
    model_name = "Qwen/Qwen2.5-0.5B"
    n_test = 100
    n_train_list = [150, 300, 500, 750, 1000]
    epochs = 3
    output_dir = "checkpoints_math_scaling"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print("=" * 80)
    print(f" MATH REASONING SCALING SUITE: {model_name}")
    print(f" Training scales: {n_train_list} | Evaluation test size: N={n_test}")
    print("=" * 80)

    t0_all = time.time()

    # Load base model & tokenizer once
    print(f"\n[Init] Loading base student model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    summary_curve = []

    for n_train in n_train_list:
        print("\n" + "#" * 80)
        print(f" RUNNING SCALE: N_train = {n_train} (N_test = {n_test})")
        print("#" * 80)
        t_scale_0 = time.time()

        train_dict, test_slice = load_math_dataset(N_train=n_train, N_test=n_test)

        # 0A Base Floor
        print(f"\n[Eval] Condition 0A: Base Floor (Zero-Shot)...")
        score_c0a, samples_c0a, _ = evaluate_math_batched(
            base_model, tokenizer, test_slice, f"Math [N={n_train}]: 0A Base Floor"
        )

        # 0B Human SFT
        print(f"\n[Train/Eval] Condition 0B: Human SFT (N={n_train})...")
        m_0b, tok_0b, tr_0b = fine_tune_lora(
            model_name, train_dict["condition_0b"], f"Math_Human_N{n_train}",
            output_dir=output_dir, epochs=epochs
        )
        score_c0b, samples_c0b, _ = evaluate_math_batched(
            m_0b, tok_0b, test_slice, f"Math [N={n_train}]: 0B Human SFT"
        )
        del m_0b
        torch.cuda.empty_cache()

        # 1 Direct Answers
        print(f"\n[Train/Eval] Condition 1: Direct Answers Only (N={n_train})...")
        m_c1, tok_c1, tr_c1 = fine_tune_lora(
            model_name, train_dict["condition_1"], f"Math_Direct_N{n_train}",
            output_dir=output_dir, epochs=epochs
        )
        score_c1, samples_c1, _ = evaluate_math_batched(
            m_c1, tok_c1, test_slice, f"Math [N={n_train}]: 1 Direct Answers"
        )
        del m_c1
        torch.cuda.empty_cache()

        # 2 Teacher GPT-3.5 Distill
        print(f"\n[Train/Eval] Condition 2: Teacher GPT-3.5 CoT Distill (N={n_train})...")
        m_c2, tok_c2, tr_c2 = fine_tune_lora(
            model_name, train_dict["condition_2"], f"Math_Distill_N{n_train}",
            output_dir=output_dir, epochs=epochs
        )
        score_c2, samples_c2, _ = evaluate_math_batched(
            m_c2, tok_c2, test_slice, f"Math [N={n_train}]: 2 Frontier Distill"
        )
        del m_c2
        torch.cuda.empty_cache()

        scale_elapsed = round(time.time() - t_scale_0, 2)
        distill_premium = round(score_c2 - score_c0b, 4)
        cot_vs_direct = round(score_c2 / max(score_c1, 0.01), 2)

        scale_result = {
            "meta": {
                "model_name": model_name,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "n_train": n_train,
                "n_test": n_test,
                "elapsed_seconds": scale_elapsed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "math_reasoning": {
                "title": f"Domain A: Math Reasoning (GSM8K <-> MetaMathQA) [N={n_train}]",
                "metric_name": "Exact-Match Benchmark Accuracy (%)",
                "scores": {
                    "c0a_base_floor": round(score_c0a, 4),
                    "c0b_human_sft": round(score_c0b, 4),
                    "c1_direct_answer": round(score_c1, 4),
                    "c2_frontier_distill": round(score_c2, 4),
                    "distill_vs_human_premium": distill_premium,
                    "cot_vs_direct_multiplier": cot_vs_direct
                },
                "training_stats": {
                    "c0b_human_tokens": tr_0b["train_tokens"],
                    "c1_direct_tokens": tr_c1["train_tokens"],
                    "c2_frontier_tokens": tr_c2["train_tokens"]
                },
                "sample_traces": {
                    "c0a": samples_c0a,
                    "c0b": samples_c0b,
                    "c1": samples_c1,
                    "c2": samples_c2
                }
            }
        }

        # Save individual scale file
        out_file = f"docs/math_scale_{n_train}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(scale_result, f, indent=2)
        print(f"[Save] Saved scale results to {out_file} in {scale_elapsed}s")

        summary_curve.append({
            "n_train": n_train,
            "c0a": round(score_c0a, 4),
            "c0b": round(score_c0b, 4),
            "c1": round(score_c1, 4),
            "c2": round(score_c2, 4),
            "distill_premium": distill_premium,
            "cot_vs_direct": cot_vs_direct,
            "elapsed_seconds": scale_elapsed
        })

    # Save summary curve file
    total_time = round(time.time() - t0_all, 2)
    summary_data = {
        "meta": {
            "model_name": model_name,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "n_test": n_test,
            "total_elapsed_seconds": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "scaling_curve": summary_curve
    }

    summary_file = "docs/math_scaling_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[Done] Finished full scaling suite in {total_time}s ({total_time/60:.1f}m). Saved to {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_math_scaling()
