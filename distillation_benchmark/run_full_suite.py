"""
Universal Multi-Domain Distillation Benchmark Suite
Supports 5 empirical domains on any model scale (0.5B, 1.5B, 3B, etc.):
  1. math_reasoning: GSM8K <-> MetaMathQA (CoT vs Direct)
  2. instruction_following: UltraFeedback General Alignment
  3. code_execution: MBPP Python Unit-Test Execution (pass@1)
  4. json_extraction: Structured JSON & Schema Validation (json.loads)
  5. mcq_reasoning: ARC-Challenge Science Multiple Choice (Exact Letter)
"""

import os
import sys
import time
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from distillation_benchmark.dataset_builder import (
    load_math_dataset,
    load_instruction_dataset,
    load_code_dataset,
    load_mbpp_test,
    load_json_dataset,
    load_mcq_dataset,
)
from distillation_benchmark.trainer import fine_tune_lora
from distillation_benchmark.evaluator import (
    evaluate_math_batched,
    evaluate_instruction_batched,
    evaluate_code_batched,
    evaluate_json_batched,
    evaluate_mcq_batched,
)


def run_domain_math(base_model, tokenizer, model_name, n_train, n_test, epochs, output_dir):
    print("\n" + "=" * 80)
    print(" [DOMAIN A] MATH REASONING (GSM8K <-> MetaMathQA)")
    print("=" * 80)
    train_dict, test_slice = load_math_dataset(N_train=n_train, N_test=n_test)

    # 0A Base Floor
    score_c0a, samples_c0a, _ = evaluate_math_batched(base_model, tokenizer, test_slice, "Math: 0A Base Floor")

    # 0B Human SFT
    m_0b, tok_0b, tr_0b = fine_tune_lora(model_name, train_dict["condition_0b"], "Math Human SFT", output_dir=output_dir, epochs=epochs)
    score_c0b, samples_c0b, _ = evaluate_math_batched(m_0b, tok_0b, test_slice, "Math: 0B Human SFT")
    del m_0b
    torch.cuda.empty_cache()

    # 1 Direct Answers
    m_c1, tok_c1, tr_c1 = fine_tune_lora(model_name, train_dict["condition_1"], "Math Direct Answers", output_dir=output_dir, epochs=epochs)
    score_c1, samples_c1, _ = evaluate_math_batched(m_c1, tok_c1, test_slice, "Math: 1 Direct Answers")
    del m_c1
    torch.cuda.empty_cache()

    # 2 Frontier Distill
    m_c2, tok_c2, tr_c2 = fine_tune_lora(model_name, train_dict["condition_2"], "Math Frontier GPT-4", output_dir=output_dir, epochs=epochs)
    score_c2, samples_c2, _ = evaluate_math_batched(m_c2, tok_c2, test_slice, "Math: 2 Frontier Distill")
    del m_c2
    torch.cuda.empty_cache()

    return {
        "title": "Domain A: Math Reasoning (GSM8K <-> MetaMathQA)",
        "metric_name": "Exact-Match Benchmark Accuracy (%)",
        "scores": {
            "c0a_base_floor": round(score_c0a, 4),
            "c0b_human_sft": round(score_c0b, 4),
            "c1_direct_answer": round(score_c1, 4),
            "c2_frontier_distill": round(score_c2, 4),
            "distill_vs_human_premium": round(score_c2 - score_c0b, 4),
            "cot_vs_direct_multiplier": round(score_c2 / max(score_c1, 0.01), 2),
        },
        "training_stats": {
            "c0b_human_tokens": tr_0b["train_tokens"],
            "c1_direct_tokens": tr_c1["train_tokens"],
            "c2_frontier_tokens": tr_c2["train_tokens"],
        },
        "sample_traces": {
            "c0a": samples_c0a,
            "c0b": samples_c0b,
            "c1": samples_c1,
            "c2": samples_c2,
        },
    }


def run_domain_json(base_model, tokenizer, model_name, n_train, n_test, epochs, output_dir):
    print("\n" + "=" * 80)
    print(" [DOMAIN D] STRUCTURED JSON & SCHEMA EXTRACTION")
    print("=" * 80)
    train_dict, test_slice = load_json_dataset(N_train=n_train, N_test=n_test)

    # 0A Base Floor
    score_c0a, samples_c0a, _ = evaluate_json_batched(base_model, tokenizer, test_slice, "JSON: 0A Base Floor")

    # 0B Weak SFT
    m_0b, tok_0b, tr_0b = fine_tune_lora(model_name, train_dict["condition_0b"], "JSON Weak SFT", output_dir=output_dir, epochs=epochs)
    score_c0b, samples_c0b, _ = evaluate_json_batched(m_0b, tok_0b, test_slice, "JSON: 0B Weak SFT")
    del m_0b
    torch.cuda.empty_cache()

    # 1 Medium Model SFT
    m_c1, tok_c1, tr_c1 = fine_tune_lora(model_name, train_dict["condition_1"], "JSON Medium Model", output_dir=output_dir, epochs=epochs)
    score_c1, samples_c1, _ = evaluate_json_batched(m_c1, tok_c1, test_slice, "JSON: 1 Medium Model")
    del m_c1
    torch.cuda.empty_cache()

    # 2 Frontier GPT-4 Distill
    m_c2, tok_c2, tr_c2 = fine_tune_lora(model_name, train_dict["condition_2"], "JSON Frontier GPT-4", output_dir=output_dir, epochs=epochs)
    score_c2, samples_c2, _ = evaluate_json_batched(m_c2, tok_c2, test_slice, "JSON: 2 Frontier Distill")
    del m_c2
    torch.cuda.empty_cache()

    return {
        "title": "Domain D: Structured JSON Extraction & Schema Adherence",
        "metric_name": "Valid Syntactic JSON & Schema Accuracy (%)",
        "scores": {
            "c0a_base_floor": round(score_c0a, 4),
            "c0b_weak_baseline": round(score_c0b, 4),
            "c1_medium_model": round(score_c1, 4),
            "c2_frontier_distill": round(score_c2, 4),
            "distill_vs_weak_premium": round(score_c2 - score_c0b, 4),
            "frontier_vs_medium_gain": round(score_c2 - score_c1, 4),
        },
        "training_stats": {
            "c0b_weak_tokens": tr_0b["train_tokens"],
            "c1_medium_tokens": tr_c1["train_tokens"],
            "c2_frontier_tokens": tr_c2["train_tokens"],
        },
        "sample_traces": {
            "c0a": samples_c0a,
            "c0b": samples_c0b,
            "c1": samples_c1,
            "c2": samples_c2,
        },
    }


def run_domain_mcq(base_model, tokenizer, model_name, n_train, n_test, epochs, output_dir):
    print("\n" + "=" * 80)
    print(" [DOMAIN E] MULTIPLE-CHOICE SCIENCE REASONING (ARC-Challenge)")
    print("=" * 80)
    train_dict, test_slice = load_mcq_dataset(N_train=n_train, N_test=n_test)

    # 0A Base Floor
    score_c0a, samples_c0a, _ = evaluate_mcq_batched(base_model, tokenizer, test_slice, "MCQ: 0A Base Floor")

    # 0B Human / Verbose SFT
    m_0b, tok_0b, tr_0b = fine_tune_lora(model_name, train_dict["condition_0b"], "MCQ Human Verbose", output_dir=output_dir, epochs=epochs)
    score_c0b, samples_c0b, _ = evaluate_mcq_batched(m_0b, tok_0b, test_slice, "MCQ: 0B Human Verbose")
    del m_0b
    torch.cuda.empty_cache()

    # 1 Direct Answer SFT
    m_c1, tok_c1, tr_c1 = fine_tune_lora(model_name, train_dict["condition_1"], "MCQ Direct Answer", output_dir=output_dir, epochs=epochs)
    score_c1, samples_c1, _ = evaluate_mcq_batched(m_c1, tok_c1, test_slice, "MCQ: 1 Direct Answer")
    del m_c1
    torch.cuda.empty_cache()

    # 2 Frontier Direct Distill
    m_c2, tok_c2, tr_c2 = fine_tune_lora(model_name, train_dict["condition_2"], "MCQ Frontier Distill", output_dir=output_dir, epochs=epochs)
    score_c2, samples_c2, _ = evaluate_mcq_batched(m_c2, tok_c2, test_slice, "MCQ: 2 Frontier Distill")
    del m_c2
    torch.cuda.empty_cache()

    return {
        "title": "Domain E: Multiple-Choice Science Reasoning (ARC-Challenge)",
        "metric_name": "Exact-Match Choice Accuracy (%)",
        "scores": {
            "c0a_base_floor": round(score_c0a, 4),
            "c0b_human_verbose": round(score_c0b, 4),
            "c1_direct_answer": round(score_c1, 4),
            "c2_frontier_distill": round(score_c2, 4),
            "distill_vs_human_premium": round(score_c2 - score_c0b, 4),
            "frontier_vs_floor_gain": round(score_c2 - score_c0a, 4),
        },
        "training_stats": {
            "c0b_human_tokens": tr_0b["train_tokens"],
            "c1_direct_tokens": tr_c1["train_tokens"],
            "c2_frontier_tokens": tr_c2["train_tokens"],
        },
        "sample_traces": {
            "c0a": samples_c0a,
            "c0b": samples_c0b,
            "c1": samples_c1,
            "c2": samples_c2,
        },
    }


def run_domain_instruction(base_model, tokenizer, model_name, n_train, n_test, epochs, output_dir):
    print("\n" + "=" * 80)
    print(" [DOMAIN B] MULTI-DOMAIN INSTRUCTION (UltraFeedback)")
    print("=" * 80)
    train_dict, test_slice = load_instruction_dataset(N_train=n_train, N_test=n_test)

    score_c0a, samples_c0a, _ = evaluate_instruction_batched(base_model, tokenizer, test_slice, "Instruct: 0A Base Floor")

    m_0b, tok_0b, tr_0b = fine_tune_lora(model_name, train_dict["condition_0b"], "Instruct Weak SFT", output_dir=output_dir, epochs=epochs)
    score_c0b, samples_c0b, _ = evaluate_instruction_batched(m_0b, tok_0b, test_slice, "Instruct: 0B Weak SFT")
    del m_0b
    torch.cuda.empty_cache()

    m_c1, tok_c1, tr_c1 = fine_tune_lora(model_name, train_dict["condition_1"], "Instruct Medium Model", output_dir=output_dir, epochs=epochs)
    score_c1, samples_c1, _ = evaluate_instruction_batched(m_c1, tok_c1, test_slice, "Instruct: 1 Medium Model")
    del m_c1
    torch.cuda.empty_cache()

    m_c2, tok_c2, tr_c2 = fine_tune_lora(model_name, train_dict["condition_2"], "Instruct Frontier GPT-4", output_dir=output_dir, epochs=epochs)
    score_c2, samples_c2, _ = evaluate_instruction_batched(m_c2, tok_c2, test_slice, "Instruct: 2 Frontier Distill")
    del m_c2
    torch.cuda.empty_cache()

    return {
        "title": "Domain B: General Instruction Emergence (UltraFeedback)",
        "metric_name": "Instruction Alignment Score (%)",
        "scores": {
            "c0a_base_floor": round(score_c0a, 4),
            "c0b_weak_baseline": round(score_c0b, 4),
            "c1_medium_model": round(score_c1, 4),
            "c2_frontier_distill": round(score_c2, 4),
            "distill_vs_weak_premium": round(score_c2 - score_c0b, 4),
            "frontier_vs_medium_gain": round(score_c2 - score_c1, 4),
        },
        "training_stats": {
            "c0b_weak_tokens": tr_0b["train_tokens"],
            "c1_medium_tokens": tr_c1["train_tokens"],
            "c2_frontier_tokens": tr_c2["train_tokens"],
        },
        "sample_traces": {
            "c0a": samples_c0a,
            "c0b": samples_c0b,
            "c1": samples_c1,
            "c2": samples_c2,
        },
    }


def run_domain_code(base_model, tokenizer, model_name, n_train, n_test, epochs, output_dir):
    print("\n" + "=" * 80)
    print(" [DOMAIN C] PYTHON CODE EXECUTION (MBPP pass@1)")
    print("=" * 80)
    train_dict = load_code_dataset(N_train=n_train)
    test_slice = load_mbpp_test(N_test=n_test)

    score_c0a, samples_c0a, _ = evaluate_code_batched(base_model, tokenizer, test_slice, "Code: 0A Base Floor")

    m_0b, tok_0b, tr_0b = fine_tune_lora(model_name, train_dict["condition_0b"], "Code Weak SFT", output_dir=output_dir, epochs=epochs)
    score_c0b, samples_c0b, _ = evaluate_code_batched(m_0b, tok_0b, test_slice, "Code: 0B Weak SFT")
    del m_0b
    torch.cuda.empty_cache()

    m_c1, tok_c1, tr_c1 = fine_tune_lora(model_name, train_dict["condition_1"], "Code Medium Model", output_dir=output_dir, epochs=epochs)
    score_c1, samples_c1, _ = evaluate_code_batched(m_c1, tok_c1, test_slice, "Code: 1 Medium Model")
    del m_c1
    torch.cuda.empty_cache()

    m_c2, tok_c2, tr_c2 = fine_tune_lora(model_name, train_dict["condition_2"], "Code Frontier GPT-4", output_dir=output_dir, epochs=epochs)
    score_c2, samples_c2, _ = evaluate_code_batched(m_c2, tok_c2, test_slice, "Code: 2 Frontier Distill")
    del m_c2
    torch.cuda.empty_cache()

    return {
        "title": "Domain C: Python Code Execution (MBPP pass@1)",
        "metric_name": "Sandboxed pass@1 Execution Accuracy (%)",
        "scores": {
            "c0a_base_floor": round(score_c0a, 4),
            "c0b_weak_baseline": round(score_c0b, 4),
            "c1_medium_model": round(score_c1, 4),
            "c2_frontier_distill": round(score_c2, 4),
            "distill_vs_weak_premium": round(score_c2 - score_c0b, 4),
            "frontier_vs_medium_gain": round(score_c2 - score_c1, 4),
        },
        "training_stats": {
            "c0b_weak_tokens": tr_0b["train_tokens"],
            "c1_medium_tokens": tr_c1["train_tokens"],
            "c2_frontier_tokens": tr_c2["train_tokens"],
        },
        "sample_traces": {
            "c0a": samples_c0a,
            "c0b": samples_c0b,
            "c1": samples_c1,
            "c2": samples_c2,
        },
    }





def main():
    parser = argparse.ArgumentParser(description="Universal Multi-Domain Distillation Suite")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--domains", type=str, default="json,mcq", help="Comma-separated or 'all'")
    parser.add_argument("--n-train", type=int, default=150)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    model_name = args.model
    tag = model_name.split("/")[-1].lower()
    output_dir = f"checkpoints_suite_{tag}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    out_file = args.output_file or f"docs/benchmark_results_{tag.replace('-', '_')}.json"

    print(f"\n[Suite] Initializing Multi-Domain Suite on {model_name}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Load existing output file if present
    results_data = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results_data = json.load(f)
        except Exception:
            results_data = {}

    results_data["meta"] = {
        "model_name": model_name,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "n_train": args.n_train,
        "n_test": args.n_test,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    selected_domains = [d.strip().lower() for d in args.domains.split(",")]
    if "all" in selected_domains:
        selected_domains = ["math", "instruction", "code", "json", "mcq"]

    if "math" in selected_domains:
        results_data["math_reasoning"] = run_domain_math(
            base_model, tokenizer, model_name, args.n_train, args.n_test, args.epochs, output_dir
        )
        with open(out_file, "w") as f:
            json.dump(results_data, f, indent=2)

    if "instruction" in selected_domains:
        results_data["instruction_following"] = run_domain_instruction(
            base_model, tokenizer, model_name, args.n_train, args.n_test, args.epochs, output_dir
        )
        with open(out_file, "w") as f:
            json.dump(results_data, f, indent=2)

    if "code" in selected_domains:
        results_data["code_execution"] = run_domain_code(
            base_model, tokenizer, model_name, args.n_train, args.n_test, args.epochs, output_dir
        )
        with open(out_file, "w") as f:
            json.dump(results_data, f, indent=2)

    if "json" in selected_domains:
        results_data["json_extraction"] = run_domain_json(
            base_model, tokenizer, model_name, args.n_train, args.n_test, args.epochs, output_dir
        )
        with open(out_file, "w") as f:
            json.dump(results_data, f, indent=2)

    if "mcq" in selected_domains:
        results_data["mcq_reasoning"] = run_domain_mcq(
            base_model, tokenizer, model_name, args.n_train, args.n_test, args.epochs, output_dir
        )
        with open(out_file, "w") as f:
            json.dump(results_data, f, indent=2)

    results_data["meta"]["total_elapsed_seconds"] = round(time.time() - t0, 2)
    with open(out_file, "w") as f:
        json.dump(results_data, f, indent=2)

    # Also keep docs/dual_benchmark_results.json updated as the active file
    with open("docs/dual_benchmark_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[Suite] Completed benchmark run in {time.time() - t0:.1f}s. Saved to {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
