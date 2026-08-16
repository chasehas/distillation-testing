"""
High-Performance Batched Empirical LLM Distillation Benchmark (GSM8K)
======================================================================
Runs on NVIDIA GeForce RTX 4070 Ti Super with Batched GPU Inference:
- Base Student Model: Qwen/Qwen2.5-0.5B-Instruct
- Benchmark: Grade School Math (GSM8K Test Split)
- Sample Size: N = 100 samples

4-Tier Experimental Hierarchy:
  * Condition 0A: Base Model Floor (Untrained zero-shot base model)
  * Condition 0B: Human SFT Baseline (Trained on human crowdworker reference solutions)
  * Condition 1:  Direct Answers Only (Zero CoT: Question -> #### Final Answer)
  * Condition 2:  Frontier AI Distillation (Trained on GPT-4 / Frontier AI CoT traces)
"""

import os
import re
import time
import json
from typing import Tuple, List, Dict
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


def extract_answer_number(text: str) -> str:
    """Extracts the final numerical answer from mathematical text."""
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans = ans.replace(",", "").replace("$", "").strip()
        match = re.search(r"[-+]?\d*\.?\d+", ans)
        if match:
            return match.group(0)

    # Search for \boxed{...}
    if "\\boxed{" in text:
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match:
            inner = match.group(1).replace(",", "").replace("$", "").strip()
            num_m = re.search(r"[-+]?\d*\.?\d+", inner)
            if num_m:
                return num_m.group(0)

    # Fallback: Find the last numerical token in the text
    matches = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if matches:
        return matches[-1]
    return ""


def clean_direct_answer(full_solution: str) -> str:
    """Extracts just '#### X' from a full GSM8K solution, stripping intermediate reasoning."""
    num = extract_answer_number(full_solution)
    return f"#### {num}"


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    N_train: int = 100,
    N_test: int = 100,
    eval_batch_size: int = 20,
    train_batch_size: int = 4,
    epochs: int = 3,
    learning_rate: float = 2e-4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("=" * 80)
    print(f" BATCHED EMPIRICAL LLM DISTILLATION BENCHMARK (GSM8K)")
    print(f" • GPU Device:          {gpu_name.upper()}")
    print(f" • Base Student Model:  {model_name}")
    print(f" • Training Budget (N): {N_train} samples")
    print(f" • Test Evaluation:     {N_test} benchmark problems (Batched GPU Inference)")
    print("=" * 80)

    start_time = time.time()

    # 1. Load Datasets
    print("\n[1/5] Loading GSM8K & MetaMathQA (GPT-4 Distillation) Datasets...")
    gsm8k = load_dataset("openai/gsm8k", "main")
    train_gsm8k = gsm8k["train"].select(range(N_train))
    test_gsm8k = gsm8k["test"].select(range(N_test))

    # Load GPT-4 distilled GSM8K solutions from MetaMathQA
    print("  [+] Loading GPT-4 distilled reasoning traces (MetaMathQA)...")
    metamath = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    
    # Collect N_train GPT-4 GSM8K traces
    gpt4_traces = []
    for item in metamath:
        if item.get("type") == "GSM_AnsAug" or "The answer is" in item.get("response", ""):
            gpt4_traces.append({"question": item["query"], "answer": item["response"]})
            if len(gpt4_traces) >= N_train:
                break

    print(f"  [+] Loaded {len(gpt4_traces)} GPT-4 distilled reasoning pairs.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left" # Required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Fast Batched Evaluation Function
    def evaluate_model_batched(model, label: str) -> Tuple[float, list]:
        print(f"\n--- Fast Batched Evaluation: {label} ({N_test} Test Problems) ---")
        model.eval()
        correct = 0
        samples_log = []
        eval_start = time.time()

        test_items = list(test_gsm8k)
        for i in range(0, len(test_items), eval_batch_size):
            batch_slice = test_items[i : i + eval_batch_size]
            prompts = [f"Question: {item['question']}\nAnswer:" for item in batch_slice]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for j, item in enumerate(batch_slice):
                gen_text = tokenizer.decode(outputs[j][inputs.input_ids[j].shape[0]:], skip_special_tokens=True)
                pred_num = extract_answer_number(gen_text)
                true_num = extract_answer_number(item["answer"])

                is_correct = (pred_num != "" and pred_num == true_num)
                if is_correct:
                    correct += 1

                if len(samples_log) < 3:
                    samples_log.append({
                        "question": item["question"],
                        "generated": gen_text.strip(),
                        "ground_truth": item["answer"].strip(),
                        "pred_num": pred_num,
                        "true_num": true_num,
                        "correct": is_correct,
                    })

            done = min(i + eval_batch_size, len(test_items))
            print(f"  [{done}/{len(test_items)}] Evaluated in {time.time() - eval_start:.1f}s | Current Acc: {correct / done:.1%}")

        final_acc = correct / len(test_items)
        print(f"  ==> {label} Final Score: {final_acc:.1%} ({correct}/{len(test_items)}) in {time.time() - eval_start:.1f}s")
        return final_acc, samples_log

    # 3. Condition 0A: Untrained Base Model Floor
    print("\n[2/5] Evaluating Condition 0A (Untrained Base Model Floor)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    acc_c0a, samples_c0a = evaluate_model_batched(base_model, "Condition 0A (Untrained Base Floor)")
    del base_model
    torch.cuda.empty_cache()

    # Helper function to fine-tune with LoRA
    def fine_tune_and_eval(condition_name: str, dataset_records: list) -> Tuple[float, list]:
        print(f"\n[+] Fine-Tuning: {condition_name} (N = {len(dataset_records)} samples)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)

        # Prepare formatted training dataset
        formatted_list = []
        for item in dataset_records:
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            enc = tokenizer(text, truncation=True, max_length=512, padding=False)
            enc["labels"] = enc["input_ids"].copy()
            formatted_list.append(enc)

        # Convert to HuggingFace Dataset
        from datasets import Dataset
        formatted_ds = Dataset.from_list(formatted_list)

        output_dir = f"checkpoints/{condition_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            logging_steps=10,
            save_strategy="no",
            eval_strategy="no",
            fp16=True,
            report_to="none",
        )

        tokenizer.padding_side = "right" # Training needs right padding
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=formatted_ds,
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        )

        train_start = time.time()
        trainer.train()
        print(f"  [+] Finished training in {time.time() - train_start:.1f}s.")

        tokenizer.padding_side = "left" # Back to left padding for generation
        acc, samples = evaluate_model_batched(model, condition_name)
        
        del model, trainer
        torch.cuda.empty_cache()
        return acc, samples

    # 4. Condition 0B: Human Reference SFT (Human crowdworker solutions)
    print("\n[3/5] Condition 0B: Human Reference SFT (Original 2021 Human Annotations)...")
    human_records = [{"question": item["question"], "answer": item["answer"]} for item in train_gsm8k]
    acc_c0b, samples_c0b = fine_tune_and_eval("Condition 0B (Human Reference SFT)", human_records)

    # 5. Condition 1: Direct Answers Only (Zero Reasoning Tokens: Question -> #### Answer)
    print("\n[4/5] Condition 1: Direct Answers Only (Defended API / Zero CoT)...")
    direct_records = [{"question": item["question"], "answer": clean_direct_answer(item["answer"])} for item in train_gsm8k]
    acc_c1, samples_c1 = fine_tune_and_eval("Condition 1 (Direct Answers Only)", direct_records)

    # 6. Condition 2: Frontier AI Distillation (GPT-4 Distilled CoT Traces)
    print("\n[5/5] Condition 2: Frontier AI Distillation (GPT-4 MetaMath Traces)...")
    acc_c2, samples_c2 = fine_tune_and_eval("Condition 2 (GPT-4 Distilled CoT)", gpt4_traces)

    elapsed_total = time.time() - start_time

    # 7. Summary Table
    print("\n" + "=" * 80)
    print(f" EMPIRICAL LLM DISTILLATION RESULTS AT LOWEST N (N = {N_train})")
    print("=" * 80)
    print(f" • Condition 0A (Untrained Base Model Floor):       {acc_c0a:.1%}")
    print(f" • Condition 0B (Human Reference SFT):              {acc_c0b:.1%} (Uplift: {acc_c0b - acc_c0a:+.1%})")
    print(f" • Condition 1  (Direct Answers Only / Zero CoT):   {acc_c1:.1%} (Uplift: {acc_c1 - acc_c0a:+.1%})")
    print(f" • Condition 2  (GPT-4 Frontier AI Distillation):   {acc_c2:.1%} (Uplift: {acc_c2 - acc_c0a:+.1%})")
    print("-" * 80)
    distill_vs_human = acc_c2 - acc_c0b
    cot_vs_direct = acc_c2 - acc_c1
    print(f" [+] FRONTIER DISTILLATION PREMIUM OVER HUMAN SFT:   {distill_vs_human:+.1%}")
    print(f" [+] NET REASONING MULTIPLIER (CoT vs Direct):       {cot_vs_direct:+.1%}")
    print(f" [+] TOTAL EXECUTION TIME:                            {elapsed_total:.1f}s")
    print("=" * 80)

    # Save export results
    payload = {
        "meta": {
            "model_name": model_name,
            "gpu_name": gpu_name,
            "N_train": N_train,
            "N_test": N_test,
            "elapsed_seconds": round(elapsed_total, 2),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": {
            "condition_0a_base_floor": round(acc_c0a, 4),
            "condition_0b_human_sft": round(acc_c0b, 4),
            "condition_1_direct_answer": round(acc_c1, 4),
            "condition_2_gpt4_distilled": round(acc_c2, 4),
            "distillation_premium_over_human": round(distill_vs_human, 4),
            "reasoning_premium_over_direct": round(cot_vs_direct, 4),
        },
        "sample_outputs": {
            "condition_0a": samples_c0a,
            "condition_0b": samples_c0b,
            "condition_1": samples_c1,
            "condition_2": samples_c2,
        },
    }

    os.makedirs("docs", exist_ok=True)
    out_file = os.path.join("docs", "llm_empirical_results.json")
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Saved real empirical LLM results to '{out_file}'.")

    return payload


if __name__ == "__main__":
    run_experiment(N_train=100, N_test=100)
