"""
Fast Batched Evaluation Module on NVIDIA RTX 4070 Ti Super
==========================================================
Left-padded batched GPU inference for Math Reasoning and General Instructions.
"""

import time
from typing import List, Dict, Tuple, Any
import torch
from .dataset_builder import extract_answer_number


def compute_token_f1(pred: str, target: str) -> float:
    """Computes token-level F1 / overlap score between predicted text and gold response."""
    pred_tokens = pred.lower().split()
    target_tokens = target.lower().split()
    
    if not pred_tokens or not target_tokens:
        return 0.0
        
    common = set(pred_tokens) & set(target_tokens)
    if not common:
        return 0.0
        
    precision = sum(min(pred_tokens.count(w), target_tokens.count(w)) for w in common) / len(pred_tokens)
    recall = sum(min(pred_tokens.count(w), target_tokens.count(w)) for w in common) / len(target_tokens)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_math_batched(
    model: Any,
    tokenizer: Any,
    test_slice: List[Dict[str, str]],
    label: str,
    batch_size: int = 20,
    max_new_tokens: int = 256,
) -> Tuple[float, List[Dict[str, Any]], float]:
    """
    Evaluates math problem solving accuracy on GSM8K using batched left-padded generation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  [Eval] Fast Batched Math Evaluation: {label} ({len(test_slice)} problems)...")
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()
    correct = 0
    samples_log = []
    t0 = time.time()
    
    for i in range(0, len(test_slice), batch_size):
        batch = test_slice[i : i + batch_size]
        prompts = [f"Question: {item['question']}\nAnswer:" for item in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        for j, item in enumerate(batch):
            gen_text = tokenizer.decode(outputs[j][inputs.input_ids[j].shape[0]:], skip_special_tokens=True)
            # Truncate at common continuation/rambling patterns
            for stop in ["\nQuestion:", "\nHuman:", "\n[Question]", "\nAssistant:", "\n\n\n"]:
                if stop in gen_text:
                    gen_text = gen_text[:gen_text.index(stop)]
            pred_num = extract_answer_number(gen_text)
            true_num = extract_answer_number(item["answer"])
            
            is_correct = (pred_num != "" and pred_num == true_num)
            if is_correct:
                correct += 1
                
            if len(samples_log) < 4:
                samples_log.append({
                    "prompt": item["question"],
                    "generated": gen_text.strip(),
                    "ground_truth": item["answer"].strip(),
                    "pred_value": pred_num,
                    "true_value": true_num,
                    "correct": is_correct,
                })
                
    elapsed = time.time() - t0
    acc = correct / len(test_slice) if test_slice else 0.0
    print(f"  [+] {label} Score: {acc:.1%} ({correct}/{len(test_slice)}) in {elapsed:.1f}s")
    
    return acc, samples_log, round(elapsed, 2)


def evaluate_instruction_batched(
    model: Any,
    tokenizer: Any,
    test_slice: List[Dict[str, str]],
    label: str,
    batch_size: int = 20,
    max_new_tokens: int = 256,
) -> Tuple[float, List[Dict[str, Any]], float]:
    """
    Evaluates multi-domain instruction following on UltraFeedback using batched left-padded generation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  [Eval] Fast Batched Instruction Evaluation: {label} ({len(test_slice)} instructions)...")
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()
    f1_scores = []
    samples_log = []
    t0 = time.time()
    
    for i in range(0, len(test_slice), batch_size):
        batch = test_slice[i : i + batch_size]
        prompts = [f"Instruction: {item['instruction']}\nResponse:" for item in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        for j, item in enumerate(batch):
            gen_text = tokenizer.decode(outputs[j][inputs.input_ids[j].shape[0]:], skip_special_tokens=True).strip()
            gold_text = item.get("gold_response", "").strip()
            
            f1 = compute_token_f1(gen_text, gold_text)
            f1_scores.append(f1)
            
            if len(samples_log) < 4:
                samples_log.append({
                    "prompt": item["instruction"],
                    "generated": gen_text,
                    "ground_truth": gold_text,
                    "f1_score": round(f1 * 100, 1),
                })
                
    elapsed = time.time() - t0
    avg_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print(f"  [+] {label} Quality Alignment Score: {avg_score:.1%} in {elapsed:.1f}s")
    
    return avg_score, samples_log, round(elapsed, 2)


# ---------------------------------------------------------------------------
# Domain C: Sandboxed Python Code Execution (pass@1 on MBPP)
# ---------------------------------------------------------------------------

def extract_python_code(text: str) -> str:
    """
    Extracts the first plausible Python code block from generated text.

    Tries, in order:
      1. Fenced ```python ... ``` block
      2. Fenced ``` ... ``` block containing 'def '
      3. Contiguous lines starting from the first 'def ' to the next blank
         or non-indented non-code line
    """
    import re

    # Strategy 1: fenced python block
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Strategy 2: generic fenced block containing a function definition
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m and "def " in m.group(1):
        return m.group(1).strip()

    # Strategy 3: grab from first 'def ' to end-of-function heuristic
    lines = text.split("\n")
    code_lines = []
    capturing = False
    for line in lines:
        if not capturing:
            if line.lstrip().startswith("def "):
                capturing = True
                code_lines.append(line)
        else:
            # Stop on clearly non-code continuations
            if line.strip() == "" and code_lines:
                code_lines.append(line)
            elif line and not line[0].isspace() and not line.startswith("def "):
                break
            else:
                code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return text.strip()


def _run_code_sandbox(code: str, test_assertions: list, timeout: int = 10) -> bool:
    """
    Executes generated code + MBPP test assertions in a subprocess sandbox.
    Returns True if all assertions pass, False otherwise.
    """
    import subprocess
    import tempfile
    import os

    test_code = code + "\n\n" + "\n".join(test_assertions) + "\n"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(test_code)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def evaluate_code_batched(
    model,
    tokenizer,
    test_slice: list,
    label: str,
    batch_size: int = 10,
    max_new_tokens: int = 512,
) -> tuple:
    """
    Evaluates code generation on MBPP problems using sandboxed pass@1 execution.

    For each MBPP problem:
      1. Prompt the model with the task description
      2. Extract Python code from the generated output
      3. Execute code + MBPP assertions in a subprocess sandbox
      4. Score: pass@1 = fraction of problems where all assertions pass
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  [Eval] Sandboxed Code Execution (pass@1): {label} ({len(test_slice)} problems)...")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    passed = 0
    samples_log = []
    t0 = time.time()

    for i in range(0, len(test_slice), batch_size):
        batch = test_slice[i : i + batch_size]
        prompts = [
            f"Write a Python function.\n{item['prompt']}\n\n```python\n"
            for item in batch
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, item in enumerate(batch):
            gen_text = tokenizer.decode(
                outputs[j][inputs.input_ids[j].shape[0] :],
                skip_special_tokens=True,
            )
            # Truncate at common continuation patterns
            for stop in ["\nQuestion:", "\nHuman:", "\n[Question]", "\nWrite a Python", "\n\n\n"]:
                if stop in gen_text:
                    gen_text = gen_text[: gen_text.index(stop)]

            code = extract_python_code(gen_text)
            test_pass = _run_code_sandbox(code, item["test_assertions"])

            if test_pass:
                passed += 1

            if len(samples_log) < 4:
                samples_log.append({
                    "task_id": item.get("task_id", "?"),
                    "prompt": item["prompt"],
                    "generated_code": code[:500],
                    "test_assertions": item["test_assertions"],
                    "passed": test_pass,
                })

    elapsed = time.time() - t0
    pass_rate = passed / len(test_slice) if test_slice else 0.0
    print(f"  [+] {label} pass@1: {pass_rate:.1%} ({passed}/{len(test_slice)}) in {elapsed:.1f}s")

    return pass_rate, samples_log, round(elapsed, 2)
