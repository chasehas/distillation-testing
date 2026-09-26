"""
Dataset Builder: Strict 1-to-1 Prompt Alignment
================================================
Loads and formats paired training datasets across model tiers for:
  1. Math Reasoning (GSM8K <-> MetaMathQA)
  2. Multi-Domain Instruction Following (UltraFeedback)
"""

import re
from typing import Dict, List, Tuple
from datasets import load_dataset


def extract_answer_number(text: str) -> str:
    """Extracts the final numerical answer from mathematical text."""
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans = ans.replace(",", "").replace("$", "").strip()
        match = re.search(r"[-+]?\d*\.?\d+", ans)
        if match:
            return match.group(0)

    if "\\boxed{" in text:
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match:
            inner = match.group(1).replace(",", "").replace("$", "").strip()
            num_m = re.search(r"[-+]?\d*\.?\d+", inner)
            if num_m:
                return num_m.group(0)

    matches = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if matches:
        return matches[-1]
    return ""


def clean_direct_answer(full_solution: str) -> str:
    """Extracts just '#### X' from a full GSM8K solution, stripping intermediate reasoning."""
    num = extract_answer_number(full_solution)
    return f"#### {num}"


def load_math_dataset(N_train: int = 250, N_test: int = 100) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    """
    Loads GSM8K questions and matches them 1-to-1 with Human and GPT-4 responses.
    """
    print(f"\n[Dataset] Loading Math Reasoning datasets (N_train={N_train}, N_test={N_test})...")
    
    # 1. Load GSM8K dataset
    gsm8k = load_dataset("openai/gsm8k", "main")
    train_slice = gsm8k["train"].select(range(N_train))
    test_slice = list(gsm8k["test"].select(range(N_test)))
    
    # Build question lookup map
    train_questions = [item["question"] for item in train_slice]
    train_human_answers = [item["answer"] for item in train_slice]
    
    # 2. Match with MetaMathQA (GPT-4 Distilled traces)
    print("  [+] Querying MetaMathQA for GPT-4 distilled solutions to exact same questions...")
    metamath = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    
    gpt4_lookup = {}
    target_q_set = set(train_questions)
    
    for item in metamath:
        q = item.get("query", "").strip()
        if q in target_q_set and q not in gpt4_lookup:
            resp = item.get("response", "").strip()
            if "####" in resp or "The answer is" in resp:
                gpt4_lookup[q] = resp
                if len(gpt4_lookup) >= len(target_q_set):
                    break
    
    # Assemble Condition datasets with STRICT 1-to-1 matching (no fallback)
    # Only include questions that have exact matches in MetaMathQA to avoid
    # contaminating Condition 2 with Condition 0B's human answers.
    cond_0b_human = []
    cond_1_direct = []
    cond_2_gpt4 = []
    skipped = 0
    
    for q, human_a in zip(train_questions, train_human_answers):
        if q not in gpt4_lookup:
            skipped += 1
            continue
        cond_0b_human.append({"question": q, "answer": human_a})
        cond_1_direct.append({"question": q, "answer": clean_direct_answer(human_a)})
        cond_2_gpt4.append({"question": q, "answer": gpt4_lookup[q]})
    
    print(f"  [+] Strictly matched {len(cond_2_gpt4)} questions (skipped {skipped} unmatched).")
    
    train_dict = {
        "condition_0b": cond_0b_human,
        "condition_1": cond_1_direct,
        "condition_2": cond_2_gpt4,
    }
    
    return train_dict, test_slice


def load_instruction_dataset(N_train: int = 250, N_test: int = 100) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    """
    Loads UltraFeedback multi-domain instructions and extracts paired responses
    from Weak, Medium, and Frontier GPT-4 model completions for identical prompts.
    """
    print(f"\n[Dataset] Loading UltraFeedback paired instruction dataset (N_train={N_train}, N_test={N_test})...")
    
    ds = load_dataset("openbmb/UltraFeedback", split="train", streaming=True)
    
    cond_0b_weak = []
    cond_1_medium = []
    cond_2_frontier = []
    test_slice = []
    
    total_needed = N_train + N_test
    collected = 0
    
    for item in ds:
        instruction = item.get("instruction", "").strip()
        completions = item.get("completions", [])
        
        if not instruction or len(completions) < 2:
            continue
        
        # Sort completions by overall helpfulness rating
        # Rating is string or int (1 to 5)
        parsed_comps = []
        for c in completions:
            resp_text = c.get("response", "").strip()
            model_name = c.get("model", "unknown")
            try:
                rating = float(c.get("annotations", {}).get("helpfulness", {}).get("Rating", 3))
            except Exception:
                rating = 3.0
            
            if resp_text:
                parsed_comps.append({
                    "model": model_name,
                    "response": resp_text,
                    "rating": rating
                })
        
        if len(parsed_comps) < 2:
            continue
            
        parsed_comps.sort(key=lambda x: x["rating"])
        
        weak_comp = parsed_comps[0]["response"]
        frontier_comp = parsed_comps[-1]["response"]
        
        # Medium completion: pick middle if 3+ completions available
        if len(parsed_comps) >= 3:
            medium_comp = parsed_comps[len(parsed_comps) // 2]["response"]
        else:
            medium_comp = weak_comp
            
        if collected < N_train:
            cond_0b_weak.append({"instruction": instruction, "response": weak_comp})
            cond_1_medium.append({"instruction": instruction, "response": medium_comp})
            cond_2_frontier.append({"instruction": instruction, "response": frontier_comp})
        elif collected < total_needed:
            test_slice.append({
                "instruction": instruction,
                "gold_response": frontier_comp,
                "weak_response": weak_comp,
            })
            
        collected += 1
        if collected >= total_needed:
            break
            
    print(f"  [+] Extracted {len(cond_2_frontier)} paired training instructions and {len(test_slice)} test instructions.")

    train_dict = {
        "condition_0b": cond_0b_weak,
        "condition_1": cond_1_medium,
        "condition_2": cond_2_frontier,
    }

    return train_dict, test_slice


def _is_python_coding_task(instruction: str) -> bool:
    """Heuristic filter for prompts that ask for Python code generation."""
    text = instruction.lower()
    # Positive signals: explicitly asks for Python code
    positive = any(kw in text for kw in [
        "write a python", "python function", "python program", "python script",
        "python code", "implement a function", "write a function that",
        "write code that", "write a program that", "def ",
        "create a function", "implement the following", "write a script",
    ])
    if not positive:
        return False
    # Negative signals: explanation/debugging tasks (no code generation)
    negative = any(kw in text for kw in [
        "explain the", "what does this", "debug the", "find the error",
        "what is the output", "trace through",
    ])
    return not negative


def _response_contains_python(response: str) -> bool:
    """Check if a response contains plausible Python code (def/class/import)."""
    return any(kw in response for kw in ["def ", "class ", "import ", "print("])


def load_code_dataset(N_train: int = 150) -> Dict[str, List[Dict]]:
    """
    Loads UltraFeedback coding tasks and extracts paired Python code responses
    from Weak, Medium, and Frontier model completions for identical prompts.

    Returns training dict only — evaluation uses MBPP test split (separate loader).
    """
    print(f"\n[Dataset] Loading UltraFeedback paired CODE dataset (N_train={N_train})...")

    ds = load_dataset("openbmb/UltraFeedback", split="train", streaming=True)

    cond_0b_weak = []
    cond_1_medium = []
    cond_2_frontier = []
    scanned = 0

    for item in ds:
        scanned += 1
        instruction = item.get("instruction", "").strip()
        completions = item.get("completions", [])

        if not instruction or not _is_python_coding_task(instruction):
            continue

        if len(completions) < 2:
            continue

        # Parse and rate completions
        parsed_comps = []
        for c in completions:
            resp_text = c.get("response", "").strip()
            model_name = c.get("model", "unknown")
            try:
                rating = float(
                    c.get("annotations", {}).get("helpfulness", {}).get("Rating", 3)
                )
            except Exception:
                rating = 3.0

            if resp_text and _response_contains_python(resp_text):
                parsed_comps.append({
                    "model": model_name,
                    "response": resp_text,
                    "rating": rating,
                })

        if len(parsed_comps) < 2:
            continue

        parsed_comps.sort(key=lambda x: x["rating"])

        weak_comp = parsed_comps[0]["response"]
        frontier_comp = parsed_comps[-1]["response"]
        medium_comp = (
            parsed_comps[len(parsed_comps) // 2]["response"]
            if len(parsed_comps) >= 3
            else weak_comp
        )

        cond_0b_weak.append({"instruction": instruction, "response": weak_comp})
        cond_1_medium.append({"instruction": instruction, "response": medium_comp})
        cond_2_frontier.append({"instruction": instruction, "response": frontier_comp})

        if len(cond_2_frontier) >= N_train:
            break

    print(
        f"  [+] Scanned {scanned} items, extracted {len(cond_2_frontier)} "
        f"paired Python coding prompts."
    )

    return {
        "condition_0b": cond_0b_weak,
        "condition_1": cond_1_medium,
        "condition_2": cond_2_frontier,
    }


def load_mbpp_test(N_test: int = 100) -> List[Dict]:
    """
    Loads MBPP (Mostly Basic Python Problems) sanitized test split for
    objective pass@1 code evaluation.

    Each item has: task_id, prompt (text), canonical code, and test assertions.
    """
    print(f"\n[Dataset] Loading MBPP sanitized test split (N_test={N_test})...")

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    test_slice = []
    for item in ds:
        if len(test_slice) >= N_test:
            break

        task_id = item.get("task_id", 0)
        text = item.get("prompt", "").strip()
        canonical = item.get("code", "").strip()
        test_list = item.get("test_list", [])

        if not text or not test_list:
            continue

        test_slice.append({
            "task_id": task_id,
            "prompt": text,
            "canonical_code": canonical,
            "test_assertions": test_list,
        })

    print(f"  [+] Loaded {len(test_slice)} MBPP test problems with assertions.")
    return test_slice
