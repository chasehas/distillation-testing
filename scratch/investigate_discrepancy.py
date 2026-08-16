import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re

def extract_answer_number(text: str) -> str:
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

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    gsm8k = load_dataset("openai/gsm8k", "main")
    test_slice_50 = list(gsm8k["test"].select(range(50)))
    test_slice_100 = list(gsm8k["test"].select(range(100)))

    for n_name, test_slice in [("N_test=50", test_slice_50), ("N_test=100", test_slice_100)]:
        correct_with_stop = 0
        correct_without_stop = 0
        
        for item in test_slice:
            prompt = f"Question: {item['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Without stop
            pred_raw = extract_answer_number(gen)
            # With stop
            gen_stop = gen
            for stop in ["\nQuestion:", "\nHuman:", "\n[Question]", "\nAssistant:", "\n\n\n"]:
                if stop in gen_stop:
                    gen_stop = gen_stop[:gen_stop.index(stop)]
            pred_stop = extract_answer_number(gen_stop)
            
            true_ans = extract_answer_number(item["answer"])
            if pred_raw == true_ans and pred_raw != "":
                correct_without_stop += 1
            if pred_stop == true_ans and pred_stop != "":
                correct_stop += 1
                
        print(f"Results on {n_name}:")
        print(f"  Without stop truncation (original bug): {correct_without_stop}/{len(test_slice)} ({correct_without_stop/len(test_slice):.1%})")
        print(f"  With stop truncation (fixed):           {correct_stop}/{len(test_slice)} ({correct_stop/len(test_slice):.1%})")

if __name__ == "__main__":
    main()
