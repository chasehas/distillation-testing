"""
Fast LoRA Fine-Tuning Module on NVIDIA RTX 4070 Ti Super
========================================================
Optimized PyTorch/PEFT training loop with FP16 and gradient accumulation.
"""

import time
import random
from typing import List, Dict, Tuple, Any
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


def fine_tune_lora(
    model_name: str,
    dataset_records: List[Dict[str, str]],
    condition_name: str,
    output_dir: str = "checkpoints",
    epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    max_length: int = 512,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Fine-tunes base model using LoRA on the given paired records.
    Returns: (fine_tuned_model, tokenizer, training_stats)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print(f"\n  [Train] Fine-Tuning: {condition_name} ({len(dataset_records)} samples)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
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
    
    # Format and tokenize
    formatted_list = []
    total_tokens = 0
    
    for item in dataset_records:
        if "question" in item:
            prompt_text = f"Question: {item['question']}\nAnswer: "
            response_text = item['answer']
        elif "instruction" in item:
            prompt_text = f"Instruction: {item['instruction']}\nResponse: "
            response_text = item['response']
        else:
            prompt_text = ""
            response_text = str(item)

        # Tokenize prompt separately to find where the answer starts
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        full_text = prompt_text + response_text
        enc = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
        
        # Mask prompt tokens in labels so loss is only computed on the answer
        prompt_len = min(len(prompt_ids), len(enc["input_ids"]))
        labels = [-100] * prompt_len + enc["input_ids"][prompt_len:]
        enc["labels"] = labels[:len(enc["input_ids"])]
        
        total_tokens += len(enc["input_ids"])
        formatted_list.append(enc)
        
    ds = Dataset.from_list(formatted_list)
    
    clean_cond = condition_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    run_output_dir = f"{output_dir}/{clean_cond}"
    
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        logging_steps=15,
        save_strategy="no",
        eval_strategy="no",
        fp16=True,
        seed=42,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )
    
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0
    
    final_loss = train_result.training_loss
    print(f"  [+] Finished training in {elapsed:.1f}s | Final Loss: {final_loss:.4f} | Total Tokens: {total_tokens:,}")
    
    stats = {
        "train_samples": len(dataset_records),
        "train_tokens": total_tokens,
        "elapsed_seconds": round(elapsed, 2),
        "final_loss": round(final_loss, 4),
    }
    
    return model, tokenizer, stats
