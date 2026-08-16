"""
Distillation Benchmark Suite Package
=====================================
Modular tools for empirical LLM distillation testing on local GPU.
"""

from .dataset_builder import load_math_dataset, load_instruction_dataset
from .trainer import fine_tune_lora
from .evaluator import evaluate_math_batched, evaluate_instruction_batched

__all__ = [
    "load_math_dataset",
    "load_instruction_dataset",
    "fine_tune_lora",
    "evaluate_math_batched",
    "evaluate_instruction_batched",
]
