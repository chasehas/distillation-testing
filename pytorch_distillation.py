"""
PyTorch CUDA-Accelerated Distillation Experiment
================================================
Leverages NVIDIA GeForce RTX 4070 Ti Super (CUDA) for high-performance distillation:
1. Ground truth teacher model (Trained on 25k samples on CUDA)
2. Counterfactual public baseline (400 samples)
3. GPU-vectorized distillation across Argmax (Cross-Entropy) & Logprob (KL-Divergence / Softmax MSE)
4. Fast tensor-based Active Uncertainty Sampling using GPU `torch.topk` entropy
5. On-distribution vs OOD Generalization Gap
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class ExperimentConfig:
    input_dim: int = 12
    num_classes: int = 6
    teacher_samples: int = 25_000
    public_samples: int = 400
    budgets: List[int] = field(default_factory=lambda: [50, 150, 400, 1000, 2500, 6000])
    seeds: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50])
    test_samples: int = 3000
    ood_shift: float = 1.25
    epochs: int = 150
    lr: float = 0.01


class MLPModel(nn.Module):
    def __init__(self, in_dim=12, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class PyTorchDistillationExperiment:
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(12345)
        self._init_ground_truth()

    def _init_ground_truth(self):
        D, K = self.config.input_dim, self.config.num_classes
        self.W1 = torch.tensor(self.rng.normal(scale=1.0, size=(D, 64)), dtype=torch.float32, device=self.device)
        self.b1 = torch.tensor(self.rng.normal(scale=0.5, size=64), dtype=torch.float32, device=self.device)
        self.W2 = torch.tensor(self.rng.normal(scale=1.0, size=(64, K)), dtype=torch.float32, device=self.device)
        self.b2 = torch.tensor(self.rng.normal(scale=0.5, size=K), dtype=torch.float32, device=self.device)

    def _ground_truth_labels(self, X: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(X @ self.W1 + self.b1)
        return torch.argmax(h @ self.W2 + self.b2, dim=1)

    def _sample(self, n: int, center: float = 0.0, rng: np.random.Generator = None) -> torch.Tensor:
        r = rng if rng is not None else self.rng
        arr = r.normal(loc=center, scale=1.0, size=(n, self.config.input_dim))
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def _train_model(self, X_train: torch.Tensor, y_train: torch.Tensor, is_prob: bool = False, seed: int = 42) -> nn.Module:
        torch.manual_seed(seed)
        model = MLPModel(self.config.input_dim, self.config.num_classes).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=1e-4)

        for _ in range(self.config.epochs):
            optimizer.zero_grad()
            logits = model(X_train)
            if is_prob:
                # Soft probability loss (KL divergence or MSE on softmax)
                loss = F.kl_div(F.log_softmax(logits, dim=-1), y_train, reduction="batchmean")
            else:
                # Hard classification loss
                loss = F.cross_entropy(logits, y_train)
            loss.backward()
            optimizer.step()

        return model

    def run(self) -> dict:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print("=" * 76)
        print(f" PYTORCH DISTILLATION ACCELERATION ({gpu_name.upper()})")
        print(f" • Device:                    {self.device}")
        print(f" • Input Dim / Classes:       {self.config.input_dim} / {self.config.num_classes}")
        print(f" • Teacher Training Samples:  {self.config.teacher_samples:,}")
        print(f" • Public Baseline Samples:    {self.config.public_samples:,}")
        print(f" • Query Budgets (Q):          {self.config.budgets}")
        print(f" • Random Seeds Tested:        {len(self.config.seeds)} ({self.config.seeds})")
        print("=" * 76)

        start_time = time.time()

        # 1. Test Sets
        X_test_on = self._sample(self.config.test_samples, center=0.0)
        y_test_on = self._ground_truth_labels(X_test_on)

        X_test_off = self._sample(self.config.test_samples, center=self.config.ood_shift)
        y_test_off = self._ground_truth_labels(X_test_off)

        # 2. Train Ground-Truth Teacher Model on GPU
        print("\n[1/4] Training Frontier Teacher Model on GPU...")
        Xt = self._sample(self.config.teacher_samples)
        yt = self._ground_truth_labels(Xt)
        teacher = self._train_model(Xt, yt, is_prob=False, seed=42)

        with torch.no_grad():
            teacher_on_acc = float((torch.argmax(teacher(X_test_on), dim=1) == y_test_on).float().mean().item())
            teacher_off_acc = float((torch.argmax(teacher(X_test_off), dim=1) == y_test_off).float().mean().item())

        print(f"  [+] Teacher On-Distribution Accuracy:  {teacher_on_acc:.2%}")
        print(f"  [+] Teacher Off-Distribution (OOD):   {teacher_off_acc:.2%}")

        # API helper functions
        def api_argmax(X):
            with torch.no_grad():
                return torch.argmax(teacher(X), dim=1)

        def api_logprob(X):
            with torch.no_grad():
                return F.softmax(teacher(X), dim=-1)

        # 3. Counterfactual Baseline
        print("\n[2/4] Measuring Counterfactual Baseline Floor (No Teacher Access)...")
        baseline_on_scores = []
        baseline_off_scores = []
        for s in self.config.seeds:
            seed_rng = np.random.default_rng(1000 + s)
            Xp = self._sample(self.config.public_samples, rng=seed_rng)
            yp = self._ground_truth_labels(Xp)
            base_model = self._train_model(Xp, yp, is_prob=False, seed=s)
            with torch.no_grad():
                on_score = (torch.argmax(base_model(X_test_on), dim=1) == y_test_on).float().mean().item()
                off_score = (torch.argmax(base_model(X_test_off), dim=1) == y_test_off).float().mean().item()
                baseline_on_scores.append(on_score)
                baseline_off_scores.append(off_score)

        base_on = float(np.mean(baseline_on_scores))
        base_off = float(np.mean(baseline_off_scores))
        base_on_std = float(np.std(baseline_on_scores))
        total_gap = teacher_on_acc - base_on
        print(f"  [+] Baseline Floor (On-Dist):  {base_on:.2%} (± {base_on_std:.2%})")
        print(f"  [+] Baseline Floor (OOD):      {base_off:.2%}")
        print(f"  [+] Proprietary Capability Gap: {total_gap:.2%}")

        # 4. Sweep conditions
        conditions = [
            ("argmax", "random", "Argmax API / Random Query"),
            ("argmax", "active", "Argmax API / Active Uncertainty"),
            ("logprob", "random", "Logprob API / Random Query"),
            ("logprob", "active", "Logprob API / Active Uncertainty (Elicitation Ceiling)"),
        ]

        total_tasks = len(self.config.budgets) * len(conditions) * len(self.config.seeds)
        print(f"\n[3/4] GPU Execution: Training {total_tasks} Student Models on {gpu_name}...")

        results_data = {
            f"{c[0]}_{c[1]}": {
                "name": c[2],
                "access": c[0],
                "strategy": c[1],
                "on_mean": [],
                "on_std": [],
                "off_mean": [],
                "off_std": [],
                "marginal_uplift": [],
                "gap_recovered_pct": [],
            }
            for c in conditions
        }

        def train_student_gpu(Q: int, access: str, strategy: str, seed: int) -> Tuple[float, float]:
            seed_rng = np.random.default_rng(5000 + seed * 200 + Q)

            # Query selection
            if strategy == "random":
                Xq = self._sample(Q, rng=seed_rng)
            else:
                # Active uncertainty sampling via GPU prediction entropy
                pool = self._sample(max(Q * 4, 3000), rng=seed_rng)
                init_q = min(max(30, Q // 5), Q)
                Xq = self._sample(init_q, rng=seed_rng)

                while len(Xq) < Q:
                    is_prob = (access == "logprob")
                    targets = api_logprob(Xq) if is_prob else api_argmax(Xq)
                    st = self._train_model(Xq, targets, is_prob=is_prob, seed=seed)

                    with torch.no_grad():
                        probs = F.softmax(st(pool), dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
                        batch_size = min(Q - len(Xq), max(40, Q // 4))
                        _, top_indices = torch.topk(entropy, batch_size)
                        Xq = torch.cat([Xq, pool[top_indices]], dim=0)

                Xq = Xq[:Q]

            # Fit student model on extracted dataset
            is_prob = (access == "logprob")
            targets = api_logprob(Xq) if is_prob else api_argmax(Xq)
            student = self._train_model(Xq, targets, is_prob=is_prob, seed=seed)

            with torch.no_grad():
                pred_on = torch.argmax(student(X_test_on), dim=1)
                pred_off = torch.argmax(student(X_test_off), dim=1)
                on_acc = (pred_on == y_test_on).float().mean().item()
                off_acc = (pred_off == y_test_off).float().mean().item()

            return on_acc, off_acc

        for Q in self.config.budgets:
            for access, strategy, _ in conditions:
                key = f"{access}_{strategy}"
                scores = [train_student_gpu(Q, access, strategy, s) for s in self.config.seeds]
                on_accs = [s[0] for s in scores]
                off_accs = [s[1] for s in scores]

                on_mean = float(np.mean(on_accs))
                on_std = float(np.std(on_accs))
                off_mean = float(np.mean(off_accs))
                off_std = float(np.std(off_accs))

                uplift = on_mean - base_on
                gap_pct = (uplift / max(1e-4, total_gap)) * 100

                results_data[key]["on_mean"].append(round(on_mean, 4))
                results_data[key]["on_std"].append(round(on_std, 4))
                results_data[key]["off_mean"].append(round(off_mean, 4))
                results_data[key]["off_std"].append(round(off_std, 4))
                results_data[key]["marginal_uplift"].append(round(uplift, 4))
                results_data[key]["gap_recovered_pct"].append(round(gap_pct, 2))

        elapsed = time.time() - start_time
        print(f"  [+] Completed ALL {total_tasks} GPU distillation training sweeps in {elapsed:.2f} seconds!")

        # Summary Output Table
        print("\n" + "=" * 76)
        print(" PYTORCH GPU RESULTS SUMMARY (On-Distribution Accuracy & Marginal Uplift)")
        print("=" * 76)
        header = f"{'Budget (Q)':>10} |"
        for c in conditions:
            short_name = f"{c[0][:3]}/{c[1][:3]}"
            header += f" {short_name:>13} |"
        print(header)
        print("-" * 76)

        for i, Q in enumerate(self.config.budgets):
            row = f"{Q:>10} |"
            for c in conditions:
                k = f"{c[0]}_{c[1]}"
                acc = results_data[k]["on_mean"][i]
                uplift = results_data[k]["marginal_uplift"][i]
                row += f" {acc:.1%} ({uplift:+.1%}) |"
            print(row)

        print("\nGeneralization Gap Analysis at Q = 2,500 (On-Distribution vs OOD Shift):")
        q_idx = self.config.budgets.index(2500) if 2500 in self.config.budgets else -1
        for c in conditions:
            k = f"{c[0]}_{c[1]}"
            on_val = results_data[k]["on_mean"][q_idx]
            off_val = results_data[k]["off_mean"][q_idx]
            drop = on_val - off_val
            print(f" • {c[2]:<52}: On={on_val:.1%}  OOD={off_val:.1%}  (Drop={drop:+.1%})")
        print(f" • {'Frontier Teacher Model':<52}: On={teacher_on_acc:.1%}  OOD={teacher_off_acc:.1%}  (Drop={teacher_on_acc-teacher_off_acc:+.1%})")

        # Compile final export dict
        export_payload = {
            "meta": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(self.device),
                "gpu_name": gpu_name,
                "input_dim": self.config.input_dim,
                "num_classes": self.config.num_classes,
                "teacher_samples": self.config.teacher_samples,
                "public_samples": self.config.public_samples,
                "budgets": self.config.budgets,
                "seeds": self.config.seeds,
                "elapsed_seconds": round(elapsed, 2),
            },
            "teacher": {
                "on_accuracy": round(teacher_on_acc, 4),
                "off_accuracy": round(teacher_off_acc, 4),
                "generalization_drop": round(teacher_on_acc - teacher_off_acc, 4),
            },
            "baseline": {
                "on_accuracy": round(base_on, 4),
                "on_std": round(base_on_std, 4),
                "off_accuracy": round(base_off, 4),
                "proprietary_gap": round(total_gap, 4),
            },
            "curves": results_data,
        }

        # Save to docs/empirical_results.json
        os.makedirs("docs", exist_ok=True)
        json_path = os.path.join("docs", "empirical_results.json")
        with open(json_path, "w") as f:
            json.dump(export_payload, f, indent=2)
        print(f"\n[4/4] Saved GPU empirical results to '{json_path}'.")

        return export_payload


if __name__ == "__main__":
    exp = PyTorchDistillationExperiment()
    results = exp.run()
