"""
Empirical Distillation Measurement Suite
========================================
Runs real empirical experiments measuring black-box capability extraction:
1. Ground truth teacher model ($N = 25,000$, $D = 12$, $K = 6$)
2. Counterfactual public-data baseline floor ($N_{public} = 400$)
3. Distillation under multiple access modes (Argmax vs Logprob) and query strategies (Random vs Active Entropy)
4. On-distribution vs OOD Generalization Gap (Transfer vs Memorization)
5. Multi-seed variance and confidence intervals

Exports results to `docs/empirical_results.json` and `empirical_frontier.png`.
"""

import json
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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


class EmpiricalExperiment:
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.rng = np.random.default_rng(12345)
        self._init_ground_truth()

    def _init_ground_truth(self):
        D, K = self.config.input_dim, self.config.num_classes
        self.W1 = self.rng.normal(scale=1.0, size=(D, 64))
        self.b1 = self.rng.normal(scale=0.5, size=64)
        self.W2 = self.rng.normal(scale=1.0, size=(64, K))
        self.b2 = self.rng.normal(scale=0.5, size=K)

    def _ground_truth_labels(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(X @ self.W1 + self.b1)
        return np.argmax(h @ self.W2 + self.b2, axis=1)

    def _sample(self, n: int, center: float = 0.0, rng: np.random.Generator = None) -> np.ndarray:
        r = rng if rng is not None else self.rng
        return r.normal(loc=center, scale=1.0, size=(n, self.config.input_dim))

    def _net_kwargs(self, seed: int = None) -> dict:
        return dict(
            hidden_layer_sizes=(80, 40),
            max_iter=300,
            alpha=1e-4,
            random_state=seed,
        )

    def run(self) -> dict:
        print("=" * 76)
        print(" RUNNING EMPIRICAL DISTILLATION EXPERIMENT")
        print(f" • Input Dim: {self.config.input_dim} | Classes: {self.config.num_classes}")
        print(f" • Teacher Training Samples: {self.config.teacher_samples:,}")
        print(f" • Public Baseline Samples:   {self.config.public_samples:,}")
        print(f" • Query Budgets (Q):         {self.config.budgets}")
        print(f" • Random Seeds Tested:       {len(self.config.seeds)} ({self.config.seeds})")
        print("=" * 76)

        start_time = time.time()

        # 1. Test Sets
        X_test_on = self._sample(self.config.test_samples, center=0.0)
        y_test_on = self._ground_truth_labels(X_test_on)

        X_test_off = self._sample(self.config.test_samples, center=self.config.ood_shift)
        y_test_off = self._ground_truth_labels(X_test_off)

        # 2. Train Ground-Truth Teacher Model
        print("\n[1/4] Training Frontier Teacher Model...")
        Xt = self._sample(self.config.teacher_samples)
        yt = self._ground_truth_labels(Xt)
        teacher = MLPClassifier(**self._net_kwargs(seed=42)).fit(Xt, yt)

        teacher_on_acc = float(accuracy_score(y_test_on, teacher.predict(X_test_on)))
        teacher_off_acc = float(accuracy_score(y_test_off, teacher.predict(X_test_off)))
        print(f"  [+] Teacher On-Distribution Accuracy:  {teacher_on_acc:.2%}")
        print(f"  [+] Teacher Off-Distribution (OOD):   {teacher_off_acc:.2%}")

        # Helper API interfaces
        def api_argmax(X):
            return teacher.predict(X)

        def api_logprob(X):
            return teacher.predict_proba(X)

        # 3. Counterfactual Baseline (Public Data Only)
        print("\n[2/4] Measuring Counterfactual Baseline Floor (No Teacher Access)...")
        baseline_on_scores = []
        baseline_off_scores = []
        for s in self.config.seeds:
            seed_rng = np.random.default_rng(1000 + s)
            Xp = self._sample(self.config.public_samples, rng=seed_rng)
            yp = self._ground_truth_labels(Xp)
            base_model = MLPClassifier(**self._net_kwargs(seed=s)).fit(Xp, yp)
            baseline_on_scores.append(accuracy_score(y_test_on, base_model.predict(X_test_on)))
            baseline_off_scores.append(accuracy_score(y_test_off, base_model.predict(X_test_off)))

        base_on = float(np.mean(baseline_on_scores))
        base_off = float(np.mean(baseline_off_scores))
        base_on_std = float(np.std(baseline_on_scores))
        print(f"  [+] Baseline Floor (On-Dist):  {base_on:.2%} (± {base_on_std:.2%})")
        print(f"  [+] Baseline Floor (OOD):      {base_off:.2%}")
        print(f"  [+] Proprietary Capability Gap: {teacher_on_acc - base_on:.2%}")

        # 4. Run Distillation Conditions
        conditions = [
            ("argmax", "random", "Argmax API / Random Query"),
            ("argmax", "active", "Argmax API / Active Uncertainty"),
            ("logprob", "random", "Logprob API / Random Query"),
            ("logprob", "active", "Logprob API / Active Uncertainty (Elicitation Ceiling)"),
        ]

        print(f"\n[3/4] Evaluating {len(conditions)} Distillation Conditions Across {len(self.config.budgets)} Query Budgets...")
        
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

        def train_student(Q: int, access: str, strategy: str, seed: int):
            seed_rng = np.random.default_rng(5000 + seed * 200 + Q)

            # Query selection
            if strategy == "random":
                Xq = self._sample(Q, rng=seed_rng)
            else:
                # Active uncertainty sampling via prediction entropy
                pool = self._sample(max(Q * 5, 4000), rng=seed_rng)
                init_q = min(max(30, Q // 5), Q)
                Xq = self._sample(init_q, rng=seed_rng)

                while len(Xq) < Q:
                    if access == "argmax":
                        st = MLPClassifier(**self._net_kwargs(seed=seed)).fit(Xq, api_argmax(Xq))
                        p = st.predict_proba(pool)
                    else:
                        st = MLPRegressor(**self._net_kwargs(seed=seed)).fit(Xq, api_logprob(Xq))
                        pred = np.clip(st.predict(pool), 1e-6, None)
                        p = pred / pred.sum(axis=1, keepdims=True)

                    entropy = -(p * np.log(p + 1e-9)).sum(axis=1)
                    batch_size = min(Q - len(Xq), max(30, Q // 4))
                    top_idx = np.argsort(entropy)[-batch_size:]
                    Xq = np.vstack([Xq, pool[top_idx]])
                Xq = Xq[:Q]

            # Fit student model
            if access == "argmax":
                student = MLPClassifier(**self._net_kwargs(seed=seed)).fit(Xq, api_argmax(Xq))
                pred_on = student.predict(X_test_on)
                pred_off = student.predict(X_test_off)
            else:
                student = MLPRegressor(**self._net_kwargs(seed=seed)).fit(Xq, api_logprob(Xq))
                pred_on = np.argmax(student.predict(X_test_on), axis=1)
                pred_off = np.argmax(student.predict(X_test_off), axis=1)

            return float(accuracy_score(y_test_on, pred_on)), float(accuracy_score(y_test_off, pred_off))

        total_gap = teacher_on_acc - base_on

        for Q in self.config.budgets:
            for access, strategy, _ in conditions:
                key = f"{access}_{strategy}"
                scores = [train_student(Q, access, strategy, s) for s in self.config.seeds]
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
        print(f"  [+] Completed all sweeps in {elapsed:.2f} seconds.")

        # Summary Output Table
        print("\n" + "=" * 76)
        print(" EMPIRICAL RESULTS SUMMARY (On-Distribution Accuracy & Marginal Uplift)")
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
        print(f"\n[4/4] Saved empirical results to '{json_path}'.")

        return export_payload


if __name__ == "__main__":
    exp = EmpiricalExperiment()
    results = exp.run()
