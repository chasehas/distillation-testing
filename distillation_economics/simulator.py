"""
Statistical Learning Simulator for Model Distillation
======================================================
Simulates capability extraction from a black-box teacher model across
different access modes (argmax, logprob, reasoning/CoT) and query strategies
(random vs active entropy sampling), measuring marginal uplift and generalization.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score

# Silence convergence warnings for clean terminal output
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class SimulationConfig:
    input_dim: int = 12
    num_classes: int = 6
    teacher_samples: int = 20_000
    public_samples: int = 400
    budgets: List[int] = field(default_factory=lambda: [100, 300, 800, 2000, 5000])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    test_samples: int = 2500
    ood_shift: float = 1.2  # Out of distribution shift magnitude


@dataclass
class SimulationResult:
    teacher_on_acc: float
    teacher_off_acc: float
    baseline_acc: float
    budgets: List[int]
    conditions: List[Tuple[str, str]]
    curves: Dict[Tuple[str, str], Dict[str, List[float]]]


class DistillationSimulator:
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(42)
        self._init_ground_truth()

    def _init_ground_truth(self):
        """Initialize fixed ground truth target distribution."""
        D, K = self.config.input_dim, self.config.num_classes
        self.W1 = self.rng.normal(scale=0.8, size=(D, 50))
        self.b1 = self.rng.normal(scale=0.5, size=50)
        self.W2 = self.rng.normal(scale=0.8, size=(50, K))
        self.b2 = self.rng.normal(scale=0.5, size=K)

    def _labels(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(X @ self.W1 + self.b1)
        return np.argmax(h @ self.W2 + self.b2, axis=1)

    def _sample(self, n: int, center: float = 0.0, rng: np.random.Generator = None) -> np.ndarray:
        r = rng if rng is not None else self.rng
        return r.normal(loc=center, scale=1.0, size=(n, self.config.input_dim))

    def _make_net_params(self, random_state: int = None) -> dict:
        return dict(
            hidden_layer_sizes=(64, 32),
            max_iter=250,
            alpha=1e-3,
            random_state=random_state,
        )

    def run(self, progress_callback=None) -> SimulationResult:
        """Run full distillation simulation across all budgets and conditions."""
        # 1. Generate test sets
        X_test_on = self._sample(self.config.test_samples, center=0.0)
        y_test_on = self._labels(X_test_on)

        X_test_off = self._sample(self.config.test_samples, center=self.config.ood_shift)
        y_test_off = self._labels(X_test_off)

        # 2. Train Teacher Model
        Xt = self._sample(self.config.teacher_samples)
        yt = self._labels(Xt)
        teacher = MLPClassifier(**self._make_net_params(random_state=42)).fit(Xt, yt)

        teacher_on_acc = float(accuracy_score(y_test_on, teacher.predict(X_test_on)))
        teacher_off_acc = float(accuracy_score(y_test_off, teacher.predict(X_test_off)))

        # API helper functions
        def api_argmax(X):
            return teacher.predict(X)

        def api_logprob(X):
            return teacher.predict_proba(X)

        def api_cot_reasoning(X):
            """Simulates reasoning traces: temperature-sharpened density with lower entropy."""
            probs = teacher.predict_proba(X)
            probs_sharp = np.power(probs, 1.6)
            return probs_sharp / probs_sharp.sum(axis=1, keepdims=True)

        # 3. Compute Counterfactual Baseline (public data only)
        baseline_scores = []
        for seed in self.config.seeds:
            seed_rng = np.random.default_rng(1000 + seed)
            Xp = self._sample(self.config.public_samples, rng=seed_rng)
            yp = self._labels(Xp)
            base_model = MLPClassifier(**self._make_net_params(random_state=seed)).fit(Xp, yp)
            baseline_scores.append(accuracy_score(y_test_on, base_model.predict(X_test_on)))
        baseline_acc = float(np.mean(baseline_scores))

        # 4. Define Distillation conditions
        conditions = [
            ("argmax", "random"),
            ("logprob", "random"),
            ("logprob", "active"),
            ("cot_reasoning", "active"),
        ]

        curves = {c: {"on": [], "off": [], "on_std": []} for c in conditions}

        # 5. Distillation evaluator for a specific budget, condition, and seed
        def evaluate_distillation(Q: int, access: str, strategy: str, seed: int) -> Tuple[float, float]:
            seed_rng = np.random.default_rng(2000 + seed * 100 + Q)

            # Query selection
            if strategy == "random":
                Xq = self._sample(Q, rng=seed_rng)
            else:
                # Active uncertainty / entropy sampling
                pool = self._sample(max(Q * 4, 3000), rng=seed_rng)
                init_q = min(max(40, Q // 5), Q)
                Xq = self._sample(init_q, rng=seed_rng)

                while len(Xq) < Q:
                    if access == "argmax":
                        st = MLPClassifier(**self._make_net_params(random_state=seed)).fit(Xq, api_argmax(Xq))
                        p = st.predict_proba(pool)
                    else:
                        st = MLPRegressor(**self._make_net_params(random_state=seed)).fit(Xq, api_logprob(Xq))
                        pred = np.clip(st.predict(pool), 1e-6, None)
                        p = pred / pred.sum(axis=1, keepdims=True)

                    entropy = -(p * np.log(p + 1e-9)).sum(axis=1)
                    batch_size = min(Q - len(Xq), max(40, Q // 4))
                    top_indices = np.argsort(entropy)[-batch_size:]
                    Xq = np.vstack([Xq, pool[top_indices]])

                Xq = Xq[:Q]

            # Fit student model on extracted queries
            if access == "argmax":
                student = MLPClassifier(**self._make_net_params(random_state=seed)).fit(Xq, api_argmax(Xq))
                pred_on = student.predict(X_test_on)
                pred_off = student.predict(X_test_off)
            elif access == "logprob":
                student = MLPRegressor(**self._make_net_params(random_state=seed)).fit(Xq, api_logprob(Xq))
                pred_on = np.argmax(student.predict(X_test_on), axis=1)
                pred_off = np.argmax(student.predict(X_test_off), axis=1)
            elif access == "cot_reasoning":
                student = MLPRegressor(**self._make_net_params(random_state=seed)).fit(Xq, api_cot_reasoning(Xq))
                pred_on = np.argmax(student.predict(X_test_on), axis=1)
                pred_off = np.argmax(student.predict(X_test_off), axis=1)
            else:
                raise ValueError(f"Unknown access type: {access}")

            return float(accuracy_score(y_test_on, pred_on)), float(accuracy_score(y_test_off, pred_off))

        # 6. Execute sweep over budgets and conditions
        total_steps = len(self.config.budgets) * len(conditions)
        step = 0
        for Q in self.config.budgets:
            for c in conditions:
                access, strategy = c
                seed_outs = [evaluate_distillation(Q, access, strategy, s) for s in self.config.seeds]
                on_scores = [s[0] for s in seed_outs]
                off_scores = [s[1] for s in seed_outs]

                curves[c]["on"].append(float(np.mean(on_scores)))
                curves[c]["off"].append(float(np.mean(off_scores)))
                curves[c]["on_std"].append(float(np.std(on_scores)))

                step += 1
                if progress_callback:
                    progress_callback(step, total_steps, Q, c)

        return SimulationResult(
            teacher_on_acc=teacher_on_acc,
            teacher_off_acc=teacher_off_acc,
            baseline_acc=baseline_acc,
            budgets=self.config.budgets,
            conditions=conditions,
            curves=curves,
        )
