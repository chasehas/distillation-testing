"""
Batched PyTorch Distillation with Variable Public Baseline N
============================================================
Measures the Matched-Budget Opportunity Cost:
At every data budget N in [50, 150, 400, 1000, 2500, 6000]:
1. Organic Public Baseline (Trained on N unassisted public samples)
2. Argmax Distillation (Trained on Q = N hard labels from Teacher)
3. Logprob Distillation (Trained on Q = N soft probability vectors)
4. Active Uncertainty Distillation (Trained on Q = N entropy-selected boundary points)
5. Matched Distillation Premium: Accuracy(Distilled N) - Accuracy(Public N)
"""

import json
import os
import time
import torch
import torch.nn.functional as F
import numpy as np


class BatchedMLP(torch.nn.Module):
    """Batched Multi-Layer Perceptron: trains B distinct models simultaneously."""
    def __init__(self, num_models: int, in_dim: int = 12, h1: int = 64, h2: int = 32, out_dim: int = 6, device="cuda"):
        super().__init__()
        self.B = num_models
        self.device = device
        
        # 3D Weight tensors: [B, In, Out]
        self.W1 = torch.nn.Parameter(torch.randn(num_models, in_dim, h1, device=device) * (2.0 / in_dim)**0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(num_models, 1, h1, device=device))
        
        self.W2 = torch.nn.Parameter(torch.randn(num_models, h1, h2, device=device) * (2.0 / h1)**0.5)
        self.b2 = torch.nn.Parameter(torch.zeros(num_models, 1, h2, device=device))
        
        self.W3 = torch.nn.Parameter(torch.randn(num_models, h2, out_dim, device=device) * (2.0 / h2)**0.5)
        self.b3 = torch.nn.Parameter(torch.zeros(num_models, 1, out_dim, device=device))

    def forward(self, x):
        # x shape: [B, N, In]
        h1 = torch.tanh(torch.bmm(x, self.W1) + self.b1)
        h2 = torch.tanh(torch.bmm(h1, self.W2) + self.b2)
        logits = torch.bmm(h2, self.W3) + self.b3
        return logits


def run_batched_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print("=" * 76)
    print(f" BATCHED PYTORCH DISTILLATION (VARIABLE N BUDGET)")
    print(f" • GPU Device:    {gpu_name.upper()}")
    print(" • Investigation: Matched-Sample Shootout (Public N vs. Distilled Q)")
    print("=" * 76)

    start_time = time.time()
    rng = np.random.default_rng(12345)

    D, K = 12, 6
    sample_budgets = [50, 150, 400, 1000, 2500, 6000]
    seeds = [10, 20, 30, 40, 50]
    conditions = [
        ("argmax", "random", "Argmax API / Random Query"),
        ("argmax", "active", "Argmax API / Active Uncertainty"),
        ("logprob", "random", "Logprob API / Random Query"),
        ("logprob", "active", "Logprob API / Active Uncertainty (Elicitation Ceiling)"),
    ]

    # 1. Ground Truth Target Network
    W1_gt = torch.tensor(rng.normal(scale=1.0, size=(D, 64)), dtype=torch.float32, device=device)
    b1_gt = torch.tensor(rng.normal(scale=0.5, size=64), dtype=torch.float32, device=device)
    W2_gt = torch.tensor(rng.normal(scale=1.0, size=(64, K)), dtype=torch.float32, device=device)
    b2_gt = torch.tensor(rng.normal(scale=0.5, size=K), dtype=torch.float32, device=device)

    def teacher_logits(X):
        h = torch.tanh(X @ W1_gt + b1_gt)
        return h @ W2_gt + b2_gt

    # Test Sets (On-distribution and OOD shifted)
    N_test = 3000
    X_test_on = torch.tensor(rng.normal(loc=0.0, scale=1.0, size=(N_test, D)), dtype=torch.float32, device=device)
    y_test_on = torch.argmax(teacher_logits(X_test_on), dim=1)

    X_test_off = torch.tensor(rng.normal(loc=1.25, scale=1.0, size=(N_test, D)), dtype=torch.float32, device=device)
    y_test_off = torch.argmax(teacher_logits(X_test_off), dim=1)

    # 2. Train Frontier Teacher Model
    print("\n[1/3] Training Frontier Teacher Model (N = 25,000)...")
    N_teacher = 25000
    Xt = torch.tensor(rng.normal(size=(N_teacher, D)), dtype=torch.float32, device=device)
    yt = torch.argmax(teacher_logits(Xt), dim=1)

    teacher_net = BatchedMLP(num_models=1, in_dim=D, out_dim=K, device=device)
    opt_t = torch.optim.Adam(teacher_net.parameters(), lr=0.015, weight_decay=1e-4)
    Xt_batch = Xt.unsqueeze(0)

    for _ in range(120):
        opt_t.zero_grad()
        loss = F.cross_entropy(teacher_net(Xt_batch).squeeze(0), yt)
        loss.backward()
        opt_t.step()

    with torch.no_grad():
        teacher_on_acc = (torch.argmax(teacher_net(X_test_on.unsqueeze(0)).squeeze(0), dim=1) == y_test_on).float().mean().item()
        teacher_off_acc = (torch.argmax(teacher_net(X_test_off.unsqueeze(0)).squeeze(0), dim=1) == y_test_off).float().mean().item()

    print(f"  [+] Teacher Frontier Accuracy: On-Dist = {teacher_on_acc:.2%}, OOD = {teacher_off_acc:.2%}")

    # 3. Train Variable Counterfactual Baseline Curve (For EVERY N in sample_budgets)
    print("\n[2/3] Training Variable Counterfactual Public Baseline for all N in budgets...")
    baseline_results = {
        "on_mean": [],
        "on_std": [],
        "off_mean": [],
        "off_std": [],
    }

    test_on_rep_seeds = X_test_on.unsqueeze(0).expand(len(seeds), -1, -1)
    test_off_rep_seeds = X_test_off.unsqueeze(0).expand(len(seeds), -1, -1)

    for N in sample_budgets:
        X_pub_list = [
            torch.tensor(np.random.default_rng(1000 + s * 10 + N).normal(size=(N, D)), dtype=torch.float32, device=device)
            for s in seeds
        ]
        X_pub = torch.stack(X_pub_list) # [5, N, D]
        y_pub = torch.stack([torch.argmax(teacher_logits(X_pub[i]), dim=1) for i in range(len(seeds))])

        base_net = BatchedMLP(num_models=len(seeds), in_dim=D, out_dim=K, device=device)
        opt_b = torch.optim.Adam(base_net.parameters(), lr=0.015, weight_decay=1e-4)

        for _ in range(120):
            opt_b.zero_grad()
            logits = base_net(X_pub)
            loss = F.cross_entropy(logits.view(-1, K), y_pub.view(-1))
            loss.backward()
            opt_b.step()

        with torch.no_grad():
            preds_on = torch.argmax(base_net(test_on_rep_seeds), dim=-1)
            preds_off = torch.argmax(base_net(test_off_rep_seeds), dim=-1)

            accs_on = [(preds_on[i] == y_test_on).float().mean().item() for i in range(len(seeds))]
            accs_off = [(preds_off[i] == y_test_off).float().mean().item() for i in range(len(seeds))]

            baseline_results["on_mean"].append(round(float(np.mean(accs_on)), 4))
            baseline_results["on_std"].append(round(float(np.std(accs_on)), 4))
            baseline_results["off_mean"].append(round(float(np.mean(accs_off)), 4))
            baseline_results["off_std"].append(round(float(np.std(accs_off)), 4))

    print(f"  [+] Variable Baseline Floor Measured across N: {baseline_results['on_mean']}")

    # 4. Batched Distillation Sweeps
    print(f"\n[3/3] Launching Batched GPU Distillation Sweeps across all N and conditions...")

    results_data = {
        f"{c[0]}_{c[1]}": {
            "name": c[2],
            "access": c[0],
            "strategy": c[1],
            "on_mean": [],
            "on_std": [],
            "off_mean": [],
            "off_std": [],
            "matched_premium": [],      # Distilled(N) - Baseline(N)
            "gap_recovered_pct": [],
        }
        for c in conditions
    }

    for b_idx, N in enumerate(sample_budgets):
        batch_X = []
        batch_Y_hard = []
        batch_Y_soft = []

        for access, strategy, _ in conditions:
            for s in seeds:
                seed_rng = np.random.default_rng(5000 + s * 200 + N)
                if strategy == "random":
                    Xq = torch.tensor(seed_rng.normal(size=(N, D)), dtype=torch.float32, device=device)
                else:
                    pool = torch.tensor(seed_rng.normal(size=(max(N * 3, 2000), D)), dtype=torch.float32, device=device)
                    init_q = min(max(30, N // 5), N)
                    Xq = pool[:init_q]
                    with torch.no_grad():
                        t_probs = F.softmax(teacher_logits(pool), dim=-1)
                        entropy = -(t_probs * torch.log(t_probs + 1e-9)).sum(dim=-1)
                        _, top_idx = torch.topk(entropy, N - init_q)
                        Xq = torch.cat([Xq, pool[top_idx]], dim=0)

                with torch.no_grad():
                    t_logits = teacher_logits(Xq)
                    y_hard = torch.argmax(t_logits, dim=-1)
                    y_soft = F.softmax(t_logits, dim=-1)

                batch_X.append(Xq)
                batch_Y_hard.append(y_hard)
                batch_Y_soft.append(y_soft)

        num_models = len(batch_X) # 20 models per budget
        tensor_X = torch.stack(batch_X) # [20, N, D]
        tensor_Y_hard = torch.stack(batch_Y_hard)
        tensor_Y_soft = torch.stack(batch_Y_soft)

        student_batch = BatchedMLP(num_models=num_models, in_dim=D, out_dim=K, device=device)
        opt_s = torch.optim.Adam(student_batch.parameters(), lr=0.015, weight_decay=1e-4)

        for _ in range(120):
            opt_s.zero_grad()
            out = student_batch(tensor_X)
            loss_hard = F.cross_entropy(out[:10].view(-1, K), tensor_Y_hard[:10].view(-1))
            loss_soft = F.kl_div(F.log_softmax(out[10:], dim=-1), tensor_Y_soft[10:], reduction="batchmean")
            loss = loss_hard + loss_soft
            loss.backward()
            opt_s.step()

        with torch.no_grad():
            test_on_rep = X_test_on.unsqueeze(0).expand(num_models, -1, -1)
            test_off_rep = X_test_off.unsqueeze(0).expand(num_models, -1, -1)
            preds_on = torch.argmax(student_batch(test_on_rep), dim=-1)
            preds_off = torch.argmax(student_batch(test_off_rep), dim=-1)

            idx = 0
            base_n_acc = baseline_results["on_mean"][b_idx]
            gap_at_n = teacher_on_acc - base_n_acc

            for access, strategy, _ in conditions:
                key = f"{access}_{strategy}"
                on_accs = [(preds_on[idx + i] == y_test_on).float().mean().item() for i in range(len(seeds))]
                off_accs = [(preds_off[idx + i] == y_test_off).float().mean().item() for i in range(len(seeds))]
                idx += len(seeds)

                on_mean = float(np.mean(on_accs))
                on_std = float(np.std(on_accs))
                off_mean = float(np.mean(off_accs))
                off_std = float(np.std(off_accs))

                premium = on_mean - base_n_acc
                gap_pct = (premium / max(1e-4, gap_at_n)) * 100

                results_data[key]["on_mean"].append(round(on_mean, 4))
                results_data[key]["on_std"].append(round(on_std, 4))
                results_data[key]["off_mean"].append(round(off_mean, 4))
                results_data[key]["off_std"].append(round(off_std, 4))
                results_data[key]["matched_premium"].append(round(premium, 4))
                results_data[key]["gap_recovered_pct"].append(round(gap_pct, 2))

    elapsed = time.time() - start_time
    print(f"\n  [+] COMPLETED ALL EXPERIMENTS WITH VARIABLE N IN {elapsed:.2f} SECONDS!")

    # Tabular Comparison
    print("\n" + "=" * 80)
    print(" MATCHED-SAMPLE OPPORTUNITY COST TABLE: PUBLIC DATA (N) vs. DISTILLED API (Q=N)")
    print("=" * 80)
    print(f"{'Budget N':>10} | {'Public Baseline':>15} | {'Argmax/Rand':>14} | {'Logprob/Active':>16} | {'Distill Premium':>16}")
    print("-" * 80)

    for i, N in enumerate(sample_budgets):
        p_base = baseline_results["on_mean"][i]
        arg_r = results_data["argmax_random"]["on_mean"][i]
        log_a = results_data["logprob_active"]["on_mean"][i]
        prem = results_data["logprob_active"]["matched_premium"][i]
        print(f"{N:>10} | {p_base:>14.1%} | {arg_r:>13.1%} | {log_a:>15.1%} | {prem:>+15.1%}")

    # Export Payload
    export_payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "gpu_name": gpu_name,
            "budgets": sample_budgets,
            "seeds": seeds,
            "elapsed_seconds": round(elapsed, 2),
        },
        "teacher": {
            "on_accuracy": round(teacher_on_acc, 4),
            "off_accuracy": round(teacher_off_acc, 4),
            "generalization_drop": round(teacher_on_acc - teacher_off_acc, 4),
        },
        "baseline_curve": baseline_results,
        "curves": results_data,
    }

    os.makedirs("docs", exist_ok=True)
    json_path = os.path.join("docs", "empirical_results.json")
    with open(json_path, "w") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\n[+] Saved complete variable-N results to '{json_path}'.")

    return export_payload


if __name__ == "__main__":
    run_batched_experiment()
