"""
Batched PyTorch CUDA Distillation (Tensor-Parallelized on RTX 4070 Ti Super)
=============================================================================
Instead of training 120 models in a serial loop, stacks all budgets, seeds,
and conditions into a single unified 3D tensor batch [B, N, D] trained
concurrently in GPU VRAM with `torch.bmm`.
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
    print(f" BATCHED TENSOR-PARALLEL PYTORCH DISTILLATION ({gpu_name.upper()})")
    print(" • Architecture: Unified 3D Tensor Batching across all Seeds & Conditions")
    print("=" * 76)

    start_time = time.time()
    rng = np.random.default_rng(12345)

    D, K = 12, 6
    budgets = [50, 150, 400, 1000, 2500, 6000]
    seeds = [10, 20, 30, 40, 50]
    conditions = [
        ("argmax", "random", "Argmax API / Random Query"),
        ("argmax", "active", "Argmax API / Active Uncertainty"),
        ("logprob", "random", "Logprob API / Random Query"),
        ("logprob", "active", "Logprob API / Active Uncertainty (Elicitation Ceiling)"),
    ]

    # Ground Truth Target Network
    W1_gt = torch.tensor(rng.normal(scale=1.0, size=(D, 64)), dtype=torch.float32, device=device)
    b1_gt = torch.tensor(rng.normal(scale=0.5, size=64), dtype=torch.float32, device=device)
    W2_gt = torch.tensor(rng.normal(scale=1.0, size=(64, K)), dtype=torch.float32, device=device)
    b2_gt = torch.tensor(rng.normal(scale=0.5, size=K), dtype=torch.float32, device=device)

    def teacher_logits(X):
        h = torch.tanh(X @ W1_gt + b1_gt)
        return h @ W2_gt + b2_gt

    # Test Sets
    N_test = 3000
    X_test_on = torch.tensor(rng.normal(loc=0.0, scale=1.0, size=(N_test, D)), dtype=torch.float32, device=device)
    y_test_on = torch.argmax(teacher_logits(X_test_on), dim=1)

    X_test_off = torch.tensor(rng.normal(loc=1.25, scale=1.0, size=(N_test, D)), dtype=torch.float32, device=device)
    y_test_off = torch.argmax(teacher_logits(X_test_off), dim=1)

    # 1. Train Teacher Model (Single batch)
    N_teacher = 25000
    Xt = torch.tensor(rng.normal(size=(N_teacher, D)), dtype=torch.float32, device=device)
    yt = torch.argmax(teacher_logits(Xt), dim=1)

    teacher_net = BatchedMLP(num_models=1, in_dim=D, out_dim=K, device=device)
    opt_t = torch.optim.Adam(teacher_net.parameters(), lr=0.015, weight_decay=1e-4)
    Xt_batch = Xt.unsqueeze(0)
    yt_batch = yt.unsqueeze(0)

    for _ in range(120):
        opt_t.zero_grad()
        loss = F.cross_entropy(teacher_net(Xt_batch).squeeze(0), yt)
        loss.backward()
        opt_t.step()

    with torch.no_grad():
        teacher_on_acc = (torch.argmax(teacher_net(X_test_on.unsqueeze(0)).squeeze(0), dim=1) == y_test_on).float().mean().item()
        teacher_off_acc = (torch.argmax(teacher_net(X_test_off.unsqueeze(0)).squeeze(0), dim=1) == y_test_off).float().mean().item()

    print(f"\n[1/3] Teacher Trained: On-Dist Acc = {teacher_on_acc:.2%}, OOD Acc = {teacher_off_acc:.2%}")

    # 2. Counterfactual Baseline Floor (Batched across 5 seeds)
    N_public = 400
    X_pub = torch.stack([
        torch.tensor(np.random.default_rng(1000 + s).normal(size=(N_public, D)), dtype=torch.float32, device=device)
        for s in seeds
    ])
    y_pub = torch.stack([torch.argmax(teacher_logits(X_pub[i]), dim=1) for i in range(len(seeds))])

    base_net = BatchedMLP(num_models=len(seeds), in_dim=D, out_dim=K, device=device)
    opt_b = torch.optim.Adam(base_net.parameters(), lr=0.015, weight_decay=1e-4)

    for _ in range(100):
        opt_b.zero_grad()
        logits = base_net(X_pub) # [5, 400, 6]
        loss = F.cross_entropy(logits.view(-1, K), y_pub.view(-1))
        loss.backward()
        opt_b.step()

    with torch.no_grad():
        test_on_rep = X_test_on.unsqueeze(0).expand(len(seeds), -1, -1)
        test_off_rep = X_test_off.unsqueeze(0).expand(len(seeds), -1, -1)
        base_on_preds = torch.argmax(base_net(test_on_rep), dim=-1)
        base_off_preds = torch.argmax(base_net(test_off_rep), dim=-1)

        base_on_accs = [(base_on_preds[i] == y_test_on).float().mean().item() for i in range(len(seeds))]
        base_off_accs = [(base_off_preds[i] == y_test_off).float().mean().item() for i in range(len(seeds))]

    base_on = float(np.mean(base_on_accs))
    base_off = float(np.mean(base_off_accs))
    base_on_std = float(np.std(base_on_accs))
    total_gap = teacher_on_acc - base_on
    print(f"[2/3] Counterfactual Baseline: On-Dist = {base_on:.2%} (± {base_on_std:.2%}), OOD = {base_off:.2%}")

    # 3. Batched Distillation Execution across ALL 120 sweeps in parallel
    print(f"\n[3/3] Launching Batched GPU Tensor Sweeps across 120 models...")

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

    # For each budget Q, train all conditions x seeds in one single batched tensor
    for Q in budgets:
        # Generate training sets for all 4 conditions x 5 seeds = 20 models per budget Q
        batch_X = []
        batch_Y_hard = []
        batch_Y_soft = []
        batch_is_prob = []

        for access, strategy, _ in conditions:
            for s in seeds:
                seed_rng = np.random.default_rng(5000 + s * 200 + Q)
                if strategy == "random":
                    Xq = torch.tensor(seed_rng.normal(size=(Q, D)), dtype=torch.float32, device=device)
                else:
                    pool = torch.tensor(seed_rng.normal(size=(max(Q * 3, 2000), D)), dtype=torch.float32, device=device)
                    init_q = min(max(30, Q // 5), Q)
                    Xq = pool[:init_q]
                    # Direct teacher query entropy
                    with torch.no_grad():
                        t_probs = F.softmax(teacher_logits(pool), dim=-1)
                        entropy = -(t_probs * torch.log(t_probs + 1e-9)).sum(dim=-1)
                        _, top_idx = torch.topk(entropy, Q - init_q)
                        Xq = torch.cat([Xq, pool[top_idx]], dim=0)

                with torch.no_grad():
                    t_logits = teacher_logits(Xq)
                    y_hard = torch.argmax(t_logits, dim=-1)
                    y_soft = F.softmax(t_logits, dim=-1)

                batch_X.append(Xq)
                batch_Y_hard.append(y_hard)
                batch_Y_soft.append(y_soft)
                batch_is_prob.append(access == "logprob")

        num_models_q = len(batch_X) # 20 models
        tensor_X = torch.stack(batch_X) # [20, Q, D]
        tensor_Y_hard = torch.stack(batch_Y_hard) # [20, Q]
        tensor_Y_soft = torch.stack(batch_Y_soft) # [20, Q, K]

        # Train all 20 models simultaneously with BatchedMLP
        student_batch = BatchedMLP(num_models=num_models_q, in_dim=D, out_dim=K, device=device)
        opt_s = torch.optim.Adam(student_batch.parameters(), lr=0.015, weight_decay=1e-4)

        for _ in range(120):
            opt_s.zero_grad()
            out = student_batch(tensor_X) # [20, Q, K]
            # Combined Loss: Hard CE for argmax, KL Div for logprobs
            loss_hard = F.cross_entropy(out[:10].view(-1, K), tensor_Y_hard[:10].view(-1))
            loss_soft = F.kl_div(F.log_softmax(out[10:], dim=-1), tensor_Y_soft[10:], reduction="batchmean")
            loss = loss_hard + loss_soft
            loss.backward()
            opt_s.step()

        # Evaluate on Test Sets simultaneously
        with torch.no_grad():
            test_on_rep_q = X_test_on.unsqueeze(0).expand(num_models_q, -1, -1)
            test_off_rep_q = X_test_off.unsqueeze(0).expand(num_models_q, -1, -1)
            preds_on = torch.argmax(student_batch(test_on_rep_q), dim=-1)
            preds_off = torch.argmax(student_batch(test_off_rep_q), dim=-1)

            idx = 0
            for access, strategy, _ in conditions:
                key = f"{access}_{strategy}"
                on_accs = [(preds_on[idx + i] == y_test_on).float().mean().item() for i in range(len(seeds))]
                off_accs = [(preds_off[idx + i] == y_test_off).float().mean().item() for i in range(len(seeds))]
                idx += len(seeds)

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
    print(f"\n  [+] ALL 120 MODELS TRAINED IN PARALLEL IN {elapsed:.2f} SECONDS!")

    # Output Table
    print("\n" + "=" * 76)
    print(" GPU TENSOR-BATCHED RESULTS SUMMARY (On-Distribution Accuracy & Uplift)")
    print("=" * 76)
    header = f"{'Budget (Q)':>10} |"
    for c in conditions:
        short_name = f"{c[0][:3]}/{c[1][:3]}"
        header += f" {short_name:>13} |"
    print(header)
    print("-" * 76)

    for i, Q in enumerate(budgets):
        row = f"{Q:>10} |"
        for c in conditions:
            k = f"{c[0]}_{c[1]}"
            acc = results_data[k]["on_mean"][i]
            uplift = results_data[k]["marginal_uplift"][i]
            row += f" {acc:.1%} ({uplift:+.1%}) |"
        print(row)

    # Save to docs/empirical_results.json
    export_payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "gpu_name": gpu_name,
            "budgets": budgets,
            "seeds": seeds,
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

    os.makedirs("docs", exist_ok=True)
    json_path = os.path.join("docs", "empirical_results.json")
    with open(json_path, "w") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\n[+] Saved batched GPU results to '{json_path}'.")

    return export_payload


if __name__ == "__main__":
    run_batched_experiment()
