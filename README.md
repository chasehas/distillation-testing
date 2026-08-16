# Distillation Economics: Measuring Commercial API Model Extraction, Capital Asymmetry, and Countermeasure Economics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Interactive Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-success.svg)](docs/index.html)

A mathematical and economic modeling suite investigating the viability, efficiency, and policy implications of **black-box commercial API distillation** (where an adversary or fast-follower lab reconstructs a frontier model's proprietary capabilities using only commercial API query outputs).

---

## 🎯 Executive Summary for Policymakers & AI Strategists

```
Frontier Lab (Teacher)              Commercial API                 Fast-Follower (Student)
┌────────────────────────┐      $0.003 / 1k tokens       ┌────────────────────────┐
│ Pretraining: $65M      │ ───► [Queries & Outputs] ───►  │ Train 8B Student Model │
│ Post-Training: $20M    │                               │ Total Cost: ~$35,000   │
│ Data & Safety: $15M    │                               │                        │
│ Total R&D: $100M       │                               │ Capability: 90%+ Match │
└────────────────────────┘                               └────────────────────────┘
                                                                     │
                                                      Cost Asymmetry: ~2,800x Leverage
                                                      Breakeven: ~2.4M queries served
```

1. **The Capital Asymmetry Dilemma**: Frontier labs spend tens or hundreds of millions of dollars on base pretraining, exploratory compute, synthetic data pipelines, and safety alignment. A fast-follower can extract **85%–95% of the model's core reasoning capability for $10,000 to $50,000** in commercial API queries plus modest fine-tuning compute ($2,000–$10,000 on an 8x H100 cluster).
2. **The Free-Rider Market Failure**: When capability reproduction costs 0.05% of primary R&D, private venture capital incentives to fund fundamental research and safety boundaries collapse without technical protections, contractual enforcement, or API auditing.
3. **Safety Alignment Stripping**: Distillation acts as a "safety filter bypass"—competitors extract core reasoning and dual-use capabilities (e.g., cyber offensive logic, biological synthesis assistance) via benign prompt distributions, training open-weights models that strip away all safety guardrails.
4. **The Elicitation Frontier**:
   - **Argmax (Hard text labels)** leaks capability at baseline rates.
   - **Top-$p$ Logprobs (Probability vectors)** leak "dark knowledge" cluster geometry, accelerating extraction by **$3\times - 5\times$**.
   - **Reasoning Traces (Chain-of-Thought tokens)** provide dense supervisory signals (as seen in the *DeepSeek-R1* paradigm), enabling smaller student models to achieve frontier capability jumps.
   - **Active Uncertainty Sampling** targets the teacher's decision boundaries directly, pushing the extraction rate to its information-theoretic ceiling.

---

## 🖥️ Live Interactive Web Demo (1–2 Minute Pitch)

The repository includes a zero-dependency, single-page web app in [`docs/`](docs/index.html) designed for live demos and deployable to **GitHub Pages** in 1 click.

### Key Interactive Features:
- **Perspective Toggle**:
  - **📊 Overview & Economics**: Real-time Capability Frontier curves, Capital Asymmetry breakdown, and Inference Breakeven trajectory.
  - **🛡️ Frontier Lab View**: An interactive **Defense Cost-Benefit Matrix** (withholding logprobs, masking CoT, entropy anomaly rate-limiting, watermarking, enterprise KYC) showing how defenses degrade attacker efficiency vs. lab overhead.
  - **🏛️ Policymaker View**: Free-rider market failure indices, safety alignment stripping risk meters, and regulatory policy levers (API KYC mandates, watermarking legal statutes, compute threshold cluster auditing).
- **⚡ 1-Click Presets**:
  - *DeepSeek-Style Reasoning Distill*
  - *7B Fast-Follower (Claude 3.5 Sonnet / GPT-4o)*
  - *Frontier Lab Defense Clamped*
  - *Policymaker Market Failure Alert*
- **Interactive 2D Decision Boundary Canvas**: Visualizes point-by-point active boundary querying vs uniform random sampling as the student reconstructs the teacher's manifold.

---

## 🚀 Quickstart: Running Locally

### 1. Requirements
Ensure Python 3.9+ is installed. Install the lightweight dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Simulation Suite
```bash
# Fast test mode (~3 seconds):
python run.py --quick

# Full publication simulation with custom frontier budget and query volume:
python run.py --teacher-rnd 150 --distill-queries 100000 --save-plots
```

### 3. Open the Interactive Web App
Simply open [`docs/index.html`](docs/index.html) in your browser:
```bash
# Or launch a local test server:
python -m http.server 8000 --directory docs
# Navigate to http://localhost:8000
```

---

## 📐 Mathematical Formulation

### 1. Capability Recovery Frontier
We model the recovered student capability $A(Q)$ as a function of query budget $Q$, access modality, and sampling strategy:
$$A(Q) = A_{\text{public}} + (A_{\text{teacher}} - A_{\text{public}}) \cdot \frac{Q^\gamma}{Q^\gamma + K_{1/2}^\gamma}$$
where:
- $A_{\text{public}}$ is the counterfactual no-access baseline trained on public data.
- $A_{\text{teacher}}$ is the frontier teacher model's capability ceiling.
- $K_{1/2}$ is the elicitation half-saturation constant, which shrinks significantly under logprob access, chain-of-thought density, and active entropy sampling.

### 2. Capital Asymmetry Leverage Ratio
$$\text{Leverage} = \frac{C_{\text{Frontier R\&D}}}{C_{\text{Distiller Total}}}$$
where:
$$C_{\text{Frontier R\&D}} = C_{\text{Pretrain}} + C_{\text{Posttrain/RL}} + C_{\text{Data}} + C_{\text{Safety}}$$
$$C_{\text{Distiller Total}} = Q \cdot (T_{\text{in}} P_{\text{in}} + T_{\text{out}} P_{\text{out}}) + C_{\text{Student SFT}} + C_{\text{Data Filter}}$$

### 3. Inference Arbitrage Breakeven Volume
The cumulative cost crossover where self-hosting a distilled student model becomes cheaper than continuously paying the commercial API:
$$V_{\text{breakeven}} = \frac{C_{\text{Distiller Total}}}{C_{\text{Teacher API per req}} - C_{\text{Student Self-Host per req}}}$$

---

## 🌐 Deploying to GitHub Pages (1-Click)

1. Push this repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Distillation Economics Suite"
   git branch -M main
   git remote add origin https://github.com/<your-username>/distillation-economics.git
   git push -u origin main
   ```
2. Go to **Repository Settings** > **Pages**.
3. Under **Build and deployment** > **Source**, select **Deploy from a branch**.
4. Set the branch to `main` and folder to `/docs`, then click **Save**.
5. Your interactive simulator is live at `https://<your-username>.github.io/distillation-economics/`!

---

## 📚 References & Research Basis

1. **Epoch AI Research**: *Trends in Training Compute, R&D Allocations, and Post-Training Dynamics* (2024).
2. **DeepSeek-AI**: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning & Distillation* (2025).
3. **Carlini et al. & Tramèr et al.**: *Stealing Machine Learning Models via Prediction APIs* (USENIX Security).
4. **Kirchenbauer et al.**: *A Watermark for Large Language Models* (ICML 2023).
5. **Anthropic & METR**: *Frontier Model Security & Alignment Extraction Elicitation Standards* (2024).

---

## 📄 License
MIT License. Open for educational, academic, and policy analysis purposes.
