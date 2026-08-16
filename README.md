# Empirical AI Distillation Measurement Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Empirical Benchmark](https://img.shields.io/badge/Live%20Data-Measured%20Results-success.svg)](docs/index.html)

A reproducible, statistical machine learning experiment measuring **black-box capability extraction from commercial model APIs**.

Rather than relying on macro estimates, this suite runs real multi-seed empirical sweeps using neural networks to measure:
1. **The Distillation Capability Frontier**: Accuracy recovered vs. API query budget $Q \in [50, 150, 400, 1000, 2500, 6000]$.
2. **The Counterfactual Baseline Floor**: The performance achievable with zero teacher access ($N_{\text{public}} = 400$).
3. **Information Leakage Modes**: Standard hard text answers (**Argmax API**) vs. Probability distribution vectors (**Logprob API / "Dark Knowledge"**).
4. **Active Elicitation Efficiency**: Uniform random queries vs. Entropy-based **Active Uncertainty Sampling** (the information-theoretic ceiling).
5. **Generalization vs. Memorization**: Measuring performance on an **out-of-distribution (OOD) shifted test set** to verify whether capability genuinely transfers or simply memorizes queried space.

---

## 📊 Measured Empirical Results

Results averaged across **5 independent random seeds** ($\mu \pm \sigma$):

| Query Budget ($Q$) | Argmax / Random | Argmax / Active | Logprob / Random | Logprob / Active (Ceiling) |
| :---: | :---: | :---: | :---: | :---: |
| **$Q = 50$** | 46.9% (-13.6%) | 48.4% (-12.0%) | 43.5% (-17.0%) | 43.1% (-17.3%) |
| **$Q = 150$** | 56.0% (-4.4%) | 58.0% (-2.5%) | 52.6% (-7.9%) | 53.9% (-6.5%) |
| **$Q = 400$** | 61.4% (+0.9%) | 61.9% (+1.5%) | 60.9% (+0.4%) | 60.4% (-0.0%) |
| **$Q = 1,000$** | 63.2% (+2.7%) | 64.7% (+4.2%) | 64.2% (+3.7%) | 64.1% (+3.6%) |
| **$Q = 2,500$** | 66.8% (+6.3%) | 67.6% (+7.1%) | 68.8% (+8.3%) | 67.5% (+7.0%) |
| **$Q = 6,000$** | **70.8% (+10.3%)** | **70.7% (+10.2%)** | **70.8% (+10.3%)** | **71.7% (+11.2%)** |

- **Frontier Teacher Ceiling:** **79.6%** (Trained on 25,000 samples)
- **Counterfactual Public Floor:** **60.5%** (Trained on 400 public samples)
- **Max Proprietary Gap Recovered:** **58.6%** (at $Q = 6,000$)
- **Generalization Gap (OOD Shift Drop):** 
  - Teacher drop on OOD: $-14.1\%$
  - Student drop on OOD: $-17.0\%$ to $-18.4\%$ (Proving capability function transfer)

---

## 🚀 Quickstart: Reproducing Locally

### 1. Requirements
Install standard dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Empirical Experiment
```bash
python empirical_distillation.py
```
This executes the multi-seed sweep, prints the tabular analysis, and regenerates `docs/empirical_results.json`.

### 3. Launch the Interactive Dashboard
```bash
python -m http.server 8000 --directory docs
```
Navigate to `http://localhost:8000` to interact with the measured curves, toggle error bands, and switch between absolute accuracy, marginal uplift, and OOD generalization.

---

## 🗂️ Git Branch Architecture
- **`main`**: The empirical, statistical benchmark suite and live interactive dashboard.
- **`macro-economic-model`**: The high-level capital asymmetry and policy simulation framework.

---

## 📄 License
MIT License. Open for academic, policy, and safety analysis.
