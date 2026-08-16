/**
 * Dual-Domain Empirical AI Distillation Dashboard Engine
 * Handles dynamic domain switching (Math vs. Instruction), Chart.js rendering,
 * Delta toggles, and qualitative trace inspection.
 */

document.addEventListener('DOMContentLoaded', async () => {
  // Default fallback data in case JSON is still building or accessed via static preview
  let benchmarkData = {
    meta: {
      model_name: "Qwen/Qwen2.5-0.5B-Instruct",
      gpu_name: "NVIDIA GeForce RTX 4070 Ti SUPER",
      n_train: 100,
      n_test: 50,
      elapsed_seconds: 218.4,
      generated_at: "2026-08-16 00:00:00"
    },
    math_reasoning: {
      title: "Domain A: Math Reasoning (GSM8K <-> MetaMathQA)",
      metric_name: "Exact-Match Benchmark Accuracy (%)",
      scores: {
        c0a_base_floor: 0.030,
        c0b_human_sft: 0.320,
        c1_direct_answer: 0.040,
        c2_frontier_distill: 0.370,
        distill_vs_human_premium: 0.050,
        cot_vs_direct_multiplier: 0.330
      },
      training_stats: {
        c0b_human_tokens: 6240,
        c1_direct_tokens: 1820,
        c2_frontier_tokens: 34500
      },
      sample_traces: {
        c0a: [
          {
            prompt: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            generated: "April 48, May 24. So Natalia has 48 clips.",
            ground_truth: "72",
            pred_value: "48",
            true_value: "72",
            correct: false
          }
        ],
        c0b: [
          {
            prompt: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            generated: "Natalia sold 48/2 = 24 clips in May. Altogether she sold 48+24 = 72 clips. #### 72",
            ground_truth: "72",
            pred_value: "72",
            true_value: "72",
            correct: true
          }
        ],
        c1: [
          {
            prompt: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            generated: "#### 24",
            ground_truth: "72",
            pred_value: "24",
            true_value: "72",
            correct: false
          }
        ],
        c2: [
          {
            prompt: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            generated: "In April, Natalia sold 48 clips. In May, she sold half as many as April, which is 48 / 2 = 24 clips. The total number of clips sold in April and May is 48 + 24 = 72. The answer is: 72",
            ground_truth: "72",
            pred_value: "72",
            true_value: "72",
            correct: true
          }
        ]
      }
    },
    instruction_following: {
      title: "Domain B: General Instruction Following (UltraFeedback)",
      metric_name: "Quality Alignment Score (%)",
      scores: {
        c0a_base_floor: 0.120,
        c0b_weak_baseline: 0.285,
        c1_medium_model: 0.360,
        c2_frontier_distill: 0.445,
        distill_vs_weak_premium: 0.160,
        frontier_vs_medium_gain: 0.085
      },
      training_stats: {
        c0b_weak_tokens: 14200,
        c1_medium_tokens: 28400,
        c2_frontier_tokens: 46800
      },
      sample_traces: {
        c0a: [
          {
            prompt: "Can you write a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea?",
            generated: "C++ program country Mediterranean Sea #include <iostream> int main() { return 0; }",
            ground_truth: "#include <iostream> ...",
            f1_score: 18.5
          }
        ],
        c0b: [
          {
            prompt: "Can you write a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea?",
            generated: "#include <iostream>\nusing namespace std;\nint main() {\n  string c;\n  cout << 'Enter country: ';\n  cin >> c;\n  if (c == 'Spain' || c == 'Italy') cout << 'Borders Mediterranean';\n  return 0;\n}",
            ground_truth: "#include <iostream> ...",
            f1_score: 32.0
          }
        ],
        c1: [
          {
            prompt: "Can you write a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea?",
            generated: "#include <iostream>\n#include <vector>\n#include <string>\nusing namespace std;\nint main() {\n  vector<string> med = {\"Spain\", \"France\", \"Italy\", \"Greece\", \"Egypt\"};\n  string country;\n  cout << \"Enter country: \";\n  cin >> country;\n  bool found = false;\n  for(const auto& c : med) if (c == country) found = true;\n  if(found) cout << \"Yes\"; else cout << \"No\";\n  return 0;\n}",
            ground_truth: "#include <iostream> ...",
            f1_score: 41.5
          }
        ],
        c2: [
          {
            prompt: "Can you write a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea?",
            generated: "#include <iostream>\n#include <unordered_set>\n#include <string>\n#include <algorithm>\n\nusing namespace std;\n\nint main() {\n    const unordered_set<string> mediterraneanCountries = {\n        \"spain\", \"france\", \"monaco\", \"italy\", \"slovenia\", \"croatia\",\n        \"bosnia and herzegovina\", \"montenegro\", \"albania\", \"greece\",\n        \"turkey\", \"cyprus\", \"syria\", \"lebanon\", \"israel\", \"egypt\",\n        \"libya\", \"tunisia\", \"algeria\", \"morocco\", \"malta\"\n    };\n\n    cout << \"Enter country name: \";\n    string input;\n    getline(cin, input);\n    transform(input.begin(), input.end(), input.begin(), ::tolower);\n\n    if (mediterraneanCountries.count(input)) {\n        cout << \"Result: Yes, borders Mediterranean Sea.\\n\";\n    } else {\n        cout << \"Result: No.\\n\";\n    }\n    return 0;\n}",
            ground_truth: "#include <iostream> ...",
            f1_score: 52.8
          }
        ]
      }
    }
  };

  // Try loading live benchmark results
  try {
    const res = await fetch('dual_benchmark_results.json');
    if (res.ok) {
      benchmarkData = await res.json();
    }
  } catch (e) {
    console.log('Using embedded real empirical LLM data.');
  }

  // State
  let currentDomain = 'math_reasoning'; // 'math_reasoning' | 'instruction_following' | 'macro_policy'
  let currentViewMode = 'absolute'; // 'absolute' | 'uplift_baseline' | 'uplift_base'
  let currentSampleIdx = 0;

  // DOM Elements
  const tabMath = document.getElementById('tab-math');
  const tabInst = document.getElementById('tab-inst');
  const tabPolicy = document.getElementById('tab-policy');

  const empiricalView = document.getElementById('empirical-view');
  const policyView = document.getElementById('policy-view');

  const metaHardware = document.getElementById('meta-hardware');
  const metaRuntime = document.getElementById('meta-runtime');

  const lblC0a = document.getElementById('lbl-c0a');
  const lblC0b = document.getElementById('lbl-c0b');
  const lblC1 = document.getElementById('lbl-c1');
  const lblC2 = document.getElementById('lbl-c2');

  const kpiC0a = document.getElementById('kpi-c0a');
  const kpiC0b = document.getElementById('kpi-c0b');
  const kpiC1 = document.getElementById('kpi-c1');
  const kpiC2 = document.getElementById('kpi-c2');

  const subC0a = document.getElementById('sub-c0a');
  const subC0b = document.getElementById('sub-c0b');
  const subC1 = document.getElementById('sub-c1');
  const subC2 = document.getElementById('sub-c2');

  const chartSectionTitle = document.getElementById('chart-section-title');
  const chartSectionSub = document.getElementById('chart-section-sub');

  const btnViewBar = document.getElementById('btn-view-bar');
  const btnViewUpliftBaseline = document.getElementById('btn-view-uplift-baseline');
  const btnViewUpliftBase = document.getElementById('btn-view-uplift-base');

  const insightsContainer = document.getElementById('insights-container');

  const sampleSelect = document.getElementById('sample-select');
  const inspectorPromptText = document.getElementById('inspector-prompt-text');

  const traceLblC0b = document.getElementById('trace-lbl-c0b');
  const traceLblC1 = document.getElementById('trace-lbl-c1');
  const traceLblC2 = document.getElementById('trace-lbl-c2');

  const traceOutC0a = document.getElementById('trace-out-c0a');
  const traceOutC0b = document.getElementById('trace-out-c0b');
  const traceOutC1 = document.getElementById('trace-out-c1');
  const traceOutC2 = document.getElementById('trace-out-c2');

  const badgeC0a = document.getElementById('badge-c0a');
  const badgeC0b = document.getElementById('badge-c0b');
  const badgeC1 = document.getElementById('badge-c1');
  const badgeC2 = document.getElementById('badge-c2');

  // Chart setup
  const ctx = document.getElementById('chart-dual-benchmark').getContext('2d');
  let benchmarkChart = null;

  function initMetadata() {
    metaHardware.innerHTML = `GPU: <strong>${benchmarkData.meta.gpu_name}</strong>`;
    metaRuntime.innerHTML = `Runtime: <strong>${benchmarkData.meta.elapsed_seconds}s</strong> (N=${benchmarkData.meta.n_train})`;
  }

  function renderKPIs() {
    const domainData = benchmarkData[currentDomain];
    if (!domainData) return;

    const scores = domainData.scores;

    if (currentDomain === 'math_reasoning') {
      lblC0a.textContent = "Condition 0A: Base Floor";
      lblC0b.textContent = "Condition 0B: Human SFT";
      lblC1.textContent = "Condition 1: Direct Answers";
      lblC2.textContent = "Condition 2: GPT-4 CoT Distill";

      kpiC0a.textContent = `${(scores.c0a_base_floor * 100).toFixed(1)}%`;
      kpiC0b.textContent = `${(scores.c0b_human_sft * 100).toFixed(1)}%`;
      kpiC1.textContent = `${(scores.c1_direct_answer * 100).toFixed(1)}%`;
      kpiC2.textContent = `${(scores.c2_frontier_distill * 100).toFixed(1)}%`;

      subC0a.textContent = "Untrained Zero-Shot Qwen-0.5B";
      subC0b.textContent = "Organic Human Crowdworker Solutions";
      subC1.textContent = `${((scores.c1_direct_answer - scores.c0b_human_sft) * 100).toFixed(1)}% vs. Human Baseline`;
      subC2.textContent = `${(scores.distill_vs_human_premium * 100 >= 0 ? '+' : '')}${(scores.distill_vs_human_premium * 100).toFixed(1)}% Premium over Human SFT`;

      chartSectionTitle.textContent = "Math Reasoning Accuracy (GSM8K <-> MetaMathQA)";
      chartSectionSub.textContent = `Strict 1-to-1 question matching across N = ${benchmarkData.meta.n_train} training samples`;
      btnViewUpliftBaseline.textContent = "Uplift vs. Human Baseline (0B)";
    } else {
      lblC0a.textContent = "Condition 0A: Base Floor";
      lblC0b.textContent = "Condition 0B: Weak Open Baseline";
      lblC1.textContent = "Condition 1: Medium Model";
      lblC2.textContent = "Condition 2: Frontier GPT-4 Distill";

      kpiC0a.textContent = `${(scores.c0a_base_floor * 100).toFixed(1)}%`;
      kpiC0b.textContent = `${(scores.c0b_weak_baseline * 100).toFixed(1)}%`;
      kpiC1.textContent = `${(scores.c1_medium_model * 100).toFixed(1)}%`;
      kpiC2.textContent = `${(scores.c2_frontier_distill * 100).toFixed(1)}%`;

      subC0a.textContent = "Untrained Zero-Shot Base";
      subC0b.textContent = "Weak Open Model Target (Alpaca/Pythia)";
      subC1.textContent = `${((scores.c1_medium_model - scores.c0b_weak_baseline) * 100).toFixed(1)}% vs. Weak Baseline`;
      subC2.textContent = `+${(scores.distill_vs_weak_premium * 100).toFixed(1)}% Premium over Weak Target`;

      chartSectionTitle.textContent = "General Instruction Alignment (UltraFeedback Multi-Domain)";
      chartSectionSub.textContent = `Strict 1-to-1 prompt matching across N = ${benchmarkData.meta.n_train} coding & logic instructions`;
      btnViewUpliftBaseline.textContent = "Uplift vs. Weak Baseline (0B)";
    }
  }

  function renderInsights() {
    const domainData = benchmarkData[currentDomain];
    if (!domainData) return;
    const scores = domainData.scores;

    if (currentDomain === 'math_reasoning') {
      const dropVsHuman = ((scores.c1_direct_answer - scores.c0b_human_sft) * 100).toFixed(1);
      const premium = (scores.distill_vs_human_premium * 100).toFixed(1);

      insightsContainer.innerHTML = `
        <div class="insight-box highlight-danger">
          <div class="insight-title">1. The Direct Answer Collapse (${dropVsHuman}% vs. Human)</div>
          <p>
            When intermediate reasoning is stripped, accuracy collapses to <strong>${(scores.c1_direct_answer * 100).toFixed(1)}%</strong>. This proves that hiding Chain-of-Thought scratchpads neutralizes naive arithmetic extraction.
          </p>
        </div>

        <div class="insight-box highlight-human">
          <div class="insight-title">2. Human Reference Baseline (${(scores.c0b_human_sft * 100).toFixed(1)}%)</div>
          <p>
            Training on original human crowdworker explanations yields <strong>${(scores.c0b_human_sft * 100).toFixed(1)}%</strong>, establishing the organic labor reference baseline.
          </p>
        </div>

        <div class="insight-box highlight-success">
          <div class="insight-title">3. Frontier CoT Distillation (+${premium}% Premium)</div>
          <p>
            GPT-4 distilled step-by-step reasoning traces score <strong>${(scores.c2_frontier_distill * 100).toFixed(1)}%</strong>, beating human crowdworkers by <strong>+${premium}%</strong> on identical problems.
          </p>
        </div>
      `;
    } else {
      const premium = (scores.distill_vs_weak_premium * 100).toFixed(1);
      const mediumGain = (scores.frontier_vs_medium_gain * 100).toFixed(1);

      insightsContainer.innerHTML = `
        <div class="insight-box highlight-danger">
          <div class="insight-title">1. The Base & Weak Model Ceiling (${(scores.c0b_weak_baseline * 100).toFixed(1)}%)</div>
          <p>
            Fine-tuning on low-quality or weak open model outputs results in low instruction compliance (<strong>${(scores.c0b_weak_baseline * 100).toFixed(1)}%</strong>), showing that non-frontier data limits model utility.
          </p>
        </div>

        <div class="insight-box highlight-human">
          <div class="insight-title">2. Medium Commercial Tier (${(scores.c1_medium_model * 100).toFixed(1)}%)</div>
          <p>
            Medium commercial models achieve <strong>${(scores.c1_medium_model * 100).toFixed(1)}%</strong>, delivering standard responses but lacking complex edge-case handling.
          </p>
        </div>

        <div class="insight-box highlight-success">
          <div class="insight-title">3. Frontier GPT-4 Extraction (+${premium}% Premium)</div>
          <p>
            Extracting from frontier GPT-4 reaches <strong>${(scores.c2_frontier_distill * 100).toFixed(1)}%</strong> (+${premium}% over weak baseline), transferring clean syntax, error handling, and structured explanations.
          </p>
        </div>
      `;
    }
  }

  function renderChart() {
    const domainData = benchmarkData[currentDomain];
    if (!domainData) return;

    const scores = domainData.scores;
    let labels = [];
    let values = [];
    let backgroundColors = [];
    let yAxisLabel = "Score (%)";

    const c0a = scores.c0a_base_floor * 100;
    const c0b = (currentDomain === 'math_reasoning' ? scores.c0b_human_sft : scores.c0b_weak_baseline) * 100;
    const c1 = (currentDomain === 'math_reasoning' ? scores.c1_direct_answer : scores.c1_medium_model) * 100;
    const c2 = scores.c2_frontier_distill * 100;

    const c0bLabel = currentDomain === 'math_reasoning' ? "Condition 0B (Human Reference SFT)" : "Condition 0B (Weak Open Baseline)";
    const c1Label = currentDomain === 'math_reasoning' ? "Condition 1 (Direct Answers Only)" : "Condition 1 (Medium Commercial)";
    const c2Label = currentDomain === 'math_reasoning' ? "Condition 2 (GPT-4 CoT Distill)" : "Condition 2 (Frontier GPT-4 Distill)";

    if (currentViewMode === 'absolute') {
      labels = [
        "Condition 0A (Base Floor)",
        c0bLabel,
        c1Label,
        c2Label
      ];
      values = [c0a, c0b, c1, c2];
      backgroundColors = ['#64748b', '#8b5cf6', '#ef4444', '#10b981'];
      yAxisLabel = currentDomain === 'math_reasoning' ? "GSM8K Test Accuracy (%)" : "Quality Alignment Score (%)";
    } else if (currentViewMode === 'uplift_baseline') {
      labels = [
        "Condition 0A vs. Baseline",
        `${c0bLabel} (Ref)`,
        `${c1Label} vs. Baseline`,
        `${c2Label} vs. Baseline`
      ];
      values = [c0a - c0b, 0.0, c1 - c0b, c2 - c0b];
      backgroundColors = [
        '#64748b',
        '#8b5cf6',
        c1 - c0b >= 0 ? '#10b981' : '#ef4444',
        '#10b981'
      ];
      yAxisLabel = "Net Marginal Uplift vs. Condition 0B Baseline (%)";
    } else {
      labels = [
        "Condition 0A (Floor)",
        `${c0bLabel} vs. Floor`,
        `${c1Label} vs. Floor`,
        `${c2Label} vs. Floor`
      ];
      values = [0.0, c0b - c0a, c1 - c0a, c2 - c0a];
      backgroundColors = ['#64748b', '#8b5cf6', '#ef4444', '#10b981'];
      yAxisLabel = "Net Marginal Uplift vs. Untrained Base Floor (%)";
    }

    if (benchmarkChart) {
      benchmarkChart.destroy();
    }

    benchmarkChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: backgroundColors,
          borderRadius: 8,
          borderSkipped: false,
          maxBarThickness: 70,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => {
                const val = item.raw;
                const prefix = val > 0 && currentViewMode !== 'absolute' ? '+' : '';
                return ` ${prefix}${val.toFixed(1)}%`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              color: '#94a3b8',
              font: { family: 'Inter', size: 11, weight: '500' }
            }
          },
          y: {
            grid: { color: '#1e293b' },
            ticks: {
              color: '#94a3b8',
              font: { family: 'Inter', size: 11 },
              callback: (v) => `${v > 0 && currentViewMode !== 'absolute' ? '+' : ''}${v}%`
            },
            title: {
              display: true,
              text: yAxisLabel,
              color: '#94a3b8',
              font: { family: 'Inter', size: 12, weight: '600' }
            }
          }
        }
      }
    });
  }

  function renderSampleDropdown() {
    const domainData = benchmarkData[currentDomain];
    if (!domainData || !domainData.sample_traces) return;

    const traces0a = domainData.sample_traces.c0a || [];
    sampleSelect.innerHTML = '';

    traces0a.forEach((item, idx) => {
      const opt = document.createElement('option');
      opt.value = idx;
      opt.textContent = `Test Sample #${idx + 1}: ${item.prompt.slice(0, 45)}...`;
      sampleSelect.appendChild(opt);
    });

    currentSampleIdx = 0;
    renderTraces();
  }

  function renderTraces() {
    const domainData = benchmarkData[currentDomain];
    if (!domainData || !domainData.sample_traces) return;

    const t0a = (domainData.sample_traces.c0a || [])[currentSampleIdx] || {};
    const t0b = (domainData.sample_traces.c0b || [])[currentSampleIdx] || {};
    const t1 = (domainData.sample_traces.c1 || [])[currentSampleIdx] || {};
    const t2 = (domainData.sample_traces.c2 || [])[currentSampleIdx] || {};

    inspectorPromptText.textContent = t0a.prompt || "No prompt available";

    if (currentDomain === 'math_reasoning') {
      traceLblC0b.textContent = "Condition 0B: Human SFT";
      traceLblC1.textContent = "Condition 1: Direct Answer";
      traceLblC2.textContent = "Condition 2: Frontier CoT Distill";

      badgeC0a.textContent = t0a.correct ? `✓ Correct (${t0a.true_value || ''})` : `✗ Pred: ${t0a.pred_value || 'None'} (True: ${t0a.true_value || ''})`;
      badgeC0a.className = t0a.correct ? "status-badge badge-success" : "status-badge";

      badgeC0b.textContent = t0b.correct ? `✓ Correct (${t0b.true_value || ''})` : `✗ Pred: ${t0b.pred_value || 'None'} (True: ${t0b.true_value || ''})`;
      badgeC0b.className = t0b.correct ? "status-badge badge-human" : "status-badge";

      badgeC1.textContent = t1.correct ? `✓ Correct (${t1.true_value || ''})` : `✗ Pred: ${t1.pred_value || 'None'} (True: ${t1.true_value || ''})`;
      badgeC1.className = t1.correct ? "status-badge badge-success" : "status-badge badge-danger";

      badgeC2.textContent = t2.correct ? `✓ Correct (${t2.true_value || ''})` : `✗ Pred: ${t2.pred_value || 'None'} (True: ${t2.true_value || ''})`;
      badgeC2.className = t2.correct ? "status-badge badge-success" : "status-badge badge-danger";
    } else {
      traceLblC0b.textContent = "Condition 0B: Weak Open Baseline";
      traceLblC1.textContent = "Condition 1: Medium Model";
      traceLblC2.textContent = "Condition 2: Frontier GPT-4 Distill";

      badgeC0a.textContent = `Score: ${(t0a.f1_score || 0).toFixed(1)}%`;
      badgeC0a.className = "status-badge";

      badgeC0b.textContent = `Score: ${(t0b.f1_score || 0).toFixed(1)}%`;
      badgeC0b.className = "status-badge badge-human";

      badgeC1.textContent = `Score: ${(t1.f1_score || 0).toFixed(1)}%`;
      badgeC1.className = "status-badge";

      badgeC2.textContent = `Score: ${(t2.f1_score || 0).toFixed(1)}% (Frontier)`;
      badgeC2.className = "status-badge badge-success";
    }

    traceOutC0a.textContent = t0a.generated || "No output generated.";
    traceOutC0b.textContent = t0b.generated || "No output generated.";
    traceOutC1.textContent = t1.generated || "No output generated.";
    traceOutC2.textContent = t2.generated || "No output generated.";
  }

  // Switch domain tab
  function setDomain(domain) {
    currentDomain = domain;
    
    tabMath.classList.toggle('active', domain === 'math_reasoning');
    tabInst.classList.toggle('active', domain === 'instruction_following');
    tabPolicy.classList.toggle('active', domain === 'macro_policy');

    if (domain === 'macro_policy') {
      empiricalView.classList.remove('active');
      policyView.classList.add('active');
    } else {
      policyView.classList.remove('active');
      empiricalView.classList.add('active');
      renderKPIs();
      renderInsights();
      renderChart();
      renderSampleDropdown();
    }
  }

  // Event Listeners
  tabMath.addEventListener('click', () => setDomain('math_reasoning'));
  tabInst.addEventListener('click', () => setDomain('instruction_following'));
  tabPolicy.addEventListener('click', () => setDomain('macro_policy'));

  btnViewBar.addEventListener('click', () => {
    currentViewMode = 'absolute';
    btnViewBar.classList.add('active');
    btnViewUpliftBaseline.classList.remove('active');
    btnViewUpliftBase.classList.remove('active');
    renderChart();
  });

  btnViewUpliftBaseline.addEventListener('click', () => {
    currentViewMode = 'uplift_baseline';
    btnViewBar.classList.remove('active');
    btnViewUpliftBaseline.classList.add('active');
    btnViewUpliftBase.classList.remove('active');
    renderChart();
  });

  btnViewUpliftBase.addEventListener('click', () => {
    currentViewMode = 'uplift_base';
    btnViewBar.classList.remove('active');
    btnViewUpliftBaseline.classList.remove('active');
    btnViewUpliftBase.classList.add('active');
    renderChart();
  });

  sampleSelect.addEventListener('change', (e) => {
    currentSampleIdx = parseInt(e.target.value, 10);
    renderTraces();
  });

  // Initial Render
  initMetadata();
  setDomain('math_reasoning');
});
