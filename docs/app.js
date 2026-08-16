/**
 * Empirical AI Distillation Dashboard Controller
 * Tells a linear narrative across 4 sections:
 *  1. The Distillation Premium (All 5 domains grouped bar chart & KPIs)
 *  2. When Does It Work? (Scaling with Data line chart + Scaling with Model Size bar chart)
 *  3. Why It Works (Qualitative Trace Inspector defaulting to Math Sample #2)
 *  4. Policy Implications
 */

document.addEventListener('DOMContentLoaded', async () => {
  // Global Data Store
  let data05b = null;
  let data15b = null;
  let dataScaling = null;

  // Chart instances
  let chartPremium = null;
  let chartScalingData = null;
  let chartScalingSize = null;

  // DOM Elements
  const sampleSelect = document.getElementById('sample-select');
  const inspectorPromptText = document.getElementById('inspector-prompt-text');

  const traceOutC0a = document.getElementById('trace-out-c0a');
  const traceOutC0b = document.getElementById('trace-out-c0b');
  const traceOutC1 = document.getElementById('trace-out-c1');
  const traceOutC2 = document.getElementById('trace-out-c2');

  const badgeC0a = document.getElementById('badge-c0a');
  const badgeC0b = document.getElementById('badge-c0b');
  const badgeC1 = document.getElementById('badge-c1');
  const badgeC2 = document.getElementById('badge-c2');

  const scalingPlaceholder = document.getElementById('scaling-placeholder');

  // Fetch helpers
  async function fetchJSON(url) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      console.warn(`Could not load ${url}:`, e);
    }
    return null;
  }

  // Initial Data Load
  data05b = await fetchJSON('benchmark_results_0_5b.json');
  data15b = await fetchJSON('benchmark_results_1_5b.json');
  dataScaling = await fetchJSON('math_scaling_summary.json');

  // Fallback defaults if static preview without server
  if (!data05b) {
    data05b = {
      math_reasoning: { scores: { c0a_base_floor: 0.03, c0b_human_sft: 0.23, c1_direct_answer: 0.00, c2_frontier_distill: 0.40 } },
      instruction_following: { scores: { c0a_base_floor: 0.289, c0b_weak_baseline: 0.313, c1_medium_model: 0.321, c2_frontier_distill: 0.325 } },
      code_execution: { scores: { c0a_base_floor: 0.04, c0b_weak_baseline: 0.05, c1_medium_model: 0.02, c2_frontier_distill: 0.02 } },
      json_extraction: { scores: { c0a_base_floor: 0.22, c0b_weak_baseline: 0.26, c1_medium_model: 0.30, c2_frontier_distill: 0.30 } },
      mcq_reasoning: { scores: { c0a_base_floor: 0.42, c0b_human_verbose: 0.46, c1_direct_answer: 0.48, c2_frontier_distill: 0.48 } }
    };
  }

  if (!data15b) {
    data15b = {
      math_reasoning: { scores: { c0a_base_floor: 0.62, c0b_human_sft: 0.54, c1_direct_answer: 0.20, c2_frontier_distill: 0.58 } },
      instruction_following: { scores: { c0a_base_floor: 0.295, c0b_weak_baseline: 0.303, c1_medium_model: 0.342, c2_frontier_distill: 0.353 } },
      code_execution: { scores: { c0a_base_floor: 0.62, c0b_weak_baseline: 0.60, c1_medium_model: 0.64, c2_frontier_distill: 0.62 } },
      json_extraction: { scores: { c0a_base_floor: 0.26, c0b_weak_baseline: 0.36, c1_medium_model: 0.48, c2_frontier_distill: 0.38 } },
      mcq_reasoning: { scores: { c0a_base_floor: 0.82, c0b_human_verbose: 0.78, c1_direct_answer: 0.78, c2_frontier_distill: 0.78 } }
    };
  }

  // =========================================================================
  // SECTION 1: THE DISTILLATION PREMIUM CHART
  // =========================================================================
  function renderSection1() {
    const ctx1 = document.getElementById('chart-premium-overview').getContext('2d');

    const domainLabels = [
      '📐 Math Reasoning',
      '💻 Instruct Following',
      '🐍 Python Code',
      '📋 Structured JSON',
      '🎯 Science MCQ'
    ];

    // Compute premiums (c2 - c0b in percentage points)
    function getPremium(data, domainKey) {
      if (!data || !data[domainKey] || !data[domainKey].scores) return 0.0;
      const s = data[domainKey].scores;
      const c2 = s.c2_frontier_distill || 0.0;
      const c0b = s.c0b_human_sft ?? s.c0b_weak_baseline ?? s.c0b_human_verbose ?? 0.0;
      return (c2 - c0b) * 100;
    }

    const premiums05b = [
      getPremium(data05b, 'math_reasoning'),
      getPremium(data05b, 'instruction_following'),
      getPremium(data05b, 'code_execution'),
      getPremium(data05b, 'json_extraction'),
      getPremium(data05b, 'mcq_reasoning')
    ];

    const premiums15b = [
      getPremium(data15b, 'math_reasoning'),
      getPremium(data15b, 'instruction_following'),
      getPremium(data15b, 'code_execution'),
      getPremium(data15b, 'json_extraction'),
      getPremium(data15b, 'mcq_reasoning')
    ];

    if (chartPremium) chartPremium.destroy();

    chartPremium = new Chart(ctx1, {
      type: 'bar',
      data: {
        labels: domainLabels,
        datasets: [
          {
            label: '0.5B Student Model',
            data: premiums05b,
            backgroundColor: '#38bdf8',
            borderRadius: 6,
            maxBarThickness: 38
          },
          {
            label: '1.5B Student Model',
            data: premiums15b,
            backgroundColor: '#818cf8',
            borderRadius: 6,
            maxBarThickness: 38
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 12, weight: '600' } }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const val = ctx.raw;
                return ` ${ctx.dataset.label}: ${val >= 0 ? '+' : ''}${val.toFixed(1)} pp`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#f8fafc', font: { family: 'Inter', size: 12, weight: '600' } }
          },
          y: {
            grid: { color: '#1e293b' },
            ticks: {
              color: '#94a3b8',
              font: { family: 'Inter', size: 11 },
              callback: (v) => `${v > 0 ? '+' : ''}${v} pp`
            },
            title: {
              display: true,
              text: 'Distillation Premium (Percentage Points Uplift vs. Condition 0B)',
              color: '#94a3b8',
              font: { family: 'Inter', size: 12, weight: '600' }
            }
          }
        }
      }
    });

    // Update KPI Cards
    const kpiBestVal = document.getElementById('kpi-best-val');
    const kpiWorstVal = document.getElementById('kpi-worst-val');
    const kpiCotVal = document.getElementById('kpi-cot-val');

    kpiBestVal.textContent = `+${Math.max(...premiums05b).toFixed(1)} pp`;
    kpiWorstVal.textContent = `${Math.min(...premiums05b).toFixed(1)} pp`;
    
    const mathC1Score = (data05b.math_reasoning?.scores?.c1_direct_answer ?? 0.0) * 100;
    kpiCotVal.textContent = `${mathC1Score.toFixed(1)}%`;
  }

  // =========================================================================
  // SECTION 2: WHEN DOES IT WORK? (DATA SCALING & MODEL SIZE SCALING)
  // =========================================================================
  function renderScalingDataChart() {
    const ctxData = document.getElementById('chart-scaling-data').getContext('2d');

    if (!dataScaling || !dataScaling.scaling_curve || dataScaling.scaling_curve.length === 0) {
      if (scalingPlaceholder) scalingPlaceholder.style.display = 'flex';
      return;
    }

    if (scalingPlaceholder) scalingPlaceholder.style.display = 'none';

    const curve = dataScaling.scaling_curve;
    const xLabels = curve.map(pt => `N=${pt.n_train}`);
    const c0aVals = curve.map(pt => (pt.c0a * 100));
    const c0bVals = curve.map(pt => (pt.c0b * 100));
    const c1Vals = curve.map(pt => (pt.c1 * 100));
    const c2Vals = curve.map(pt => (pt.c2 * 100));

    if (chartScalingData) chartScalingData.destroy();

    chartScalingData = new Chart(ctxData, {
      type: 'line',
      data: {
        labels: xLabels,
        datasets: [
          {
            label: 'C2: Frontier GPT-4 CoT Distill',
            data: c2Vals,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.15)',
            borderWidth: 3,
            tension: 0.25,
            pointRadius: 5,
            pointBackgroundColor: '#10b981',
            fill: '+1' // Fill down to C0B
          },
          {
            label: 'C0B: Human SFT Baseline',
            data: c0bVals,
            borderColor: '#8b5cf6',
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            tension: 0.25,
            pointRadius: 4,
            pointBackgroundColor: '#8b5cf6'
          },
          {
            label: 'C1: Direct Answers Only (Stripped)',
            data: c1Vals,
            borderColor: '#ef4444',
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.25,
            pointRadius: 3,
            pointBackgroundColor: '#ef4444'
          },
          {
            label: 'C0A: Untrained Base Floor',
            data: c0aVals,
            borderColor: '#64748b',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            borderDash: [3, 3],
            tension: 0,
            pointRadius: 2,
            pointBackgroundColor: '#64748b'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 10, weight: '600' }, boxWidth: 12 }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(1)}%`
            }
          }
        },
        scales: {
          x: {
            grid: { color: '#1e293b' },
            ticks: { color: '#f8fafc', font: { family: 'Inter', size: 11, weight: '600' } }
          },
          y: {
            grid: { color: '#1e293b' },
            ticks: {
              color: '#94a3b8',
              font: { family: 'Inter', size: 11 },
              callback: (v) => `${v}%`
            },
            title: {
              display: true,
              text: 'GSM8K Accuracy (%)',
              color: '#94a3b8',
              font: { family: 'Inter', size: 11, weight: '600' }
            }
          }
        }
      }
    });
  }

  function renderScalingSizeChart() {
    const ctxSize = document.getElementById('chart-scaling-size').getContext('2d');

    const m05 = data05b?.math_reasoning?.scores || { c0a_base_floor: 0.03, c0b_human_sft: 0.23, c2_frontier_distill: 0.40 };
    const m15 = data15b?.math_reasoning?.scores || { c0a_base_floor: 0.62, c0b_human_sft: 0.54, c2_frontier_distill: 0.58 };

    const labels = ['0.5B Student Model', '1.5B Student Model'];

    const baseVals = [m05.c0a_base_floor * 100, m15.c0a_base_floor * 100];
    const humanVals = [m05.c0b_human_sft * 100, m15.c0b_human_sft * 100];
    const distillVals = [m05.c2_frontier_distill * 100, m15.c2_frontier_distill * 100];

    if (chartScalingSize) chartScalingSize.destroy();

    chartScalingSize = new Chart(ctxSize, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Condition 0A: Base Floor',
            data: baseVals,
            backgroundColor: '#64748b',
            borderRadius: 6,
            maxBarThickness: 32
          },
          {
            label: 'Condition 0B: Human SFT',
            data: humanVals,
            backgroundColor: '#8b5cf6',
            borderRadius: 6,
            maxBarThickness: 32
          },
          {
            label: 'Condition 2: Frontier Distill',
            data: distillVals,
            backgroundColor: '#10b981',
            borderRadius: 6,
            maxBarThickness: 32
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 10, weight: '600' }, boxWidth: 12 }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(1)}%`
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#f8fafc', font: { family: 'Inter', size: 12, weight: '600' } }
          },
          y: {
            grid: { color: '#1e293b' },
            ticks: {
              color: '#94a3b8',
              font: { family: 'Inter', size: 11 },
              callback: (v) => `${v}%`
            },
            title: {
              display: true,
              text: 'GSM8K Accuracy (%)',
              color: '#94a3b8',
              font: { family: 'Inter', size: 11, weight: '600' }
            }
          }
        }
      }
    });
  }

  // =========================================================================
  // SECTION 3: QUALITATIVE TRACE INSPECTOR (DEFAULT SAMPLE #2 ROBE PROBLEM)
  // =========================================================================
  function renderTraceInspector(sampleIndex = 1) {
    const traces = data15b?.math_reasoning?.sample_traces || data05b?.math_reasoning?.sample_traces;
    if (!traces) return;

    const t0a = (traces.c0a || [])[sampleIndex] || {};
    const t0b = (traces.c0b || [])[sampleIndex] || {};
    const t1 = (traces.c1 || [])[sampleIndex] || {};
    const t2 = (traces.c2 || [])[sampleIndex] || {};

    inspectorPromptText.textContent = t0a.prompt || "No prompt available";

    traceOutC0a.textContent = t0a.generated || "--";
    traceOutC0b.textContent = t0b.generated || "--";
    traceOutC1.textContent = t1.generated || "--";
    traceOutC2.textContent = t2.generated || "--";

    badgeC0a.textContent = t0a.correct ? `✓ Correct (${t0a.true_value || ''})` : `✗ Pred: ${t0a.pred_value || 'None'}`;
    badgeC0a.className = t0a.correct ? "status-badge badge-success" : "status-badge";

    badgeC0b.textContent = t0b.correct ? `✓ Correct (${t0b.true_value || ''})` : `✗ Pred: ${t0b.pred_value || 'None'}`;
    badgeC0b.className = t0b.correct ? "status-badge badge-human" : "status-badge";

    badgeC1.textContent = t1.correct ? `✓ Correct (${t1.true_value || ''})` : `✗ Pred: ${t1.pred_value || 'None'}`;
    badgeC1.className = t1.correct ? "status-badge badge-success" : "status-badge badge-danger";

    badgeC2.textContent = t2.correct ? `✓ Correct (${t2.true_value || ''})` : `✗ Pred: ${t2.pred_value || 'None'}`;
    badgeC2.className = t2.correct ? "status-badge badge-success" : "status-badge badge-danger";
  }

  // Event listener for sample select dropdown
  if (sampleSelect) {
    sampleSelect.addEventListener('change', (e) => {
      renderTraceInspector(parseInt(e.target.value, 10));
    });
  }

  // Polling for scaling summary if still running
  async function checkScalingDataPoll() {
    if (!dataScaling) {
      dataScaling = await fetchJSON('math_scaling_summary.json');
      if (dataScaling) {
        renderScalingDataChart();
      }
    }
  }

  // Render everything
  renderSection1();
  renderScalingDataChart();
  renderScalingSizeChart();
  renderTraceInspector(1); // Default to Sample #2 (Robe fiber problem)

  // Poll every 30 seconds if scaling data wasn't ready
  setInterval(checkScalingDataPoll, 30000);
});
