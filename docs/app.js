/**
 * The Distillation Premium: Focused Investigative Benchmark Dashboard
 * Visualizes the quantitative impact of distillation across:
 *  - Section 1: Hero Math Distillation Premium (0.5B vs. 1.5B) + Dynamic KPIs
 *  - Section 2: Data Scaling Curve (N = 150 to 1000)
 */

document.addEventListener('DOMContentLoaded', async () => {
  let data05b = null;
  let data15b = null;
  let dataScaling = null;

  let chartHero = null;
  let chartScaling = null;

  const scalingPlaceholder = document.getElementById('scaling-placeholder');

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

  // Load JSON datasets
  data05b = await fetchJSON('benchmark_results_0_5b.json');
  data15b = await fetchJSON('benchmark_results_1_5b.json');
  dataScaling = await fetchJSON('math_scaling_summary.json');

  // Fallbacks if opening as static file without web server
  if (!data05b) {
    data05b = {
      math_reasoning: {
        scores: {
          c0a_base_floor: 0.03,
          c0b_human_sft: 0.23,
          c1_direct_answer: 0.00,
          c2_frontier_distill: 0.40
        }
      }
    };
  }

  if (!data15b) {
    data15b = {
      math_reasoning: {
        scores: {
          c0a_base_floor: 0.62,
          c0b_human_sft: 0.54,
          c1_direct_answer: 0.20,
          c2_frontier_distill: 0.58
        }
      },
      instruction_following: {
        scores: {
          c0a_base_floor: 0.295,
          c0b_weak_baseline: 0.303,
          c1_medium_model: 0.342,
          c2_frontier_distill: 0.353
        }
      },
      code_execution: {
        scores: {
          c0a_base_floor: 0.62,
          c0b_weak_baseline: 0.60,
          c1_medium_model: 0.64,
          c2_frontier_distill: 0.62
        }
      }
    };
  }

  // =========================================================================
  // SECTION 1: HERO MATH DISTILLATION PREMIUM CHART & KPIS
  // =========================================================================
  function renderSection1() {
    const ctxHero = document.getElementById('chart-hero-math').getContext('2d');

    const m05 = data05b?.math_reasoning?.scores || { c0a_base_floor: 0.03, c0b_human_sft: 0.23, c2_frontier_distill: 0.40 };
    const m15 = data15b?.math_reasoning?.scores || { c0a_base_floor: 0.62, c0b_human_sft: 0.54, c2_frontier_distill: 0.58 };

    const c0a05 = (m05.c0a_base_floor ?? 0.03) * 100;
    const c0b05 = (m05.c0b_human_sft ?? 0.23) * 100;
    const c205 = (m05.c2_frontier_distill ?? 0.40) * 100;

    const c0a15 = (m15.c0a_base_floor ?? 0.62) * 100;
    const c0b15 = (m15.c0b_human_sft ?? 0.54) * 100;
    const c215 = (m15.c2_frontier_distill ?? 0.58) * 100;

    const premium05 = c205 - c0b05;
    const premium15 = c215 - c0b15;
    const humanHurt15 = c0b15 - c0a15;

    // Update KPI Cards
    const kpiSmallVal = document.getElementById('kpi-small-val');
    const kpiSmallSub = document.getElementById('kpi-small-sub');
    const kpiLargeVal = document.getElementById('kpi-large-val');
    const kpiLargeSub = document.getElementById('kpi-large-sub');
    const kpiHurtVal = document.getElementById('kpi-hurt-val');
    const kpiHurtSub = document.getElementById('kpi-hurt-sub');

    if (kpiSmallVal) kpiSmallVal.textContent = `+${premium05.toFixed(0)} pp`;
    if (kpiSmallSub) kpiSmallSub.textContent = `0.5B model: ${c0b05.toFixed(0)}% with human data → ${c205.toFixed(0)}% with GPT-4 data`;

    if (kpiLargeVal) kpiLargeVal.textContent = `+${premium15.toFixed(0)} pp`;
    if (kpiLargeSub) kpiLargeSub.textContent = `1.5B model: ${c0b15.toFixed(0)}% with human data → ${c215.toFixed(0)}% with GPT-4 data`;

    if (kpiHurtVal) kpiHurtVal.textContent = `${humanHurt15 >= 0 ? '+' : '−'}${Math.abs(humanHurt15).toFixed(0)} pp`;
    if (kpiHurtSub) kpiHurtSub.textContent = `1.5B base scores ${c0a15.toFixed(0)}% — drops to ${c0b15.toFixed(0)}% after training on human solutions`;

    // Render Hero Grouped Bar Chart
    if (chartHero) chartHero.destroy();

    chartHero = new Chart(ctxHero, {
      type: 'bar',
      data: {
        labels: ['Qwen 0.5B Student', 'Qwen 1.5B Student'],
        datasets: [
          {
            label: 'Untrained Base',
            data: [c0a05, c0a15],
            backgroundColor: '#64748b',
            borderRadius: 6,
            maxBarThickness: 48
          },
          {
            label: 'Trained on Human Solutions',
            data: [c0b05, c0b15],
            backgroundColor: '#8b5cf6',
            borderRadius: 6,
            maxBarThickness: 48
          },
          {
            label: 'Trained on GPT-4 Solutions',
            data: [c205, c215],
            backgroundColor: '#10b981',
            borderRadius: 6,
            maxBarThickness: 48
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
            labels: {
              color: '#94a3b8',
              font: { family: 'Inter', size: 12, weight: '600' },
              padding: 16
            }
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
            ticks: {
              color: '#f8fafc',
              font: { family: 'Inter', size: 13, weight: '700' }
            }
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
              text: 'GSM8K Test Accuracy (%)',
              color: '#94a3b8',
              font: { family: 'Inter', size: 12, weight: '600' }
            },
            max: 75
          }
        }
      }
    });
  }

  // =========================================================================
  // SECTION 2: DATA SCALING CHART
  // =========================================================================
  function renderScalingChart() {
    const ctxScaling = document.getElementById('chart-scaling-data').getContext('2d');

    if (!dataScaling || !dataScaling.scaling_curve || dataScaling.scaling_curve.length === 0) {
      if (scalingPlaceholder) scalingPlaceholder.style.display = 'flex';
      return;
    }

    if (scalingPlaceholder) scalingPlaceholder.style.display = 'none';

    const curve = dataScaling.scaling_curve;
    const xLabels = curve.map(pt => pt.n_train.toString());
    const c0aVals = curve.map(pt => pt.c0a * 100);
    const c0bVals = curve.map(pt => pt.c0b * 100);
    const c1Vals = curve.map(pt => pt.c1 * 100);
    const c2Vals = curve.map(pt => pt.c2 * 100);

    if (chartScaling) chartScaling.destroy();

    chartScaling = new Chart(ctxScaling, {
      type: 'line',
      data: {
        labels: xLabels,
        datasets: [
          {
            label: 'Trained on GPT-4 Solutions',
            data: c2Vals,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.16)',
            borderWidth: 3,
            tension: 0.2,
            pointRadius: 5,
            pointBackgroundColor: '#10b981',
            fill: '+1' // Shaded gap down to Human Solutions
          },
          {
            label: 'Trained on Human Solutions',
            data: c0bVals,
            borderColor: '#8b5cf6',
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            tension: 0.2,
            pointRadius: 4,
            pointBackgroundColor: '#8b5cf6'
          },
          {
            label: 'GPT-4 Answers Only (No Reasoning)',
            data: c1Vals,
            borderColor: '#ef4444',
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.2,
            pointRadius: 3,
            pointBackgroundColor: '#ef4444'
          },
          {
            label: 'Untrained Base',
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
            labels: {
              color: '#94a3b8',
              font: { family: 'Inter', size: 11, weight: '600' },
              padding: 14,
              boxWidth: 12
            }
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
            ticks: { color: '#f8fafc', font: { family: 'Inter', size: 11, weight: '600' } },
            title: {
              display: true,
              text: 'Training Samples',
              color: '#94a3b8',
              font: { family: 'Inter', size: 12, weight: '600' }
            }
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
              font: { family: 'Inter', size: 12, weight: '600' }
            }
          }
        }
      }
    });
  }

  // Polling for scaling summary if still running
  async function checkScalingDataPoll() {
    if (!dataScaling) {
      dataScaling = await fetchJSON('math_scaling_summary.json');
      if (dataScaling) {
        renderScalingChart();
      }
    }
  }

  // Initial Render
  renderSection1();
  renderScalingChart();

  // Poll every 30 seconds
  setInterval(checkScalingDataPoll, 30000);
});
