/**
 * The Distillation Premium: Focused Investigative Benchmark Dashboard (Qwen 0.5B)
 * Visualizes the quantitative impact of distillation across:
 *  - Section 1: Hero Math Distillation Premium (Qwen2.5-0.5B) + Delta Annotation
 *  - Section 2: Data Scaling Curve (N = 150 to 1000)
 */

document.addEventListener('DOMContentLoaded', async () => {
  let data05b = null;
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

  // Load JSON datasets (0.5B only)
  data05b = await fetchJSON('benchmark_results_0_5b.json');
  dataScaling = await fetchJSON('math_scaling_summary.json');

  // Fallback defaults if opening as a static file without a web server
  if (!data05b) {
    data05b = {
      math_reasoning: {
        scores: {
          c0a_base_floor: 0.29,
          c0b_human_sft: 0.32,
          c1_direct_answer: 0.14,
          c2_frontier_distill: 0.46
        }
      }
    };
  }

  // =========================================================================
  // SECTION 1: HERO MATH DISTILLATION PREMIUM CHART (Qwen2.5-0.5B)
  // =========================================================================
  function renderSection1() {
    const ctxHero = document.getElementById('chart-hero-math').getContext('2d');

    const m05 = data05b?.math_reasoning?.scores || {
      c0a_base_floor: 0.29,
      c0b_human_sft: 0.32,
      c1_direct_answer: 0.14,
      c2_frontier_distill: 0.46
    };

    const c0a = (m05.c0a_base_floor ?? 0.29) * 100;
    const c0b = (m05.c0b_human_sft ?? 0.32) * 100;
    const c2 = (m05.c2_frontier_distill ?? 0.46) * 100;
    const premium = c2 - c0b;

    // Render Hero Bar Chart (Single Qwen2.5-0.5B Group)
    if (chartHero) chartHero.destroy();

    // Custom inline plugin to draw delta annotation above Human -> Distill bars
    const deltaPlugin = {
      id: 'deltaAnnotation',
      afterDatasetsDraw(chart) {
        const { ctx } = chart;
        const meta = chart.getDatasetMeta(0);
        if (!meta.data || meta.data.length < 3) return;

        const humanBar = meta.data[1];
        const distillBar = meta.data[2];

        const xPos = (humanBar.x + distillBar.x) / 2;
        const yTop = Math.min(humanBar.y, distillBar.y) - 26;

        ctx.save();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Draw pill background
        ctx.fillStyle = '#10b981';
        const text = `+${premium.toFixed(0)} Percentage Point Premium`;
        ctx.font = 'bold 12px Inter, system-ui';
        const textWidth = ctx.measureText(text).width;
        const padding = 10;
        const rectHeight = 22;
        const rectWidth = textWidth + padding * 2;
        const rectX = xPos - rectWidth / 2;
        const rectY = yTop - rectHeight / 2;

        ctx.beginPath();
        ctx.roundRect(rectX, rectY, rectWidth, rectHeight, 6);
        ctx.fill();

        // Draw text
        ctx.fillStyle = '#0b0f19';
        ctx.fillText(text, xPos, yTop);

        // Draw connecting bracket line
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(humanBar.x, humanBar.y - 8);
        ctx.lineTo(humanBar.x, yTop);
        ctx.lineTo(distillBar.x, yTop);
        ctx.lineTo(distillBar.x, distillBar.y - 8);
        ctx.stroke();

        ctx.restore();
      }
    };

    chartHero = new Chart(ctxHero, {
      type: 'bar',
      data: {
        labels: [
          'Untrained Base',
          'Trained on Human Solutions',
          'Trained on GPT-3.5 Solutions'
        ],
        datasets: [
          {
            data: [c0a, c0b, c2],
            backgroundColor: [
              '#64748b', // Untrained Base
              '#8b5cf6', // Human Solutions
              '#10b981'  // GPT-4 Solutions
            ],
            borderRadius: 8,
            maxBarThickness: 76
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` Accuracy: ${ctx.raw.toFixed(1)}%`
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
            max: 60,
            beginAtZero: true
          }
        }
      },
      plugins: [deltaPlugin]
    });
  }

  // =========================================================================
  // SECTION 2: DATA SCALING CHART (3 Lines: GPT-4, Human, Untrained Base)
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
    const c2Vals = curve.map(pt => pt.c2 * 100);

    if (chartScaling) chartScaling.destroy();

    chartScaling = new Chart(ctxScaling, {
      type: 'line',
      data: {
        labels: xLabels,
        datasets: [
          {
            label: 'Trained on GPT-3.5 Solutions',
            data: c2Vals,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.16)',
            borderWidth: 3,
            tension: 0.2,
            pointRadius: 5,
            pointBackgroundColor: '#10b981',
            fill: '+1' // Shaded corridor down to Human Solutions
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
            },
            beginAtZero: true,
            max: 60
          }
        }
      }
    });
  }

  // Polling for scaling summary if needed
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
