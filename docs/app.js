/**
 * Empirical LLM Distillation Dashboard Engine
 * Dynamically loads and renders real GPU results from llm_empirical_results.json.
 */

document.addEventListener('DOMContentLoaded', async () => {
  let experimentData = {
    "meta": {
      "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
      "gpu_name": "NVIDIA GeForce RTX 4070 Ti SUPER",
      "N_train": 100,
      "N_test": 100,
      "elapsed_seconds": 218.4,
      "generated_at": "2026-08-15 23:01:40"
    },
    "results": {
      "condition_0a_base_floor": 0.030,
      "condition_0b_human_sft": 0.320,
      "condition_1_direct_answer": 0.040,
      "condition_2_gpt4_distilled": 0.370,
      "distillation_premium_over_human": 0.050,
      "reasoning_premium_over_direct": 0.330
    }
  };

  try {
    const res = await fetch('llm_empirical_results.json');
    if (res.ok) {
      experimentData = await res.json();
    }
  } catch (e) {
    console.log('Using embedded real empirical LLM data.');
  }

  // DOM Elements
  const kpiC0a = document.getElementById('kpi-c0a');
  const kpiC1 = document.getElementById('kpi-c1');
  const kpiC0b = document.getElementById('kpi-c0b');
  const kpiC2 = document.getElementById('kpi-c2');
  const metaGpu = document.getElementById('meta-gpu');

  const btnViewBar = document.getElementById('btn-view-bar');
  const btnViewUplift = document.getElementById('btn-view-uplift');

  function initKPIs() {
    metaGpu.textContent = `${experimentData.meta.gpu_name} (${experimentData.meta.elapsed_seconds}s)`;
    kpiC0a.textContent = `${(experimentData.results.condition_0a_base_floor * 100).toFixed(1)}%`;
    kpiC1.textContent = `${(experimentData.results.condition_1_direct_answer * 100).toFixed(1)}%`;
    kpiC0b.textContent = `${(experimentData.results.condition_0b_human_sft * 100).toFixed(1)}%`;
    kpiC2.textContent = `${(experimentData.results.condition_2_gpt4_distilled * 100).toFixed(1)}%`;
  }

  // Chart setup
  const ctx = document.getElementById('chart-llm-benchmark').getContext('2d');
  let chart;
  let currentMode = 'absolute'; // 'absolute' or 'uplift'

  const labels = [
    'Condition 0A: Base Floor (Zero-Shot)',
    'Condition 1: Direct Answers (No CoT)',
    'Condition 0B: Human SFT (Crowdworkers)',
    'Condition 2: GPT-4 Distillation (CoT Traces)',
  ];

  const absoluteData = [
    experimentData.results.condition_0a_base_floor * 100,
    experimentData.results.condition_1_direct_answer * 100,
    experimentData.results.condition_0b_human_sft * 100,
    experimentData.results.condition_2_gpt4_distilled * 100,
  ];

  const baseFloor = experimentData.results.condition_0a_base_floor * 100;
  const upliftData = absoluteData.map(v => v - baseFloor);

  const backgroundColors = [
    '#64748b', // Base floor
    '#ef4444', // Direct answer
    '#8b5cf6', // Human SFT
    '#10b981', // GPT-4 Distilled
  ];

  function initChart() {
    chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'GSM8K Accuracy (%)',
            data: absoluteData,
            backgroundColor: backgroundColors,
            borderRadius: 6,
            borderWidth: 1,
            borderColor: 'rgba(255, 255, 255, 0.1)',
            barThickness: 42,
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
              label: (ctx) => {
                const val = ctx.parsed.y;
                return currentMode === 'absolute' 
                  ? `Accuracy: ${val.toFixed(1)}%` 
                  : `Marginal Uplift: +${val.toFixed(1)}% over Base Floor`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11, weight: '500' } }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            beginAtZero: true,
            max: 50,
            ticks: {
              color: '#94a3b8',
              font: { size: 11 },
              callback: (v) => `${v}%`
            },
            title: {
              display: true,
              text: 'Benchmark Accuracy on 100 Test Problems (%)',
              color: '#64748b',
              font: { size: 11 }
            }
          }
        }
      }
    });
  }

  function updateView(mode) {
    currentMode = mode;
    if (mode === 'absolute') {
      btnViewBar.classList.add('active');
      btnViewUplift.classList.remove('active');
      chart.data.datasets[0].data = absoluteData;
      chart.data.datasets[0].label = 'GSM8K Accuracy (%)';
      chart.options.scales.y.title.text = 'Benchmark Accuracy on 100 Test Problems (%)';
    } else {
      btnViewUplift.classList.add('active');
      btnViewBar.classList.remove('active');
      chart.data.datasets[0].data = upliftData;
      chart.data.datasets[0].label = 'Marginal Uplift over Base Floor (+%)';
      chart.options.scales.y.title.text = 'Net Marginal Uplift (+%) over Base Model';
    }
    chart.update();
  }

  btnViewBar.addEventListener('click', () => updateView('absolute'));
  btnViewUplift.addEventListener('click', () => updateView('uplift'));

  initKPIs();
  initChart();
});
