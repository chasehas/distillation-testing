/**
 * Empirical Distillation Measurement Dashboard Engine
 * Directly powers the web interface with real experimental data.
 */

document.addEventListener('DOMContentLoaded', async () => {
  // Exact empirical data generated from empirical_distillation.py
  let experimentData = {
    "meta": {
      "generated_at": "2026-08-15 22:11:10",
      "input_dim": 12,
      "num_classes": 6,
      "teacher_samples": 25000,
      "public_samples": 400,
      "budgets": [50, 150, 400, 1000, 2500, 6000],
      "seeds": [10, 20, 30, 40, 50],
      "elapsed_seconds": 367.97
    },
    "teacher": {
      "on_accuracy": 0.7960,
      "off_accuracy": 0.6553,
      "generalization_drop": 0.1407
    },
    "baseline": {
      "on_accuracy": 0.6048,
      "on_std": 0.0079,
      "off_accuracy": 0.4392,
      "proprietary_gap": 0.1912
    },
    "curves": {
      "argmax_random": {
        "name": "Argmax API / Random Query",
        "on_mean": [0.4689, 0.5604, 0.6139, 0.6319, 0.6683, 0.7079],
        "on_std": [0.0226, 0.0205, 0.0101, 0.0096, 0.0046, 0.0082],
        "off_mean": [0.3603, 0.4131, 0.4471, 0.4651, 0.4987, 0.5342],
        "off_std": [0.0182, 0.0154, 0.0112, 0.0098, 0.0075, 0.0081],
        "marginal_uplift": [-0.1359, -0.0444, 0.0091, 0.0271, 0.0635, 0.1031],
        "gap_recovered_pct": [-71.08, -23.22, 4.76, 14.17, 33.21, 53.92]
      },
      "argmax_active": {
        "name": "Argmax API / Active Uncertainty",
        "on_mean": [0.4844, 0.5796, 0.6193, 0.6468, 0.6756, 0.7067],
        "on_std": [0.0195, 0.0178, 0.0115, 0.0084, 0.0052, 0.0074],
        "off_mean": [0.3712, 0.4285, 0.4562, 0.4781, 0.4941, 0.5302],
        "off_std": [0.0165, 0.0142, 0.0105, 0.0089, 0.0068, 0.0075],
        "marginal_uplift": [-0.1204, -0.0252, 0.0145, 0.0420, 0.0708, 0.1019],
        "gap_recovered_pct": [-62.97, -13.18, 7.58, 21.97, 37.03, 53.29]
      },
      "logprob_random": {
        "name": "Logprob API / Random Query",
        "on_mean": [0.4352, 0.5256, 0.6085, 0.6421, 0.6881, 0.7083],
        "on_std": [0.0210, 0.0185, 0.0120, 0.0091, 0.0061, 0.0070],
        "off_mean": [0.3421, 0.3952, 0.4412, 0.4685, 0.4864, 0.5281],
        "off_std": [0.0190, 0.0161, 0.0118, 0.0092, 0.0071, 0.0069],
        "marginal_uplift": [-0.1696, -0.0792, 0.0037, 0.0373, 0.0833, 0.1035],
        "gap_recovered_pct": [-88.70, -41.42, 1.93, 19.51, 43.57, 54.13]
      },
      "logprob_active": {
        "name": "Logprob API / Active Uncertainty (Elicitation Ceiling)",
        "on_mean": [0.4314, 0.5392, 0.6044, 0.6412, 0.6749, 0.7169],
        "on_std": [0.0205, 0.0169, 0.0108, 0.0079, 0.0048, 0.0062],
        "off_mean": [0.3395, 0.4061, 0.4485, 0.4721, 0.4908, 0.5375],
        "off_std": [0.0185, 0.0150, 0.0102, 0.0081, 0.0060, 0.0065],
        "marginal_uplift": [-0.1734, -0.0656, -0.0004, 0.0364, 0.0701, 0.1121],
        "gap_recovered_pct": [-90.69, -34.31, -0.21, 19.04, 36.66, 58.63]
      }
    }
  };

  // Try to fetch live empirical JSON if hosted via server
  try {
    const res = await fetch('empirical_results.json');
    if (res.ok) {
      experimentData = await res.json();
    }
  } catch (e) {
    console.log('Loaded embedded empirical benchmark dataset.');
  }

  // DOM Elements
  const metaRunTime = document.getElementById('meta-run-time');
  const kpiTeacherAcc = document.getElementById('kpi-teacher-acc');
  const kpiBaselineAcc = document.getElementById('kpi-baseline-acc');
  const kpiMaxUplift = document.getElementById('kpi-max-uplift');
  const kpiDarkKnowledge = document.getElementById('kpi-dark-knowledge');

  const metricBtns = document.querySelectorAll('.tool-btn');
  const tableBody = document.getElementById('table-body');

  const toggleLogprobActive = document.getElementById('toggle-logprob-active');
  const toggleLogprobRandom = document.getElementById('toggle-logprob-random');
  const toggleArgmaxActive = document.getElementById('toggle-argmax-active');
  const toggleArgmaxRandom = document.getElementById('toggle-argmax-random');

  // Chart setup
  const ctx = document.getElementById('chart-empirical-frontier').getContext('2d');
  let chart;
  let currentMetric = 'on_mean'; // 'on_mean', 'marginal_uplift', 'gap_recovered_pct', 'off_mean'

  function initKPIs() {
    metaRunTime.textContent = experimentData.meta.generated_at || 'Measured Benchmark';
    kpiTeacherAcc.textContent = `${(experimentData.teacher.on_accuracy * 100).toFixed(1)}%`;
    kpiBaselineAcc.textContent = `${(experimentData.baseline.on_accuracy * 100).toFixed(1)}%`;

    const maxAcc = experimentData.curves.logprob_active.on_mean.slice(-1)[0];
    const maxUplift = maxAcc - experimentData.baseline.on_accuracy;
    kpiMaxUplift.textContent = `+${(maxUplift * 100).toFixed(1)}%`;

    // Maximum gap recovery
    const maxGapRec = experimentData.curves.logprob_active.gap_recovered_pct.slice(-1)[0];
    kpiDarkKnowledge.textContent = `${maxGapRec.toFixed(1)}%`;
  }

  function initTable() {
    tableBody.innerHTML = '';
    const budgets = experimentData.meta.budgets;

    budgets.forEach((q, i) => {
      const tr = document.createElement('tr');
      
      const tdQ = document.createElement('td');
      tdQ.textContent = `Q = ${Number(q).toLocaleString()}`;
      tr.appendChild(tdQ);

      const condKeys = ['argmax_random', 'argmax_active', 'logprob_random', 'logprob_active'];
      condKeys.forEach(k => {
        const td = document.createElement('td');
        const mean = (experimentData.curves[k].on_mean[i] * 100).toFixed(1);
        const std = (experimentData.curves[k].on_std[i] * 100).toFixed(1);
        const uplift = (experimentData.curves[k].marginal_uplift[i] * 100).toFixed(1);
        
        const sign = uplift >= 0 ? '+' : '';
        const upliftColor = uplift >= 0 ? '#38bdf8' : '#ef4444';
        td.innerHTML = `<strong>${mean}%</strong> <span style="color:#64748b;font-size:0.75rem;">(±${std}%)</span> <span style="color:${upliftColor};font-size:0.75rem;">[${sign}${uplift}%]</span>`;
        tr.appendChild(td);
      });

      tableBody.appendChild(tr);
    });
  }

  function initChart() {
    const budgets = experimentData.meta.budgets;
    const labels = budgets.map(b => b >= 1000 ? `${b / 1000}k` : `${b}`);

    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            id: 'logprob_active',
            label: 'Logprob API / Active (Elicitation Ceiling)',
            data: experimentData.curves.logprob_active.on_mean,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2.5,
            tension: 0.25,
            pointRadius: 4,
          },
          {
            id: 'logprob_random',
            label: 'Logprob API / Random Query',
            data: experimentData.curves.logprob_random.on_mean,
            borderColor: '#38bdf8',
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 3.5,
          },
          {
            id: 'argmax_active',
            label: 'Argmax API / Active Uncertainty',
            data: experimentData.curves.argmax_active.on_mean,
            borderColor: '#f59e0b',
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 3.5,
          },
          {
            id: 'argmax_random',
            label: 'Argmax API / Random Query',
            data: experimentData.curves.argmax_random.on_mean,
            borderColor: '#ef4444',
            borderWidth: 1.8,
            borderDash: [4, 4],
            tension: 0.25,
            pointRadius: 3,
          },
          {
            id: 'teacher_line',
            label: `Teacher Ceiling (${(experimentData.teacher.on_accuracy * 100).toFixed(1)}%)`,
            data: budgets.map(() => experimentData.teacher.on_accuracy),
            borderColor: '#94a3b8',
            borderWidth: 1.8,
            pointRadius: 0,
            fill: false,
          },
          {
            id: 'baseline_line',
            label: `Counterfactual Floor (${(experimentData.baseline.on_accuracy * 100).toFixed(1)}%)`,
            data: budgets.map(() => experimentData.baseline.on_accuracy),
            borderColor: '#64748b',
            borderDash: [6, 6],
            borderWidth: 1.6,
            pointRadius: 0,
            fill: false,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 }, boxWidth: 12 }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const val = ctx.parsed.y;
                if (currentMetric === 'gap_recovered_pct') {
                  return `${ctx.dataset.label}: ${val.toFixed(1)}% gap recovered`;
                }
                return `${ctx.dataset.label}: ${(val * 100).toFixed(1)}%`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { size: 11 } },
            title: { display: true, text: 'Query Budget Q (Log-Spaced Samples)', color: '#64748b', font: { size: 11 } }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: {
              color: '#94a3b8',
              font: { size: 11 },
              callback: (v) => currentMetric === 'gap_recovered_pct' ? `${v}%` : `${(v * 100).toFixed(0)}%`
            }
          }
        }
      }
    });
  }

  function updateChartMetric(metricKey) {
    currentMetric = metricKey;
    const budgets = experimentData.meta.budgets;

    // Update datasets
    const keys = ['logprob_active', 'logprob_random', 'argmax_active', 'argmax_random'];
    keys.forEach(k => {
      const ds = chart.data.datasets.find(d => d.id === k);
      if (ds) {
        ds.data = experimentData.curves[k][metricKey];
      }
    });

    // Reference lines
    const dsTeacher = chart.data.datasets.find(d => d.id === 'teacher_line');
    const dsBase = chart.data.datasets.find(d => d.id === 'baseline_line');

    if (metricKey === 'on_mean') {
      dsTeacher.hidden = false;
      dsBase.hidden = false;
      dsTeacher.data = budgets.map(() => experimentData.teacher.on_accuracy);
      dsBase.data = budgets.map(() => experimentData.baseline.on_accuracy);
      dsTeacher.label = `Teacher Ceiling (${(experimentData.teacher.on_accuracy * 100).toFixed(1)}%)`;
      dsBase.label = `Counterfactual Floor (${(experimentData.baseline.on_accuracy * 100).toFixed(1)}%)`;
    } else if (metricKey === 'off_mean') {
      dsTeacher.hidden = false;
      dsBase.hidden = false;
      dsTeacher.data = budgets.map(() => experimentData.teacher.off_accuracy);
      dsBase.data = budgets.map(() => experimentData.baseline.off_accuracy);
      dsTeacher.label = `Teacher OOD Shift (${(experimentData.teacher.off_accuracy * 100).toFixed(1)}%)`;
      dsBase.label = `Baseline OOD Floor (${(experimentData.baseline.off_accuracy * 100).toFixed(1)}%)`;
    } else if (metricKey === 'marginal_uplift') {
      dsTeacher.hidden = false;
      dsBase.hidden = false;
      dsTeacher.data = budgets.map(() => experimentData.baseline.proprietary_gap);
      dsBase.data = budgets.map(() => 0.0);
      dsTeacher.label = `Max Proprietary Gap (+${(experimentData.baseline.proprietary_gap * 100).toFixed(1)}%)`;
      dsBase.label = `No-Access Baseline (0.0%)`;
    } else if (metricKey === 'gap_recovered_pct') {
      dsTeacher.hidden = false;
      dsBase.hidden = false;
      dsTeacher.data = budgets.map(() => 100.0);
      dsBase.data = budgets.map(() => 0.0);
      dsTeacher.label = `100% Gap Recovered (Ceiling)`;
      dsBase.label = `0% Gap Recovered (Floor)`;
    }

    chart.update();
  }

  function updateConditionVisibility() {
    const map = {
      'logprob_active': toggleLogprobActive.checked,
      'logprob_random': toggleLogprobRandom.checked,
      'argmax_active': toggleArgmaxActive.checked,
      'argmax_random': toggleArgmaxRandom.checked,
    };

    Object.keys(map).forEach(k => {
      const ds = chart.data.datasets.find(d => d.id === k);
      if (ds) {
        ds.hidden = !map[k];
      }
    });

    chart.update();
  }

  // Event Listeners
  metricBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      metricBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const m = btn.getAttribute('data-metric');
      updateChartMetric(m);
    });
  });

  [toggleLogprobActive, toggleLogprobRandom, toggleArgmaxActive, toggleArgmaxRandom].forEach(t => {
    t.addEventListener('change', updateConditionVisibility);
  });

  // Init
  initKPIs();
  initTable();
  initChart();
});
