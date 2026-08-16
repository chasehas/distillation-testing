/**
 * Post-Training Distillation Economics Engine
 * Isolates the Reasoning & Alignment layer holding Base Pretraining Equal
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const sliderTeacherPosttrain = document.getElementById('slider-teacher-posttrain');
  const sliderQueries = document.getElementById('slider-queries');
  const radioModes = document.querySelectorAll('input[name="api-mode"]');

  const lblTeacherPosttrain = document.getElementById('lbl-teacher-posttrain');
  const lblQueries = document.getElementById('lbl-queries');

  // First-Mover Elements
  const resLabCost = document.getElementById('res-lab-cost');
  const resLabAnnot = document.getElementById('res-lab-annot');
  const resLabFailedRl = document.getElementById('res-lab-failed-rl');
  const resLabSafety = document.getElementById('res-lab-safety');

  // Competitor Elements
  const resCompCost = document.getElementById('res-comp-cost');
  const resCompApi = document.getElementById('res-comp-api');
  const resCompSft = document.getElementById('res-comp-sft');
  const resCompCap = document.getElementById('res-comp-cap');

  // Punchline
  const resLeverage = document.getElementById('res-leverage');
  const resPunchlineDesc = document.getElementById('res-punchline-desc');

  // Perspectives & Defenses
  const pBtns = document.querySelectorAll('.p-btn');
  const panelLab = document.getElementById('panel-lab-defenses');
  const panelPolicy = document.getElementById('panel-policy-insights');
  const chkMaskCot = document.getElementById('chk-mask-cot');
  const chkBlockLogprobs = document.getElementById('chk-block-logprobs');

  // Chart
  const ctx = document.getElementById('mini-chart').getContext('2d');
  let miniChart;

  function initChart() {
    miniChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: ['50k', '200k', '500k', '1.0M', '2.0M', '3.0M'],
        datasets: [
          {
            label: 'Reasoning Capability Extracted via API',
            data: [65, 82, 91, 95.2, 97.4, 98.2],
            borderColor: '#38bdf8',
            backgroundColor: 'rgba(56, 189, 248, 0.12)',
            borderWidth: 2.5,
            fill: true,
            tension: 0.3,
            pointRadius: 3,
          },
          {
            label: 'Un-Assisted Post-Training Ceiling (No Teacher)',
            data: [50, 50, 50, 50, 50, 50],
            borderColor: '#64748b',
            borderDash: [5, 5],
            borderWidth: 1.8,
            pointRadius: 0,
            fill: false,
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
            labels: { color: '#94a3b8', font: { size: 10 }, boxWidth: 12 }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}% match`
            }
          }
        },
        scales: {
          x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8', font: { size: 10 } } },
          y: {
            min: 40,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', callback: (v) => `${v}%`, font: { size: 10 } }
          }
        }
      }
    });
  }

  function formatDollar(val) {
    if (val >= 1_000_000) {
      return `$${(val / 1_000_000).toFixed(1)}M`;
    }
    return `$${Math.round(val / 1_000)}k`;
  }

  function calculate() {
    const teacherPosttrainMillions = parseFloat(sliderTeacherPosttrain.value);
    const teacherPosttrainCost = teacherPosttrainMillions * 1_000_000;
    const queries = parseInt(sliderQueries.value);

    let apiMode = 'reasoning';
    radioModes.forEach(r => { if (r.checked) apiMode = r.value; });

    const hideCot = chkMaskCot ? chkMaskCot.checked : false;
    const blockLogprobs = chkBlockLogprobs ? chkBlockLogprobs.checked : false;

    // 1. First-Mover Post-Training Breakdown
    const labAnnot = teacherPosttrainCost * 0.30;     // Expert PhD annotations ($15M on $50M)
    const labFailedRl = teacherPosttrainCost * 0.50;  // Failed RL search runs & reward model exploration ($25M)
    const labSafety = teacherPosttrainCost * 0.20;    // Safety RLHF, red-teaming, alignment ($10M)

    // 2. Competitor Post-Training Breakdown
    // API tokens: 400 in, 800 out @ $3/M in, $15/M out = $0.0132/query
    const costPerQuery = 0.0132;
    const compApiCost = Math.round(queries * costPerQuery);

    // SFT on cluster (e.g. 8x H100s for a few days)
    const compSftCost = 30_000;

    const totalCompetitorCost = compApiCost + compSftCost;

    // 3. Reasoning Capability Extraction Curve
    // Base raw model before post-training has ~50% reasoning capability
    const baseFloor = 50.0;
    const maxLift = 48.0; // up to 98%

    let efficiency = (apiMode === 'reasoning') ? 1.0 : 0.45;
    if (hideCot && apiMode === 'reasoning') efficiency *= 0.55;
    if (blockLogprobs) efficiency *= 0.85;

    const halfSatQueries = 250000 / efficiency;
    const qP = Math.pow(queries, 0.82);
    const kP = Math.pow(halfSatQueries, 0.82);
    const compCap = Math.min(98.2, baseFloor + (maxLift * qP) / (qP + kP));

    const leverage = Math.round(teacherPosttrainCost / Math.max(1, totalCompetitorCost));
    const costPct = ((totalCompetitorCost / teacherPosttrainCost) * 100).toFixed(2);

    return {
      teacherPosttrainMillions,
      teacherPosttrainCost,
      labAnnot,
      labFailedRl,
      labSafety,
      queries,
      compApiCost,
      compSftCost,
      totalCompetitorCost,
      compCap,
      leverage,
      costPct,
      efficiency,
    };
  }

  function update() {
    const res = calculate();

    // Inputs labels
    lblTeacherPosttrain.textContent = `$${res.teacherPosttrainMillions} Million`;
    lblQueries.textContent = `${(res.queries >= 1000000) ? (res.queries / 1000000).toFixed(1) + 'M' : Math.round(res.queries / 1000) + 'k'} queries`;

    // First-Mover Lab Card
    resLabCost.textContent = `$${res.teacherPosttrainMillions}.0M`;
    resLabAnnot.textContent = formatDollar(res.labAnnot);
    resLabFailedRl.textContent = formatDollar(res.labFailedRl);
    resLabSafety.textContent = formatDollar(res.labSafety);

    // Competitor Card
    resCompCost.textContent = formatDollar(res.totalCompetitorCost);
    resCompApi.textContent = `${formatDollar(res.compApiCost)} (${(res.queries >= 1000000) ? (res.queries / 1000000).toFixed(1) + 'M' : Math.round(res.queries / 1000) + 'k'} CoT)`;
    resCompSft.textContent = formatDollar(res.compSftCost);
    resCompCap.textContent = `${res.compCap.toFixed(1)}% Match`;

    // Punchline
    resLeverage.textContent = `${res.leverage}x Leverage`;
    resPunchlineDesc.innerHTML = `Competitor captured <strong>${res.compCap.toFixed(1)}%</strong> of the first-mover's reasoning capability for <strong>${formatDollar(res.totalCompetitorCost)} (${res.costPct}% of post-training R&D)</strong>.`;

    // Chart Update
    if (miniChart) {
      const qSteps = [50000, 200000, 500000, 1000000, 2000000, 3000000];

      const distillCurve = qSteps.map(q => {
        const halfSat = 250000 / res.efficiency;
        const qP = Math.pow(q, 0.82);
        const kP = Math.pow(halfSat, 0.82);
        return Math.min(98.2, 50.0 + (48.0 * qP) / (qP + kP));
      });

      miniChart.data.datasets[0].data = distillCurve;
      miniChart.update();
    }
  }

  // Event Listeners
  sliderTeacherPosttrain.addEventListener('input', update);
  sliderQueries.addEventListener('input', update);
  radioModes.forEach(r => r.addEventListener('change', update));
  if (chkMaskCot) chkMaskCot.addEventListener('change', update);
  if (chkBlockLogprobs) chkBlockLogprobs.addEventListener('change', update);

  // Perspective Buttons
  pBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      pBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const v = btn.getAttribute('data-view');
      if (v === 'lab') {
        panelLab.classList.remove('hidden');
        panelPolicy.classList.add('hidden');
      } else if (v === 'policy') {
        panelLab.classList.add('hidden');
        panelPolicy.classList.remove('hidden');
      } else {
        panelLab.classList.add('hidden');
        panelPolicy.classList.add('hidden');
      }
      update();
    });
  });

  initChart();
  update();
});
