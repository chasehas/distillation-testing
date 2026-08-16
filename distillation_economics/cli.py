"""
Command-Line Interface for Distillation Economics Suite
========================================================
Parses arguments, executes the ML simulation and economic models,
and prints structured policy and strategic reports to the console.
"""

import argparse
import json
import sys
from .simulator import DistillationSimulator, SimulationConfig
from .economics import EconomicModel, FrontierLabCostBreakdown, FrontierDefense, PolicyLevers
from .plotter import generate_frontier_plot, generate_economics_plot


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distillation Economics: Measure capability extraction, capital asymmetry, and policy risks."
    )
    parser.add_argument("--quick", action="store_true", help="Run with fewer seeds for rapid testing (~2s).")
    parser.add_argument("--teacher-rnd", type=float, default=100.0, help="Frontier Lab total R&D budget in $ Millions (default: $100M).")
    parser.add_argument("--api-input-price", type=float, default=3.0, help="API input token price per million ($ USD).")
    parser.add_argument("--api-output-price", type=float, default=15.0, help="API output token price per million ($ USD).")
    parser.add_argument("--distill-queries", type=int, default=50000, help="Total queries attacker executes against API (default: 50,000).")
    parser.add_argument("--enable-defenses", action="store_true", help="Simulate Frontier Lab enabling all defensive countermeasures.")
    parser.add_argument("--enable-policy", action="store_true", help="Simulate Policymaker enforcing API KYC and legal protections.")
    parser.add_argument("--save-plots", action="store_true", default=True, help="Save PNG plots (default: True).")
    parser.add_argument("--json-out", type=str, default="", help="Optional path to dump results JSON.")
    return parser.parse_args()


def print_banner():
    print("=" * 78)
    print(" DISTILLATION ECONOMICS & POLICY BENCHMARK SUITE")
    print(" Threat Model: Black-Box Frontier Capability Extraction via Commercial API")
    print("=" * 78)


def main():
    args = parse_args()
    print_banner()

    # Configure simulation
    seeds = [0, 1] if args.quick else [0, 1, 2, 3]
    budgets = [100, 300, 800, 2000] if args.quick else [100, 300, 800, 2000, 5000]
    config = SimulationConfig(budgets=budgets, seeds=seeds)
    simulator = DistillationSimulator(config)

    print(f"\n[1/3] Running ML Distillation Simulation (seeds={len(seeds)}, budgets={budgets})...")
    sim_res = simulator.run()

    # Configure Economics
    teacher_breakdown = FrontierLabCostBreakdown(
        pretraining_compute_usd=args.teacher_rnd * 1_000_000 * 0.65,
        posttraining_rl_usd=args.teacher_rnd * 1_000_000 * 0.20,
        data_curation_usd=args.teacher_rnd * 1_000_000 * 0.10,
        safety_alignment_usd=args.teacher_rnd * 1_000_000 * 0.05,
    )

    econ_model = EconomicModel(
        input_price_per_million=args.api_input_price,
        output_price_per_million=args.api_output_price,
        teacher_breakdown=teacher_breakdown,
    )

    defenses = econ_model.default_defenses()
    if args.enable_defenses:
        for d in defenses:
            d.enabled = True

    policy_levers = PolicyLevers(
        kyc_api_mandate=args.enable_policy,
        export_controls_api=args.enable_policy,
        watermarking_legal_framework=args.enable_policy,
        compute_threshold_auditing=args.enable_policy,
    )

    print("\n[2/3] Computing Capital Asymmetry & Economic Breakdowns...")
    econ_res = econ_model.calculate(
        query_count=args.distill_queries,
        defenses=defenses,
        policy_levers=policy_levers,
    )

    # 1. Statistical Report
    print("\n" + "-" * 78)
    print(" A. STATISTICAL CAPABILITY EXTRACTION")
    print("-" * 78)
    print(f"Frontier Teacher Capability:  {sim_res.teacher_on_acc:.1%} (On-Dist) | {sim_res.teacher_off_acc:.1%} (OOD Shift)")
    print(f"No-Access Baseline (Public):  {sim_res.baseline_acc:.1%}  <-- Counterfactual Floor")
    gap = sim_res.teacher_on_acc - sim_res.baseline_acc
    print(f"Total Proprietary Gap:       {gap:.1%}\n")

    print(f"{'Budget (Q)':>10} | {'Argmax/Rand':>13} | {'Logprob/Rand':>14} | {'Logprob/Active':>16} | {'CoT/Active':>13}")
    print("-" * 78)
    for i, Q in enumerate(sim_res.budgets):
        row = f"{Q:>10} |"
        for cond in [("argmax", "random"), ("logprob", "random"), ("logprob", "active"), ("cot_reasoning", "active")]:
            if cond in sim_res.curves:
                acc = sim_res.curves[cond]["on"][i]
                uplift = acc - sim_res.baseline_acc
                row += f" {acc:.1%} ({uplift:+.1%}) |"
        print(row)

    best_cond = ("cot_reasoning", "active") if ("cot_reasoning", "active") in sim_res.curves else ("logprob", "active")
    max_acc = sim_res.curves[best_cond]["on"][-1]
    pct_gap_recovered = (max_acc - sim_res.baseline_acc) / max(0.001, gap)
    print(f"\nMax Capability Gap Recovered (at Q={sim_res.budgets[-1]}): {pct_gap_recovered:.1%}")

    # 2. Economic Report
    print("\n" + "-" * 78)
    print(" B. CAPITAL ASYMMETRY & DISTILLATION ECONOMICS")
    print("-" * 78)
    print(f"Frontier Lab Total R&D:       ${econ_res.teacher_cost.total_rnd_cost:,.0f}")
    for name, pct in econ_res.teacher_cost.get_percentages().items():
        print(f"  * {name:<30}: {pct:5.1f}% (${pct/100 * econ_res.teacher_cost.total_rnd_cost:,.0f})")

    print(f"\nAdversary Distillation Total:  ${econ_res.distiller_cost.total_distiller_cost:,.0f} (for {args.distill_queries:,} queries)")
    for name, pct in econ_res.distiller_cost.get_percentages().items():
        print(f"  * {name:<30}: {pct:5.1f}% (${pct/100 * econ_res.distiller_cost.total_distiller_cost:,.0f})")

    print(f"\nCapital Asymmetry Ratio:       {econ_res.asymmetry_leverage:,.0f}x Advantage to Distiller")
    print(f"Inference Arbitrage Breakeven: {econ_res.breakeven_queries:,.0f} queries served locally")

    # 3. Policy & Strategy Levers
    print("\n" + "-" * 78)
    print(" C. POLICY & STRATEGIC RISK INDICES")
    print("-" * 78)
    print(f"Market Failure / Free-Rider Risk: {econ_res.market_failure_risk_score:.1f} / 100")
    print(f"Safety Alignment Stripping Risk:  {econ_res.safety_stripping_risk_score:.1f} / 100")
    print(f"Frontier Defenses Enabled:         {args.enable_defenses}")
    print(f"Policy Levers Active:              {args.enable_policy}")

    # 4. Generate Plots
    if args.save_plots:
        print("\n[3/3] Generating Visualizations...")
        p1 = generate_frontier_plot(sim_res, "distillation_frontier.png")
        p2 = generate_economics_plot(econ_res, "economic_asymmetry.png")
        print(f"  [+] Saved {p1}")
        print(f"  [+] Saved {p2}")

    # Optional JSON Dump
    if args.json_out:
        out_data = {
            "teacher_on_acc": sim_res.teacher_on_acc,
            "baseline_acc": sim_res.baseline_acc,
            "asymmetry_leverage": econ_res.asymmetry_leverage,
            "teacher_cost_usd": econ_res.teacher_cost.total_rnd_cost,
            "distiller_cost_usd": econ_res.distiller_cost.total_distiller_cost,
            "breakeven_queries": econ_res.breakeven_queries,
            "market_failure_risk": econ_res.market_failure_risk_score,
            "safety_stripping_risk": econ_res.safety_stripping_risk_score,
        }
        with open(args.json_out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"  [+] Saved JSON results to {args.json_out}")

    print("\n" + "=" * 78)
    print(" Run complete! Open 'docs/index.html' in your browser for the interactive demo.")
    print("=" * 78)


if __name__ == "__main__":
    main()
