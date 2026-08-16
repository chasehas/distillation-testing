"""
Plotting and Visualization Generator
====================================
Produces publication-quality Matplotlib figures for the distillation frontier
and economic asymmetry analysis.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .simulator import SimulationResult
from .economics import EconomicResult


def generate_frontier_plot(sim_res: SimulationResult, output_path: str = "distillation_frontier.png"):
    """Generate capability vs budget frontier plot."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=140)

    # Style dictionary
    style_map = {
        ("argmax", "random"): ("#e05656", "--", "o", "Argmax API (Random Queries)"),
        ("logprob", "random"): ("#4a90e2", "-.", "s", "Logprob API (Random Queries)"),
        ("logprob", "active"): ("#27ae60", "-", "^", "Logprob API (Active Entropy Elicitation)"),
        ("cot_reasoning", "active"): ("#8e44ad", "-", "D", "Reasoning CoT (Active Elicitation)"),
    }

    for cond, (color, ls, marker, label) in style_map.items():
        if cond in sim_res.curves:
            y_vals = sim_res.curves[cond]["on"]
            ax.plot(
                sim_res.budgets,
                y_vals,
                linestyle=ls,
                color=color,
                marker=marker,
                linewidth=2.2,
                markersize=6.5,
                label=label,
            )

    # Teacher capability ceiling
    ax.axhline(
        sim_res.teacher_on_acc,
        color="#2c3e50",
        linestyle="-",
        linewidth=1.8,
        label=f"Frontier Teacher Capability ({sim_res.teacher_on_acc:.1%})",
    )

    # No-access counterfactual baseline
    ax.axhline(
        sim_res.baseline_acc,
        color="#7f8c8d",
        linestyle=":",
        linewidth=1.8,
        label=f"Public Data Baseline ({sim_res.baseline_acc:.1%})",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Adversary Query Budget (Q, log scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Recovered Capability (On-Dist Accuracy)", fontsize=11, fontweight="bold")
    ax.set_title("Distillation Capability Frontier: Marginal Extraction vs. Query Budget", fontsize=12, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def generate_economics_plot(econ_res: EconomicResult, output_path: str = "economic_asymmetry.png"):
    """Generate cost breakdown and breakeven economics figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=140)

    # 1. Cost Comparison (Log scale bar chart)
    categories = ["Frontier Lab\nTotal R&D", "Adversary\nDistillation Cost"]
    costs = [econ_res.teacher_cost.total_rnd_cost, econ_res.distiller_cost.total_distiller_cost]
    colors = ["#2c3e50", "#e74c3c"]

    bars = ax1.bar(categories, costs, color=colors, width=0.45)
    ax1.set_yscale("log")
    ax1.set_ylabel("Cost in USD (log scale)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Capital Asymmetry: {econ_res.asymmetry_leverage:,.0f}x Leverage Ratio", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Annotate bar values
    for bar, cost in zip(bars, costs):
        y = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            y * 1.3,
            f"${cost:,.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # 2. Cumulative Serving Cost & Breakeven Curve
    query_volumes = np.linspace(0, econ_res.breakeven_queries * 2.5, 200)
    cost_api_serving = query_volumes * econ_res.cost_per_teacher_request_usd
    cost_distilled_serving = (
        econ_res.distiller_cost.total_distiller_cost
        + query_volumes * econ_res.cost_per_student_request_usd
    )

    ax2.plot(query_volumes / 1e6, cost_api_serving, color="#e74c3c", lw=2.2, label="Pay Commercial API Ongoing")
    ax2.plot(query_volumes / 1e6, cost_distilled_serving, color="#27ae60", lw=2.2, label="Distill & Self-Host Student")

    # Breakeven point
    be_x = econ_res.breakeven_queries / 1e6
    be_y = econ_res.breakeven_queries * econ_res.cost_per_teacher_request_usd
    ax2.plot(be_x, be_y, marker="o", markersize=8, color="#f39c12")
    ax2.annotate(
        f"Breakeven:\n{be_x:.2f}M Queries (${be_y:,.0f})",
        xy=(be_x, be_y),
        xytext=(be_x * 1.15, be_y * 0.75),
        arrowprops=dict(facecolor="#333", shrink=0.08, width=1, headwidth=6),
        fontweight="bold",
        fontsize=9,
    )

    ax2.set_xlabel("Inference Queries Served (Millions)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Total Cumulative Cost ($ USD)", fontsize=11, fontweight="bold")
    ax2.set_title("Inference Arbitrage: Self-Hosting Breakeven Trajectory", fontsize=12, fontweight="bold")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y/1e3:,.0f}k" if y < 1e6 else f"${y/1e6:,.1f}M"))
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=9.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
