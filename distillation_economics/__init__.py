"""
Distillation Economics: Simulation & Policy Suite
=================================================
A mathematical and economic modeling suite investigating the efficiency,
cost asymmetry, and security implications of black-box commercial API distillation.
"""

from .simulator import DistillationSimulator, SimulationConfig, SimulationResult
from .economics import (
    EconomicModel,
    FrontierLabCostBreakdown,
    DistillationCostBreakdown,
    EconomicResult,
    FrontierDefense,
    PolicyLevers,
)
from .plotter import generate_frontier_plot, generate_economics_plot

__version__ = "1.0.0"
