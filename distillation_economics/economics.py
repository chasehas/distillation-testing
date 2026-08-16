"""
Economic Modeling Engine for AI Model Distillation
===================================================
Models the capital asymmetry, cost breakdowns, inference arbitrage breakeven,
Frontier Lab defensive features (cost/benefit), and Policymaker market failure metrics.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class FrontierLabCostBreakdown:
    pretraining_compute_usd: float = 65_000_000.0   # GPU clusters, power, cluster amortization
    posttraining_rl_usd: float = 20_000_000.0       # RLHF, RLVR, reasoning exploration
    data_curation_usd: float = 10_000_000.0         # Web scraping, synthetic datasets, human annotation
    safety_alignment_usd: float = 5_000_000.0       # Red teaming, CBRN filters, safety evals

    @property
    def total_rnd_cost(self) -> float:
        return (
            self.pretraining_compute_usd
            + self.posttraining_rl_usd
            + self.data_curation_usd
            + self.safety_alignment_usd
        )

    def get_percentages(self) -> Dict[str, float]:
        total = self.total_rnd_cost
        if total == 0:
            return {}
        return {
            "Pretraining Compute": (self.pretraining_compute_usd / total) * 100,
            "Post-Training & Reasoning RL": (self.posttraining_rl_usd / total) * 100,
            "Data Collection & Prep": (self.data_curation_usd / total) * 100,
            "Safety Alignment & Red Teaming": (self.safety_alignment_usd / total) * 100,
        }


@dataclass
class DistillationCostBreakdown:
    api_query_cost_usd: float                       # Cost of tokens from commercial API
    student_finetuning_usd: float = 8_000.0         # 8x H100 GPU compute for SFT/RL (e.g. 48h)
    data_filtering_rejection_usd: float = 4_000.0   # Automated verifiers, rejection sampling
    base_model_amortization_usd: float = 0.0        # Uses open-weights base (or small pretrain)

    @property
    def total_distiller_cost(self) -> float:
        return (
            self.api_query_cost_usd
            + self.student_finetuning_usd
            + self.data_filtering_rejection_usd
            + self.base_model_amortization_usd
        )

    def get_percentages(self) -> Dict[str, float]:
        total = self.total_distiller_cost
        if total == 0:
            return {}
        return {
            "API Query Tokens": (self.api_query_cost_usd / total) * 100,
            "Student Fine-Tuning Compute": (self.student_finetuning_usd / total) * 100,
            "Data Filtering & Rejection": (self.data_filtering_rejection_usd / total) * 100,
            "Base Model License/Compute": (self.base_model_amortization_usd / total) * 100,
        }


@dataclass
class FrontierDefense:
    name: str
    description: str
    effectiveness_reduction_pct: float  # How much it reduces adversary distillation efficiency (%)
    cost_to_lab_usd: float              # Added engineering / proxy / UX cost
    latency_impact_ms: float            # Added latency to legitimate API users
    enabled: bool = False


@dataclass
class PolicyLevers:
    kyc_api_mandate: bool = False           # Require identity verification for high-volume API accounts
    export_controls_api: bool = False       # Restrict frontier API access to allied nations
    watermarking_legal_framework: bool = False  # Statutory damages for training on watermarked outputs
    compute_threshold_auditing: bool = False    # Mandatory registration for clusters > 10^24 FLOPs


@dataclass
class EconomicResult:
    teacher_cost: FrontierLabCostBreakdown
    distiller_cost: DistillationCostBreakdown
    asymmetry_leverage: float               # Teacher R&D / Distiller Total
    breakeven_queries: int                  # Inference queries to pay off distillation
    cost_per_teacher_request_usd: float
    cost_per_student_request_usd: float
    market_failure_risk_score: float        # 0 - 100
    safety_stripping_risk_score: float      # 0 - 100
    defenses: List[FrontierDefense]
    policy_levers: PolicyLevers


class EconomicModel:
    def __init__(
        self,
        input_price_per_million: float = 3.00,    # $3.00 / 1M input tokens (e.g. Claude 3.5 Sonnet)
        output_price_per_million: float = 15.00,  # $15.00 / 1M output tokens
        avg_prompt_tokens: int = 400,
        avg_output_tokens: int = 800,
        student_serving_cost_per_million: float = 0.20,  # Self-hosted 8B model cost
        teacher_breakdown: FrontierLabCostBreakdown = None,
    ):
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million
        self.avg_prompt_tokens = avg_prompt_tokens
        self.avg_output_tokens = avg_output_tokens
        self.student_serving_cost_per_million = student_serving_cost_per_million
        self.teacher_breakdown = teacher_breakdown or FrontierLabCostBreakdown()

    def default_defenses(self) -> List[FrontierDefense]:
        return [
            FrontierDefense(
                name="Withhold Full Logprobs (Argmax Only)",
                description="Clamps API to hard labels / top-1 tokens. Blocks dark knowledge probability leak.",
                effectiveness_reduction_pct=45.0,
                cost_to_lab_usd=0.0,
                latency_impact_ms=0.0,
                enabled=False,
            ),
            FrontierDefense(
                name="Mask Reasoning Traces (Hidden CoT)",
                description="Hides raw internal chain-of-thought tokens, returning only the synthesized final answer.",
                effectiveness_reduction_pct=50.0,
                cost_to_lab_usd=50_000.0,
                latency_impact_ms=5.0,
                enabled=False,
            ),
            FrontierDefense(
                name="Entropy / Anomaly Rate-Limiting",
                description="Detects active uncertainty querying patterns and dynamically throttles or injects noise.",
                effectiveness_reduction_pct=30.0,
                cost_to_lab_usd=150_000.0,
                latency_impact_ms=12.0,
                enabled=False,
            ),
            FrontierDefense(
                name="Statistical Watermarking & Canary Trap Tokens",
                description="Embeds subtle cryptographic probability biases to legally prove weight extraction in court.",
                effectiveness_reduction_pct=15.0,
                cost_to_lab_usd=80_000.0,
                latency_impact_ms=2.0,
                enabled=False,
            ),
            FrontierDefense(
                name="Tiered Enterprise Quotas & Verification",
                description="Restricts queries >100k/day to verified enterprise accounts with contractual non-competes.",
                effectiveness_reduction_pct=40.0,
                cost_to_lab_usd=300_000.0,
                latency_impact_ms=0.0,
                enabled=False,
            ),
        ]

    def compute_api_cost_for_queries(self, query_count: int, reasoning_multiplier: float = 1.0) -> float:
        """Calculate the total API cost for Q distillation queries."""
        input_tokens = query_count * self.avg_prompt_tokens
        output_tokens = query_count * self.avg_output_tokens * reasoning_multiplier

        input_cost = (input_tokens / 1_000_000.0) * self.input_price_per_million
        output_cost = (output_tokens / 1_000_000.0) * self.output_price_per_million
        return float(input_cost + output_cost)

    def calculate(
        self,
        query_count: int = 50_000,
        reasoning_multiplier: float = 1.0,
        defenses: List[FrontierDefense] = None,
        policy_levers: PolicyLevers = None,
    ) -> EconomicResult:
        defenses = defenses or self.default_defenses()
        policy_levers = policy_levers or PolicyLevers()

        # Base API query cost
        base_api_cost = self.compute_api_cost_for_queries(query_count, reasoning_multiplier)

        # Distiller cost setup
        distiller = DistillationCostBreakdown(
            api_query_cost_usd=base_api_cost,
            student_finetuning_usd=8_000.0,
            data_filtering_rejection_usd=4_000.0,
            base_model_amortization_usd=0.0,
        )

        teacher_total = self.teacher_breakdown.total_rnd_cost
        distiller_total = distiller.total_distiller_cost

        # Cost asymmetry leverage: e.g. $100M / $25k = 4,000x
        asymmetry_leverage = teacher_total / max(1.0, distiller_total)

        # Serving unit economics
        tokens_per_request = self.avg_prompt_tokens + self.avg_output_tokens
        teacher_cost_per_req = (
            (self.avg_prompt_tokens / 1e6) * self.input_price_per_million
            + (self.avg_output_tokens / 1e6) * self.output_price_per_million
        )
        student_cost_per_req = (tokens_per_request / 1e6) * self.student_serving_cost_per_million

        # Breakeven volume: Volume * (teacher_req_cost - student_req_cost) = distiller_fixed_cost
        unit_savings = max(0.0001, teacher_cost_per_req - student_cost_per_req)
        breakeven_queries = int(distiller_total / unit_savings)

        # Policymaker Risk Metrics
        # 1. Market failure risk: Higher when leverage is huge and defenses are zero
        defense_mitigation = sum(d.effectiveness_reduction_pct for d in defenses if d.enabled)
        defense_mitigation = min(90.0, defense_mitigation)

        policy_mitigation = 0.0
        if policy_levers.kyc_api_mandate:
            policy_mitigation += 20.0
        if policy_levers.export_controls_api:
            policy_mitigation += 15.0
        if policy_levers.watermarking_legal_framework:
            policy_mitigation += 25.0
        if policy_levers.compute_threshold_auditing:
            policy_mitigation += 15.0

        total_mitigation = min(95.0, defense_mitigation + policy_mitigation)

        # Base market failure score proportional to log-leverage
        raw_market_risk = min(100.0, (math.log10(max(10.0, asymmetry_leverage)) / 5.0) * 100.0)
        market_failure_risk_score = max(5.0, raw_market_risk * (1.0 - (total_mitigation / 100.0)))

        # Safety alignment stripping score
        raw_safety_risk = 85.0  # High baseline because weights are unaligned
        safety_stripping_risk_score = max(10.0, raw_safety_risk * (1.0 - (total_mitigation * 0.7 / 100.0)))

        return EconomicResult(
            teacher_cost=self.teacher_breakdown,
            distiller_cost=distiller,
            asymmetry_leverage=asymmetry_leverage,
            breakeven_queries=breakeven_queries,
            cost_per_teacher_request_usd=teacher_cost_per_req,
            cost_per_student_request_usd=student_cost_per_req,
            market_failure_risk_score=market_failure_risk_score,
            safety_stripping_risk_score=safety_stripping_risk_score,
            defenses=defenses,
            policy_levers=policy_levers,
        )
