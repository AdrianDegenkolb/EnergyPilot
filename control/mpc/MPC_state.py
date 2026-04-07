"""
MPC history dataclasses.

MPCStep: snapshot of one complete MPC iteration — the step index and
the full optimization result (which carries all signal arrays and decisions).

MPCHistory: ordered sequence of MPCStep records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..milp.household import HouseholdResult, HouseholdOptimizationProblem


@dataclass
class MPCStep:
    """Snapshot of a single MPC iteration."""

    step: int
    timestamp: datetime
    optimization_problem: HouseholdOptimizationProblem
    result: HouseholdResult


@dataclass
class MPCHistory:
    """Ordered sequence of MPC step records."""

    steps: list[MPCStep] = field(default_factory=list)

    def append(self, step: MPCStep) -> None:
        self.steps.append(step)

    def __len__(self) -> int:
        return len(self.steps)

    def step_cost(self, i: int) -> float:
        """
        Energy cost incurred during step i [€].

        dt is derived from the timestamps of steps i and i+1.
        Raises IndexError for the last step — it has no successor.
        """
        if i >= len(self.steps) - 1:
            raise IndexError(f"Step {i} is the last step; cost requires a successor timestamp.")
        dt_h = (self.steps[i + 1].timestamp - self.steps[i].timestamp).total_seconds() / 3600
        r = self.steps[i].result
        return float((r.p_buy[0] * r.price_buy[0] - r.p_sell[0] * r.price_sell[0]) * dt_h)

    def cumulative_cost(self) -> np.ndarray:
        """Cumulative energy cost over all steps except the last [€]."""
        return np.cumsum([self.step_cost(i) for i in range(len(self.steps) - 1)])
