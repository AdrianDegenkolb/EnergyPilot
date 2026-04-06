"""
MPC history dataclasses.

MPCStep: snapshot of one complete MPC iteration — the step index and
the full optimization result (which carries all signal arrays and decisions).

MPCHistory: ordered sequence of MPCStep records.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from optimization.milp.household import HouseholdResult, HouseholdOptimizationProblem


@dataclass
class MPCStep:
    """Snapshot of a single MPC iteration."""

    step: int
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
