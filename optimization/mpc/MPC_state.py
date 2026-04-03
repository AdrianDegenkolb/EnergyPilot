"""
MPC history dataclasses.

MPCStep: snapshot of one complete MPC iteration.
MPCHistory: ordered sequence of MPCStep records.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from forecasting.Observation import Observation


@dataclass(frozen=True)
class MPCStep:
    """
    Snapshot of a single MPC iteration.

    Captures the observation, the grid decisions, and all entity
    decisions taken in this step.
    """

    step: int
    observation: Observation
    total_cost: float
    p_buy: float                        # grid purchase [kW]
    p_sell: float                       # grid feed-in [kW]
    decisions: dict[str, float]         # entity decisions, e.g. x_bat, x_ev, x_hp


@dataclass
class MPCHistory:
    """Ordered sequence of MPC step records."""

    steps: list[MPCStep] = field(default_factory=list)

    def append(self, step: MPCStep) -> None:
        """Append a completed MPC step."""
        self.steps.append(step)

    def __len__(self) -> int:
        return len(self.steps)