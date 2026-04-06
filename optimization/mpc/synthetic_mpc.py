"""
SyntheticMPC: MPC subclass for synthetic (simulated) experiments.

Adds a signal_driver hook that advances passive signal states (price, load,
generation, outdoor temperature) before each MPC step. In production the
signals come from real sensors; in synthetic mode this closure drives them
from the same diurnal patterns used by the forecasters.

Signal driver contract:
  signal_driver(step: int) → None
  Called once per MPC step, before series are observed. Writes new values
  into the SyntheticState objects via state.set(value).
"""

from __future__ import annotations
from typing import Callable

from .mpc import MPC
from .MPC_state import MPCStep


class SyntheticMPC(MPC):
    """
    MPC with a per-step signal driver for synthetic experiments.

    Args:
        signal_driver: Called at the start of each step with the step index.
                       Must update all passive SyntheticState values before
                       the series are observed.
        **kwargs:      All other arguments forwarded to MPC.__init__.
    """

    def __init__(
        self,
        signal_driver: Callable[[int], None],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._signal_driver = signal_driver

    def _step(self, step: int, T_h: int) -> MPCStep | None:
        self._signal_driver(step)
        return super()._step(step, T_h)
