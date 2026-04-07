"""SyntheticActuator: sets power on a SyntheticState and advances it via physics."""

from __future__ import annotations

from .state import SyntheticState
from ..actuator import Actuator


class SyntheticActuator(Actuator[float, float]):
    """
    Actuator that applies a power setpoint to a SyntheticState.

    The SyntheticState's update_fn computes the resulting next state from
    the power command, timestep, and previous state. This mirrors the real
    world: the actuator commands a power level; the physics determine the
    resulting state (SoC, temperature, etc.).

    Created via SyntheticState.make_sensor_actuator() — do not instantiate
    directly unless you have a specific reason.
    """

    def __init__(self, state: SyntheticState) -> None:
        self._state = state

    def execute(self, command: float, expected_next_state: float) -> None:
        """
        Sets the internal state to the expected_next_state computed by the optimization process
        """
        self._state.set(expected_next_state)
