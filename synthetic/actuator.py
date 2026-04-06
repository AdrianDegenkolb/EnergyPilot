"""SyntheticActuator: sets power on a SyntheticState and advances it via physics."""

from __future__ import annotations

from real_world_interfaces import Actuator
from .state import SyntheticState


class SyntheticActuator(Actuator[float]):
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

    def execute(self, command: float) -> None:
        """
        Apply the power command to the device and advance the simulated state.

        Args:
            command: Power setpoint [kW] as computed by the optimizer.
                     Positive = consuming/charging, negative = producing/discharging.
        """
        self._state.step(command)
