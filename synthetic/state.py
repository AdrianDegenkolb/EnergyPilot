"""
SyntheticState: shared mutable float representing one simulated physical quantity.

For entity states (battery SoC, indoor temperature), the physics of state
evolution are captured in update_fn. The actuator calls apply(power) after
each MPC solve, advancing the state by one timestep.

For passive signals (price, load, generation, outdoor temperature), the
SyntheticMPC directly calls set(value) to drive the signal from outside.
These states have no actuator and no update_fn.
"""

from __future__ import annotations
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .sensor import SyntheticSensor
    from .actuator import SyntheticActuator


class SyntheticState:
    """
    Mutable float state shared between a SyntheticSensor and SyntheticActuator.

    Args:
        initial: Starting value of the simulated quantity.
        dt: Timestep duration [h]. Required when update_fn is provided.
        update_fn: Physics model — (prev_state, power, dt) -> next_state.
                   Captures device-specific constants (efficiency, C_therm,
                   etc.) in the closure. Required for entity states; omit for
                   passively driven signal states.

    Example — battery:
        SyntheticState(
            initial=4.0, dt=0.25,
            update_fn=lambda soc, power, dt: soc + power * dt * efficiency,
        )

    Example — price signal (no physics):
        SyntheticState(initial=0.28)
    """

    def __init__(
        self,
        initial: float,
        dt: float = 0.0,
        update_fn: Optional[Callable[[float, float, float], float]] = None,
    ) -> None:
        self._value = initial
        self._dt = dt
        self._step_dynamics = update_fn

    def read(self) -> float:
        """Return the current simulated value."""
        return self._value

    def set(self, value: float) -> None:
        """
        Directly overwrite the state value.

        Used by SyntheticMPC to drive passive signal states (price, load,
        generation, outdoor temperature) from synthetic diurnal patterns.
        """
        self._value = value

    def step(self, power: float) -> None:
        """
        Advance the state by one timestep given the power setpoint.

        Called by SyntheticActuator after each MPC solve. Runs the physics
        model captured in update_fn.

        Args:
            power: Power setpoint applied to the device [kW]. Positive =
                   consuming / charging, negative = producing / discharging.

        Raises:
            RuntimeError: If no update_fn was provided at construction.
        """
        if self._step_dynamics is None:
            raise RuntimeError(
                "SyntheticState.apply() called but no update_fn was provided. "
                "Passive signal states must be driven via set(), not apply()."
            )
        self._value = self._step_dynamics(self._value, power, self._dt)

    def make_sensor(self) -> SyntheticSensor:
        """
        Create a SyntheticSensor that reads from this state.

        Use for passive signal states that have no corresponding actuator.
        """
        from .sensor import SyntheticSensor
        return SyntheticSensor(self)

    def make_sensor_actuator(self) -> tuple[SyntheticSensor, SyntheticActuator]:
        """
        Create a matched SyntheticSensor / SyntheticActuator pair.

        Both reference this state. The sensor reads the current value; the
        actuator calls apply(power) to advance it via physics.

        Raises:
            RuntimeError: If no update_fn was provided (passive signal state).
        """
        if self._step_dynamics is None:
            raise RuntimeError(
                "make_sensor_actuator() requires an update_fn. "
                "Use make_sensor() for passive signal states."
            )
        from .sensor import SyntheticSensor
        from .actuator import SyntheticActuator
        return SyntheticSensor(self), SyntheticActuator(self)
