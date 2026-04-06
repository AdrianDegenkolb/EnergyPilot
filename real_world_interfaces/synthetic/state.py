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
from typing import TYPE_CHECKING

from forecasting.time_series_data import TimeSeriesData
from forecasting.time import Time

if TYPE_CHECKING:
    from .sensor import SyntheticSensor
    from .actuator import SyntheticActuator


class SyntheticState:
    """
    Mutable float state shared between a SyntheticSensor and SyntheticActuator.

    Args:
        initial: Starting value of the simulated quantity.

    Example — battery:
        SyntheticState(initial=4.0)
    """

    def __init__(
        self,
        initial: float,
    ) -> None:
        self._value = initial

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

    def make_sensor(self) -> SyntheticSensor:
        """
        Create a SyntheticSensor that reads from this state.
        """
        from .sensor import SyntheticSensor
        return SyntheticSensor(self)

    def make_sensor_actuator(self) -> tuple[SyntheticSensor, SyntheticActuator]:
        """
        Create a matched SyntheticSensor / SyntheticActuator pair.

        Both reference this state. The sensor reads the current value; the
        actuator calls set().
        """
        from .sensor import SyntheticSensor
        from .actuator import SyntheticActuator
        return SyntheticSensor(self), SyntheticActuator(self)

class SyntheticExternalState:
    """
    Read-only state backed by a TimeSeriesData object.

    Intended for passive external signals (grid price, solar irradiance,
    outdoor temperature) whose synthetic trajectory is known in advance. On each
    read() the current simulation time is looked up in the time series
    and the interpolated value is returned.

    Args:
        data: TimeSeriesData covering the full simulation horizon.
    """

    def __init__(self, data: TimeSeriesData) -> None:
        self._data = data

    def read(self) -> float:
        """Return the interpolated value at the current simulation time."""
        timestamp = Time.get_instance().get()
        return self._data.get_value_at(timestamp)

    def make_sensor(self) -> SyntheticSensor:
        """Create a SyntheticSensor that reads from this state."""
        from .sensor import SyntheticSensor
        return SyntheticSensor(self)
