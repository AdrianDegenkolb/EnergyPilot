"""SyntheticSensor: reads from a SyntheticState."""

from __future__ import annotations

from .state import SyntheticState, SyntheticExternalState
from ..sensor import Sensor


class SyntheticSensor(Sensor[float]):
    """
    Sensor that reads from a SyntheticState or SyntheticExternalState.

    Used in place of a real hardware sensor (BMS, smart meter, thermostat)
    during synthetic simulation. Paired with SyntheticActuator via
    SyntheticState.make_sensor_actuator().
    """

    def __init__(self, state: SyntheticState | SyntheticExternalState) -> None:
        self._state = state

    def read(self) -> float:
        return self._state.read()
