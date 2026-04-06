"""SyntheticSensor: reads from a SyntheticState."""

from __future__ import annotations

from real_world_interfaces import Sensor
from .state import SyntheticState


class SyntheticSensor(Sensor[float]):
    """
    Sensor that reads from a SyntheticState.

    Used in place of a real hardware sensor (BMS, smart meter, thermostat)
    during synthetic simulation. Paired with SyntheticActuator via
    SyntheticState.make_sensor_actuator().
    """

    def __init__(self, state: SyntheticState) -> None:
        self._state = state

    def read(self) -> float:
        return self._state.read()
