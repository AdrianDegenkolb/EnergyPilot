"""
PhysicalEntity: pairs an OptimizationComponent with its Sensor and Actuator.

A PhysicalEntity represents a real-world controllable device (battery, EV,
heat pump). It holds the three components of the sense-plan-act loop:

    Sensor                → reads current device state → seeds OptimizationComponent,
    OptimizationComponent → contributes to the MILP
    Actuator              → receives first-step optimal power command → commands the device
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from real_world_interfaces import Sensor, Actuator
from .opt_components import OptimizationComponent

if TYPE_CHECKING:
    from optimization.milp.household import HouseholdResult


class PhysicalEntity:
    """
    A controllable physical device in the household energy system.

    Args:
        sensor: Reads current device state (SoC [kWh] or temperature [°C]).
        model: OptimizationComponent contributing to the MILP.
        actuator: Executes the optimal first-step power command on the device.
    """

    def __init__(
        self,
        sensor: Sensor[float],
        model: OptimizationComponent,
        actuator: Actuator,
    ) -> None:
        self.sensor = sensor
        self.model = model
        self.actuator = actuator

    def observe(self) -> float:
        """
        Read the current sensor value and write it into the model.

        Returns:
            The sensor reading.
        """
        value = self.sensor.read()
        self.model.set_initial_state(value)
        return value

    def act(self, result: HouseholdResult) -> float:
        """
        Extract the optimal first-step power setpoint and execute it.

        Args:
            result: Full MILP result from HouseholdOptimizationProblem.solve().

        Returns:
            The power command sent to the actuator [kW].
        """
        power_setpoint = self.model.extract_optimal_power_setpoint(result)
        expected_next_state = self.model.extract_expected_next_step_state(result)
        self.actuator.execute(power_setpoint, expected_next_state)
        return power_setpoint
