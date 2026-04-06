"""
Synthetic simulation package.

SyntheticState is a mutable float that represents one simulated physical
quantity (battery SoC, indoor temperature, grid price, ...). It is the
shared state between a SyntheticSensor (read side) and a SyntheticActuator
(write side), ensuring the simulated world stays consistent across MPC steps.

Two usage patterns:

  Passive signal (externally driven — no actuator):
      state = SyntheticState(initial_value=0.28)
      sensor = state.make_sensor()
      # SyntheticMPC calls state.set(new_value) each step

  Entity state (physics-driven — has actuator):
      state = SyntheticState(
          initial_value=4.0,
          dt=0.25,
          update_fn=lambda prev, power, dt: prev + power * dt,
      )
      sensor, actuator = state.make_sensor_actuator()
      # SyntheticActuator calls state.apply(power) after each solve

Validation: SyntheticMPC checks at construction that every entity sensor is a
SyntheticSensor and every entity actuator is a SyntheticActuator, preventing
accidental mixing of real and synthetic components.
"""

from .state import SyntheticState
from .sensor import SyntheticSensor
from .actuator import SyntheticActuator

__all__ = ["SyntheticState", "SyntheticSensor", "SyntheticActuator"]
