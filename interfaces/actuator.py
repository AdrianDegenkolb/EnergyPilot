"""
Abstract actuator interface and stub concrete implementations.

An Actuator receives the action extracted from an optimization result and
applies it to the physical (or simulated) device. It is the write-side
counterpart to SensorInterface's read side.

execute() takes a dict so that multi-output devices (e.g. a heat pump that
controls both power and a valve) can pass all signals in one call without
changing the interface signature.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar("T")
S = TypeVar("S")

class Actuator(ABC, Generic[T, S]):
    """
    Abstract base for all actuator types.

    Subclasses send commands to real hardware, external APIs, or synthetic
    simulation state. Callers only interact with execute()
    """

    @abstractmethod
    def execute(self, action: T, expected_next_state: S) -> None:
        """
        Apply the given action of type T to the device.
        Alternatively, the expected_next_state of type S can be set directly

        Args:
            action: Mapping of signal names to values, as returned by
                    OptimizationComponent.extract_action(). The keys and
                    their semantics are defined by the specific actuator
                    and its paired OptimizationComponent.
            expected_next_state: The expected next state of the device after
                    executing the action, as computed by the device's physics
        """
        ...
