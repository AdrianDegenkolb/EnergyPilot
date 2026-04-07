"""
Abstract sensor interface and stub concrete implementations.

Sensor[T] is the common interface for all data sources — physical devices,
external APIs, synthetic simulations, and derived computations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Sensor(ABC, Generic[T]):
    """Abstract base for all sensor types."""

    @abstractmethod
    def read(self) -> T:
        """Return the current reading from this sensor."""
        ...
