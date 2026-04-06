"""
TimeSeries: per-signal observation history and forecast.

Couples a Sensor (current value source) with a SeriesForecaster
(history -> trajectory) and maintains a rolling history of observed values.

Forecast caching: forecast() returns a cached result and only recomputes
after a new observe() call. Multiple calls within one MPC step are safe.

Step ordering contract: observe() must be called before forecast() each step.
"""

from __future__ import annotations
from collections import deque

import numpy as np

from real_world_interfaces import Sensor
from .series_forecaster import SeriesForecaster


class TimeSeries:
    """
    Combines a sensor, a rolling history, and a per-series forecaster.

    Args:
        sensor: Data source for the current value.
        forecaster: Maps observation history to a forecast trajectory.
        max_history: Maximum number of past observations to retain.
    """

    def __init__(
        self,
        sensor: Sensor[float],
        forecaster: SeriesForecaster,
        max_history: int = 512,
    ) -> None:
        self._sensor = sensor
        self._forecaster = forecaster
        self._history: deque[float] = deque(maxlen=max_history)
        self._cached_forecast: np.ndarray | None = None
        self._dirty: bool = True

    def observe(self) -> float:
        """
        Read the current sensor value and append it to history.

        Invalidates the forecast cache.
        """
        value = self._sensor.read()
        self._history.append(value)
        self._dirty = True
        return value

    def forecast(self, T: int, dt: float) -> np.ndarray:
        """
        Return the forecast trajectory for the next T timesteps.

        Returns a cached result if observe() has not been called since the
        last forecast(). Raises if no observations have been recorded yet.
        """
        if not self._history:
            raise RuntimeError("forecast() called before observe().")
        if self._dirty or self._cached_forecast is None or len(self._cached_forecast) != T:
            self._cached_forecast = self._forecaster.forecast(list(self._history), T, dt)
            self._dirty = False
        return self._cached_forecast

    @property
    def latest(self) -> float:
        """Return the most recently observed value."""
        if not self._history:
            raise RuntimeError("No observations recorded yet.")
        return self._history[-1]

    @property
    def history(self) -> list[float]:
        """Return all observations as a list, oldest first."""
        return list(self._history)
