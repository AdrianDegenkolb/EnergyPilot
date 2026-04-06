"""
SeriesForecaster: per-series forecasting interface.

SeriesForecaster is a parallel, simpler forecasting abstraction to the
existing Forecaster. Where Forecaster operates on a full ObservationHistory
and returns a bundled ForecastParams, SeriesForecaster operates on the
scalar history of a single TimeSeries and returns a plain np.ndarray.

This keeps forecasting concerns local to each TimeSeries and avoids the
need for a global observation bundle.

The existing Forecaster / SyntheticForecaster remain unchanged until the
old MPCRunner path is removed.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class SeriesForecaster(ABC):
    """
    Abstract forecaster for a single scalar TimeSeries.

    Args (via forecast):
        history: Observed values so far, oldest first. May be empty on the
                 first step — subclasses must handle this gracefully.
        T: Planning horizon in timesteps.
        dt: Timestep duration [h].

    Returns:
        np.ndarray of shape (T,) representing the mean forecast trajectory.
        t=0 must be pinned to history[-1] (the latest observation) so that
        the optimizer sees the true current value at the start of the horizon.
    """

    @abstractmethod
    def forecast(self, history: list[float], T: int, dt: float) -> np.ndarray:
        """Produce a mean forecast trajectory of length T."""
        ...

    def _pin(self, mean: np.ndarray, current: float) -> np.ndarray:
        """Pin t=0 of the forecast to the latest observed value."""
        mean[0] = current
        return mean
