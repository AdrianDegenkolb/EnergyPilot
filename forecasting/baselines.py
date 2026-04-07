"""
Baseline SeriesForecaster implementations for POC use.

LookupForecaster: Perfect forecaster that reads future values directly from a TimeSeriesData object.
ConstantForecaster: Forecast that repeats the last observed value for the entire horizon.
"""

from __future__ import annotations
from datetime import timedelta

import numpy as np

from forecasting.series_forecaster import SeriesForecaster
from core.time import Time
from core.time_series_data import TimeSeriesData


class LookupForecaster(SeriesForecaster):
    """
    Perfect forecaster that reads future values directly from a TimeSeriesData object.

    Suitable for synthetic simulations where the full signal trajectory is
    known in advance.
    """

    def __init__(self, data: TimeSeriesData) -> None:
        self._data = data

    def forecast(self, history: list[float], T: int, dt: timedelta) -> np.ndarray:
        current_time = Time.get_instance().get()
        return np.array([
            self._data.get_value_at(current_time + dt * (i + 1))
            for i in range(T)
        ])


class ConstantForecaster(SeriesForecaster):
    """Forecast that repeats the last observed value for the entire horizon."""

    def forecast(self, history: list[float], T: int, dt: timedelta) -> np.ndarray:
        return np.full(T, history[-1])
