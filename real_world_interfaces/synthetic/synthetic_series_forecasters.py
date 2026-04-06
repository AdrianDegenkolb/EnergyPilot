"""
Synthetic SeriesForecaster implementations for POC use.

One forecaster per signal, each replicating the diurnal pattern logic
from SyntheticForecaster. These are constructed with the total number of
timesteps in the simulation (T_total) so that fractional-day position can
be inferred from history length.

All forecasters pin t=0 to history[-1] (the latest observation).
Replace with trained models once real data is available.
"""

from __future__ import annotations
from datetime import timedelta

import numpy as np

from forecasting.series_forecaster import SeriesForecaster
from forecasting.time import Time
from forecasting.time_series_data import TimeSeriesData


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


class SyntheticPriceBuyForecaster(SeriesForecaster):
    """
    Buy price forecast: higher at night/morning, lower at midday.

    Pattern: 0.28 + 0.08 * cos(2π * frac)
    """

    def __init__(self, T_total: int) -> None:
        self._T_total = T_total

    def forecast(self, history: list[float], T: int, dt: float) -> np.ndarray:
        step = len(history) - 1
        t = np.arange(T)
        frac = (step + t) / self._T_total
        mean = 0.28 + 0.08 * np.cos(2 * np.pi * frac)
        return self._pin(mean, history[-1])


class SyntheticPriceSellForecaster(SeriesForecaster):
    """
    Sell price forecast: fixed feed-in tariff.
    """

    def forecast(self, history: list[float], T: int, dt: float) -> np.ndarray:
        mean = np.full(T, 0.08)
        return self._pin(mean, history[-1])


class SyntheticGenForecaster(SeriesForecaster):
    """
    Generation (PV) forecast: Gaussian bell curve peaking at solar noon.

    Pattern: 3.5 * exp(-0.5 * ((frac - 0.5) / 0.18)²), clipped to >= 0.
    """

    def __init__(self, T_total: int) -> None:
        self._T_total = T_total

    def forecast(self, history: list[float], T: int, dt: float) -> np.ndarray:
        step = len(history) - 1
        t = np.arange(T)
        frac = (step + t) / self._T_total
        mean = np.maximum(0.0, 3.5 * np.exp(-0.5 * ((frac - 0.5) / 0.18) ** 2))
        return self._pin(mean, history[-1])


class SyntheticLoadForecaster(SeriesForecaster):
    """
    Base load forecast: higher in morning and evening, lower at midday.

    Pattern: 0.6 + 0.25 * |sin(π * frac)|, clipped to >= 0.
    """

    def __init__(self, T_total: int) -> None:
        self._T_total = T_total

    def forecast(self, history: list[float], T: int, dt: float) -> np.ndarray:
        step = len(history) - 1
        t = np.arange(T)
        frac = (step + t) / self._T_total
        mean = np.maximum(0.0, 0.6 + 0.25 * np.abs(np.sin(np.pi * frac)))
        return self._pin(mean, history[-1])


class SyntheticTempOutForecaster(SeriesForecaster):
    """
    Outdoor temperature forecast: warming through the day.

    Pattern: 4.0 + 4.0 * sin(π * frac)
    """

    def __init__(self, T_total: int) -> None:
        self._T_total = T_total

    def forecast(self, history: list[float], T: int, dt: float) -> np.ndarray:
        step = len(history) - 1
        t = np.arange(T)
        frac = (step + t) / self._T_total
        mean = 4.0 + 4.0 * np.sin(np.pi * frac)
        return self._pin(mean, history[-1])
