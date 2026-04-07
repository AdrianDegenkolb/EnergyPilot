"""
TimeSeries: per-signal observation history and forecast.

Couples a Sensor (current value source) with a SeriesForecaster
(history -> trajectory) and maintains a rolling history of observed values.

Forecast caching: forecast() returns a cached result and only recomputes
after a new observe() call. Multiple calls within one MPC step are safe.

Step ordering contract: observe() must be called before forecast() each step.

Timeline model: _history and _cached_forecast form one unified timeline.
get_value_at() interpolates seamlessly across both. On each observe(), stale
forecast entries (timestamp <= latest observation) are pruned.
"""

from __future__ import annotations

import bisect
from datetime import datetime, timedelta

import numpy as np

from core.time import Time
from core.time_series_data import TimeSeriesData
from interfaces import Sensor
from .series_forecaster import SeriesForecaster


class TimeSeries:
    """
    Combines a sensor, a rolling observation history, and a forecaster.

    Args:
        sensor:      Data source for the current value.
        forecaster:  Maps observation history to a forecast trajectory.
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
        self._max_history = max_history
        self._history = TimeSeriesData()
        self._cached_forecast: list[tuple[datetime, float]] = []
        self._dirty: bool = True

    def observe(self) -> float:
        """
        Read the current sensor value and append it to history.

        Prunes forecast entries with timestamps <= the new observation timestamp.
        Invalidates the forecast cache. Trims history to max_history.
        """
        value = self._sensor.read()
        timestamp: datetime = Time.get_instance().get()
        self.add_measurement(timestamp=timestamp, value=value)
        return value

    def add_measurement(self, timestamp: datetime, value: float) -> None:
        """
        Manually insert a (timestamp, value) pair into history.

        Prunes stale forecast entries and trims history to max_history.
        """
        self._history.add_point(timestamp, value)
        self._dirty = True

        self._cached_forecast = [
            (t, v) for (t, v) in self._cached_forecast if t > timestamp
        ]

        self._history.trim_oldest(self._max_history)

    def forecast(self, T: int, dt: timedelta) -> np.ndarray:
        """
        Predict values for the next T timesteps with spacing dt.

        Resamples history at dt resolution to build the input array,
        then calls the forecaster. The result is cached until the next observe().
        Raises if no observations have been recorded yet.

        Args:
            T:  Number of future timesteps to forecast.
            dt: Spacing between forecast timesteps.

        Returns:
            List of values for the forecast horizon in specified resolution.
        """
        if len(self._history) == 0:
            raise RuntimeError("forecast() called before any observations.")

        if not self._dirty and len(self._cached_forecast) == T:
            return np.array([v for (_, v) in self._cached_forecast])

        history_values = self._sample_history_equidistant(dt)
        raw_forecast: np.ndarray = self._forecaster.forecast(history_values, T, dt)

        latest_timestamp: datetime = self._history[-1][0]
        self._cached_forecast = [
            (latest_timestamp + dt * (i + 1), float(raw_forecast[i]))
            for i in range(T)
        ]
        self._dirty = False
        return raw_forecast

    def get_value_at(self, timestamp: datetime) -> float:
        """
        Interpolate the value at the given timestamp across the unified timeline
        (history + cached forecast).

        Raises if the timestamp is out of the covered range or no data exists.
        """
        timeline = self._history.points + self._cached_forecast
        if not timeline:
            raise RuntimeError("No data available.")

        keys = [t for (t, _) in timeline]
        idx = bisect.bisect_left(keys, timestamp)

        if idx == 0:
            if keys[0] == timestamp:
                return timeline[0][1]
            raise ValueError(f"Timestamp {timestamp} is before the earliest entry.")
        if idx == len(timeline):
            raise ValueError(f"Timestamp {timestamp} is after the latest entry.")

        left_t, left_v = timeline[idx - 1]
        right_t, right_v = timeline[idx]

        span = (right_t - left_t).total_seconds()
        alpha = (timestamp - left_t).total_seconds() / span
        return left_v + alpha * (right_v - left_v)

    @property
    def latest(self) -> float:
        """Return the most recently observed value."""
        if len(self._history) == 0:
            raise RuntimeError("No observations recorded yet.")
        return self._history[-1][1]

    @property
    def history(self) -> list[tuple[datetime, float]]:
        """Return all observations as a list of (timestamp, value), oldest first."""
        return self._history.points

    def _sample_history_equidistant(self, dt: timedelta) -> list[float]:
        """
        Resample history at dt intervals via interpolation, oldest to latest.

        The number of samples scales with history length: a long history yields
        many samples, a short history yields few.
        """
        if len(self._history) == 1:
            return [self._history[0][1]]

        oldest_t: datetime = self._history[0][0]
        latest_t: datetime = self._history[-1][0]
        n = int((latest_t - oldest_t).total_seconds() / dt.total_seconds()) + 1

        sample_timestamps = [oldest_t + dt * i for i in range(n)]

        points = self._history.points
        keys = [t for (t, _) in points]
        values = [v for (_, v) in points]

        return [
            TimeSeriesData._interpolate_in_sorted(keys, values, t)
            for t in sample_timestamps
        ]
