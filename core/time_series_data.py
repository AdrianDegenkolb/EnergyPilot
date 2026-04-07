"""
TimeSeriesData: a sorted list of (datetime, float) pairs with linear interpolation.
"""

from __future__ import annotations

import bisect
from datetime import datetime

from sortedcontainers import SortedList


class TimeSeriesData:
    """
    A sorted list of (datetime, float) pairs with linear interpolation.

    Args:
        points: Optional initial list of (timestamp, value) pairs.
    """

    def __init__(self, points: list[tuple[datetime, float]] | None = None) -> None:
        self._points: SortedList = SortedList(key=lambda x: x[0])
        if points:
            for timestamp, value in points:
                self._points.add((timestamp, value))

    def add_point(self, timestamp: datetime, value: float) -> None:
        self._points.add((timestamp, value))

    def get_value_at(self, timestamp: datetime) -> float:
        """
        Interpolate the value at the given timestamp.

        Raises if the timestamp is out of the covered range or no data exists.
        """
        if not self._points:
            raise RuntimeError("No data available.")

        keys = [t for (t, _) in self._points]
        values = [v for (_, v) in self._points]
        idx = bisect.bisect_left(keys, timestamp)

        if idx == 0:
            if keys[0] == timestamp:
                return values[0]
            raise ValueError(f"Timestamp {timestamp} is before the earliest entry.")
        if idx == len(keys):
            raise ValueError(f"Timestamp {timestamp} is after the latest entry.")

        return self._interpolate_in_sorted(keys, values, timestamp)

    def trim_oldest(self, max_size: int) -> None:
        """Remove oldest entries until at most max_size entries remain."""
        while len(self._points) > max_size:
            self._points.pop(0)

    @property
    def points(self) -> list[tuple[datetime, float]]:
        """Return all points as a list of (timestamp, value), oldest first."""
        return list(self._points)

    def __len__(self) -> int:
        return len(self._points)

    def __getitem__(self, idx: int) -> tuple[datetime, float]:
        return self._points[idx]

    @staticmethod
    def _interpolate_in_sorted(
        keys: list[datetime],
        values: list[float],
        query: datetime,
    ) -> float:
        """
        Linear interpolation within a sorted (keys, values) pair.

        Clamps to boundary values if query is outside the range.
        """
        idx = bisect.bisect_left(keys, query)
        if idx == 0:
            return values[0]
        if idx == len(keys):
            return values[-1]

        left_t, right_t = keys[idx - 1], keys[idx]
        left_v, right_v = values[idx - 1], values[idx]
        alpha = (query - left_t).total_seconds() / (right_t - left_t).total_seconds()
        return left_v + alpha * (right_v - left_v)
