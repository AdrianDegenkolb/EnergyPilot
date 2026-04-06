"""
Time singleton that provides current time, allowing for synthetic time during simulation and real time otherwise.
"""

from datetime import datetime
from typing import Optional


class Time:
    def __init__(self):
        self._synthetic_current_time: Optional[datetime] = None

    def get(self) -> datetime:
        if self._synthetic_current_time is None:
            return datetime.now()                   # real time when not simulating
        else:
            return self._synthetic_current_time     # synthetic time when simulating

    def set(self, synthetic_time: datetime):
        """Set the current time for synthetic simulation."""
        self._synthetic_current_time = synthetic_time

    @staticmethod
    def get_instance():
        """Singleton instance of Time."""
        if not hasattr(Time, "_instance"):
            Time._instance = Time()
        return Time._instance
