"""
Sensor observations from the household at a single point in time.
"""

from dataclasses import dataclass
from collections import deque
from typing import Optional


@dataclass(frozen=True)
class Observation:
    """
    Real sensor readings at a single timestep.

    All values represent instantaneous measurements, not forecasts.
    SoC values come from battery management systems (BMS).
    Temperature values come from smart thermometers.
    """

    price_buy: float            # current electricity buy price [€/kWh]
    price_sell: float           # current feed-in tariff [€/kWh]
    pv: float                   # current PV generation [kW]
    load: float                 # current uncontrollable load [kW]
    temp_out: float             # current outdoor temperature [°C]
    temp_in: float              # current indoor temperature [°C]
    soc_bat: Optional[float]    # battery state of charge [kWh], None if no battery
    soc_ev: Optional[float]     # EV state of charge [kWh], None if EV absent


class ObservationHistory:
    """
    Ordered sequence of past observations, most recent last.

    Acts as a fixed-size sliding window — older observations are
    dropped automatically when max_len is reached.
    """

    def __init__(self, max_len: int = 512) -> None:
        """
        Args:
            max_len: Maximum number of observations to retain.
        """
        self._history: deque[Observation] = deque(maxlen=max_len)

    def append(self, obs: Observation) -> None:
        """Add a new observation to the history."""
        self._history.append(obs)

    @property
    def latest(self) -> Observation:
        """Return the most recent observation."""
        if not self._history:
            raise ValueError("No observations recorded yet")
        return self._history[-1]

    @property
    def observations(self) -> list[Observation]:
        """Return all observations as a list, oldest first."""
        return list(self._history)

    def __len__(self) -> int:
        return len(self._history)