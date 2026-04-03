"""
Forecaster hierarchy for generating forecast distributions.

Forecaster is the abstract interface — subclasses implement forecast()
using different strategies (synthetic, trained model, etc.).

The forecaster receives the full ObservationHistory so that a trained
model can condition on past values. The current observation (t=0) is
always pinned to the latest real measurement (std=0).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from .Observation import Observation, ObservationHistory
from .Forecast import ForecastParams


class Forecaster(ABC):
    """
    Abstract base class for all forecasters.

    A forecaster maps an ObservationHistory to a ForecastParams over
    a planning horizon of T timesteps.
    """

    @abstractmethod
    def forecast(
        self,
        history: ObservationHistory,
        T: int,
        dt: float,
    ) -> ForecastParams:
        """
        Produce forecast distributions for the next T timesteps.

        The first timestep (t=0) must be pinned to the latest real
        observation by setting std[0]=0 and mean[0]=observed value.

        Args:
            history: All past observations, most recent last.
            T: Planning horizon in timesteps.
            dt: Timestep duration [h].

        Returns:
            ForecastParams with mean and std arrays of shape (T,).
        """
        ...


class SyntheticForecaster(Forecaster):
    """
    Synthetic forecaster for POC use — no trained model required.

    Generates simple parametric forecast distributions based on
    the current observation and synthetic diurnal patterns.
    The current observation pins t=0 exactly (std=0).

    Replace with TrainedForecaster once forecasting models are available.
    """

    def forecast(
        self,
        history: ObservationHistory,
        T: int,
        dt: float,
    ) -> ForecastParams:
        """
        Generate synthetic forecast distributions over T timesteps.

        Uses the latest observation for t=0 and simple sinusoidal
        patterns for future timesteps.
        """
        obs = history.latest
        t = np.arange(T)

        price_buy_mean, price_buy_std = self._forecast_price_buy(obs, t, T)
        price_sell_mean, price_sell_std = self._forecast_price_sell(obs, t, T)
        pv_mean, pv_std = self._forecast_pv(obs, t, T)
        load_mean, load_std = self._forecast_load(obs, t, T)
        temp_out_mean, temp_out_std = self._forecast_temp_out(obs, t, T)

        return ForecastParams(
            T=T, dt=dt,
            price_buy_mean=price_buy_mean,
            price_buy_std=price_buy_std,
            price_sell_mean=price_sell_mean,
            price_sell_std=price_sell_std,
            pv_mean=pv_mean,
            pv_std=pv_std,
            load_mean=load_mean,
            load_std=load_std,
            temp_out_mean=temp_out_mean,
            temp_out_std=temp_out_std,
        )

    def _pin(self, mean: np.ndarray, std: np.ndarray, value: float) -> tuple[np.ndarray, np.ndarray]:
        """Pin t=0 to the real observed value with zero uncertainty."""
        mean[0] = value
        std[0] = 0.0
        return mean, std

    def _forecast_price_buy(
        self, obs: Observation, t: np.ndarray, T: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Buy price: higher morning/evening, lower midday."""
        mean = 0.28 + 0.08 * np.cos(2 * np.pi * t / T)
        std = np.full(T, 0.02)
        return self._pin(mean, std, obs.price_buy)

    def _forecast_price_sell(
        self, obs: Observation, t: np.ndarray, T: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sell price: approximately fixed feed-in tariff."""
        mean = np.full(T, 0.08)
        std = np.full(T, 0.005)
        return self._pin(mean, std, obs.price_sell)

    def _forecast_pv(
        self, obs: Observation, t: np.ndarray, T: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """PV generation: bell curve peaking at midday."""
        mean = np.maximum(0.0, 3.5 * np.exp(-0.5 * ((t - T * 0.5) / (T * 0.18)) ** 2))
        std = 0.2 * mean + 0.05
        return self._pin(mean, std, obs.pv)

    def _forecast_load(
        self, obs: Observation, t: np.ndarray, T: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Base load: slightly higher in morning and evening."""
        mean = np.maximum(0.0, 0.6 + 0.25 * np.abs(np.sin(np.pi * t / T)))
        std = np.full(T, 0.1)
        return self._pin(mean, std, obs.load)

    def _forecast_temp_out(
        self, obs: Observation, t: np.ndarray, T: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Outdoor temperature: warming through the day."""
        mean = 4.0 + 4.0 * np.sin(np.pi * t / T)
        std = np.full(T, 1.0)
        return self._pin(mean, std, obs.temp_out)