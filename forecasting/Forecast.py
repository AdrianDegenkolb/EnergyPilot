"""
Forecast trajectories for the planning horizon.

At t=0 real observed values are used. For t>0, values are sampled
from distributions representing forecast uncertainty.
Each quantity is modeled as a normal distribution with a mean and std
trajectory over the horizon.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ForecastParams:
    """
    Parameters defining the forecast distributions over a horizon of T steps.

    All array fields have shape (T,). The first entry (t=0) represents the
    current real observation; subsequent entries are forecast distributions.
    """

    T: int
    dt: float                       # timestep duration [h]

    price_buy_mean: np.ndarray      # electricity buy price [€/kWh]
    price_buy_std: np.ndarray

    price_sell_mean: np.ndarray     # electricity sell/feed-in price [€/kWh]
    price_sell_std: np.ndarray

    pv_mean: np.ndarray             # PV generation [kW]
    pv_std: np.ndarray

    load_mean: np.ndarray           # uncontrollable base load [kW]
    load_std: np.ndarray

    temp_out_mean: np.ndarray       # outdoor temperature [°C]
    temp_out_std: np.ndarray


@dataclass
class ForecastTrajectory:
    """
    A single sampled trajectory over T timesteps.

    t=0 uses real observations (std=0), t>0 are sampled from distributions.
    """

    T: int
    dt: float

    price_buy: np.ndarray           # [€/kWh]
    price_sell: np.ndarray          # [€/kWh]
    pv: np.ndarray                  # [kW]
    load: np.ndarray                # [kW]
    temp_out: np.ndarray            # [°C]

    @staticmethod
    def sample(
        params: ForecastParams,
        rng: np.random.Generator,
    ) -> "ForecastTrajectory":
        """
        Sample a single forecast trajectory from the given distributions.

        The first timestep uses the mean directly (real observation, std=0
        is assumed for t=0 by the caller by setting std[0]=0).
        All values are clipped to physically plausible ranges.
        """
        def _sample(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
            return mean + std * rng.standard_normal(len(mean))

        return ForecastTrajectory(
            T=params.T,
            dt=params.dt,
            price_buy=np.maximum(0.0, _sample(params.price_buy_mean, params.price_buy_std)),
            price_sell=np.maximum(0.0, _sample(params.price_sell_mean, params.price_sell_std)),
            pv=np.maximum(0.0, _sample(params.pv_mean, params.pv_std)),
            load=np.maximum(0.0, _sample(params.load_mean, params.load_std)),
            temp_out=_sample(params.temp_out_mean, params.temp_out_std),
        )
