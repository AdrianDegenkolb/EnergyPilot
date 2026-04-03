"""
MPC runner and environment interface.

Environment is the abstract interface for receiving observations.
MPCRunner orchestrates the MPC loop over a list of entities.

At each step:
  1. Observe current sensor readings from the environment
  2. Append observation to history
  3. Call update() on each entity with the latest obs and trajectory
  4. Sample a forecast trajectory
  5. Build and solve the household MILP
  6. Record the step in history
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from forecasting import Observation, ObservationHistory, ForecastTrajectory, Forecaster
from optimization.milp import Entity, Household
from .MPC_state import MPCStep, MPCHistory


class Environment(ABC):
    """
    Abstract interface for receiving real sensor observations.

    In production: reads from smart meter, BMS, weather API, etc.
    In POC: generates synthetic observations.
    """

    @abstractmethod
    def observe(self, step: int) -> Observation:
        """
        Return the current real observation at the given step.

        Args:
            step: Current MPC timestep index.
        """
        ...


class SyntheticEnvironment(Environment):
    """
    Synthetic environment for POC — simulates sensor readings.

    Generates observations from simple diurnal patterns with noise,
    mimicking a real household over a single day.
    """

    def __init__(
        self,
        T_total: int,
        rng: np.random.Generator,
        soc_bat_init: float = 4.0,
        soc_ev_init: float = 15.0,
        temp_in_init: float = 18.5,
    ) -> None:
        """
        Args:
            T_total: Total number of MPC steps in the simulation.
            rng: Random number generator for reproducible noise.
            soc_bat_init: Initial battery SoC [kWh].
            soc_ev_init: Initial EV SoC [kWh].
            temp_in_init: Initial indoor temperature [°C].
        """
        self.T_total = T_total
        self.rng = rng
        self._soc_bat = soc_bat_init
        self._soc_ev = soc_ev_init
        self._temp_in = temp_in_init

    def update_state(self, soc_bat: float, soc_ev: float, temp_in: float) -> None:
        """
        Update simulated state from MILP result.

        In production this is not needed — the real sensors report back.
        """
        self._soc_bat = soc_bat
        self._soc_ev = soc_ev
        self._temp_in = temp_in

    def observe(self, step: int) -> Observation:
        """Return a noisy synthetic observation for the current step."""
        frac = step / self.T_total
        return Observation(
            price_buy=float(
                0.28 + 0.08 * np.cos(2 * np.pi * frac) + self.rng.normal(0, 0.01)
            ),
            price_sell=0.08,
            pv=float(max(
                0.0,
                3.5 * np.exp(-0.5 * ((frac - 0.5) / 0.18) ** 2) + self.rng.normal(0, 0.1),
            )),
            load=float(max(
                0.0,
                0.6 + 0.25 * abs(np.sin(np.pi * frac)) + self.rng.normal(0, 0.05),
            )),
            temp_out=float(4.0 + 4.0 * np.sin(np.pi * frac) + self.rng.normal(0, 0.3)),
            temp_in=self._temp_in,
            soc_bat=self._soc_bat,
            soc_ev=self._soc_ev,
        )


class MPCRunner:
    """
    Orchestrates the MPC loop over a fixed list of entities.

    The entity list defines the household configuration — omitting an
    entity (e.g. ElectricVehicle) simply excludes it from scheduling.
    Each entity is updated in-place via update() before each solve,
    so entities are created once and reused across MPC steps.
    """

    def __init__(
        self,
        environment: Environment,
        forecaster: Forecaster,
        entities: list[Entity],
        T_horizon: int,
        dt: float,
        rng: np.random.Generator,
    ) -> None:
        """
        Args:
            environment: Source of real sensor observations.
            forecaster: Produces forecast distributions from history.
            entities: All household entities — defines the configuration.
            T_horizon: Planning horizon in timesteps.
            dt: Timestep duration [h].
            rng: Random number generator for trajectory sampling.
        """
        self.environment = environment
        self.forecaster = forecaster
        self.entities = entities
        self.T_horizon = T_horizon
        self.dt = dt
        self.rng = rng
        self._obs_history = ObservationHistory()

    def run(self, T_total: int) -> MPCHistory:
        """
        Run the MPC loop for T_total steps.

        Args:
            T_total: Number of MPC steps to execute.

        Returns:
            MPCHistory containing all completed steps.
        """
        history = MPCHistory()

        print(f"{'t':>3}  {'Cost':>7}  {'p_buy':>6}  {'p_sell':>6}  {'Decisions'}")
        print("─" * 65)

        for step in range(T_total):
            T_h = min(self.T_horizon, T_total - step)
            mpc_step = self._step(step, T_h)
            if mpc_step is None:
                continue
            history.append(mpc_step)
            dec_str = "  ".join(f"{k}={v:.2f}" for k, v in mpc_step.decisions.items())
            print(f"{step:>3}  {mpc_step.total_cost:>7.4f}  "
                  f"{mpc_step.p_buy:>6.2f}  {mpc_step.p_sell:>6.2f}  {dec_str}")

        return history

    def _step(self, step: int, T_h: int) -> MPCStep | None:
        """Execute a single MPC step. Returns None if solve failed."""
        obs = self.environment.observe(step)
        self._obs_history.append(obs)

        params = self.forecaster.forecast(self._obs_history, T_h, self.dt)
        trajectory = ForecastTrajectory.sample(params, self.rng)

        # Update each entity with latest observation and trajectory
        for entity in self.entities:
            entity.update(obs, trajectory)

        result = Household(entities=self.entities).solve(trajectory)

        if not result.success:
            print(f"{step:>3}  FAILED: {result.message}")
            return None

        # Feed state back to synthetic environment (no-op in production)
        if isinstance(self.environment, SyntheticEnvironment):
            self.environment.update_state(
                soc_bat=float(result.states.get("soc_bat", [obs.soc_bat, obs.soc_bat])[1]),
                soc_ev=float(result.states.get("soc_ev", [obs.soc_ev, obs.soc_ev])[1]),
                temp_in=float(result.states.get("temp_in", [obs.temp_in, obs.temp_in])[1]),
            )

        return MPCStep(
            step=step,
            observation=obs,
            total_cost=result.total_cost,
            p_buy=float(result.p_buy[0]),
            p_sell=float(result.p_sell[0]),
            decisions={k: float(v[0]) for k, v in result.decisions.items()},
        )