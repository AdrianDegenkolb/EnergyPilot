"""
MPC: Model Predictive Control orchestrator.

MPC owns all entities, the four required TimeSeries, any extra series (e.g.
temp_out), and builds the HouseholdOptimizationProblem internally.

Step sequence per iteration:
  1. observe all TimeSeries (sensor read → append to history → invalidate cache)
  2. observe all PhysicalEntities (sensor read → set_initial_state on model)
  3. household.solve(T_h, dt) — reads forecasts from series internally
  4. act: each entity extracts its power command and executes via its actuator
  5. record MPCStep snapshot

Ordering contract:
  Series must be observed before entities (step 1 before 2) so that any
  series whose sensor is shared with an entity sensor is read first.
  Series must be observed before solve (step 1 before 3) so that
  HeatPumpModel.contribute() reads a fresh temp_out forecast from its series.
"""

from __future__ import annotations

from datetime import timedelta
from time import sleep

from forecasting.time import Time
from forecasting.time_series import TimeSeries
from optimization.milp.physical_entity import PhysicalEntity
from optimization.milp.household import HouseholdOptimizationProblem
from .MPC_state import MPCStep, MPCHistory


class MPC:
    """
    Orchestrates the MPC loop over a set of physical entities and time series.

    Args:
        entities:    Controllable physical devices — each has a sensor, model,
                     and actuator.
        price_buy:   Buy price TimeSeries [€/kWh].
        price_sell:  Sell price TimeSeries [€/kWh].
        load:        Base load TimeSeries [kW].
        gen:         Generation (PV) TimeSeries [kW].
        extra_series: Additional TimeSeries to observe each step (e.g. temp_out
                      consumed by a HeatPumpModel). Observed but not passed to
                      HouseholdOptimizationProblem directly.
        T_horizon:   Planning horizon in timesteps.
        dt:          Timestep duration [h].
    """

    def __init__(
        self,
        entities: list[PhysicalEntity],
        price_buy: TimeSeries,
        price_sell: TimeSeries,
        load: TimeSeries,
        gen: TimeSeries,
        extra_series: list[TimeSeries],
        T_horizon: int,
        dt: timedelta,
    ) -> None:
        self.entities = entities
        self._price_buy = price_buy
        self._price_sell = price_sell
        self._load = load
        self._gen = gen
        self._extra_series = extra_series
        self.T_horizon = T_horizon
        self.dt = dt
        self.household = HouseholdOptimizationProblem(
            components=[e.model for e in entities],
            price_buy=price_buy,
            price_sell=price_sell,
            load=load,
            gen=gen,
        )

    def run(self, T_total: int, fast_forward: bool) -> MPCHistory:
        """
        Run the MPC loop for T_total steps.

        Args:
            T_total: Number of MPC steps to execute.
            fast_forward: If false, wait for self.dt between steps. If false assume a simulation setup and fast forward
            time using Time.set(...)

        Returns:
            MPCHistory containing all completed steps.
        """
        history = MPCHistory()

        print(f"{'t':>3}  {'Cost':>7}  {'p_buy':>6}  {'p_sell':>6}  {'Variables'}")
        print("─" * 65)

        for step in range(T_total):
            T_h = min(self.T_horizon, T_total - step)
            mpc_step = self._step(step, T_h)
            if mpc_step is None:
                continue
            history.append(mpc_step)

            if fast_forward:
                now = Time.get_instance().get()
                Time.get_instance().set(now + self.dt)
            else:
                sleep(self.dt.total_seconds())

            dec_str = "  ".join(
                f"{k}={v[0]:.2f}" for k, v in mpc_step.result.variables.items()
            )
            print(
                f"{step:>3}  {mpc_step.result.total_cost:>7.4f}  "
                f"{mpc_step.result.p_buy[0]:>6.2f}  "
                f"{mpc_step.result.p_sell[0]:>6.2f}  {dec_str}"
            )

        return history

    def _step(self, step: int, T_h: int) -> MPCStep | None:
        """Execute a single MPC step. Returns None if solve failed."""
        timestamp = Time.get_instance().get()

        # 1. Observe and forecast all series
        for ts in self._extra_series + [self._price_buy, self._price_sell, self._load, self._gen]:
            ts.observe()
            ts.forecast(T_h, self.dt)

        # 2. Observe all entities (sensor read → set_initial_state on model)
        for entity in self.entities:
            entity.observe()

        # 3. Solve
        result = self.household.solve(T_h, self.dt)

        if not result.success:
            print(f"{step:>3}  FAILED: {result.message}")
            return None

        # 4. Execute actions on all entities
        for entity in self.entities:
            entity.act(result)

        return MPCStep(step=step, timestamp=timestamp, optimization_problem=self.household, result=result)
