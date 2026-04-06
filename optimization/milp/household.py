"""
Household energy scheduling MILP.

HouseholdOptimizationProblem assembles OptimizationComponents and four
required TimeSeries (price_buy, price_sell, load, gen) into a MILP.

Load and generation are passive forecast signals — they have no decision
variables and need no dedicated component.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from forecasting.time_series import TimeSeries
from .opt_components import OptimizationComponent, VariableRegistry, Constraints


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HouseholdResult:
    """Result of a single household MILP solve."""

    success: bool
    total_cost: float
    message: str

    p_buy: Optional[np.ndarray] = None              # grid purchase [kW] per timestep
    p_sell: Optional[np.ndarray] = None             # grid feed-in [kW] per timestep

    load: Optional[np.ndarray] = None               # base load forecast [kW]
    gen: Optional[np.ndarray] = None                # generation forecast [kW]
    price_buy: Optional[np.ndarray] = None          # buy price forecast [€/kWh]
    price_sell: Optional[np.ndarray] = None         # sell price forecast [€/kWh]

    variables: dict = field(default_factory=dict)   # variables by name


# ---------------------------------------------------------------------------
# Optimization problem
# ---------------------------------------------------------------------------

class HouseholdOptimizationProblem:
    """
    Assembles OptimizationComponents and TimeSeries into a household MILP.

    Components (Battery, EV, HeatPump) contribute decision variables,
    state dynamics, and constraints. Load and generation are passive forecast
    signals read directly from the provided TimeSeries.

    Args:
        components: Controllable OptimizationComponents (battery, EV, HP, ...).
        price_buy:  Buy price TimeSeries [€/kWh].
        price_sell: Sell price TimeSeries [€/kWh].
        load:       Base load TimeSeries [kW].
        gen:        Generation (PV) TimeSeries [kW].
    """

    def __init__(
        self,
        components: list[OptimizationComponent],
        price_buy: TimeSeries,
        price_sell: TimeSeries,
        load: TimeSeries,
        gen: TimeSeries,
    ) -> None:
        self._components = components
        self._price_buy = price_buy
        self._price_sell = price_sell
        self._load = load
        self._gen = gen

    def solve(self, T: int, dt: timedelta) -> HouseholdResult:
        """
        Formulate and solve the household scheduling MILP.

        All forecast arrays are read from the injected TimeSeries. The series
        must have been observed (cache refreshed) before this call.

        Args:
            T: Planning horizon in timesteps.
            dt: Timestep duration.

        Returns:
            HouseholdResult with optimal decisions, state trajectories, and
            the signal arrays used in this solve.
        """
        dt_in_hours = dt.seconds / 3600
        price_buy = self._price_buy.forecast(T, dt)
        price_sell = self._price_sell.forecast(T, dt)
        load = self._load.forecast(T, dt)
        gen = self._gen.forecast(T, dt)

        reg = VariableRegistry()
        constraints = Constraints()

        for component in self._components:
            component.register(reg, T)

        idx_buy = reg.register("p_buy", T)
        idx_sell = reg.register("p_sell", T)

        n = reg.n_vars
        objective = np.zeros(n)
        lb = np.full(n, -np.inf)
        ub = np.full(n, np.inf)

        for t in range(T):
            objective[idx_buy[t]] = price_buy[t] * dt_in_hours
            objective[idx_sell[t]] = -price_sell[t] * dt_in_hours

        lb[idx_buy.start: idx_buy.stop] = 0.0
        lb[idx_sell.start: idx_sell.stop] = 0.0

        for component in self._components:
            component.contribute(reg, constraints, objective, lb, ub, T, dt_in_hours)

        # Grid balance: p_buy - p_sell = sum(component_net_power) + load - gen
        for t in range(T):
            row = np.zeros(n)
            row[idx_buy[t]] = 1.0
            row[idx_sell[t]] = -1.0

            rhs = float(load[t]) - float(gen[t])
            for component in self._components:
                coeffs, component_rhs = component.net_power(n, t)
                row -= coeffs
                rhs += component_rhs

            constraints.add_eq(row, rhs)

        if not constraints.rows:
            return HouseholdResult(success=False, total_cost=0.0, message="No constraints built")

        A = np.vstack(constraints.rows)
        lin_constraints = LinearConstraint(A, constraints.lo, constraints.hi)
        result = milp(
            c=objective,
            constraints=lin_constraints,
            bounds=Bounds(lb=lb, ub=ub),
        )

        if not result.success:
            return HouseholdResult(success=False, total_cost=0.0, message=result.message)

        return self._extract_result(
            result.x, reg, objective, idx_buy, idx_sell, T,
            price_buy, price_sell, load, gen,
        )

    @staticmethod
    def _extract_result(
        x: np.ndarray,
        reg: VariableRegistry,
        objective: np.ndarray,
        idx_buy: range,
        idx_sell: range,
        T: int,
        price_buy: np.ndarray,
        price_sell: np.ndarray,
        load: np.ndarray,
        gen: np.ndarray,
    ) -> HouseholdResult:
        variables: dict = {}

        for name in reg.get_registered_names():
            variables[name] = np.array([x[i] for i in reg[name]])

        return HouseholdResult(
            success=True,
            total_cost=float(objective @ x),
            message="Optimal",
            p_buy=np.array([x[idx_buy[t]] for t in range(T)]),
            p_sell=np.array([x[idx_sell[t]] for t in range(T)]),
            load=load,
            gen=gen,
            price_buy=price_buy,
            price_sell=price_sell,
            variables=variables,
        )
