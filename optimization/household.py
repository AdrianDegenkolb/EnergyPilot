"""
Household: assembles entities and solves the energy scheduling MILP.

Formulation:
    Decision variables: determined by entities (x_bat, x_ev, x_hp, ...)
    Auxiliary variables: p_buy^t, p_sell^t >= 0
    State variables:     determined by entities (soc_bat, soc_ev, temp_in, ...)

    Objective:
        min  sum_t  price_buy^t * p_buy^t - price_sell^t * p_sell^t

    Grid balance (per timestep, fully generic):
        p_buy^t - p_sell^t = sum_e net_power_e^t

    Each entity self-contains its constraints AND bounds via contribute().
    Household has no knowledge of concrete entity types.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from .entities import Entity, VariableRegistry, Constraints
from .forecast import ForecastTrajectory


@dataclass
class OptimisationResult:
    """Result of a single MILP solve."""

    success: bool
    total_cost: float
    message: str

    p_buy: Optional[np.ndarray] = None      # grid purchase [kW] per timestep
    p_sell: Optional[np.ndarray] = None     # grid feed-in [kW] per timestep

    decisions: dict = field(default_factory=dict)   # decision variables per entity
    states: dict = field(default_factory=dict)      # state trajectories per entity


class Household:
    """
    Assembles a set of entities into a household energy system.

    solve(trajectory) formulates and solves the MILP for a given
    forecast trajectory. All constraints and bounds come from the
    entities themselves — Household has no knowledge of their internals.
    """

    def __init__(self, entities: list[Entity]) -> None:
        """
        Args:
            entities: All household entities in any order.
        """
        self._entities = entities

    def solve(self, trajectory: ForecastTrajectory) -> OptimisationResult:
        """
        Formulate and solve the household scheduling MILP.

        Args:
            trajectory: Sampled forecast trajectory for the planning horizon.

        Returns:
            OptimisationResult with optimal decisions and state trajectories.
        """
        T = trajectory.T
        dt = trajectory.dt

        reg = VariableRegistry()
        constraints = Constraints()

        # Register all entity variables
        for entity in self._entities:
            entity.register(reg, T)

        # Register auxiliary grid variables: p_buy^t, p_sell^t
        idx_buy = reg.register("p_buy", T)
        idx_sell = reg.register("p_sell", T)

        n = reg.n_vars
        objective = np.zeros(n)
        lb = np.full(n, -np.inf)
        ub = np.full(n, np.inf)

        # Objective: min sum_t price_buy * p_buy * dt - price_sell * p_sell * dt
        for t in range(T):
            objective[idx_buy[t]] = trajectory.price_buy[t] * dt
            objective[idx_sell[t]] = -trajectory.price_sell[t] * dt

        # p_buy, p_sell >= 0
        lb[idx_buy.start: idx_buy.stop] = 0.0
        lb[idx_sell.start: idx_sell.stop] = 0.0

        # Each entity contributes its own constraints AND bounds
        for entity in self._entities:
            entity.contribute(reg, constraints, objective, lb, ub, T, dt)

        # Grid balance per timestep — fully generic, no entity type checks
        # p_buy^t - p_sell^t = sum_e (coeffs_e^t @ x + rhs_e^t)
        for t in range(T):
            row = np.zeros(n)
            row[idx_buy[t]] = 1.0
            row[idx_sell[t]] = -1.0

            rhs = 0.0
            for entity in self._entities:
                coeffs, entity_rhs = entity.net_power(n, t)
                row -= coeffs
                rhs += entity_rhs

            constraints.add_eq(row, rhs)

        # Solve
        if not constraints.rows:
            return OptimisationResult(
                success=False, total_cost=0.0, message="No constraints built"
            )

        A = np.vstack(constraints.rows)
        lin_constraints = LinearConstraint(A, constraints.lo, constraints.hi)
        result = milp(
            c=objective,
            constraints=lin_constraints,
            bounds=Bounds(lb=lb, ub=ub),
        )

        if not result.success:
            return OptimisationResult(
                success=False, total_cost=0.0, message=result.message
            )

        return self._extract_result(result.x, reg, objective, idx_buy, idx_sell, T)

    def _extract_result(
        self,
        x: np.ndarray,
        reg: VariableRegistry,
        objective: np.ndarray,
        idx_buy: range,
        idx_sell: range,
        T: int,
    ) -> OptimisationResult:
        """
        Extract named results from the raw solution vector.

        Reads variable blocks by name from the registry — no isinstance needed.
        """
        decisions: dict = {}
        states: dict = {}

        name_map = {
            "battery_x": ("x_bat", decisions),
            "battery_soc": ("soc_bat", states),
            "ev_x": ("x_ev", decisions),
            "ev_soc": ("soc_ev", states),
            "hp_x": ("x_hp", decisions),
            "hp_temp": ("temp_in", states),
        }

        for reg_key, (result_key, target) in name_map.items():
            if reg_key in reg._blocks:
                target[result_key] = np.array([x[i] for i in reg[reg_key]])

        return OptimisationResult(
            success=True,
            total_cost=float(objective @ x),
            message="Optimal",
            p_buy=np.array([x[idx_buy[t]] for t in range(T)]),
            p_sell=np.array([x[idx_sell[t]] for t in range(T)]),
            decisions=decisions,
            states=states,
        )