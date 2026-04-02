"""
Household: assembles entities and solves the energy scheduling MILP.

Formulation:
    Decision variables: x_bat^t, x_ev^t, x_hp^t
    Auxiliary variables: p_buy^t, p_sell^t >= 0
    State variables:     soc_bat^t, soc_ev^t, temp_in^t

    Objective:
        min  sum_t  price_buy^t * p_buy^t - price_sell^t * p_sell^t

    Grid balance (per timestep):
        p_buy^t - p_sell^t = load^t + x_hp^t + x_ev^t + x_bat^t - gen^t

    Entity dynamics and constraints are contributed by each entity.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from .entities import Battery, ElectricVehicle, HeatPump, Entity, VariableRegistry, Constraints
from .forecast import ForecastTrajectory


@dataclass
class OptimisationResult:
    """Result of a single MILP solve."""

    success: bool
    total_cost: float
    message: str

    # Optimal decisions per timestep
    x_bat: Optional[np.ndarray] = None     # battery power [kW]
    x_ev: Optional[np.ndarray] = None      # EV charging power [kW]
    x_hp: Optional[np.ndarray] = None      # heat pump power [kW]
    p_buy: Optional[np.ndarray] = None     # grid purchase [kW]
    p_sell: Optional[np.ndarray] = None    # grid feed-in [kW]

    # State trajectories (length T+1)
    soc_bat: Optional[np.ndarray] = None
    soc_ev: Optional[np.ndarray] = None
    temp_in: Optional[np.ndarray] = None


class Household:
    """
    Assembles a set of entities into a household energy system.

    solve(trajectory) formulates and solves the MILP for a given
    forecast trajectory, returning optimal decisions for all timesteps.
    """

    def __init__(
        self,
        battery: Optional[Battery] = None,
        ev: Optional[ElectricVehicle] = None,
        heat_pump: Optional[HeatPump] = None,
    ) -> None:
        """
        Args:
            battery: Home battery storage entity.
            ev: Electric vehicle entity.
            heat_pump: Heat pump entity.
        """
        self.battery = battery
        self.ev = ev
        self.heat_pump = heat_pump
        self._entities: list[Entity] = [
            e for e in [battery, ev, heat_pump] if e is not None
        ]

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

        # Register entity variables
        for entity in self._entities:
            entity.register(reg, T)

        # Register auxiliary grid variables: p_buy^t, p_sell^t
        idx_buy = reg.register("p_buy", T)
        idx_sell = reg.register("p_sell", T)

        n = reg.n_vars
        objective = np.zeros(n)

        # Objective: min sum_t price_buy * p_buy - price_sell * p_sell
        for t in range(T):
            objective[idx_buy[t]] = trajectory.price_buy[t] * dt
            objective[idx_sell[t]] = -trajectory.price_sell[t] * dt

        # Entity constraints
        for entity in self._entities:
            entity.contribute(reg, constraints, objective, T, dt)

        # Grid balance: p_buy^t - p_sell^t = load^t + net_controllable^t - gen^t
        # Rearranged: p_buy^t - p_sell^t - sum(x_entity^t) = load^t - gen^t
        for t in range(T):
            row = np.zeros(n)
            row[idx_buy[t]] = 1.0
            row[idx_sell[t]] = -1.0
            for entity in self._entities:
                # net_power contribution at timestep t
                net = entity.net_power(reg, np.eye(n), T)
                # net is a (T, n) matrix — row t gives coefficients for x
                row -= net[t]
            rhs = float(trajectory.load[t] - trajectory.pv[t])
            constraints.add_eq(row, rhs)

        # Bounds
        lb = np.full(n, -np.inf)
        ub = np.full(n, np.inf)

        for entity in self._entities:
            if hasattr(entity, "bounds"):
                e_lb, e_ub = entity.bounds(T)
                # find the variable range for this entity
                keys = [k for k in ["battery_x", "battery_soc",
                                     "ev_x", "ev_soc",
                                     "hp_x", "hp_temp"] if k in reg._blocks]
                # apply bounds via the registered ranges
                start = None
                if isinstance(entity, Battery):
                    start = reg["battery_x"].start
                elif isinstance(entity, ElectricVehicle):
                    start = reg["ev_x"].start
                elif isinstance(entity, HeatPump):
                    start = reg["hp_x"].start
                if start is not None:
                    end = start + len(e_lb)
                    lb[start:end] = e_lb
                    ub[start:end] = e_ub

        # p_buy, p_sell >= 0
        lb[idx_buy.start: idx_buy.stop] = 0.0
        lb[idx_sell.start: idx_sell.stop] = 0.0

        # Solve
        if not constraints.rows:
            return OptimisationResult(success=False, total_cost=0.0, message="No constraints built")

        A = np.vstack(constraints.rows)
        lin_constraints = LinearConstraint(A, constraints.lo, constraints.hi)
        result = milp(
            c=objective,
            constraints=lin_constraints,
            bounds=Bounds(lb=lb, ub=ub),
        )

        if not result.success:
            return OptimisationResult(success=False, total_cost=0.0, message=result.message)

        x = result.x
        return OptimisationResult(
            success=True,
            total_cost=float(objective @ x),
            message=result.message,
            x_bat=np.array([x[reg["battery_x"][t]] for t in range(T)]) if self.battery else None,
            x_ev=np.array([x[reg["ev_x"][t]] for t in range(T)]) if self.ev else None,
            x_hp=np.array([x[reg["hp_x"][t]] for t in range(T)]) if self.heat_pump else None,
            p_buy=np.array([x[idx_buy[t]] for t in range(T)]),
            p_sell=np.array([x[idx_sell[t]] for t in range(T)]),
            soc_bat=np.array([x[reg["battery_soc"][t]] for t in range(T + 1)]) if self.battery else None,
            soc_ev=np.array([x[reg["ev_soc"][t]] for t in range(T + 1)]) if self.ev else None,
            temp_in=np.array([x[reg["hp_temp"][t]] for t in range(T + 1)]) if self.heat_pump else None,
        )
