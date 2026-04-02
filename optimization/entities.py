"""
Entities that make up a household.

Each entity contributes:
  - decision variables and their bounds
  - state variables and their dynamics (equality constraints)
  - additional inequality constraints (e.g. deadlines)
  - terms to the objective function (currently none besides grid cost)

Naming convention for variable indices within the flat optimisation vector:
  Each entity receives a slice of the variable vector via register().
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Variable registry
# ---------------------------------------------------------------------------

class VariableRegistry:
    """
    Assigns contiguous index ranges to named variable blocks.

    After all entities have registered their variables, n_vars gives
    the total number of optimisation variables.
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._blocks: dict[str, range] = {}
        self._n = 0

    def register(self, name: str, count: int) -> range:
        """
        Reserve `count` consecutive indices for a named block.

        Returns the range of indices assigned to this block.
        """
        r = range(self._n, self._n + count)
        self._blocks[name] = r
        self._n += count
        return r

    @property
    def n_vars(self) -> int:
        """Total number of registered variables."""
        return self._n

    def __getitem__(self, name: str) -> range:
        return self._blocks[name]


# ---------------------------------------------------------------------------
# Constraint accumulators
# ---------------------------------------------------------------------------

@dataclass
class Constraints:
    """Accumulates linear constraints A_row @ x in [lo, hi]."""

    rows: list[np.ndarray] = field(default_factory=list)
    lo: list[float] = field(default_factory=list)
    hi: list[float] = field(default_factory=list)

    def add_eq(self, row: np.ndarray, val: float) -> None:
        """Add equality constraint row @ x == val."""
        self.rows.append(row)
        self.lo.append(val)
        self.hi.append(val)

    def add_ineq(self, row: np.ndarray, lo: float, hi: float) -> None:
        """Add inequality constraint lo <= row @ x <= hi."""
        self.rows.append(row)
        self.lo.append(lo)
        self.hi.append(hi)


# ---------------------------------------------------------------------------
# Abstract entity
# ---------------------------------------------------------------------------

class Entity(ABC):
    """
    Abstract base class for all household entities.

    Subclasses implement register() to claim variable indices and
    contribute() to add constraints and objective terms.
    """

    @abstractmethod
    def register(self, reg: VariableRegistry, T: int) -> None:
        """
        Register decision and state variables with the registry.

        Called once before optimisation to assign variable indices.
        """
        ...

    @abstractmethod
    def contribute(
        self,
        reg: VariableRegistry,
        constraints: Constraints,
        objective: np.ndarray,
        T: int,
        dt: float,
    ) -> None:
        """
        Add constraints and objective terms for this entity.

        Args:
            reg: Variable registry with assigned indices.
            constraints: Constraint accumulator to append to.
            objective: Objective coefficient vector to modify in-place.
            T: Number of timesteps.
            dt: Timestep duration [h].
        """
        ...

    @abstractmethod
    def net_power(self, reg: VariableRegistry, x: np.ndarray, T: int) -> np.ndarray:
        """
        Return net power consumption [kW] for each timestep given solution x.

        Positive = consuming from grid, negative = feeding into grid.
        """
        ...


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------

class Battery(Entity):
    """
    Home battery storage.

    Decision variable x_bat^t in [−discharge_max, charge_max]:
      positive = charging (consuming power)
      negative = discharging (providing power)

    State: soc_bat^t in [soc_min, capacity]
    Dynamics: soc_bat^{t+1} = soc_bat^t + x_bat^t * dt
    """

    def __init__(
        self,
        capacity: float,
        soc_init: float,
        charge_max: float,
        discharge_max: float,
        soc_min: float = 0.0,
        efficiency: float = 1.0,
    ) -> None:
        """
        Args:
            capacity: Maximum SoC [kWh].
            soc_init: Initial SoC [kWh].
            charge_max: Maximum charging power [kW].
            discharge_max: Maximum discharging power [kW].
            soc_min: Minimum allowed SoC [kWh].
            efficiency: Round-trip efficiency [0, 1].
        """
        self.capacity = capacity
        self.soc_init = soc_init
        self.charge_max = charge_max
        self.discharge_max = discharge_max
        self.soc_min = soc_min
        self.efficiency = efficiency

        self._x: Optional[range] = None    # decision variable indices
        self._soc: Optional[range] = None  # state variable indices

    def register(self, reg: VariableRegistry, T: int) -> None:
        """Register x_bat^t (T vars) and soc_bat^t (T+1 vars)."""
        self._x = reg.register("battery_x", T)
        self._soc = reg.register("battery_soc", T + 1)

    def contribute(
        self,
        reg: VariableRegistry,
        constraints: Constraints,
        objective: np.ndarray,
        T: int,
        dt: float,
    ) -> None:
        """Add SoC dynamics and bounds constraints."""
        n = objective.size

        # Fix initial SoC
        row = np.zeros(n)
        row[self._soc[0]] = 1.0
        constraints.add_eq(row, self.soc_init)

        for t in range(T):
            # SoC dynamics: soc[t+1] = soc[t] + x[t] * dt
            row = np.zeros(n)
            row[self._soc[t + 1]] = 1.0
            row[self._soc[t]] = -1.0
            row[self._x[t]] = -dt * self.efficiency
            constraints.add_eq(row, 0.0)

    def bounds(self, T: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (lb, ub) for battery variables."""
        lb = np.concatenate([
            np.full(T, -self.discharge_max),
            np.full(T + 1, self.soc_min),
        ])
        ub = np.concatenate([
            np.full(T, self.charge_max),
            np.full(T + 1, self.capacity),
        ])
        return lb, ub

    def net_power(self, reg: VariableRegistry, x: np.ndarray, T: int) -> np.ndarray:
        """Return x_bat^t — positive = charging = consuming."""
        return np.array([x[self._x[t]] for t in range(T)])


# ---------------------------------------------------------------------------
# Electric Vehicle
# ---------------------------------------------------------------------------

class ElectricVehicle(Entity):
    """
    Electric vehicle battery.

    Like Battery but with an optional deadline constraint on SoC,
    and no discharging by default (no V2G).
    """

    def __init__(
        self,
        capacity: float,
        soc_init: float,
        charge_max: float,
        soc_min: float = 0.0,
        target_soc: Optional[float] = None,
        target_timestep: Optional[int] = None,
        efficiency: float = 1.0,
    ) -> None:
        """
        Args:
            capacity: Maximum SoC [kWh].
            soc_init: Initial SoC [kWh].
            charge_max: Maximum charging power [kW].
            soc_min: Minimum allowed SoC [kWh].
            target_soc: Desired minimum SoC at target_timestep [kWh].
            target_timestep: Timestep by which target_soc must be reached.
            efficiency: Charging efficiency [0, 1].
        """
        self.capacity = capacity
        self.soc_init = soc_init
        self.charge_max = charge_max
        self.soc_min = soc_min
        self.target_soc = target_soc
        self.target_timestep = target_timestep
        self.efficiency = efficiency

        self._x: Optional[range] = None
        self._soc: Optional[range] = None

    def register(self, reg: VariableRegistry, T: int) -> None:
        """Register x_ev^t (T vars) and soc_ev^t (T+1 vars)."""
        self._x = reg.register("ev_x", T)
        self._soc = reg.register("ev_soc", T + 1)

    def contribute(
        self,
        reg: VariableRegistry,
        constraints: Constraints,
        objective: np.ndarray,
        T: int,
        dt: float,
    ) -> None:
        """Add SoC dynamics and optional deadline constraint."""
        n = objective.size

        # Fix initial SoC
        row = np.zeros(n)
        row[self._soc[0]] = 1.0
        constraints.add_eq(row, self.soc_init)

        for t in range(T):
            # SoC dynamics
            row = np.zeros(n)
            row[self._soc[t + 1]] = 1.0
            row[self._soc[t]] = -1.0
            row[self._x[t]] = -dt * self.efficiency
            constraints.add_eq(row, 0.0)

        # Deadline constraint
        if self.target_soc is not None and self.target_timestep is not None:
            deadline = min(self.target_timestep, T)
            row = np.zeros(n)
            row[self._soc[deadline]] = 1.0
            constraints.add_ineq(row, self.target_soc, np.inf)

    def bounds(self, T: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (lb, ub) — no discharging (x_ev >= 0)."""
        lb = np.concatenate([
            np.zeros(T),
            np.full(T + 1, self.soc_min),
        ])
        ub = np.concatenate([
            np.full(T, self.charge_max),
            np.full(T + 1, self.capacity),
        ])
        return lb, ub

    def net_power(self, reg: VariableRegistry, x: np.ndarray, T: int) -> np.ndarray:
        """Return x_ev^t — always non-negative (only charging)."""
        return np.array([x[self._x[t]] for t in range(T)])


# ---------------------------------------------------------------------------
# Heat Pump
# ---------------------------------------------------------------------------

class HeatPump(Entity):
    """
    Heat pump with room temperature dynamics.

    Decision variable x_hp^t in [0, x_hp_max]: electrical power input [kW].

    State: temp_in^t
    Dynamics:
        temp_in^{t+1} = temp_in^t
                       + cop^t * x_hp^t * dt / C_therm
                       - lambda_ * (temp_in^t - temp_out^t) * dt

    COP is precomputed per timestep using temp_in^t ≈ temp_min^t (lower bound)
    to maintain linearity. This is conservative (underestimates COP slightly).

    Comfort constraint: temp_in^t in [temp_min^t, temp_max^t]
    """

    def __init__(
        self,
        temp_init: float,
        temp_min: np.ndarray,
        temp_max: np.ndarray,
        temp_out: np.ndarray,
        max_power: float,
        C_therm: float,
        lambda_: float,
        cop_eta: float = 0.4,
    ) -> None:
        """
        Args:
            temp_init: Initial indoor temperature [°C].
            temp_min: Minimum allowed indoor temperature per timestep [°C].
            temp_max: Maximum allowed indoor temperature per timestep [°C].
            temp_out: Outdoor temperature per timestep [°C] (from forecast).
            max_power: Maximum electrical input power [kW].
            C_therm: Thermal capacity of house [kWh/°C].
            lambda_: Heat loss coefficient [kW/°C].
            cop_eta: Carnot efficiency factor for COP model (typical: 0.3-0.5).
        """
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.temp_out = temp_out
        self.max_power = max_power
        self.C_therm = C_therm
        self.lambda_ = lambda_
        self.cop_eta = cop_eta

        self._x: Optional[range] = None
        self._temp: Optional[range] = None

    def _compute_cop(self, T: int) -> np.ndarray:
        """
        Precompute COP per timestep using temp_min as indoor temperature.

        Uses simplified Carnot model: COP = eta * T_in_K / (T_in_K - T_out_K).
        Clipped to [1, 6] for physical plausibility.
        """
        T_in_K = self.temp_min[:T] + 273.15
        T_out_K = self.temp_out[:T] + 273.15
        delta = np.maximum(T_in_K - T_out_K, 1.0)  # avoid division by zero
        cop = self.cop_eta * T_in_K / delta
        return np.clip(cop, 1.0, 6.0)

    def register(self, reg: VariableRegistry, T: int) -> None:
        """Register x_hp^t (T vars) and temp_in^t (T+1 vars)."""
        self._x = reg.register("hp_x", T)
        self._temp = reg.register("hp_temp", T + 1)

    def contribute(
        self,
        reg: VariableRegistry,
        constraints: Constraints,
        objective: np.ndarray,
        T: int,
        dt: float,
    ) -> None:
        """Add temperature dynamics and comfort constraints."""
        n = objective.size
        cop = self._compute_cop(T)

        # Fix initial temperature
        row = np.zeros(n)
        row[self._temp[0]] = 1.0
        constraints.add_eq(row, self.temp_init)

        for t in range(T):
            # Temperature dynamics (linear since cop^t is precomputed)
            # temp[t+1] = temp[t] * (1 - lambda*dt/C) + cop*x*dt/C + lambda*dt*T_out/C
            row = np.zeros(n)
            row[self._temp[t + 1]] = 1.0
            row[self._temp[t]] = -(1.0 - self.lambda_ * dt / self.C_therm)
            row[self._x[t]] = -cop[t] * dt / self.C_therm
            rhs = self.lambda_ * dt * self.temp_out[t] / self.C_therm
            constraints.add_eq(row, rhs)

            # Comfort bounds on temp[t+1]
            row = np.zeros(n)
            row[self._temp[t + 1]] = 1.0
            constraints.add_ineq(row, self.temp_min[t], self.temp_max[t])

    def bounds(self, T: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (lb, ub) for heat pump variables."""
        lb = np.concatenate([np.zeros(T), np.full(T + 1, -np.inf)])
        ub = np.concatenate([np.full(T, self.max_power), np.full(T + 1, np.inf)])
        return lb, ub

    def net_power(self, reg: VariableRegistry, x: np.ndarray, T: int) -> np.ndarray:
        """Return x_hp^t — electrical power consumed."""
        return np.array([x[self._x[t]] for t in range(T)])
