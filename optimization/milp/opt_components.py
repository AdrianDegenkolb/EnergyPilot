"""
Optimization components that make up a household energy system.

Each OptimizationComponent represents a controllable physical device
(battery, EV, heat pump). It contributes to the household MILP by:
  - registering decision and state variables
  - adding constraints (dynamics, bounds, comfort, deadlines)
  - contributing to the net power balance

State key convention (set_initial_state / extract_command):
  BatteryModel:  set_initial_state receives current SoC [kWh]
                 extract_command returns scheduled charging power [kW]
  EVModel:       set_initial_state receives current SoC [kWh]
                 extract_command returns scheduled charging power [kW]
  HeatPumpModel: set_initial_state receives current indoor temperature [°C]
                 extract_command returns scheduled electrical input power [kW]

Physical interpretation of commands:
  The command is the power setpoint sent to the physical device.
  Positive = consuming from grid (charging / heating).
  Negative = injecting into grid (discharging).
  The SyntheticActuator's update_fn translates this power into a state change.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from forecasting.time_series import TimeSeries
    from .household import HouseholdResult


# ---------------------------------------------------------------------------
# Variable registry
# ---------------------------------------------------------------------------

class VariableRegistry:
    """Assigns contiguous index ranges to named variable blocks."""

    def __init__(self) -> None:
        self._blocks: dict[str, range] = {}
        self._n = 0

    def register(self, name: str, count: int) -> range:
        r = range(self._n, self._n + count)
        self._blocks[name] = r
        self._n += count
        return r

    @property
    def n_vars(self) -> int:
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
        self.rows.append(row)
        self.lo.append(val)
        self.hi.append(val)

    def add_ineq(self, row: np.ndarray, lo: float, hi: float) -> None:
        self.rows.append(row)
        self.lo.append(lo)
        self.hi.append(hi)


# ---------------------------------------------------------------------------
# Abstract optimization component
# ---------------------------------------------------------------------------

class OptimizationComponent(ABC):
    """
    Abstract base for all household optimization components. These components act as configurable loads meaning we can schedule
    their power consumption/production (while respecting certain constraints) to optimize the household's energy costs.

    Represents a controllable physical device in the household. Each component:
      - registers its variables with the MILP registry
      - contributes constraints and bounds
      - reports its net power contribution to the grid balance
      - accepts its current measured state before each solve
      - extracts the optimal power command after each solve
    """

    @abstractmethod
    def register(self, reg: VariableRegistry, T: int) -> None:
        """Register decision and state variables."""
        ...

    @abstractmethod
    def contribute(
        self,
        reg: VariableRegistry,
        constraints: Constraints,
        objective: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        T: int,
        dt: float,
    ) -> None:
        """Add constraints, bounds, and objective terms."""
        ...

    @abstractmethod
    def net_power(self, n: int, t: int) -> tuple[np.ndarray, float]:
        """
        Return (coeffs, rhs) describing this device's net power at timestep t.

        Net power = coeffs @ x + rhs.
        Positive = consuming from grid. Negative = feeding into grid.
        """
        ...

    @abstractmethod
    def set_initial_state(self, value: float) -> None:
        """
        Seed the initial state from the current sensor reading.

        Called by PhysicalEntity.observe() once per MPC step before solve().

        Args:
            value: Current measured state (SoC [kWh] or temperature [°C]).
        """
        ...

    @abstractmethod
    def extract_command(self, result: HouseholdResult) -> float:
        """
        Extract the optimal first-step power command from the solve result.

        Called by PhysicalEntity.act() after each solve. The returned value
        is passed directly to the Actuator as the power setpoint [kW].

        Args:
            result: Full MILP result from HouseholdOptimizationProblem.solve().
        """
        ...


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------

class BatteryModel(OptimizationComponent):
    """
    Home battery storage.

    Decision variable x_bat^t in [−discharge_max, charge_max]:
      positive = charging (consuming power from grid)
      negative = discharging (injecting power into grid)

    State: soc_bat^t in [soc_min, capacity]
    Dynamics: soc_bat^{t+1} = soc_bat^t + x_bat^t * dt * efficiency

    Command sent to actuator: x_bat^0 (first-step charging power [kW]).
    The SyntheticActuator's update_fn must implement the same SoC dynamics.
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
        self.capacity = capacity
        self.soc_init = soc_init
        self.charge_max = charge_max
        self.discharge_max = discharge_max
        self.soc_min = soc_min
        self.efficiency = efficiency

        self._x: Optional[range] = None
        self._soc: Optional[range] = None

    def register(self, reg: VariableRegistry, T: int) -> None:
        self._x = reg.register("battery_x", T)
        self._soc = reg.register("battery_soc", T + 1)

    def contribute(self, reg, constraints, objective, lb, ub, T, dt) -> None:
        n = objective.size

        for t in range(T):
            lb[self._x[t]] = -self.discharge_max
            ub[self._x[t]] = self.charge_max
        for t in range(T + 1):
            lb[self._soc[t]] = self.soc_min
            ub[self._soc[t]] = self.capacity

        row = np.zeros(n)
        row[self._soc[0]] = 1.0
        constraints.add_eq(row, self.soc_init)

        for t in range(T):
            row = np.zeros(n)
            row[self._soc[t + 1]] = 1.0
            row[self._soc[t]] = -1.0
            row[self._x[t]] = -dt * self.efficiency
            constraints.add_eq(row, 0.0)

    def net_power(self, n: int, t: int) -> tuple[np.ndarray, float]:
        coeffs = np.zeros(n)
        coeffs[self._x[t]] = 1.0
        return coeffs, 0.0

    def set_initial_state(self, value: float) -> None:
        self.soc_init = value

    def extract_command(self, result: HouseholdResult) -> float:
        return float(result.decisions["x_bat"][0])


# ---------------------------------------------------------------------------
# Electric Vehicle
# ---------------------------------------------------------------------------

class EVModel(OptimizationComponent):
    """
    Electric vehicle battery.

    Like BatteryModel but with an optional deadline constraint on SoC.
    No V2G by default (discharge_max=0).

    target_timestep is the absolute simulation step by which target_soc must
    be reached. Set at construction; the MPC horizon clips it in contribute().

    Command sent to actuator: x_ev^0 (first-step charging power [kW]).
    """

    def __init__(
        self,
        capacity: float,
        soc_init: float,
        charge_max: float,
        discharge_max: float = 0.0,
        soc_min: float = 0.0,
        target_soc: Optional[float] = None,
        target_timestep: Optional[int] = None,
        efficiency: float = 1.0,
    ) -> None:
        self.capacity = capacity
        self.soc_init = soc_init
        self.charge_max = charge_max
        self.discharge_max = discharge_max
        self.soc_min = soc_min
        self.target_soc = target_soc
        self.target_timestep = target_timestep
        self.efficiency = efficiency

        self._x: Optional[range] = None
        self._soc: Optional[range] = None

    def register(self, reg: VariableRegistry, T: int) -> None:
        self._x = reg.register("ev_x", T)
        self._soc = reg.register("ev_soc", T + 1)

    def contribute(self, reg, constraints, objective, lb, ub, T, dt) -> None:
        n = objective.size

        for t in range(T):
            lb[self._x[t]] = -self.discharge_max
            ub[self._x[t]] = self.charge_max
        for t in range(T + 1):
            lb[self._soc[t]] = self.soc_min
            ub[self._soc[t]] = self.capacity

        row = np.zeros(n)
        row[self._soc[0]] = 1.0
        constraints.add_eq(row, self.soc_init)

        for t in range(T):
            row = np.zeros(n)
            row[self._soc[t + 1]] = 1.0
            row[self._soc[t]] = -1.0
            row[self._x[t]] = -dt * self.efficiency
            constraints.add_eq(row, 0.0)

        if self.target_soc is not None and self.target_timestep is not None:
            deadline = min(self.target_timestep, T)
            row = np.zeros(n)
            row[self._soc[deadline]] = 1.0
            constraints.add_ineq(row, self.target_soc, np.inf)

    def net_power(self, n: int, t: int) -> tuple[np.ndarray, float]:
        coeffs = np.zeros(n)
        coeffs[self._x[t]] = 1.0
        return coeffs, 0.0

    def set_initial_state(self, value: float) -> None:
        self.soc_init = value

    def extract_command(self, result: HouseholdResult) -> float:
        return float(result.decisions["x_ev"][0])


# ---------------------------------------------------------------------------
# Heat Pump
# ---------------------------------------------------------------------------

class HeatPumpModel(OptimizationComponent):
    """
    Heat pump with room temperature dynamics.

    Decision variable x_hp^t in [0, max_power]: electrical input power [kW].

    State: temp_in^t
    Dynamics:
        temp_in^{t+1} = temp_in^t * (1 - lambda*dt/C)
                       + cop^t * x_hp^t * dt / C
                       + lambda * dt * temp_out^t / C

    temp_out is read from a TimeSeries injected at construction — the only
    component with a TimeSeries dependency, made explicit via the constructor.

    Command sent to actuator: x_hp^0 (first-step electrical input power [kW]).
    The SyntheticActuator's update_fn must implement the same heat dynamics.
    """

    def __init__(
        self,
        temp_init: float,
        temp_min: np.ndarray,
        temp_max: np.ndarray,
        temp_out_series: TimeSeries,
        max_power: float,
        C_therm: float,
        lambda_: float,
        cop_eta: float = 0.4,
    ) -> None:
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.temp_max = temp_max
        self._temp_out_series = temp_out_series
        self.max_power = max_power
        self.C_therm = C_therm
        self.lambda_ = lambda_
        self.cop_eta = cop_eta

        self._x: Optional[range] = None
        self._temp: Optional[range] = None

    def _compute_cop(self, temp_out: np.ndarray, T: int) -> np.ndarray:
        T_in_K = self.temp_min[:T] + 273.15
        T_out_K = temp_out[:T] + 273.15
        delta = np.maximum(T_in_K - T_out_K, 1.0)
        cop = self.cop_eta * T_in_K / delta
        return np.clip(cop, 1.0, 6.0)

    def register(self, reg: VariableRegistry, T: int) -> None:
        self._x = reg.register("hp_x", T)
        self._temp = reg.register("hp_temp", T + 1)

    def contribute(self, reg, constraints, objective, lb, ub, T, dt) -> None:
        temp_out = self._temp_out_series.forecast(T, dt)
        cop = self._compute_cop(temp_out, T)
        n = objective.size

        for t in range(T):
            lb[self._x[t]] = 0.0
            ub[self._x[t]] = self.max_power

        row = np.zeros(n)
        row[self._temp[0]] = 1.0
        constraints.add_eq(row, self.temp_init)

        for t in range(T):
            row = np.zeros(n)
            row[self._temp[t + 1]] = 1.0
            row[self._temp[t]] = -(1.0 - self.lambda_ * dt / self.C_therm)
            row[self._x[t]] = -cop[t] * dt / self.C_therm
            rhs = self.lambda_ * dt * temp_out[t] / self.C_therm
            constraints.add_eq(row, rhs)

            row = np.zeros(n)
            row[self._temp[t + 1]] = 1.0
            constraints.add_ineq(row, self.temp_min[t], self.temp_max[t])

    def net_power(self, n: int, t: int) -> tuple[np.ndarray, float]:
        coeffs = np.zeros(n)
        coeffs[self._x[t]] = 1.0
        return coeffs, 0.0

    def set_initial_state(self, value: float) -> None:
        self.temp_init = value

    def extract_command(self, result: HouseholdResult) -> float:
        return float(result.decisions["x_hp"][0])
