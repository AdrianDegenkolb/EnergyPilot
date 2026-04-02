# EnergyPilot

EnergyPilot is a modular framework for cost-optimal household energy scheduling. It coordinates controllable loads — battery storage, electric vehicle charging, and heat pump operation — to minimise electricity costs by exploiting forecast electricity prices and local PV generation.

## Motivation

Electricity prices vary significantly throughout the day. At the same time, modern households have increasing flexibility in *when* they consume energy: a battery can be charged during cheap hours, an EV does not need to start charging the moment it is plugged in, and a heat pump can pre-heat a building before prices rise. EnergyPilot makes these decisions automatically, guided by forecasts of prices, solar generation, and weather.

## Core Idea

The scheduling problem is formulated as a **Mixed Integer Linear Program (MILP)** solved within a **Model Predictive Control (MPC)** loop:

- At each timestep, the current system state is observed (battery SoC, EV SoC, indoor temperature).
- A forecast trajectory is sampled for the planning horizon — prices, PV generation, base load, and outdoor temperature.
- A MILP is solved over the full horizon, finding optimal decisions for all future timesteps.
- Only the first timestep's decisions are applied. The process then repeats with fresh observations and an updated forecast.

This rolling-horizon approach means the system continuously reacts to new information without committing to a fixed plan far into the future.

## Energy Model

Each controllable asset in the household is modelled as an **entity** with its own decision variable, physical dynamics, and constraints. The grid is not explicitly modelled as a network — instead, a single power balance equation ties everything together:

```
p_buy^t − p_sell^t = load^t + x_hp^t + x_ev^t + x_bat^t − gen^t
```

Where `p_buy` and `p_sell` are auxiliary variables representing grid import and export, and the right-hand side is the net power demand from all controllable and uncontrollable assets minus PV generation.

The **objective** is to minimise total electricity cost over the horizon:

```
min  Σ_t  price_buy^t · p_buy^t  −  price_sell^t · p_sell^t
```

### Entities

**Battery**
- Decision variable: `x_bat^t ∈ [−discharge_max, charge_max]` (positive = charging)
- State: `soc_bat^t ∈ [soc_min, capacity]`
- Dynamics: `soc^{t+1} = soc^t + x_bat^t · dt`

**Electric Vehicle**
- Decision variable: `x_ev^t ∈ [0, charge_max]` (charging only, no V2G)
- State: `soc_ev^t ∈ [0, capacity]`
- Dynamics: same as battery
- Deadline constraint: `soc_ev^{T*} ≥ target_soc`

**Heat Pump**
- Decision variable: `x_hp^t ∈ [0, max_power]` (electrical input)
- State: indoor temperature `temp_in^t`
- Dynamics:
  ```
  temp_in^{t+1} = temp_in^t
                + cop^t · x_hp^t · dt / C_therm
                − λ · (temp_in^t − temp_out^t) · dt
  ```
- Comfort constraint: `temp_in^t ∈ [temp_min^t, temp_max^t]`
- COP is precomputed per timestep from outdoor temperature (Carnot model), keeping the dynamics linear.

**Base Load & PV**
Uncontrollable base load and PV generation are treated as forecast parameters, not decision variables. The base load aggregates all non-schedulable consumption (fridge, standby, etc.).

## Forecast

Uncertainty in prices, PV output, and weather is modelled explicitly. Each quantity is represented as a **normal distribution** per timestep, from which a trajectory is sampled at the start of each MPC step:

- `t = 0`: real observed values (zero uncertainty)
- `t > 0`: sampled from forecast distributions

This design is intentionally decoupled — the forecast module can be replaced with a trained model (e.g. a price forecaster or irradiance model) without changing the optimisation layer.

## MPC Loop

```
for each timestep t:
    1. Observe real values at t=0
    2. Sample forecast trajectory for t=1..T
    3. Rebuild entities with current state as initial conditions
    4. Formulate and solve MILP over horizon [t, t+T]
    5. Apply decisions from first timestep
    6. Advance state (soc_bat, soc_ev, temp_in)
```

## Project Structure

```
EnergyPilot/
├── forecast.py     # ForecastParams and ForecastTrajectory
├── entities.py     # Battery, ElectricVehicle, HeatPump + VariableRegistry
├── household.py    # Household: assembles entities and solves MILP
└── main.py         # MPC loop, synthetic scenario, plotting
```

## Dependencies

- Python ≥ 3.11
- `numpy`
- `scipy` ≥ 1.9 (provides `scipy.optimize.milp` backed by HiGHS)
- `matplotlib`

## Limitations & Next Steps

- **Forecast model**: Currently uses synthetic distributions. The next step is to plug in real trained models for electricity prices and PV irradiance.
- **Stochastic optimisation**: The current formulation optimises against a single sampled trajectory. Scenario-based stochastic programming (optimising over multiple sampled trajectories simultaneously) would yield more robust decisions.
- **Schedulable appliances**: Washing machines, dishwashers, etc. can be modelled as binary-scheduled loads with a one-shot constraint, adding a MILP integer variable per appliance per day.
- **V2G**: Vehicle-to-grid discharge can be enabled by extending the EV decision variable to negative values.
- **Nonlinear COP**: COP is currently linearised by fixing indoor temperature to its lower bound. A nonlinear solver (e.g. CasADi + IPOPT) would handle the full temperature-dependent model.
