# EnergyPilot

EnergyPilot is a modular framework for cost-optimal household energy scheduling. It coordinates controllable loads — battery storage, electric vehicle charging, and heat pump operation — to minimise electricity costs by exploiting forecast electricity prices and local PV generation.

## Motivation

Electricity prices vary significantly throughout the day. Modern households have increasing flexibility in *when* they consume energy: a battery can be charged during cheap hours, an EV does not need to start charging the moment it is plugged in, and a heat pump can pre-heat a building before prices rise. EnergyPilot makes these decisions automatically, guided by forecasts of prices, solar generation, and weather.

## Approach

The scheduling problem is formulated as a **Mixed Integer Linear Program (MILP)** solved within a **Model Predictive Control (MPC)** loop. At each timestep, the current system state is observed, a forecast trajectory is produced for the planning horizon, and the MILP finds optimal power setpoints for all controllable devices. Only the first step's commands are applied — then the loop repeats with fresh observations.

This rolling-horizon approach lets the system continuously react to new information without committing to a fixed long-term plan.

## Energy Model

A single power balance ties everything together:

```
p_buy − p_sell = load + x_bat + x_ev + x_hp − gen
```

Each controllable device has a decision variable (power setpoint), a state variable (SoC or temperature), and physical dynamics encoded as linear constraints. The MILP minimises total electricity cost — buying cheap, selling surplus, and pre-conditioning devices ahead of expensive periods.

Base load and PV generation are uncontrollable forecast signals, not decision variables.

![optimisation diagram](diagrams/optimization_components.svg)

### Battery
- Decision variable: `x_bat^t ∈ [−discharge_max, charge_max]` (positive = charging)
- State: `soc_bat^t ∈ [soc_min, capacity]`
- Dynamics: `soc^{t+1} = soc^t + x_bat^t · dt`

### Electric Vehicle
- Decision variable: `x_ev^t ∈ [−discharge_max, charge_max]`
- State: `soc_ev^t ∈ [0, capacity]`
- Dynamics: same as battery
- Deadline constraint: `soc_ev^{T*} ≥ target_soc`

### Heat Pump
- Decision variable: `x_hp^t ∈ [0, max_power]` (electrical input)
- State: indoor temperature `temp_in^t`
- Dynamics:
  ```
  temp_in^{t+1} = temp_in^t
                + cop^t · x_hp^t · dt / C_therm
                − λ · (temp_in^t − temp_out^t) · dt
  ```
- Comfort constraint: `temp_in^t ∈ [temp_min^t, temp_max^t]`
- COP is precomputed per timestep from the outdoor temperature forecast (Carnot model), keeping the dynamics linear.

## Forecasting

Each signal — prices, load, generation, outdoor temperature — is backed by a forecaster that maps observed history to a trajectory over the planning horizon. The first value is always pinned to the latest observation; subsequent steps use signal-specific patterns (diurnal curves, Gaussian bell for PV, etc.).

The forecasting layer is intentionally decoupled: any forecaster — a trained ML model, a weather API, a simple heuristic — can be dropped in without touching the optimization layer.

## Synthetic Simulation

For development and testing, all sensors, actuators, and signals are replaced with synthetic counterparts. Device physics (SoC evolution, thermal dynamics) are simulated locally. Signal dynamics (diurnal price and load patterns) are driven by a step hook in `SyntheticMPC`. The real-world interfaces remain unchanged — the distinction between synthetic and production lives entirely inside individual sensor and actuator implementations.

## Project Structure

```
EnergyPilot/
├── main.py                  Entry point
├── plot.py                  Result visualisation
├── real_world_interfaces/   Sensor and Actuator abstractions
├── synthetic/               Synthetic simulation components
├── forecasting/             Time series, forecaster interface and implementations
├── optimization/
│   ├── milp/                Device models and MILP formulation
│   └── mpc/                 MPC loop orchestration
└── outputs/                 Saved plots
```

## Dependencies

- Python ≥ 3.11
- `numpy`
- `scipy` ≥ 1.9 — MILP solver (HiGHS backend via `scipy.optimize.milp`)
- `matplotlib`

## Limitations & Next Steps

- **Forecast model** — currently uses synthetic patterns; replace with trained models for production use
- **Synthetic environment** — the scattered synthetic state objects are a candidate for consolidation into a unified environment class
- **Stochastic optimisation** — optimising over multiple forecast scenarios simultaneously would yield more robust decisions
- **Schedulable appliances** — washing machines, dishwashers, etc. fit naturally as one-shot binary-scheduled loads
- **Nonlinear COP** — COP is currently linearised; a nonlinear solver would handle the full temperature-dependent model
