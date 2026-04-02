"""
MPC loop for household energy scheduling.

At each timestep:
  1. Populate t=0 with real observed values (std=0)
  2. Sample a forecast trajectory for t=1..T
  3. Solve MILP over full horizon
  4. Apply first timestep decisions
  5. Advance state
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optimization.entities import Battery, ElectricVehicle, HeatPump, BaseLoad, PVGenerator
from optimization.forecast import ForecastParams, ForecastTrajectory
from optimization.household import Household, OptimisationResult


def make_forecast_params(
    T: int,
    dt: float,
    current_obs: dict,
    rng: np.random.Generator,
) -> ForecastParams:
    """
    Build forecast parameters for the planning horizon.

    t=0 uses real observations (std=0).
    t>0 uses simple synthetic forecast distributions.

    Args:
        T: Planning horizon in timesteps.
        dt: Timestep duration [h].
        current_obs: Dictionary of current real observations.
        rng: Random number generator.

    Returns:
        ForecastParams with mean/std arrays of shape (T,).
    """
    t = np.arange(T)

    # Buy price: higher morning/evening, lower midday, slight noise for t>0
    price_buy_mean = 0.28 + 0.08 * np.cos(2 * np.pi * t / T)
    price_buy_mean[0] = current_obs["price_buy"]
    price_buy_std = np.concatenate([[0.0], np.full(T - 1, 0.02)])

    # Sell price: fixed feed-in tariff with small uncertainty
    price_sell_mean = np.full(T, 0.08)
    price_sell_mean[0] = current_obs["price_sell"]
    price_sell_std = np.concatenate([[0.0], np.full(T - 1, 0.005)])

    # PV: bell curve peaking at midday
    pv_mean = 3.5 * np.exp(-0.5 * ((t - T * 0.5) / (T * 0.18)) ** 2)
    pv_mean[0] = current_obs["pv"]
    pv_std = np.concatenate([[0.0], 0.2 * pv_mean[1:] + 0.05])

    # Base load: slightly higher in morning and evening
    load_mean = 0.6 + 0.25 * np.abs(np.sin(np.pi * t / T))
    load_mean[0] = current_obs["load"]
    load_std = np.concatenate([[0.0], np.full(T - 1, 0.1)])

    # Outdoor temperature: warming through the day
    temp_out_mean = 4.0 + 4.0 * np.sin(np.pi * t / T)
    temp_out_mean[0] = current_obs["temp_out"]
    temp_out_std = np.concatenate([[0.0], np.full(T - 1, 1.0)])

    return ForecastParams(
        T=T, dt=dt,
        price_buy_mean=price_buy_mean,
        price_buy_std=price_buy_std,
        price_sell_mean=price_sell_mean,
        price_sell_std=price_sell_std,
        pv_mean=pv_mean,
        pv_std=pv_std,
        load_mean=load_mean,
        load_std=load_std,
        temp_out_mean=temp_out_mean,
        temp_out_std=temp_out_std,
    )


def run_mpc(
    T_total: int = 12,
    T_horizon: int = 12,
    dt: float = 0.5,
    seed: int = 0,
) -> None:
    """
    Run MPC loop over T_total timesteps.

    Args:
        T_total: Total number of MPC steps to simulate.
        T_horizon: Planning horizon length in timesteps.
        dt: Timestep duration [h].
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Initial state
    state = {
        "soc_bat": 4.0,
        "soc_ev": 15.0,
        "temp_in": 21.0,
    }

    # Simulated real observations (would come from sensors in production)
    def get_observations(step: int) -> dict:
        """Return synthetic real observations for the current timestep."""
        return {
            "price_buy": 0.28 + 0.08 * np.cos(2 * np.pi * step / T_total) + rng.normal(0, 0.01),
            "price_sell": 0.08,
            "pv": max(0.0, 3.5 * np.exp(-0.5 * ((step - T_total * 0.5) / (T_total * 0.18)) ** 2)
                      + rng.normal(0, 0.1)),
            "load": max(0.0, 0.6 + 0.25 * abs(np.sin(np.pi * step / T_total)) + rng.normal(0, 0.05)),
            "temp_out": 4.0 + 4.0 * np.sin(np.pi * step / T_total) + rng.normal(0, 0.3),
        }

    history = {
        "soc_bat": [state["soc_bat"]],
        "soc_ev": [state["soc_ev"]],
        "temp_in": [state["temp_in"]],
        "p_buy": [], "p_sell": [],
        "x_bat": [], "x_ev": [], "x_hp": [],
        "price_buy": [], "pv": [], "load": [],
    }

    print(f"{'t':>3}  {'Cost':>7}  {'SoC bat':>8}  {'SoC EV':>7}  {'Temp':>6}  {'p_buy':>6}  {'p_sell':>6}")
    print("─" * 65)

    for step in range(T_total):
        T_h = min(T_horizon, T_total - step)
        obs = get_observations(step)

        params = make_forecast_params(T_h, dt, obs, rng)
        trajectory = ForecastTrajectory.sample(params, rng)

        # Rebuild entities with current state
        battery = Battery(
            capacity=10.0, soc_init=state["soc_bat"],
            charge_max=3.0, discharge_max=3.0, soc_min=1.0,
        )
        ev = ElectricVehicle(
            capacity=60.0, soc_init=state["soc_ev"],
            charge_max=11.0, soc_min=0.0,
            target_soc=50.0, target_timestep=T_h - 1,
        )
        heat_pump = HeatPump(
            temp_init=state["temp_in"],
            temp_min=np.full(T_h, 20.0),
            temp_max=np.full(T_h, 23.0),
            temp_out=trajectory.temp_out,
            max_power=3.0,
            C_therm=5.0,
            lambda_=0.25,
            cop_eta=0.4,
        )
        base_load = BaseLoad(trajectory.load)
        pv = PVGenerator(trajectory.pv)

        household = Household(entities=[battery, ev, heat_pump, base_load, pv])
        result: OptimisationResult = household.solve(trajectory)

        if not result.success:
            print(f"{step:>3}  {'—':>7}  {state['soc_bat']:>8.2f}  {state['soc_ev']:>7.2f}  "
                  f"{state['temp_in']:>6.1f}  FAILED: {result.message}")
            continue

        # Advance state using first timestep
        state["soc_bat"] = float(result.states['soc_bat'][1])
        state["soc_ev"] = float(result.states['soc_ev'][1])
        state["temp_in"] = float(result.states['temp_in'][1])

        history["soc_bat"].append(state["soc_bat"])
        history["soc_ev"].append(state["soc_ev"])
        history["temp_in"].append(state["temp_in"])
        history["p_buy"].append(float(result.p_buy[0]))
        history["p_sell"].append(float(result.p_sell[0]))
        history["x_bat"].append(float(result.decisions['x_bat'][0]))
        history["x_ev"].append(float(result.decisions['x_ev'][0]))
        history["x_hp"].append(float(result.decisions['x_hp'][0]))
        history["price_buy"].append(obs["price_buy"])
        history["pv"].append(obs["pv"])
        history["load"].append(obs["load"])

        print(f"{step:>3}  {result.total_cost:>7.4f}  {state['soc_bat']:>8.2f}  "
              f"{state['soc_ev']:>7.2f}  {state['temp_in']:>6.1f}  "
              f"{result.p_buy[0]:>6.2f}  {result.p_sell[0]:>6.2f}")

    plot_results(history, dt)


def plot_results(history: dict, dt: float) -> None:
    """Plot MPC simulation results."""
    steps = np.arange(len(history["p_buy"])) * dt

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("Household Energy Scheduling — MPC POC", fontsize=13, fontweight="bold")

    # SoC
    ax = axes[0]
    ax.plot(steps, history["soc_bat"][1:], "o-", label="Battery SoC [kWh]", color="#2196F3")
    ax.plot(steps, history["soc_ev"][1:], "s-", label="EV SoC [kWh]", color="#4CAF50")
    ax.set_ylabel("SoC [kWh]")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Temperature
    ax = axes[1]
    ax.plot(steps, history["temp_in"][1:], "o-", color="#F44336", label="Indoor temp [°C]")
    ax.axhline(20.0, linestyle="--", color="gray", alpha=0.6, label="Min 20°C")
    ax.axhline(23.0, linestyle="--", color="gray", alpha=0.4, label="Max 23°C")
    ax.set_ylabel("Temperature [°C]")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Power decisions
    ax = axes[2]
    ax.bar(steps, history["x_hp"], width=0.4*dt, label="Heat pump [kW]", color="#FF9800", alpha=0.8)
    ax.bar(steps, history["x_bat"], width=0.4*dt, bottom=history["x_hp"],
           label="Battery [kW]", color="#2196F3", alpha=0.8)
    ax.bar(steps, history["x_ev"], width=0.4*dt,
           bottom=np.array(history["x_hp"]) + np.array(history["x_bat"]),
           label="EV [kW]", color="#4CAF50", alpha=0.8)
    ax.set_ylabel("Power [kW]")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Grid + price
    ax = axes[3]
    ax2 = ax.twinx()
    ax.bar(steps - 0.2*dt, history["p_buy"], width=0.35*dt,
           label="Grid buy [kW]", color="#E91E63", alpha=0.8)
    ax.bar(steps + 0.2*dt, history["p_sell"], width=0.35*dt,
           label="Grid sell [kW]", color="#9C27B0", alpha=0.8)
    ax2.plot(steps, history["price_buy"], "k--", alpha=0.5, label="Buy price [€/kWh]")
    ax.set_ylabel("Power [kW]")
    ax2.set_ylabel("Price [€/kWh]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path("outputs/mpc_test_results.png")
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print("\nPlot saved under " + str(save_path.absolute()))


if __name__ == "__main__":
    run_mpc()
