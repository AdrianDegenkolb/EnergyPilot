"""
Entry point for the household energy scheduling POC.

Wires together environment, forecaster, entity list, and MPC runner,
then plots the results.

The household configuration is defined by the entity list passed to
MPCRunner — omit an entity to exclude it from scheduling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from optimization import Battery, ElectricVehicle, HeatPump, BaseLoad, PVGenerator, MPCHistory, MPCRunner, MPCStep, SyntheticEnvironment
from forecasting import SyntheticForecaster

# Consistent color palette
C_BATTERY = "#1976D2"
C_EV = "#388E3C"
C_HP = "#F57C00"
C_PV = "#FBC02D"
C_LOAD = "#D32F2F"
C_BUY = "#C2185B"
C_SELL = "#7B1FA2"
C_PRICE = "#455A64"
C_TEMP_IN = "#E53935"
C_TEMP_OUT = "#42A5F5"
C_MPC = "#1565C0"
C_NAIVE = "#EF6C00"


def _style_ax(ax: plt.Axes, ylabel: str, legend_loc: str = "best") -> None:
    """Apply consistent styling to an axes."""
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc=legend_loc, framealpha=0.7)


def _stacked_bars(
        ax: plt.Axes,
        steps: np.ndarray,
        components: list[tuple[str, np.ndarray, str]],
        width: float,
) -> None:
    """
    Plot stacked bars with positive and negative values stacking separately.

    Positive values stack upward, negative values stack downward —
    bars never overlap regardless of sign.

    Args:
        ax: Target axes.
        steps: Bar x-positions.
        components: List of (label, values_array, color).
        width: Bar width.
    """
    pos_bottom = np.zeros(len(steps))
    neg_bottom = np.zeros(len(steps))

    for label, values, color in components:
        values = np.asarray(values, dtype=float)
        pos = np.where(values > 0, values, 0.0)
        neg = np.where(values < 0, values, 0.0)
        if pos.any():
            ax.bar(steps, pos, width=width, bottom=pos_bottom,
                   label=label, color=color, alpha=0.85)
            pos_bottom += pos
        if neg.any():
            ax.bar(steps, neg, width=width, bottom=neg_bottom,
                   label=f"{label} (↓)" if pos.any() else label,
                   color=color, alpha=0.45)
            neg_bottom += neg


def _step_cost(s: MPCStep, dt: float) -> float:
    """
    Compute realised cost for a single executed MPC timestep.

    Uses p_buy/p_sell from the first timestep of the optimal plan,
    not total_cost which covers the entire planning horizon.
    """
    return (s.p_buy * s.observation.price_buy
            - s.p_sell * s.observation.price_sell) * dt


def _naive_step_cost(s: MPCStep, dt: float) -> float:
    """
    Compute naive baseline cost for a fair comparison with MPC.

    The naive system fulfils the same HP and EV demands as MPC but has
    no battery — it buys or sells the net demand immediately at the
    current spot price without any temporal shifting.

    net = load + x_hp + x_ev - pv
    cost = max(0, net) * price_buy - max(0, -net) * price_sell
    """
    x_hp = s.decisions.get("x_hp", 0.0)
    x_ev = s.decisions.get("x_ev", 0.0)
    net = s.observation.load + x_hp + x_ev - s.observation.pv
    buy = max(0.0, net) * s.observation.price_buy * dt
    sell = max(0.0, -net) * s.observation.price_sell * dt
    return buy - sell


def plot_results(history: MPCHistory, dt: float) -> None:
    """
    Plot a dashboard of MPC simulation results.

    Layout (4 rows):
        [1] Energy prices             [2] Load & PV generation
        [3] Grid import / export      [4] Scheduling decisions
        [5] State of charge           [6] Heat pump & temperature
        [7] Cumulative cost (full width)
    """
    steps = np.array([s.step for s in history.steps]) * dt
    w = 0.55 * dt

    # --- Extract data ---
    price_buy = np.array([s.observation.price_buy for s in history.steps])
    price_sell = np.array([s.observation.price_sell for s in history.steps])
    p_buy = np.array([s.p_buy for s in history.steps])
    p_sell = np.array([s.p_sell for s in history.steps])
    load = np.array([s.observation.load for s in history.steps])
    pv = np.array([s.observation.pv for s in history.steps])
    temp_in = np.array([s.observation.temp_in for s in history.steps])
    temp_out = np.array([s.observation.temp_out for s in history.steps])
    soc_bat = np.array([s.observation.soc_bat for s in history.steps])
    soc_ev = np.array([s.observation.soc_ev for s in history.steps])
    x_bat = np.array([s.decisions.get("x_bat", 0.0) for s in history.steps])
    x_ev = np.array([s.decisions.get("x_ev", 0.0) for s in history.steps])
    x_hp = np.array([s.decisions.get("x_hp", 0.0) for s in history.steps])

    mpc_costs = np.array([_step_cost(s, dt) for s in history.steps])
    naive_costs = np.array([_naive_step_cost(s, dt) for s in history.steps])

    # --- Layout ---
    fig = plt.figure(figsize=(14, 16))
    fig.suptitle("Household Energy Scheduling — MPC", fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])  # Energy prices
    ax2 = fig.add_subplot(gs[0, 1])  # Load & PV generation
    ax3 = fig.add_subplot(gs[1, 0])  # Grid import / export
    ax4 = fig.add_subplot(gs[1, 1])  # Scheduling decisions
    ax5 = fig.add_subplot(gs[2, 0])  # State of charge
    ax6 = fig.add_subplot(gs[2, 1])  # Heat pump & temperature
    ax7 = fig.add_subplot(gs[3, :])  # Cumulative cost (full width)

    # --- [1] Energy prices ---
    ax1.plot(steps, price_buy, "-", color=C_BUY, lw=1.5, label="Buy price [€/kWh]")
    ax1.plot(steps, price_sell, "--", color=C_SELL, lw=1.5, label="Sell price [€/kWh]")
    ax1.set_title("Energy Prices", fontsize=9, fontweight="bold")
    _style_ax(ax1, "Price [€/kWh]")

    # --- [2] Load & PV generation ---
    ax2.plot(steps, load, "-", color=C_LOAD, lw=1.5, label="Load [kW]")
    ax2.plot(steps, pv, "--", color=C_PV, lw=1.5, label="PV gen [kW]")
    ax2.set_title("Load & PV Generation", fontsize=9, fontweight="bold")
    _style_ax(ax2, "Power [kW]")

    # --- [3] Grid import / export ---
    ax3.bar(steps, p_buy, width=w, color=C_BUY, alpha=0.85, label="Import [kW]")
    ax3.bar(steps, -p_sell, width=w, color=C_SELL, alpha=0.85, label="Export [kW]")
    ax3.axhline(0, color="black", lw=0.4)
    ax3.set_title("Grid Import / Export", fontsize=9, fontweight="bold")
    _style_ax(ax3, "Power [kW]")

    # --- [4] Scheduling decisions ---
    _stacked_bars(ax4, steps, [
        ("Battery", x_bat, C_BATTERY),
        ("EV", x_ev, C_EV),
        ("Heat pump", x_hp, C_HP),
    ], width=w)
    ax4.axhline(0, color="black", lw=0.4)
    ax4.set_title("Scheduling Decisions", fontsize=9, fontweight="bold")
    _style_ax(ax4, "Scheduled power [kW]")

    # --- [5] State of charge ---
    ax5.plot(steps, soc_bat, "o-", color=C_BATTERY, lw=1.5, ms=4, label="Battery [kWh]")
    ax5.plot(steps, soc_ev, "s-", color=C_EV, lw=1.5, ms=4, label="EV [kWh]")
    ax5.set_title("State of Charge", fontsize=9, fontweight="bold")
    _style_ax(ax5, "SoC [kWh]")

    # --- [6] Heat pump & temperature (no HP bars, single combined legend) ---
    ax6.plot(steps, temp_in, "-", color=C_TEMP_IN, lw=1.5, label="Indoor [°C]")
    ax6.plot(steps, temp_out, "--", color=C_TEMP_OUT, lw=1.5, label="Outdoor [°C]")
    ax6.axhline(20.0, color=C_TEMP_IN, lw=0.8, ls=":", alpha=0.6, label="Min comfort [°C]")
    ax6.axhline(23.0, color=C_TEMP_IN, lw=0.8, ls=":", alpha=0.4, label="Max comfort [°C]")
    ax6.set_title("Heat Pump & Temperature", fontsize=9, fontweight="bold")
    _style_ax(ax6, "Temperature [°C]")

    # --- [7] Cumulative cost (full width) ---
    cum_mpc = np.cumsum(mpc_costs)
    cum_naive = np.cumsum(naive_costs)
    final_mpc = cum_mpc[-1]
    final_naive = cum_naive[-1]
    savings = final_naive - final_mpc

    ax7.plot(steps, cum_mpc, "o-", color=C_MPC, lw=2, ms=4,
             label=f"MPC  —  final: {final_mpc:.2f} €")
    ax7.plot(steps, cum_naive, "s--", color=C_NAIVE, lw=2, ms=4,
             label=f"Naive  —  final: {final_naive:.2f} €")
    ax7.fill_between(
        steps, cum_mpc, cum_naive,
        where=cum_naive >= cum_mpc,
        alpha=0.12, color="#4CAF50", label=f"Savings: {savings:.2f} €",
    )
    ax7.set_title("Cumulative Cost: MPC vs Naive", fontsize=9, fontweight="bold")
    _style_ax(ax7, "Cumulative cost [€]")

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.set_xlabel("Time [h]", fontsize=8)

    plt.savefig("outputs/mpc_v5_results.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved.")
    print(f"  MPC total cost:   {final_mpc:.4f} €")
    print(f"  Naive total cost: {final_naive:.4f} €")
    print(f"  Savings (MPC):    {savings:.4f} €")


def main() -> None:
    """Wire up components and run the MPC simulation."""
    dt = 1.0  # 1 hour timestep
    T_total = int(24 / dt)
    T_horizon = int(12 / dt)
    seed = 0

    rng = np.random.default_rng(seed)

    # Define household configuration — add or remove entities freely
    entities = [
        BaseLoad(load=np.zeros(T_horizon)),       # updated each step via update()
        PVGenerator(generation=np.zeros(T_horizon)),
        Battery(capacity=10.0, soc_init=4.0, charge_max=3.0, discharge_max=3.0, soc_min=1.0),
        ElectricVehicle(capacity=60.0, soc_init=15.0, charge_max=11.0, discharge_max=11.0, target_soc=50.0),
        HeatPump(
            temp_init=18.5,
            temp_min=np.full(T_horizon, 20.0),
            temp_max=np.full(T_horizon, 23.0),
            temp_out=np.zeros(T_horizon),         # updated each step via update()
            max_power=3.0,
            C_therm=5.0,
            lambda_=0.25,
            cop_eta=0.4,
        ),
    ]

    runner = MPCRunner(
        environment=SyntheticEnvironment(
            T_total=T_total, rng=rng,
            soc_bat_init=4.0, soc_ev_init=15.0, temp_in_init=18.5,
        ),
        forecaster=SyntheticForecaster(),
        entities=entities,
        T_horizon=T_horizon,
        dt=dt,
        rng=rng,
    )

    history = runner.run(T_total)
    plot_results(history, dt)


if __name__ == "__main__":
    main()