"""Plotting logic for MPC simulation results."""

from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from control import HeatPumpModel, BatteryModel, EVModel, MPCHistory, MPCStep

C_BATTERY = "#1976D2"
C_EV      = "#388E3C"
C_HP      = "#F57C00"
C_PV      = "#FBC02D"
C_LOAD    = "#D32F2F"
C_BUY     = "#C2185B"
C_SELL    = "#7B1FA2"
C_TEMP_IN = "#E53935"
C_MPC     = "#1565C0"

_COMPARISON_COLORS = ["#1565C0", "#EF6C00", "#388E3C", "#C2185B", "#7B1FA2", "#5D4037"]


def _style_ax(ax: plt.Axes, ylabel: str, legend_loc: str = "best") -> None:
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


def plot_results(history: MPCHistory, dt: timedelta, path: str = "outputs/mpc_results.png", show: bool = True) -> None:
    dt_h  = dt.total_seconds() / 3600
    steps = np.array([s.step for s in history.steps]) * dt_h
    w     = 0.55 * dt_h

    heat_pump = next((c for c in history.steps[0].optimization_problem._components if isinstance(c, HeatPumpModel)), None)
    temp_variable_name        = heat_pump._temp_variable_name if heat_pump else None
    power_setpoint_heat_pump  = heat_pump._power_setpoint_variable_name if heat_pump else None
    battery = next((c for c in history.steps[0].optimization_problem._components if isinstance(c, BatteryModel)), None)
    soc_battery_variable_name = battery._soc_variable_name if battery else None
    power_setpoint_battery    = battery._power_setpoint_variable_name if battery else None
    ev = next((c for c in history.steps[0].optimization_problem._components if isinstance(c, EVModel)), None)
    soc_ev_variable_name = ev._soc_variable_name if ev else None
    power_setpoint_ev    = ev._power_setpoint_variable_name if ev else None

    price_buy  = np.array([s.result.price_buy[0]  for s in history.steps])
    price_sell = np.array([s.result.price_sell[0] for s in history.steps])
    p_buy      = np.array([s.result.p_buy[0]      for s in history.steps])
    p_sell     = np.array([s.result.p_sell[0]     for s in history.steps])
    load       = np.array([s.result.load[0]        for s in history.steps])
    gen        = np.array([s.result.gen[0]          for s in history.steps])
    temp_in    = np.array([s.result.variables.get(temp_variable_name,       np.zeros(2))[0] for s in history.steps])
    soc_bat    = np.array([s.result.variables.get(soc_battery_variable_name, np.zeros(2))[0] for s in history.steps])
    soc_ev     = np.array([s.result.variables.get(soc_ev_variable_name,     np.zeros(2))[0] for s in history.steps])
    x_bat      = np.array([float(s.result.variables.get(power_setpoint_battery,    np.zeros(1))[0]) for s in history.steps])
    x_ev       = np.array([float(s.result.variables.get(power_setpoint_ev,         np.zeros(1))[0]) for s in history.steps])
    x_hp       = np.array([float(s.result.variables.get(power_setpoint_heat_pump,  np.zeros(1))[0]) for s in history.steps])

    cum_cost = history.cumulative_cost()

    fig = plt.figure(figsize=(14, 16))
    fig.suptitle("Household Energy Scheduling — MPC", fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[3, :])

    ax1.plot(steps, price_buy,  "-",  color=C_BUY,  lw=1.5, label="Buy [€/kWh]")
    ax1.plot(steps, price_sell, "--", color=C_SELL, lw=1.5, label="Sell [€/kWh]")
    ax1.set_title("Energy Prices", fontsize=9, fontweight="bold")
    _style_ax(ax1, "Price [€/kWh]")

    ax2.plot(steps, load, "-",  color=C_LOAD, lw=1.5, label="Load [kW]")
    ax2.plot(steps, gen,  "--", color=C_PV,   lw=1.5, label="PV gen [kW]")
    ax2.set_title("Load & Generation", fontsize=9, fontweight="bold")
    _style_ax(ax2, "Power [kW]")

    ax3.bar(steps,  p_buy,  width=w, color=C_BUY,  alpha=0.85, label="Import [kW]")
    ax3.bar(steps, -p_sell, width=w, color=C_SELL, alpha=0.85, label="Export [kW]")
    ax3.axhline(0, color="black", lw=0.4)
    ax3.set_title("Grid Import / Export", fontsize=9, fontweight="bold")
    _style_ax(ax3, "Power [kW]")

    _stacked_bars(ax4, steps, [
        ("Battery",   x_bat, C_BATTERY),
        ("EV",        x_ev,  C_EV),
        ("Heat pump", x_hp,  C_HP),
    ], width=w)
    ax4.axhline(0, color="black", lw=0.4)
    ax4.set_title("Scheduling Decisions", fontsize=9, fontweight="bold")
    _style_ax(ax4, "Power [kW]")

    ax5.plot(steps, soc_bat, "o-", color=C_BATTERY, lw=1.5, ms=4, label="Battery [kWh]")
    ax5.plot(steps, soc_ev,  "s-", color=C_EV,      lw=1.5, ms=4, label="EV [kWh]")
    ax5.set_title("State of Charge", fontsize=9, fontweight="bold")
    _style_ax(ax5, "SoC [kWh]")

    ax6.plot(steps, temp_in, "-", color=C_TEMP_IN, lw=1.5, label="Indoor [°C]")
    ax6.axhline(20.0, color=C_TEMP_IN, lw=0.8, ls=":", alpha=0.6, label="Min comfort")
    ax6.axhline(23.0, color=C_TEMP_IN, lw=0.8, ls=":", alpha=0.4, label="Max comfort")
    ax6.set_title("Heat Pump & Temperature", fontsize=9, fontweight="bold")
    _style_ax(ax6, "Temperature [°C]")

    final_cost = cum_cost[-1]
    ax7.plot(steps[:-1], cum_cost, "o-", color=C_MPC, lw=2, ms=4,
             label=f"MPC  —  final: {final_cost:.2f} €")
    ax7.set_title("Cumulative Cost", fontsize=9, fontweight="bold")
    _style_ax(ax7, "Cumulative cost [€]")

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.set_xlabel("Time [h]", fontsize=8)

    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    print(f"\nPlot saved to {path}")
    print(f"  MPC total cost: {final_cost:.4f} €")


def plot_forecast_comparison(
    histories: dict[str, MPCHistory],
    dt: timedelta,
    path: str = "outputs/mpc_comparison.png",
    show: bool = True,
) -> None:
    """
    Plot cumulative costs for all forecasting modes on a single axis.

    Shades the area between the cheapest and most expensive mode,
    and annotates the final cost difference in the legend.
    """
    dt_h = dt.total_seconds() / 3600

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mode, history in histories.items():
        steps    = np.array([s.step for s in history.steps[:-1]]) * dt_h
        cum_cost = history.cumulative_cost()
        curves[mode] = (steps, cum_cost)

    final_costs = {mode: cum[-1] for mode, (_, cum) in curves.items()}
    cheapest    = min(final_costs, key=final_costs.get)
    most_expensive = max(final_costs, key=final_costs.get)
    delta = final_costs[most_expensive] - final_costs[cheapest]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Forecast Mode Comparison — Cumulative Cost", fontsize=13, fontweight="bold")

    for (mode, (steps, cum)), color in zip(curves.items(), _COMPARISON_COLORS):
        ax.plot(steps, cum, "o-", color=color, lw=2, ms=3,
                label=f"{mode}  —  {final_costs[mode]:.2f} €")

    # Shade between cheapest and most expensive
    steps_ref = curves[cheapest][0]
    ax.fill_between(
        steps_ref,
        curves[cheapest][1],
        curves[most_expensive][1],
        alpha=0.12,
        color="gray",
        label=f"Δ ({cheapest} vs {most_expensive}) = {delta:.2f} €",
    )

    ax.set_xlabel("Time [h]", fontsize=9)
    _style_ax(ax, "Cumulative cost [€]", legend_loc="upper left")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    print(f"\nComparison plot saved to {path}")
    for mode, cost in sorted(final_costs.items(), key=lambda x: x[1]):
        print(f"  {mode}: {cost:.4f} €")
