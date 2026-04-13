"""Dash dashboard for EnergyPilot — streams live updates during MPC simulation."""

import threading
from datetime import timedelta

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots

from control import BatteryModel, EVModel, HeatPumpModel
from control.mpc.MPC_state import MPCStep

# ── Colours ───────────────────────────────────────────────────────────────────
C_BUY  = "#1565C0"
C_SELL = "#E65100"
C_LOAD = "#880E4F"
C_GEN  = "#1B5E20"

C_BATTERY = "#1976D2"
C_EV      = "#388E3C"
C_HP      = "#F57C00"
C_TEMP    = "#E53935"

_MODE_COLORS = ["#1565C0", "#EF6C00", "#388E3C", "#C2185B", "#7B1FA2", "#5D4037"]

_VLINE_STYLE  = dict(color="#444", width=1.5, dash="dash")
_XAXIS_COMMON = dict(title_text="Time [h]", range=[0, 24])

# Base style shared by every per-subplot legend
_LEG = dict(xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor="#ccc", borderwidth=1)


# ── Live data store ───────────────────────────────────────────────────────────

class LiveStore:
    """Thread-safe store of MPC steps written by simulation, read by dashboard."""

    def __init__(self, mode_names: list[str]) -> None:
        self._lock = threading.Lock()
        self._steps: dict[str, list[MPCStep]] = {m: [] for m in mode_names}
        self._active_mode: str | None = None
        self._done = False

    def set_active_mode(self, mode: str) -> None:
        with self._lock:
            self._active_mode = mode

    def add_step(self, mode: str, step: MPCStep) -> None:
        with self._lock:
            self._steps[mode].append(step)

    def mark_done(self) -> None:
        with self._lock:
            self._done = True

    def snapshot(self) -> tuple[dict[str, list[MPCStep]], str | None, bool]:
        with self._lock:
            return (
                {m: list(steps) for m, steps in self._steps.items()},
                self._active_mode,
                self._done,
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_fig(message: str = "No data yet") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=16, color="#aaa"),
    )
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      uirevision="constant")
    return fig


def _with_alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _var_names(steps: list[MPCStep]) -> dict[str, str]:
    if not steps:
        return {}
    names: dict[str, str] = {}
    for c in steps[0].optimization_problem._components:
        if isinstance(c, BatteryModel):
            names["bat_soc"]   = c._soc_variable_name
            names["bat_power"] = c._power_setpoint_variable_name
        elif isinstance(c, EVModel):
            names["ev_soc"]   = c._soc_variable_name
            names["ev_power"] = c._power_setpoint_variable_name
        elif isinstance(c, HeatPumpModel):
            names["hp_temp"]  = c._temp_variable_name
            names["hp_power"] = c._power_setpoint_variable_name
    return names


def _get_var(variables: dict, vn: dict[str, str], role: str):
    key = vn.get(role)
    return variables.get(key) if key else None


def _first_vals(steps: list[MPCStep], n: int, vn: dict[str, str], role: str) -> list[float]:
    key = vn.get(role)
    if not key:
        return [0.0] * n
    return [float(steps[i].result.variables.get(key, [0.0])[0]) for i in range(n)]


def _cumcost(steps: list[MPCStep], dt_h: float) -> tuple[list[float], np.ndarray]:
    if len(steps) < 2:
        return [], np.array([])
    hours, costs = [], []
    for i in range(len(steps) - 1):
        r = steps[i].result
        costs.append(float((r.p_buy[0] * r.price_buy[0] - r.p_sell[0] * r.price_sell[0]) * dt_h))
        hours.append(i * dt_h)
    return hours, np.cumsum(costs)


# ── Figure builders ───────────────────────────────────────────────────────────

def _costs_fig(all_steps: dict[str, list[MPCStep]], dt_h: float) -> go.Figure:
    """Tab 1 — live cumulative cost curves."""
    fig = go.Figure()
    any_data = False
    all_curves: dict[str, tuple[list[float], np.ndarray]] = {}

    for (mode, steps), color in zip(all_steps.items(), _MODE_COLORS):
        hours, cum = _cumcost(steps, dt_h)
        if len(cum) == 0:
            continue
        any_data = True
        all_curves[mode] = (hours, cum)
        fig.add_trace(go.Scattergl(
            x=hours, y=cum,
            name=f"{mode}  —  {cum[-1]:.2f} €",
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(color=color, width=2),
        ))

    if not any_data:
        return _empty_fig("Waiting for first MPC steps…")

    if len(all_curves) >= 2:
        by_final = sorted(all_curves.items(), key=lambda kv: kv[1][1][-1])
        h_cheap, cum_cheap = by_final[0][1]
        h_price, cum_price = by_final[-1][1]
        if len(cum_cheap) == len(cum_price):
            delta = cum_price[-1] - cum_cheap[-1]
            # Use go.Scatter (not Scattergl) — fill="toself" requires the SVG renderer
            fig.add_trace(go.Scatter(
                x=h_cheap + list(reversed(h_price)),
                y=list(cum_cheap) + list(reversed(cum_price)),
                fill="toself",
                fillcolor="rgba(128,128,128,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=True,
                name=f"Δ = {delta:.2f} €",
            ))

    fig.update_layout(
        title=dict(text="Cumulative Cost by Forecasting Mode"),
        xaxis=dict(**_XAXIS_COMMON),
        yaxis=dict(title="Cumulative cost [€]"),
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, **_LEG),
        uirevision="constant",
    )
    return fig


def _signals_fig(steps: list[MPCStep], slider_val: int, dt_h: float) -> go.Figure:
    """Tab 2 — observed signals and forecasts in two subplots.

    Subplot layout (1×2, horizontal_spacing=0.10):
      col 1 x-domain ≈ [0.00, 0.45]  → legend  at x=0.01
      col 2 x-domain ≈ [0.55, 1.00]  → legend2 at x=0.56
    """
    if not steps:
        return _empty_fig()

    n_obs     = min(slider_val + 1, len(steps))
    current_h = slider_val * dt_h

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["<b>Energy Prices</b>", "<b>Load & PV Generation</b>"],
        horizontal_spacing=0.10,
    )

    # ── Observed (left of line) ──
    if n_obs > 0:
        obs_h = [i * dt_h for i in range(n_obs)]
        fig.add_trace(go.Scattergl(
            x=obs_h, y=[steps[i].result.price_buy[0]  for i in range(n_obs)],
            name="Buy (past)", mode="lines", line=dict(color=C_BUY,  width=2.5),
            legend="legend",
        ), row=1, col=1)
        fig.add_trace(go.Scattergl(
            x=obs_h, y=[steps[i].result.price_sell[0] for i in range(n_obs)],
            name="Sell (past)", mode="lines", line=dict(color=C_SELL, width=2.5),
            legend="legend",
        ), row=1, col=1)
        fig.add_trace(go.Scattergl(
            x=obs_h, y=[steps[i].result.load[0] for i in range(n_obs)],
            name="Load (past)", mode="lines", line=dict(color=C_LOAD, width=2.5),
            legend="legend2",
        ), row=1, col=2)
        fig.add_trace(go.Scattergl(
            x=obs_h, y=[steps[i].result.gen[0]  for i in range(n_obs)],
            name="PV (past)", mode="lines", line=dict(color=C_GEN, width=2.5),
            legend="legend2",
        ), row=1, col=2)

    fig.add_vline(x=current_h, line=_VLINE_STYLE,
                  annotation_text=" now", annotation_position="top right")

    # ── Forecast at slider position ──
    r  = steps[slider_val].result
    T  = len(r.price_buy)
    fc = [current_h + j * dt_h for j in range(T)]

    fig.add_trace(go.Scattergl(
        x=fc, y=r.price_buy,  name="Buy (forecast)",
        mode="lines", line=dict(color=_with_alpha(C_BUY,  0.40), dash="dash", width=1.5),
        legend="legend",
    ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=fc, y=r.price_sell, name="Sell (forecast)",
        mode="lines", line=dict(color=_with_alpha(C_SELL, 0.40), dash="dash", width=1.5),
        legend="legend",
    ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=fc, y=r.load, name="Load (forecast)",
        mode="lines", line=dict(color=_with_alpha(C_LOAD, 0.40), dash="dash", width=1.5),
        legend="legend2",
    ), row=1, col=2)
    fig.add_trace(go.Scattergl(
        x=fc, y=r.gen,  name="PV (forecast)",
        mode="lines", line=dict(color=_with_alpha(C_GEN,  0.40), dash="dash", width=1.5),
        legend="legend2",
    ), row=1, col=2)

    fig.update_xaxes(**_XAXIS_COMMON)
    fig.update_yaxes(title_text="Price [€/kWh]", row=1, col=1)
    fig.update_yaxes(title_text="Power [kW]",    row=1, col=2)
    fig.update_layout(
        title=dict(text="Signals: Observed History & Forecasts"),
        hovermode="x unified",
        height=500,
        legend =dict(x=0.01, y=0.99, **_LEG),
        legend2=dict(x=0.56, y=0.99, **_LEG),
        uirevision="constant",
    )
    return fig


def _schedule_fig(steps: list[MPCStep], slider_val: int, dt_h: float) -> go.Figure:
    """Tab 3 — grid schedule and device states.

    Subplot layout (2×2, vertical_spacing=0.14, horizontal_spacing=0.10):
      row1/col1 → legend  (x=0.01, y=0.97)
      row1/col2 → legend2 (x=0.56, y=0.97)
      row2/col1 → legend3 (x=0.01, y=0.42)
      row2/col2 → legend4 (x=0.56, y=0.42)
    """
    if not steps:
        return _empty_fig()

    vn        = _var_names(steps)
    current_h = slider_val * dt_h
    bar_w     = dt_h * 0.85

    n_past    = min(slider_val, len(steps))
    plan_idx  = min(slider_val, len(steps) - 1)
    plan      = steps[plan_idx].result
    T_plan    = len(plan.p_buy)

    past_h     = [i * dt_h for i in range(n_past)]
    future_h   = [(slider_val + j) * dt_h for j in range(T_plan)]
    future_soc = [(slider_val + j) * dt_h for j in range(T_plan + 1)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["<b>Grid Schedule</b>", "<b>Device Power</b>",
                        "<b>State of Charge</b>", "<b>Temperature</b>"],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # ── Past: executed decisions ──
    if n_past > 0:
        p_buy_p  = [steps[i].result.p_buy[0]  for i in range(n_past)]
        p_sell_p = [steps[i].result.p_sell[0] for i in range(n_past)]
        fig.add_trace(go.Bar(x=past_h, y=p_buy_p, width=bar_w,
                             name="Import (past)", marker_color=C_BUY, opacity=0.85,
                             legend="legend"), row=1, col=1)
        fig.add_trace(go.Bar(x=past_h, y=[-v for v in p_sell_p], width=bar_w,
                             name="Export (past)", marker_color=C_SELL, opacity=0.85,
                             legend="legend"), row=1, col=1)

        bat_pow = _first_vals(steps, n_past, vn, "bat_power")
        ev_pow  = _first_vals(steps, n_past, vn, "ev_power")
        hp_pow  = _first_vals(steps, n_past, vn, "hp_power")
        fig.add_trace(go.Bar(x=past_h, y=bat_pow, width=bar_w,
                             name="Battery (past)", marker_color=C_BATTERY, opacity=0.85,
                             legend="legend2"), row=1, col=2)
        fig.add_trace(go.Bar(x=past_h, y=ev_pow, width=bar_w,
                             name="EV (past)", marker_color=C_EV, opacity=0.85,
                             legend="legend2"), row=1, col=2)
        fig.add_trace(go.Bar(x=past_h, y=hp_pow, width=bar_w,
                             name="Heat pump (past)", marker_color=C_HP, opacity=0.85,
                             legend="legend2"), row=1, col=2)

        n_soc     = min(slider_val + 1, len(steps))
        soc_h     = [i * dt_h for i in range(n_soc)]
        bat_soc_p = _first_vals(steps, n_soc, vn, "bat_soc")
        ev_soc_p  = _first_vals(steps, n_soc, vn, "ev_soc")
        fig.add_trace(go.Scattergl(x=soc_h, y=bat_soc_p, name="Battery SoC (past)",
                                   mode="lines", line=dict(color=C_BATTERY, width=2),
                                   legend="legend3"), row=2, col=1)
        fig.add_trace(go.Scattergl(x=soc_h, y=ev_soc_p, name="EV SoC (past)",
                                   mode="lines", line=dict(color=C_EV, width=2),
                                   legend="legend3"), row=2, col=1)

        hp_temp_p = _first_vals(steps, n_soc, vn, "hp_temp")
        fig.add_trace(go.Scattergl(x=soc_h, y=hp_temp_p, name="Temperature (past)",
                                   mode="lines", line=dict(color=C_TEMP, width=2),
                                   legend="legend4"), row=2, col=2)

    # ── "Now" vertical lines ──
    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.add_vline(x=current_h, line=_VLINE_STYLE, row=r, col=c, annotation_text=" now")

    # ── Future: optimal plan ──
    fig.add_trace(go.Bar(x=future_h, y=plan.p_buy, width=bar_w,
                         name="Import (plan)", marker_color=_with_alpha(C_BUY, 0.45),
                         marker_line=dict(width=0), legend="legend"), row=1, col=1)
    fig.add_trace(go.Bar(x=future_h, y=-plan.p_sell, width=bar_w,
                         name="Export (plan)", marker_color=_with_alpha(C_SELL, 0.45),
                         marker_line=dict(width=0), legend="legend"), row=1, col=1)

    bat_pow_f = _get_var(plan.variables, vn, "bat_power")
    ev_pow_f  = _get_var(plan.variables, vn, "ev_power")
    hp_pow_f  = _get_var(plan.variables, vn, "hp_power")
    if bat_pow_f is not None:
        fig.add_trace(go.Bar(x=future_h, y=bat_pow_f, width=bar_w, name="Battery (plan)",
                             marker_color=_with_alpha(C_BATTERY, 0.45),
                             marker_line=dict(width=0), legend="legend2"), row=1, col=2)
    if ev_pow_f is not None:
        fig.add_trace(go.Bar(x=future_h, y=ev_pow_f, width=bar_w, name="EV (plan)",
                             marker_color=_with_alpha(C_EV, 0.45),
                             marker_line=dict(width=0), legend="legend2"), row=1, col=2)
    if hp_pow_f is not None:
        fig.add_trace(go.Bar(x=future_h, y=hp_pow_f, width=bar_w, name="Heat pump (plan)",
                             marker_color=_with_alpha(C_HP, 0.45),
                             marker_line=dict(width=0), legend="legend2"), row=1, col=2)

    bat_soc_f = _get_var(plan.variables, vn, "bat_soc")
    ev_soc_f  = _get_var(plan.variables, vn, "ev_soc")
    hp_temp_f = _get_var(plan.variables, vn, "hp_temp")
    if bat_soc_f is not None:
        fig.add_trace(go.Scattergl(x=future_soc, y=bat_soc_f, name="Battery SoC (plan)",
                                   mode="lines",
                                   line=dict(color=_with_alpha(C_BATTERY, 0.5), width=2),
                                   legend="legend3"), row=2, col=1)
    if ev_soc_f is not None:
        fig.add_trace(go.Scattergl(x=future_soc, y=ev_soc_f, name="EV SoC (plan)",
                                   mode="lines",
                                   line=dict(color=_with_alpha(C_EV, 0.5), width=2),
                                   legend="legend3"), row=2, col=1)
    if hp_temp_f is not None:
        fig.add_trace(go.Scattergl(x=future_soc, y=hp_temp_f, name="Temperature (plan)",
                                   mode="lines",
                                   line=dict(color=_with_alpha(C_TEMP, 0.5), width=2),
                                   legend="legend4"), row=2, col=2)
        fig.add_hline(y=20.0, line=dict(color=C_TEMP, width=1, dash="dot"),
                      annotation_text="min 20°C", row=2, col=2)
        fig.add_hline(y=23.0, line=dict(color=C_TEMP, width=1, dash="dot"),
                      annotation_text="max 23°C", row=2, col=2)

    fig.add_hline(y=0, line=dict(color="black", width=0.5), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="black", width=0.5), row=1, col=2)

    fig.update_xaxes(**_XAXIS_COMMON)
    fig.update_yaxes(title_text="Power [kW]",       row=1, col=1)
    fig.update_yaxes(title_text="Power [kW]",       row=1, col=2)
    fig.update_yaxes(title_text="SoC [kWh]",        row=2, col=1)
    fig.update_yaxes(title_text="Temperature [°C]", row=2, col=2)
    fig.update_layout(
        title=dict(text=f"Schedule  |  {current_h:.2f} h  |  Plan horizon: {T_plan} steps"),
        hovermode="x unified",
        barmode="relative",
        height=700,
        legend =dict(x=0.01, y=0.97, **_LEG),
        legend2=dict(x=0.56, y=0.97, **_LEG),
        legend3=dict(x=0.01, y=0.40, **_LEG),
        legend4=dict(x=0.56, y=0.40, **_LEG),
        uirevision="constant",
    )
    return fig


def _devices_fig(steps: list[MPCStep], dt_h: float) -> go.Figure:
    """Tab 4 — full-day device trajectories.

    Only subplot 3 (Heat Pump) has a legend showing temp + min/max bounds.
    """
    if not steps:
        return _empty_fig()

    vn  = _var_names(steps)
    n   = len(steps)
    hrs = [i * dt_h for i in range(n)]

    bat_soc = _first_vals(steps, n, vn, "bat_soc")
    ev_soc  = _first_vals(steps, n, vn, "ev_soc")
    hp_temp = _first_vals(steps, n, vn, "hp_temp")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=["<b>Battery</b>", "<b>Electric Vehicle</b>", "<b>Heat Pump</b>"],
        shared_xaxes=True,
        vertical_spacing=0.10,
    )

    # Battery & EV — no legend, subplot title is sufficient
    fig.add_trace(go.Scattergl(x=hrs, y=bat_soc, name="Battery SoC",
                               mode="lines", line=dict(color=C_BATTERY, width=2),
                               showlegend=False), row=1, col=1)
    fig.add_trace(go.Scattergl(x=hrs, y=ev_soc, name="EV SoC",
                               mode="lines", line=dict(color=C_EV, width=2),
                               showlegend=False), row=2, col=1)

    # Heat pump — legend with current temp + constraint bounds.
    # Use real Scatter traces for min/max so they appear in the legend.
    x_span = [0, 24]
    fig.add_trace(go.Scatter(x=x_span, y=[20.0, 20.0], name="Min (20 °C)",
                             mode="lines", line=dict(color=C_TEMP, width=1, dash="dot"),
                             legend="legend"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_span, y=[23.0, 23.0], name="Max (23 °C)",
                             mode="lines", line=dict(color=C_TEMP, width=1, dash="dot"),
                             legend="legend"), row=3, col=1)
    fig.add_trace(go.Scattergl(x=hrs, y=hp_temp, name="Indoor Temperature",
                               mode="lines", line=dict(color=C_TEMP, width=2),
                               legend="legend"), row=3, col=1)

    # Dynamic y-range: cover actual data and both constraint bounds, pad 5 °C each side
    if hp_temp:
        t_lo = min(min(hp_temp), 20.0)
        t_hi = max(max(hp_temp), 23.0)
    else:
        t_lo, t_hi = 20.0, 23.0
    fig.update_yaxes(range=[t_lo - 5, t_hi + 5], row=3, col=1)

    fig.update_xaxes(**_XAXIS_COMMON)
    fig.update_yaxes(title_text="SoC [kWh]",        row=1, col=1)
    fig.update_yaxes(title_text="SoC [kWh]",        row=2, col=1)
    fig.update_yaxes(title_text="Temperature [°C]",  row=3, col=1)
    fig.update_layout(
        title=dict(text="Device Trajectories"),
        hovermode="x unified",
        barmode="relative",
        height=700,
        # legend sits at top of row 3 (paper y ≈ 0.27 for 3×1 with spacing=0.10)
        legend=dict(x=0.01, y=0.26, **_LEG),
        uirevision="constant",
    )
    return fig


# ── App factory ───────────────────────────────────────────────────────────────

def create_dashboard(
    store: LiveStore,
    dt: timedelta,
    mode_names: list[str],
    T_total: int,
) -> dash.Dash:
    """Build and return the Dash app. Call .run() to serve it (blocking)."""
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    dt_h = dt.total_seconds() / 3600

    step_per_2h = max(1, round(1.0 / dt_h))
    hour_marks  = {i: f"{i * dt_h:.0f}h" for i in range(0, T_total + 1, step_per_2h)}

    dropdown_opts = [{"label": m, "value": m} for m in mode_names]
    _drop_style   = {"width": "200px", "display": "inline-block", "verticalAlign": "middle"}
    _slider_wrap  = {"padding": "0 24px 20px"}

    _btn_style = {
        "padding": "6px 14px", "cursor": "pointer",
        "fontFamily": "sans-serif", "fontSize": "14px",
        "border": "1px solid #ccc", "borderRadius": "4px",
        "background": "#fff", "lineHeight": "1",
    }

    app.layout = html.Div([
        html.Div(
            [
                html.H2("EnergyPilot — Dashboard", style={"fontFamily": "sans-serif"}),
                html.Div([
                    dcc.Dropdown(id="forecasting-mode", options=dropdown_opts,
                                 value=mode_names[0], style=_drop_style),
                    html.Button("↺", id="reload-btn", n_clicks=0, style=_btn_style),
                ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
            ],
            style={
                "display": "flex", "justifyContent": "space-between",
                "alignItems": "center", "margin": "20px 40px 10px 40px",
            }
        ),
        html.Div(id="status-bar", style={
            "margin": "10px 40px 20px 40px", "padding": "8px 14px",
            "borderRadius": "6px", "background": "#f0f4ff",
            "fontFamily": "monospace", "fontSize": "14px", "color": "#333",
        }),
        dcc.Interval(id="interval", interval=800, n_intervals=0),

        dcc.Tabs(style={"fontFamily": "sans-serif"}, children=[

            dcc.Tab(label="Costs", children=[
                dcc.Graph(id="costs-fig", style={"height": "500px"}),
            ]),

            dcc.Tab(label="Signals", children=[
                html.Div([
                    html.H2(""),
                    dcc.Slider(id="signals-slider", min=0, max=T_total - 1,
                               value=0, step=None, marks=hour_marks,
                               updatemode="mouseup", dots=True,
                               tooltip=None),
                ], style=_slider_wrap),
                dcc.Graph(id="signals-fig", style={"height": "500px"}),
            ]),

            dcc.Tab(label="Schedule", children=[
                html.Div([
                    html.H2(""),
                    dcc.Slider(id="schedule-slider", min=0, max=T_total - 1,
                               value=0, step=None, marks=hour_marks,
                               updatemode="mouseup",
                               tooltip=None),
                ], style=_slider_wrap),
                dcc.Graph(id="schedule-fig", style={"height": "700px"}),
            ]),

            dcc.Tab(label="Devices", children=[
                dcc.Graph(id="devices-fig", style={"height": "700px"}),
            ]),
        ],
                 parent_style={"margin": "10px 40px 20px 40px"}),
    ])

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @app.callback(
        Output("status-bar", "children"),
        Output("costs-fig",  "figure"),
        Output("interval",   "disabled"),
        Input("interval",    "n_intervals"),
        Input("reload-btn",  "n_clicks"),
    )
    def update_costs(n: int, _r: int):
        all_steps, active_mode, done = store.snapshot()
        total = sum(len(s) for s in all_steps.values())
        if done:
            status = f"Simulation complete — {total} steps across {len(mode_names)} mode(s)"
        elif active_mode:
            status = f"Running: {active_mode} — step {len(all_steps.get(active_mode, []))} / {T_total}"
        else:
            status = "Waiting for simulation to start…"
        return status, _costs_fig(all_steps, dt_h), done

    @app.callback(
        Output("signals-fig",    "figure"),
        Output("signals-slider", "max"),
        Input("forecasting-mode", "value"),
        Input("signals-slider",   "value"),
        Input("interval",         "n_intervals"),
        Input("reload-btn",       "n_clicks"),
    )
    def update_signals(mode: str, slider_val: int, _n: int, _r: int):
        all_steps, _, _ = store.snapshot()
        steps = all_steps.get(mode, [])
        return _signals_fig(steps, slider_val, dt_h), max(len(steps) - 1, 0)

    @app.callback(
        Output("schedule-fig",    "figure"),
        Output("schedule-slider", "max"),
        Input("forecasting-mode",  "value"),
        Input("schedule-slider",   "value"),
        Input("interval",          "n_intervals"),
        Input("reload-btn",        "n_clicks"),
    )
    def update_schedule(mode: str, slider_val: int, _n: int, _r: int):
        all_steps, _, _ = store.snapshot()
        steps = all_steps.get(mode, [])
        return _schedule_fig(steps, slider_val, dt_h), max(len(steps) - 1, 0)

    @app.callback(
        Output("devices-fig",     "figure"),
        Input("interval",         "n_intervals"),
        Input("forecasting-mode", "value"),
        Input("reload-btn",       "n_clicks"),
    )
    def update_devices(n: int, mode: str, _r: int) -> go.Figure:
        all_steps, _, _ = store.snapshot()
        return _devices_fig(all_steps.get(mode, []), dt_h)

    return app
