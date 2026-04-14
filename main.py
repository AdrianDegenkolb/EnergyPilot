"""Entry point for the household energy scheduling POC."""

import threading
from datetime import datetime, timedelta

import numpy as np

from control import MPC, BatteryModel, EVModel, HeatPumpModel, PhysicalEntity, MPCHistory
from dashboard.dashboard import LiveStore, create_dashboard
from core import Time, TimeSeriesData
from forecasting import TimeSeries, LookupForecaster, ConstantForecaster
from interfaces.synthetic import SyntheticState, SyntheticExternalState
from plot import plot_results, plot_forecast_comparison

FORECASTING_MODES = ["lookup", "constant"]


def _build_signals(
    t0: datetime,
    T: int,
    dt: timedelta,
    rng: np.random.Generator,
) -> dict[str, TimeSeriesData]:
    """Precompute passive signal trajectories as TimeSeriesData objects."""
    signals = {k: TimeSeriesData() for k in ("price_buy", "price_sell", "load", "gen", "temp_out")}
    T_day = int(timedelta(hours=24) / dt)

    for step in range(T):
        t    = t0 + dt * step
        frac = step / T_day
        signals["price_buy"].add_point(t, float(0.28 + 0.08 * np.cos(2 * np.pi * frac) + rng.normal(0, 0.01)))
        signals["price_sell"].add_point(t, 0.08)
        signals["load"].add_point(t, float(max(0.0, 0.6 + 0.25 * abs(np.sin(np.pi * frac)) + rng.normal(0, 0.05))))
        signals["gen"].add_point(t, float(max(0.0, 1.5 * np.exp(-0.5 * ((frac - 0.5) / 0.18) ** 2) + rng.normal(0, 0.1))))
        signals["temp_out"].add_point(t, float(4.0 + 4.0 * np.sin(np.pi * frac) + rng.normal(0, 0.3)))

    return signals


def _make_series(mode: str, signals: dict[str, TimeSeriesData]) -> dict[str, TimeSeries]:
    def _ts(key: str) -> TimeSeries:
        sensor = SyntheticExternalState(signals[key]).make_sensor()
        forecaster = LookupForecaster(signals[key]) if mode == "lookup" else ConstantForecaster()
        return TimeSeries(sensor, forecaster)

    return {k: _ts(k) for k in ("price_buy", "price_sell", "load", "gen", "temp_out")}


def _run_mode(
    mode: str,
    signals: dict[str, TimeSeriesData],
    t0: datetime,
    dt: timedelta,
    T_total: int,
    T_horizon: int,
    on_step=None,
) -> MPCHistory:
    Time.get_instance().set(t0)

    series = _make_series(mode, signals)

    bat_state = SyntheticState(4.0)
    ev_state  = SyntheticState(15.0)
    hp_state  = SyntheticState(21.5)

    bat_model = BatteryModel(capacity=10.0, charge_max=3.0, discharge_max=3.0, soc_min=1.0, efficiency=0.95, wearing_cost_per_kwh=0.01)
    ev_model  = EVModel(capacity=60.0, charge_max=11.0, discharge_max=11.0, target_soc=50.0, target_timestep=T_total + 1, efficiency=0.95, wearing_cost_per_kwh=0.02)
    hp_model  = HeatPumpModel(
        temp_min=np.full(T_horizon, 20.0),
        temp_max=np.full(T_horizon, 23.0),
        temp_out_series=series["temp_out"],
        max_power=3.0, C_therm=10.0, lambda_=0.25, cop_eta=0.4,
        wearing_cost_per_kwh=0.01,
    )

    bat_sensor, bat_actuator = bat_state.make_sensor_actuator()
    ev_sensor,  ev_actuator  = ev_state.make_sensor_actuator()
    hp_sensor,  hp_actuator  = hp_state.make_sensor_actuator()

    entities = [
        PhysicalEntity(bat_sensor, bat_model, bat_actuator),
        PhysicalEntity(ev_sensor,  ev_model,  ev_actuator),
        PhysicalEntity(hp_sensor,  hp_model,  hp_actuator),
    ]

    mpc = MPC(
        entities=entities,
        price_buy=series["price_buy"],
        price_sell=series["price_sell"],
        load=series["load"],
        gen=series["gen"],
        extra_series=[series["temp_out"]],
        T_horizon=T_horizon,
        dt=dt,
    )

    print(f"\n{'='*65}")
    print(f"  Forecasting mode: {mode}")
    print(f"{'='*65}")
    return mpc.run(T_total, fast_forward=True, on_step=on_step)


def main() -> None:
    dt        = timedelta(hours=0.25)
    T_total   = int(timedelta(hours=24) / dt)
    T_horizon = int(timedelta(hours=12) / dt)
    t0        = datetime(2024, 1, 1)

    rng     = np.random.default_rng(0)
    signals = _build_signals(t0, T_total + T_horizon, dt, rng)

    live_store = LiveStore(FORECASTING_MODES)

    def _run_simulation() -> None:
        histories: dict[str, MPCHistory] = {}
        for mode in FORECASTING_MODES:
            live_store.set_active_mode(mode)
            history = _run_mode(
                mode, signals, t0, dt, T_total, T_horizon,
                on_step=lambda step, m=mode: live_store.add_step(m, step),
            )
            histories[mode] = history
            plot_results(history, dt=dt, path=f"outputs/mpc_results_{mode}.png", show=False)

        plot_forecast_comparison(histories, dt=dt, path="outputs/mpc_comparison.png", show=False)
        live_store.mark_done()

    sim_thread = threading.Thread(target=_run_simulation, daemon=True)
    sim_thread.start()

    print("\nLaunching dashboard at http://localhost:8050  (simulation running in background)")
    create_dashboard(live_store, dt, FORECASTING_MODES, T_total).run(debug=False, port=8050)


if __name__ == "__main__":
    main()
