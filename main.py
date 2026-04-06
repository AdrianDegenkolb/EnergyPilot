"""Entry point for the household energy scheduling POC."""

from datetime import datetime, timedelta

import numpy as np

from forecasting import TimeSeries
from forecasting.time import Time
from forecasting.time_series_data import TimeSeriesData
from optimization import MPC
from optimization.milp import BatteryModel, EVModel, HeatPumpModel, PhysicalEntity
from plot import plot_results
from real_world_interfaces.synthetic import SyntheticState, SyntheticExternalState
from real_world_interfaces.synthetic.synthetic_series_forecasters import LookupForecaster


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
        signals["gen"].add_point(t, float(max(0.0, 3.5 * np.exp(-0.5 * ((frac - 0.5) / 0.18) ** 2) + rng.normal(0, 0.1))))
        signals["temp_out"].add_point(t, float(4.0 + 4.0 * np.sin(np.pi * frac) + rng.normal(0, 0.3)))

    return signals


def main() -> None:
    dt        = timedelta(hours=0.25)
    T_total   = int(timedelta(hours=24) / dt)
    T_horizon = int(timedelta(hours=12) / dt)
    t0        = datetime(2024, 1, 1)

    Time.get_instance().set(t0)
    rng = np.random.default_rng(0)

    # --- Precompute signals — T_total + T_horizon covers forecast lookahead at every step ---
    signals = _build_signals(t0, T_total + T_horizon, dt, rng)

    # --- Passive series: sensor reads current value from data, forecaster looks up future values ---
    price_buy_series  = TimeSeries(SyntheticExternalState(signals["price_buy"]).make_sensor(),  LookupForecaster(signals["price_buy"]))
    price_sell_series = TimeSeries(SyntheticExternalState(signals["price_sell"]).make_sensor(), LookupForecaster(signals["price_sell"]))
    load_series       = TimeSeries(SyntheticExternalState(signals["load"]).make_sensor(),       LookupForecaster(signals["load"]))
    gen_series        = TimeSeries(SyntheticExternalState(signals["gen"]).make_sensor(),        LookupForecaster(signals["gen"]))
    temp_out_series   = TimeSeries(SyntheticExternalState(signals["temp_out"]).make_sensor(),   LookupForecaster(signals["temp_out"]))

    # --- Controllable states ---
    bat_state = SyntheticState(4.0)
    ev_state  = SyntheticState(15.0)
    hp_state  = SyntheticState(21.5)

    # --- Optimization models ---
    bat_model = BatteryModel(capacity=10.0, charge_max=3.0, discharge_max=3.0, soc_min=1.0, efficiency=0.95)
    ev_model  = EVModel(capacity=60.0, charge_max=11.0, discharge_max=11.0, target_soc=50.0, target_timestep=T_total + 1, efficiency=0.95)
    hp_model  = HeatPumpModel(
        temp_min=np.full(T_horizon, 20.0),
        temp_max=np.full(T_horizon, 23.0),
        temp_out_series=temp_out_series,
        max_power=3.0, C_therm=5.0, lambda_=0.25, cop_eta=0.4,
    )

    # --- Sensors, actuators, entities ---
    bat_sensor, bat_actuator = bat_state.make_sensor_actuator()
    ev_sensor,  ev_actuator  = ev_state.make_sensor_actuator()
    hp_sensor,  hp_actuator  = hp_state.make_sensor_actuator()

    entities = [
        PhysicalEntity(bat_sensor, bat_model, bat_actuator),
        PhysicalEntity(ev_sensor,  ev_model,  ev_actuator),
        PhysicalEntity(hp_sensor,  hp_model,  hp_actuator),
    ]

    # --- Run ---
    mpc = MPC(
        entities=entities,
        price_buy=price_buy_series,
        price_sell=price_sell_series,
        load=load_series,
        gen=gen_series,
        extra_series=[temp_out_series],
        T_horizon=T_horizon,
        dt=dt,
    )

    history = mpc.run(T_total, fast_forward=True)
    plot_results(history, dt.seconds / 3600, show=True)


if __name__ == "__main__":
    main()
