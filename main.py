"""Entry point for the household energy scheduling POC."""

import numpy as np

from synthetic import SyntheticState
from forecasting import TimeSeries
from synthetic.synthetic_series_forecasters import (
    SyntheticPriceBuyForecaster,
    SyntheticPriceSellForecaster,
    SyntheticGenForecaster,
    SyntheticLoadForecaster,
    SyntheticTempOutForecaster,
)
from optimization.milp import BatteryModel, EVModel, HeatPumpModel, PhysicalEntity
from optimization.mpc.synthetic_mpc import SyntheticMPC
from plot import plot_results


def main() -> None:
    dt       = 0.25
    T_total  = int(24 / dt)
    T_horizon = int(12 / dt)

    # --- Passive signal states ---
    price_buy_state  = SyntheticState(0.28)
    price_sell_state = SyntheticState(0.08)
    load_state       = SyntheticState(0.7)
    gen_state        = SyntheticState(0.0)
    temp_out_state   = SyntheticState(5.0)

    # --- Series  ---
    price_buy_series  = TimeSeries(price_buy_state.make_sensor(),  SyntheticPriceBuyForecaster(T_total))
    price_sell_series = TimeSeries(price_sell_state.make_sensor(), SyntheticPriceSellForecaster())
    load_series       = TimeSeries(load_state.make_sensor(),       SyntheticLoadForecaster(T_total))
    gen_series        = TimeSeries(gen_state.make_sensor(),        SyntheticGenForecaster(T_total))
    temp_out_series   = TimeSeries(temp_out_state.make_sensor(),   SyntheticTempOutForecaster(T_total))

    # --- Optimization models ---
    bat_model = BatteryModel(
        capacity=10.0, soc_init=4.0,
        charge_max=3.0, discharge_max=3.0, soc_min=1.0, efficiency=0.95,
    )
    ev_model = EVModel(
        capacity=60.0, soc_init=15.0,
        charge_max=11.0, discharge_max=11.0,
        target_soc=50.0, target_timestep=T_total + 1, efficiency=0.95,
    )
    hp_model = HeatPumpModel(
        temp_init=21.5,
        temp_min=np.full(T_horizon, 20.0),
        temp_max=np.full(T_horizon, 23.0),
        temp_out_series=temp_out_series,
        max_power=3.0, C_therm=5.0, lambda_=0.25, cop_eta=0.4,
    )

    # --- Entity states with physics ---
    bat_state = SyntheticState(
        initial=bat_model.soc_init, dt=dt,
        update_fn=lambda soc, p, _dt: soc + p * _dt * bat_model.efficiency,
    )
    ev_state = SyntheticState(
        initial=ev_model.soc_init, dt=dt,
        update_fn=lambda soc, p, _dt: soc + p * _dt * ev_model.efficiency,
    )

    def hp_update(temp_in: float, p: float, _dt: float) -> float:
        temp_out = temp_out_state.read()
        T_in_K   = temp_in  + 273.15
        T_out_K  = temp_out + 273.15
        cop      = min(max(hp_model.cop_eta * T_in_K / max(T_in_K - T_out_K, 1.0), 1.0), 6.0)
        return (temp_in * (1 - hp_model.lambda_ * _dt / hp_model.C_therm)
                + cop * p * _dt / hp_model.C_therm
                + hp_model.lambda_ * _dt * temp_out / hp_model.C_therm)

    hp_state = SyntheticState(initial=hp_model.temp_init, dt=dt, update_fn=hp_update)

    # --- Sensors, actuators, entities ---
    bat_sensor, bat_actuator = bat_state.make_sensor_actuator()
    ev_sensor,  ev_actuator  = ev_state.make_sensor_actuator()
    hp_sensor,  hp_actuator  = hp_state.make_sensor_actuator()

    entities = [
        PhysicalEntity(bat_sensor, bat_model, bat_actuator),
        PhysicalEntity(ev_sensor,  ev_model,  ev_actuator),
        PhysicalEntity(hp_sensor,  hp_model,  hp_actuator),
    ]

    # --- Signal driver ---
    rng = np.random.default_rng(0)

    def drive(step: int) -> None:
        frac = step / T_total
        price_buy_state.set(float(0.28 + 0.08 * np.cos(2 * np.pi * frac) + rng.normal(0, 0.01)))
        gen_state.set(float(max(0.0, 3.5 * np.exp(-0.5 * ((frac - 0.5) / 0.18) ** 2) + rng.normal(0, 0.1))))
        load_state.set(float(max(0.0, 0.6 + 0.25 * abs(np.sin(np.pi * frac)) + rng.normal(0, 0.05))))
        temp_out_state.set(float(4.0 + 4.0 * np.sin(np.pi * frac) + rng.normal(0, 0.3)))

    # --- Run ---
    mpc = SyntheticMPC(
        signal_driver=drive,
        entities=entities,
        price_buy=price_buy_series,
        price_sell=price_sell_series,
        load=load_series,
        gen=gen_series,
        extra_series=[temp_out_series],
        T_horizon=T_horizon,
        dt=dt,
    )

    history = mpc.run(T_total)
    plot_results(history, dt, show=True)


if __name__ == "__main__":
    main()
