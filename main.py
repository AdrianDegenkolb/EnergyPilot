"""Entry point for the household energy scheduling POC."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from core.config import AppConfig, load_config
from control import MPC, BatteryModel, EVModel, HeatPumpModel, PhysicalEntity
from core import Time, TimeSeriesData
from forecasting import TimeSeries, LookupForecaster, ConstantForecaster
from interfaces.synthetic import SyntheticState, SyntheticExternalState
from plot import plot_results, plot_forecast_comparison


def _build_signals(
    t0: datetime,
    T: int,
    dt: timedelta,
    rng: np.random.Generator,
    config: AppConfig,
) -> dict[str, TimeSeriesData]:
    """Precompute passive signal trajectories as TimeSeriesData objects."""
    signals = {k: TimeSeriesData() for k in ("price_buy", "price_sell", "load", "gen", "temp_out")}
    T_day = int(timedelta(hours=24) / dt)

    sig_cfg = config.synthetic_signals

    for step in range(T):
        t = t0 + dt * step
        frac = step / T_day

        signals["price_buy"].add_point(
            t,
            float(
                sig_cfg.price_buy.base
                + sig_cfg.price_buy.amplitude * np.cos(2 * np.pi * frac)
                + rng.normal(0, sig_cfg.price_buy.noise_std)
            ),
        )

        signals["price_sell"].add_point(
            t,
            float(sig_cfg.price_sell.constant),
        )

        signals["load"].add_point(
            t,
            float(
                max(
                    0.0,
                    sig_cfg.load.base
                    + sig_cfg.load.amplitude * abs(np.sin(np.pi * frac))
                    + rng.normal(0, sig_cfg.load.noise_std),
                )
            ),
        )

        signals["gen"].add_point(
            t,
            float(
                max(
                    0.0,
                    sig_cfg.gen.peak * np.exp(-0.5 * ((frac - 0.5) / sig_cfg.gen.width) ** 2)
                    + rng.normal(0, sig_cfg.gen.noise_std),
                )
            ),
        )

        signals["temp_out"].add_point(
            t,
            float(
                sig_cfg.temp_out.base
                + sig_cfg.temp_out.amplitude * np.sin(np.pi * frac)
                + rng.normal(0, sig_cfg.temp_out.noise_std)
            ),
        )

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
    config: AppConfig,
) -> "MPCHistory":
    sim_cfg = config.simulation
    entity_cfg = config.entities
    hp_cfg = entity_cfg.heat_pump

    Time.get_instance().set(sim_cfg.t0)

    series = _make_series(mode, signals)

    bat_state = SyntheticState(entity_cfg.battery.initial_soc)
    ev_state = SyntheticState(entity_cfg.ev.initial_soc)
    hp_state = SyntheticState(hp_cfg.initial_temp)

    bat_model = BatteryModel(
        capacity=entity_cfg.battery.capacity,
        charge_max=entity_cfg.battery.charge_max,
        discharge_max=entity_cfg.battery.discharge_max,
        soc_min=entity_cfg.battery.soc_min,
        efficiency=entity_cfg.battery.efficiency,
    )

    ev_model = EVModel(
        capacity=entity_cfg.ev.capacity,
        charge_max=entity_cfg.ev.charge_max,
        discharge_max=entity_cfg.ev.discharge_max,
        target_soc=entity_cfg.ev.target_soc,
        target_timestep=sim_cfg.t_total + 1,
        efficiency=entity_cfg.ev.efficiency,
    )

    hp_model = HeatPumpModel(
        temp_min=np.full(sim_cfg.t_horizon, hp_cfg.temp_min),
        temp_max=np.full(sim_cfg.t_horizon, hp_cfg.temp_max),
        temp_out_series=series["temp_out"],
        max_power=hp_cfg.max_power,
        C_therm=hp_cfg.c_therm,
        lambda_=hp_cfg.lambda_,
        cop_eta=hp_cfg.cop_eta,
    )

    bat_sensor, bat_actuator = bat_state.make_sensor_actuator()
    ev_sensor, ev_actuator = ev_state.make_sensor_actuator()
    hp_sensor, hp_actuator = hp_state.make_sensor_actuator()

    entities = [
        PhysicalEntity(bat_sensor, bat_model, bat_actuator),
        PhysicalEntity(ev_sensor, ev_model, ev_actuator),
        PhysicalEntity(hp_sensor, hp_model, hp_actuator),
    ]

    mpc = MPC(
        entities=entities,
        price_buy=series["price_buy"],
        price_sell=series["price_sell"],
        load=series["load"],
        gen=series["gen"],
        extra_series=[series["temp_out"]],
        T_horizon=sim_cfg.t_horizon,
        dt=sim_cfg.dt,
    )

    print(f"\n{'=' * 65}")
    print(f"  Forecasting mode: {mode}")
    print(f"{'=' * 65}")
    return mpc.run(sim_cfg.t_total, fast_forward=True)


def main() -> None:
    config = load_config("config/sim_default.yaml")

    output_dir = Path(config.outputs.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.simulation.seed)
    signals = _build_signals(
        t0=config.simulation.t0,
        T=config.simulation.t_total + config.simulation.t_horizon,
        dt=config.simulation.dt,
        rng=rng,
        config=config,
    )

    histories: dict[str, "MPCHistory"] = {}
    for mode in config.forecasting.modes:
        history = _run_mode(mode, signals, config)
        histories[mode] = history

        plot_results(
            history,
            dt=config.simulation.dt,
            path=str(output_dir / f"mpc_results_{mode}.png"),
            show=config.outputs.show_individual_plots,
        )

    plot_forecast_comparison(
        histories,
        dt=config.simulation.dt,
        path=str(output_dir / "mpc_comparison.png"),
        show=config.outputs.show_comparison_plot,
    )


if __name__ == "__main__":
    main()