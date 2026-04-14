"""Entry point for the household energy scheduling POC."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from control import MPC, BatteryModel, EVModel, HeatPumpModel, PhysicalEntity, MPCHistory
from core import Time, TimeSeriesData
from core.config import AppConfig, SystemConfig, EntityConfig, load_config
from dashboard.dashboard import LiveStore, create_dashboard
from forecasting import TimeSeries, LookupForecaster, ConstantForecaster
from interfaces.synthetic import SyntheticState, SyntheticExternalState
from plot import plot_results, plot_forecast_comparison


def _build_signals(
    t0: datetime,
    T: int,
    dt: timedelta,
    rng: np.random.Generator,
    system_cfg: SystemConfig,
) -> dict[str, TimeSeriesData]:
    """Precompute passive signal trajectories as TimeSeriesData objects."""
    signals = {k: TimeSeriesData() for k in ("price_buy", "price_sell", "load", "gen", "temp_out")}
    T_day = int(timedelta(hours=24) / dt)

    sig_cfg = system_cfg.synthetic_signals

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

        signals["price_sell"].add_point(t, float(sig_cfg.price_sell.constant))

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


def _build_series(mode: str, signals: dict[str, TimeSeriesData]) -> dict[str, TimeSeries]:
    def _ts(key: str) -> TimeSeries:
        sensor = SyntheticExternalState(signals[key]).make_sensor()
        forecaster = LookupForecaster(signals[key]) if mode == "lookup" else ConstantForecaster()
        return TimeSeries(sensor, forecaster)

    return {k: _ts(k) for k in ("price_buy", "price_sell", "load", "gen", "temp_out")}


def _build_physical_entity(
    entity_cfg: EntityConfig,
    series: dict[str, TimeSeries],
    t_horizon: int,
    t_total: int,
) -> PhysicalEntity | None:
    if not entity_cfg.enabled:
        return None

    p = entity_cfg.parameters
    s = entity_cfg.initial_state

    if entity_cfg.type == "battery":
        state = SyntheticState(s["soc"])
        sensor, actuator = state.make_sensor_actuator()
        model = BatteryModel(
            capacity=p["capacity"],
            charge_max=p["charge_max"],
            discharge_max=p["discharge_max"],
            soc_min=p["soc_min"],
            efficiency=p["efficiency"],
            wearing_cost_per_kwh=p["wearing_cost_per_kwh"],
        )
    elif entity_cfg.type == "ev":
        state = SyntheticState(s["soc"])
        sensor, actuator = state.make_sensor_actuator()
        model = EVModel(
            capacity=p["capacity"],
            charge_max=p["charge_max"],
            discharge_max=p["discharge_max"],
            target_soc=p["target_soc"],
            target_timestep=t_total + 1,
            efficiency=p["efficiency"],
            wearing_cost_per_kwh=p["wearing_cost_per_kwh"],
        )
    elif entity_cfg.type == "heat_pump":
        state = SyntheticState(s["temperature"])
        sensor, actuator = state.make_sensor_actuator()
        model = HeatPumpModel(
            temp_min=np.full(t_horizon, p["temp_min"]),
            temp_max=np.full(t_horizon, p["temp_max"]),
            temp_out_series=series["temp_out"],
            max_power=p["max_power"],
            C_therm=p["c_therm"],
            lambda_=p["lambda_"],
            cop_eta=p["cop_eta"],
            wearing_cost_per_kwh=p["wearing_cost_per_kwh"],
        )
    else:
        raise ValueError(f"Unknown entity type: {entity_cfg.type}")

    return PhysicalEntity(sensor, model, actuator)


def _run_mode(
    mode: str,
    config: AppConfig,
    system_cfg: SystemConfig,
    on_step=None,
) -> "MPCHistory":
    sim_cfg = config.simulation

    Time.get_instance().set(sim_cfg.t0)

    rng = np.random.default_rng(config.simulation.seed)
    signals = _build_signals(
        t0=config.simulation.t0,
        T=config.simulation.t_total + config.simulation.t_horizon,
        dt=config.simulation.dt,
        rng=rng,
        system_cfg=system_cfg,
    )

    series = _build_series(mode, signals)
    entities = [
        _build_physical_entity(
            entity_cfg=entity_cfg,
            series=series,
            t_horizon=sim_cfg.t_horizon,
            t_total=sim_cfg.t_total,
        ) for entity_cfg in system_cfg.entities
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
    print(f"  System: {system_cfg.id} | Forecasting mode: {mode}")
    print(f"{'=' * 65}")
    return mpc.run(sim_cfg.t_total, fast_forward=True, on_step=on_step)


def main() -> None:
    config = load_config("config/sim_default.yaml")

    output_dir = Path(config.outputs.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    enabled_systems = [system for system in config.energy_world if system.enabled]
    if not enabled_systems:
        raise ValueError("No enabled systems found in config.")

    system_cfg = enabled_systems[0]

    live_store = LiveStore(config.forecasting.modes)

    def _run_simulation():
        histories: dict[str, MPCHistory] = {}
        for mode in config.forecasting.modes:
            live_store.set_active_mode(mode)
            history = _run_mode(
                mode, config, system_cfg,
                on_step=lambda step, m=mode: live_store.add_step(m, step),
            )
            histories[mode] = history

            plot_results(
                history,
                dt=config.simulation.dt,
                path=str(output_dir / f"{system_cfg.id}_mpc_results_{mode}.png"),
                show=config.outputs.show_individual_plots,
            )

        plot_forecast_comparison(
            histories,
            dt=config.simulation.dt,
            path=str(output_dir / f"{system_cfg.id}_mpc_comparison.png"),
            show=config.outputs.show_comparison_plot,
        )
        live_store.mark_done()

    sim_thread = threading.Thread(target=_run_simulation, daemon=True)
    sim_thread.start()

    create_dashboard(live_store, config.simulation.dt, config.forecasting.modes, config.simulation.t_total).run(debug=False, port=8050)


if __name__ == "__main__":
    main()