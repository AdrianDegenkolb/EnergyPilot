from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml


@dataclass
class SimulationConfig:
    start_time: datetime
    dt_minutes: int
    duration_hours: int
    horizon_hours: int
    seed: int

    @property
    def dt(self) -> timedelta:
        return timedelta(minutes=self.dt_minutes)

    @property
    def t0(self) -> datetime:
        return self.start_time

    @property
    def t_total(self) -> int:
        return int(timedelta(hours=self.duration_hours) / self.dt)

    @property
    def t_horizon(self) -> int:
        return int(timedelta(hours=self.horizon_hours) / self.dt)


@dataclass
class PriceBuySignalConfig:
    base: float
    amplitude: float
    noise_std: float


@dataclass
class PriceSellSignalConfig:
    constant: float


@dataclass
class LoadSignalConfig:
    base: float
    amplitude: float
    noise_std: float


@dataclass
class GenSignalConfig:
    peak: float
    width: float
    noise_std: float


@dataclass
class TempOutSignalConfig:
    base: float
    amplitude: float
    noise_std: float


@dataclass
class SyntheticSignalsConfig:
    price_buy: PriceBuySignalConfig
    price_sell: PriceSellSignalConfig
    load: LoadSignalConfig
    gen: GenSignalConfig
    temp_out: TempOutSignalConfig


@dataclass
class ForecastingConfig:
    modes: list[str]


@dataclass
class BatteryConfig:
    initial_soc: float
    capacity: float
    charge_max: float
    discharge_max: float
    soc_min: float
    efficiency: float


@dataclass
class EVConfig:
    initial_soc: float
    capacity: float
    charge_max: float
    discharge_max: float
    target_soc: float
    efficiency: float


@dataclass
class HeatPumpConfig:
    initial_temp: float
    temp_min: float
    temp_max: float
    max_power: float
    c_therm: float
    lambda_: float
    cop_eta: float


@dataclass
class EntitiesConfig:
    battery: BatteryConfig
    ev: EVConfig
    heat_pump: HeatPumpConfig


@dataclass
class OutputsConfig:
    directory: str
    show_individual_plots: bool
    show_comparison_plot: bool


@dataclass
class AppConfig:
    simulation: SimulationConfig
    forecasting: ForecastingConfig
    synthetic_signals: SyntheticSignalsConfig
    entities: EntitiesConfig
    outputs: OutputsConfig


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        simulation=SimulationConfig(
            start_time=datetime.fromisoformat(raw["simulation"]["start_time"]),
            dt_minutes=raw["simulation"]["dt_minutes"],
            duration_hours=raw["simulation"]["duration_hours"],
            horizon_hours=raw["simulation"]["horizon_hours"],
            seed=raw["simulation"]["seed"],
        ),
        forecasting=ForecastingConfig(
            modes=list(raw["forecasting"]["modes"]),
        ),
        synthetic_signals=SyntheticSignalsConfig(
            price_buy=PriceBuySignalConfig(**raw["synthetic_signals"]["price_buy"]),
            price_sell=PriceSellSignalConfig(**raw["synthetic_signals"]["price_sell"]),
            load=LoadSignalConfig(**raw["synthetic_signals"]["load"]),
            gen=GenSignalConfig(**raw["synthetic_signals"]["gen"]),
            temp_out=TempOutSignalConfig(**raw["synthetic_signals"]["temp_out"]),
        ),
        entities=EntitiesConfig(
            battery=BatteryConfig(**raw["entities"]["battery"]),
            ev=EVConfig(**raw["entities"]["ev"]),
            heat_pump=HeatPumpConfig(**raw["entities"]["heat_pump"]),
        ),
        outputs=OutputsConfig(**raw["outputs"]),
    )