from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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
class ForecastingConfig:
    modes: list[str]


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
class EntityConfig:
    type: str
    id: str
    role: str | None
    enabled: bool
    initial_state: dict[str, Any]
    parameters: dict[str, Any]


@dataclass
class SystemConfig:
    id: str
    type: str
    enabled: bool
    synthetic_signals: SyntheticSignalsConfig
    entities: list[EntityConfig]


@dataclass
class OutputsConfig:
    directory: str
    show_individual_plots: bool
    show_comparison_plot: bool


@dataclass
class AppConfig:
    simulation: SimulationConfig
    forecasting: ForecastingConfig
    energy_world: list[SystemConfig]
    outputs: OutputsConfig


def _load_synthetic_signals(raw: dict[str, Any]) -> SyntheticSignalsConfig:
    return SyntheticSignalsConfig(
        price_buy=PriceBuySignalConfig(**raw["price_buy"]),
        price_sell=PriceSellSignalConfig(**raw["price_sell"]),
        load=LoadSignalConfig(**raw["load"]),
        gen=GenSignalConfig(**raw["gen"]),
        temp_out=TempOutSignalConfig(**raw["temp_out"]),
    )


def _load_entities(raw_entities: list[dict[str, Any]]) -> list[EntityConfig]:
    return [
        EntityConfig(
            type=entity["type"],
            id=entity.get("id", f"{entity['type']}_{uuid.uuid4()}"),
            role=entity.get("role"),
            enabled=entity.get("enabled", True),
            initial_state=entity.get("initial_state", {}),
            parameters=entity.get("parameters", {}),
        )
        for entity in raw_entities
    ]


def _load_systems(raw_energy_world: list[dict[str, Any]]) -> list[SystemConfig]:
    return [
        SystemConfig(
            id=system["id"],
            type=system["type"],
            enabled=system.get("enabled", True),
            synthetic_signals=_load_synthetic_signals(system["synthetic_signals"]),
            entities=_load_entities(system.get("entities", [])),
        )
        for system in raw_energy_world
    ]


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
        energy_world=_load_systems(raw["energy_world"]),
        outputs=OutputsConfig(**raw["outputs"]),
    )