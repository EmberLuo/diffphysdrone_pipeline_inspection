"""Wind sampling helpers aligned with training_tasks robust_target_hover."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


WIND_MODES = (
    "constant",
    "gust",
    "side",
    "vertical",
    "mixed",
    "constant_wind",
    "gust_wind",
    "side_wind",
    "vertical_wind",
)


@dataclass
class WindSample:
    policy_step: int
    wind_train_world: list[float]
    wind_airsim_ned: list[float]
    wind_norm: float
    resampled: bool
    sample_index: int


def add_wind_args(parser) -> None:
    parser.add_argument("--use_wind", default=False, action="store_true")
    parser.add_argument("--wind_mode", default="mixed", choices=WIND_MODES)
    parser.add_argument("--wind_mean_range", type=float, nargs="+", default=[-2.0, 2.0])
    parser.add_argument("--wind_gust_range", type=float, nargs="+", default=[-3.0, 3.0])
    parser.add_argument("--wind_vertical_range", type=float, nargs="+", default=[0.0, 1.5])
    parser.add_argument("--wind_side_range", type=float, nargs="+", default=[0.0, 12.0])
    parser.add_argument("--wind_update_interval", type=int, default=8)
    parser.add_argument("--wind_randomize_prob", type=float, default=0.85)


def wind_config_from_args(args) -> dict[str, Any]:
    return {
        "use_wind": bool(getattr(args, "use_wind", False)),
        "wind_mode": normalize_wind_mode(getattr(args, "wind_mode", "mixed")),
        "wind_mode_raw": str(getattr(args, "wind_mode", "mixed")),
        "wind_mean_range": _as_float_list(getattr(args, "wind_mean_range", [])),
        "wind_gust_range": _as_float_list(getattr(args, "wind_gust_range", [])),
        "wind_vertical_range": _as_float_list(getattr(args, "wind_vertical_range", [])),
        "wind_side_range": _as_float_list(getattr(args, "wind_side_range", [])),
        "wind_update_interval": int(getattr(args, "wind_update_interval", 8)),
        "wind_randomize_prob": float(getattr(args, "wind_randomize_prob", 0.85)),
    }


def normalize_wind_mode(mode: str) -> str:
    aliases = {
        "constant_wind": "constant",
        "gust_wind": "gust",
        "side_wind": "side",
        "vertical_wind": "vertical",
    }
    return aliases.get(str(mode).lower(), str(mode).lower())


def train_world_to_airsim_ned(wind_train_world: list[float] | np.ndarray) -> list[float]:
    wind = [float(x) for x in wind_train_world]
    return [wind[0], -wind[1], -wind[2]]


class WindSampler:
    def __init__(self, config: dict[str, Any], rng=None):
        self.config = dict(config)
        self.rng = rng if rng is not None else np.random
        self.use_wind = bool(self.config.get("use_wind", False))
        self.mode = normalize_wind_mode(self.config.get("wind_mode", "mixed"))
        self.interval = max(1, int(self.config.get("wind_update_interval", 8)))
        self.current = np.zeros(3, dtype=float)
        self.last_step: int | None = None
        self.sample_index = 0

    def step(self, policy_step: int) -> WindSample:
        resampled = self._should_resample(policy_step)
        if resampled:
            self.current = self._sample_wind() if self.use_wind else np.zeros(3, dtype=float)
            self.sample_index += 1
        wind_train = self.current.astype(float).tolist()
        wind_airsim = train_world_to_airsim_ned(wind_train)
        self.last_step = int(policy_step)
        return WindSample(
            policy_step=int(policy_step),
            wind_train_world=wind_train,
            wind_airsim_ned=wind_airsim,
            wind_norm=float(np.linalg.norm(self.current)),
            resampled=bool(resampled),
            sample_index=int(self.sample_index),
        )

    def _should_resample(self, policy_step: int) -> bool:
        if self.last_step is None:
            return True
        return self.mode in {"gust", "mixed"} and policy_step > 0 and policy_step % self.interval == 0

    def _sample_wind(self) -> np.ndarray:
        wind = _sample_uniform(self.config.get("wind_mean_range", []), 3, self.rng)
        if self.mode in {"side", "mixed"}:
            wind[1] += _sample_signed_component(self.config.get("wind_side_range", []), self.rng)
        if self.mode in {"vertical", "mixed"}:
            wind[2] += _sample_signed_component(self.config.get("wind_vertical_range", []), self.rng)
        if self.mode in {"gust", "mixed"}:
            wind += _sample_uniform(self.config.get("wind_gust_range", []), 3, self.rng)

        prob = min(1.0, max(0.0, float(self.config.get("wind_randomize_prob", 1.0))))
        if prob < 1.0 and float(self.rng.random()) >= prob:
            wind = np.zeros(3, dtype=float)
        return wind.astype(float)


def summarize_wind_trace(wind_records: list[dict[str, Any]], config: dict[str, Any], seed: int) -> dict[str, Any]:
    norms = [float(rec.get("wind_norm", 0.0)) for rec in wind_records]
    nonzero = [value for value in norms if value > 1e-9]
    return {
        "seed": int(seed),
        "config": dict(config),
        "record_count": len(wind_records),
        "sample_count": sum(1 for rec in wind_records if rec.get("resampled")),
        "effective_wind_ratio": (len(nonzero) / len(norms)) if norms else 0.0,
        "mean_wind_norm": (sum(norms) / len(norms)) if norms else 0.0,
        "max_wind_norm": max(norms) if norms else 0.0,
    }


def wind_sample_to_record(sample: WindSample, sim_time: float, wall_time: float) -> dict[str, Any]:
    return {
        "policy_step": sample.policy_step,
        "sim_time": float(sim_time),
        "wall_time": float(wall_time),
        "wind_train_world": sample.wind_train_world,
        "wind_airsim_ned": sample.wind_airsim_ned,
        "wind_norm": sample.wind_norm,
        "resampled": sample.resampled,
        "sample_index": sample.sample_index,
    }


def _sample_uniform(value: Any, dim: int, rng) -> np.ndarray:
    flat = _as_float_list(value)
    if len(flat) == 0:
        return np.zeros(dim, dtype=float)
    if len(flat) == 1:
        return np.full(dim, flat[0], dtype=float)
    if len(flat) == 2:
        lo, hi = flat
        return rng.uniform(lo, hi, size=dim).astype(float)
    if len(flat) == dim:
        return np.asarray(flat, dtype=float).copy()
    if len(flat) == dim * 2:
        pairs = np.asarray(flat, dtype=float).reshape(dim, 2)
        return rng.uniform(pairs[:, 0], pairs[:, 1]).astype(float)
    raise ValueError(f"Cannot interpret range {value!r} for dimension {dim}")


def _sample_signed_component(value: Any, rng) -> float:
    flat = _as_float_list(value)
    if len(flat) == 2 and flat[0] >= 0.0:
        mag = float(rng.uniform(flat[0], flat[1]))
        sign = -1.0 if float(rng.random()) < 0.5 else 1.0
        return mag * sign
    return float(_sample_uniform(value, 1, rng)[0])


def _as_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, str):
        cleaned = value.replace(",", " ").split()
        return [float(x) for x in cleaned]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return [float(value)]
    return [float(value)]
