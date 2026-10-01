"""Configuration loading and project paths."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Paths:
    data: Path
    results: Path

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.interim, self.processed, self.results):
            p.mkdir(parents=True, exist_ok=True)
        return self


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, encoding="utf-8") as f:
            cfg = _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def get_paths(cfg: dict) -> Paths:
    p = cfg["paths"]
    data, results = Path(p["data"]), Path(p["results"])
    if not data.is_absolute():
        data = ROOT / data
    if not results.is_absolute():
        results = ROOT / results
    return Paths(data, results).ensure()
