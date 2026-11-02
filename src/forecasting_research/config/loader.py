"""Load and validate YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..domain.exceptions import ConfigError
from .schemas import AppConfig


def format_validation_error(err: ValidationError) -> str:
    lines = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "<root>"
        lines.append(f"  - {loc}: {e['msg']}")
    return "Invalid configuration:\n" + "\n".join(lines)


def config_from_dict(data: dict[str, Any] | None) -> AppConfig:
    try:
        return AppConfig.model_validate(data or {})
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc)) from exc


def config_from_yaml_text(text: str) -> AppConfig:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML syntax error: {exc}") from exc
    if data is not None and not isinstance(data, dict):
        raise ConfigError("Configuration root must be a mapping")
    return config_from_dict(data)


def load_config(path: str | Path | None) -> AppConfig:
    """Load a YAML file; `None` returns the validated defaults."""
    if path is None:
        return AppConfig()
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Configuration file not found: {p}")
    return config_from_yaml_text(p.read_text(encoding="utf-8"))


def dump_config_yaml(cfg: AppConfig) -> str:
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
