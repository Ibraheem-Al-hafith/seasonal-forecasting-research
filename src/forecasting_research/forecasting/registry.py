"""Method registry: stable names -> forecaster factories."""

from __future__ import annotations

from collections.abc import Callable

from ..config.schemas import ForecastingCfg
from ..domain.exceptions import ConfigError
from .base import Forecaster

Factory = Callable[[ForecastingCfg], Forecaster]
_REGISTRY: dict[str, Factory] = {}


def register(name: str, factory: Factory | None = None):
    """Register a factory (usable as a function or as a decorator on a class with `from_config`)."""

    def _do(obj):
        fac = obj if factory is not None and obj is factory else getattr(obj, "from_config", obj)
        _REGISTRY[name] = fac
        return obj

    if factory is not None:
        _REGISTRY[name] = factory
        return factory
    return _do


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def available_methods() -> list[str]:
    _ensure_builtin()
    return sorted(_REGISTRY)


def validate_method_names(names: list[str]) -> None:
    _ensure_builtin()
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise ConfigError(f"Unknown forecasting method(s): {unknown}. Available: {sorted(_REGISTRY)}")


def create_method(name: str, cfg: ForecastingCfg) -> Forecaster:
    _ensure_builtin()
    if name not in _REGISTRY:
        raise ConfigError(f"Unknown forecasting method: {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)


def _ensure_builtin() -> None:
    # Importing the modules triggers their @register decorators.
    from . import classical_decomposition, holt_winters, sarima  # noqa: F401
