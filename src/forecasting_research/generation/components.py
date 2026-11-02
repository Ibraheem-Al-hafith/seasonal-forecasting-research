"""Mathematics of the synthetic series (see docs/research_methodology.md).

Time index t = 0..n-1.

Trend           : base                       (none)
                  base + slope * t           (linear)
Seasonal profile: p(t) = zero-mean, max|p| = 1, exactly periodic with period m,
                  fundamental sinusoid + second harmonic (amplitude h2), random phase.
Additive        : Y = Trend + A * p + eps,           eps ~ N(0, sigma * base)
Multiplicative  : Y = Trend * (1 + a * p) * (1 + e), e   ~ N(0, sigma), clipped to +-clip
"""

from __future__ import annotations

import numpy as np

from ..config.schemas import GeneratorParams
from ..domain.contracts import GroundTruth
from ..domain.exceptions import DataError


def trend_component(trend_type: str, n: int, params: GeneratorParams) -> np.ndarray:
    t = np.arange(n, dtype=float)
    if trend_type == "none":
        return np.full(n, params.base_level, dtype=float)
    if trend_type == "linear":
        return params.base_level + params.trend_slope * t
    raise DataError(f"Unknown trend type: {trend_type}")


def seasonal_profile(period: int, n: int, phase: float, harmonic2: float) -> np.ndarray:
    """Zero-mean (over a cycle), unit-max-amplitude, exactly periodic profile of length n."""
    c = np.arange(period, dtype=float)
    cycle = np.sin(2 * np.pi * c / period + phase) + harmonic2 * np.sin(4 * np.pi * c / period + 2 * phase)
    cycle = cycle - cycle.mean()
    peak = np.max(np.abs(cycle))
    if peak < 1e-12:  # pathological phase; fall back to the plain fundamental
        cycle = np.sin(2 * np.pi * c / period)
        cycle = cycle - cycle.mean()
        peak = np.max(np.abs(cycle))
    cycle = cycle / peak
    return cycle[np.arange(n) % period]


def build_series(
    seasonality_type: str,
    period: int,
    trend_type: str,
    noise_level: str,
    n: int,
    seed: int,
    params: GeneratorParams,
) -> tuple[np.ndarray, GroundTruth, dict]:
    """Generate observed series + ground truth. Draw order (phase, then noise) is fixed."""
    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(0.0, 2 * np.pi))
    sigma = params.noise_sigma[noise_level]
    trend = trend_component(trend_type, n, params)
    profile = seasonal_profile(period, n, phase, params.harmonic2_amplitude)

    if seasonality_type == "additive":
        seasonal = params.additive_amplitude * profile
        noise = rng.normal(0.0, sigma * params.base_level, size=n)
        observed = trend + seasonal + noise
        noise_kind = "additive"
    elif seasonality_type == "multiplicative":
        seasonal = 1.0 + params.multiplicative_amplitude * profile
        eps = np.clip(rng.normal(0.0, sigma, size=n), -params.multiplicative_noise_clip, params.multiplicative_noise_clip)
        noise = 1.0 + eps  # positive factor
        observed = trend * seasonal * noise
        noise_kind = "multiplicative_factor"
        if np.any(observed <= 0) or np.any(seasonal <= 0) or np.any(noise <= 0):
            raise DataError("Multiplicative generation produced non-positive values")
    else:
        raise DataError(f"Unknown seasonality type: {seasonality_type}")

    if not np.all(np.isfinite(observed)):
        raise DataError("Generated series contains non-finite values")
    truth = GroundTruth(trend=trend, seasonal=seasonal, noise=noise, noise_kind=noise_kind)
    return observed, truth, {"phase": phase, "sigma": sigma}
