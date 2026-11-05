"""Pydantic configuration schema (validated before anything executes)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..domain.enums import NoiseLevel, SeasonalityType, TrendType
from .defaults import DEFAULT_METHODS, DEFAULT_NOISE_SIGMA, MIN_TRAIN_CYCLES


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentCfg(_Strict):
    name: str = "seasonal_forecasting_comparison"
    description: str = ""


class GeneratorParams(_Strict):
    """Exact generator mathematics. Recorded in every dataset's metadata and ID hash."""

    base_level: float = Field(100.0, gt=0)
    trend_slope: float = Field(0.3, ge=0)
    additive_amplitude: float = Field(15.0, gt=0)
    multiplicative_amplitude: float = Field(0.15, gt=0, lt=1)
    harmonic2_amplitude: float = Field(0.4, ge=0, le=1)
    noise_sigma: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_NOISE_SIGMA))
    multiplicative_noise_clip: float = Field(0.5, gt=0, lt=1)

    @field_validator("noise_sigma")
    @classmethod
    def _check_sigma(cls, v: dict[str, float]) -> dict[str, float]:
        missing = {n.value for n in NoiseLevel} - set(v)
        if missing:
            raise ValueError(f"noise_sigma missing levels: {sorted(missing)}")
        if any(s <= 0 for s in v.values()):
            raise ValueError("noise_sigma values must be positive")
        return v


def _unique_nonempty(values: list, name: str) -> list:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates")
    return values


class GenerationCfg(_Strict):
    seasonality_types: list[SeasonalityType] = Field(
        default_factory=lambda: [SeasonalityType.ADDITIVE, SeasonalityType.MULTIPLICATIVE]
    )
    seasonal_periods: list[int] = Field(default_factory=lambda: [4, 7, 12])
    trends: list[TrendType] = Field(default_factory=lambda: [TrendType.NONE, TrendType.LINEAR])
    noise_levels: list[NoiseLevel] = Field(
        default_factory=lambda: [NoiseLevel.LOW, NoiseLevel.MEDIUM, NoiseLevel.HIGH]
    )
    dataset_lengths: list[int] = Field(default_factory=lambda: [100, 200, 500])
    replications: int = Field(5, ge=1, le=1000)
    base_seed: int = Field(42, ge=0, lt=2**32)
    params: GeneratorParams = Field(default_factory=GeneratorParams)

    @field_validator("seasonality_types", "trends", "noise_levels", "seasonal_periods", "dataset_lengths")
    @classmethod
    def _lists(cls, v: list, info) -> list:
        return _unique_nonempty(v, info.field_name)

    @field_validator("seasonal_periods")
    @classmethod
    def _periods(cls, v: list[int]) -> list[int]:
        if any(p < 2 for p in v):
            raise ValueError("seasonal periods must be >= 2")
        return v

    @field_validator("dataset_lengths")
    @classmethod
    def _lengths(cls, v: list[int]) -> list[int]:
        if any(n < 10 for n in v):
            raise ValueError("dataset lengths must be >= 10")
        return v

    def combination_count(self) -> int:
        return (
            len(self.seasonality_types)
            * len(self.seasonal_periods)
            * len(self.trends)
            * len(self.noise_levels)
            * len(self.dataset_lengths)
        )

    def dataset_count(self) -> int:
        return self.combination_count() * self.replications


class EvaluationCfg(_Strict):
    train_fraction: float = Field(0.8, gt=0.0, lt=1.0)
    forecast_horizon: int | None = Field(None, ge=1)  # None -> one seasonal cycle
    seasonal_mode: Literal["matched", "additive", "multiplicative"] = "matched"


SARIMA_METHOD_VERSION = 2  # v1 = original implementation (seasonal-strength D, level-only KPSS, full grid, AIC)


class SarimaCfg(_Strict):
    """SARIMA search configuration (method version 2). Every field is recorded with each experiment."""

    method_version: Literal[2] = SARIMA_METHOD_VERSION

    # ---- seasonal differencing (D) -------------------------------------------------------------------
    # fixed_one  : D = 1 (default). A SARIMA without seasonal regressors can only represent a FIXED seasonal pattern
    #              through seasonal differencing (with a seasonal MA term); measured on this generator it is as accurate
    #              as Holt-Winters / classical decomposition, whereas D = 0 is ~20 % worse and leaves autocorrelated
    #              residuals in ~90 % of fits (docs/sarima_methodology.md).
    # fixed_zero : D = 0. Theoretically cleanest for deterministic seasonality (no seasonal unit root) but handicaps
    #              SARIMA; use it as an explicit sensitivity experiment, not silently.
    # ocsb / ch  : data-driven seasonal unit-root tests (OCSB / Canova-Hansen) via the optional pmdarima package
    # seasonal_strength : legacy rule "D = 1 if strength > threshold". NOT a unit-root test; kept only for comparison.
    seasonal_differencing_strategy: Literal["fixed_zero", "fixed_one", "ocsb", "ch", "seasonal_strength"] = "fixed_one"
    seasonal_strength_threshold: float = Field(0.64, gt=0, lt=1)  # used only by strategy seasonal_strength

    # ---- regular differencing (d) and deterministic trend ----------------------------------------------
    # c    : KPSS around a constant; rejection -> d = 1 (legacy behaviour)
    # ct   : KPSS around a linear trend; rejection -> d = 1, otherwise a deterministic linear trend, d = 0
    # auto : 'c' first; if rejected, 'ct'. Trend-stationary series keep d = 0 and get a deterministic trend
    #        (no information about the generator is used: this is decided from the training data only)
    kpss_regression: Literal["c", "ct", "auto"] = "auto"
    kpss_alpha: float = Field(0.05, gt=0, lt=1)

    # ---- search space ----------------------------------------------------------------------------------------
    max_pq_sum: int = Field(2, ge=0, le=4)  # p + q <= this
    max_seasonal_pq_sum: int = Field(1, ge=0, le=2)  # P + Q <= this
    selection_criterion: Literal["aic", "aicc", "bic"] = "aicc"
    max_iterations: int = Field(100, ge=10, le=2000)
    time_budget_seconds: float = Field(300.0, gt=0)

    # ---- search strategy -------------------------------------------------------------------------------------
    search_strategy: Literal["full_grid", "two_stage"] = "two_stage"
    stage1_max_pq_sum: int | None = Field(None, ge=0, le=4)  # None = auto: min(1, max_pq_sum)
    stage1_max_seasonal_pq_sum: int | None = Field(None, ge=0, le=2)  # None = auto: min(1, max_seasonal_pq_sum)
    # when to run stage 2 (always on TRAINING data only):
    #   boundary_or_ljung_box : best stage-1 model sits on a stage-1 limit OR training residuals are autocorrelated
    #   boundary | ljung_box  : only that indicator          always : full grid (== full_grid, but recorded as staged)
    stage2_trigger: Literal["boundary_or_ljung_box", "boundary", "ljung_box", "always"] = "boundary_or_ljung_box"
    ljung_box_alpha: float = Field(0.05, gt=0, lt=1)

    log_transform_multiplicative: bool = False

    @model_validator(mode="after")
    def _check_stage_limits(self) -> SarimaCfg:
        if (self.stage1_max_pq_sum or 0) > self.max_pq_sum or (self.stage1_max_seasonal_pq_sum or 0) > self.max_seasonal_pq_sum:
            raise ValueError("stage1 limits must not exceed max_pq_sum / max_seasonal_pq_sum")
        return self

    def stage1_pq(self) -> int:
        """Effective stage-1 limit for p + q (explicit value, or auto = min(1, max_pq_sum))."""
        return self.stage1_max_pq_sum if self.stage1_max_pq_sum is not None else min(1, self.max_pq_sum)

    def stage1_spq(self) -> int:
        """Effective stage-1 limit for P + Q."""
        return self.stage1_max_seasonal_pq_sum if self.stage1_max_seasonal_pq_sum is not None else min(1, self.max_seasonal_pq_sum)

    def candidate_count(self) -> int:
        from ..forecasting.sarima_components.candidates import full_grid

        return len(full_grid(self))


class HoltWintersCfg(_Strict):
    trend_options: list[Literal["none", "add", "mul"]] = Field(default_factory=lambda: ["none", "add", "mul"])
    selection_criterion: Literal["aicc"] = "aicc"

    @field_validator("trend_options")
    @classmethod
    def _nonempty(cls, v: list[str]) -> list[str]:
        return _unique_nonempty(v, "trend_options")


class ForecastingCfg(_Strict):
    methods: list[str] = Field(default_factory=lambda: list(DEFAULT_METHODS))
    sarima: SarimaCfg = Field(default_factory=SarimaCfg)
    holt_winters: HoltWintersCfg = Field(default_factory=HoltWintersCfg)

    @field_validator("methods")
    @classmethod
    def _methods(cls, v: list[str]) -> list[str]:
        return _unique_nonempty(v, "methods")


class ExecutionCfg(_Strict):
    """How independent evaluations are executed. Machine-specific: does not affect scientific results."""

    strategy: Literal["sequential", "parallel"] = "sequential"
    workers: int | Literal["auto"] = "auto"  # auto = CPU cores - 1 (capped at 8); or an integer >= 1
    start_method: Literal["spawn", "forkserver", "fork"] | None = None  # None -> "spawn" (safe everywhere)
    worker_modules: list[str] = Field(default_factory=list)  # imported in every worker (custom methods)
    max_tasks_per_worker: int | None = Field(None, ge=1)  # recycle workers (Python >= 3.11, not with fork)
    limit_blas_threads: bool = True  # 1 BLAS thread per worker: avoids CPU oversubscription
    max_worker_restarts: int = Field(3, ge=0, le=100)  # pool rebuilds after crashed workers

    @field_validator("workers")
    @classmethod
    def _workers(cls, v: int | str) -> int | str:
        if isinstance(v, int) and not isinstance(v, bool) and v < 1:
            raise ValueError("workers must be >= 1 or 'auto'")
        return v


class WorkspaceCfg(_Strict):
    path: str = "./workspace"


class LoggingCfg(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class AppConfig(_Strict):
    experiment: ExperimentCfg = Field(default_factory=ExperimentCfg)
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    evaluation: EvaluationCfg = Field(default_factory=EvaluationCfg)
    forecasting: ForecastingCfg = Field(default_factory=ForecastingCfg)
    execution: ExecutionCfg = Field(default_factory=ExecutionCfg)
    workspace: WorkspaceCfg = Field(default_factory=WorkspaceCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)

    @model_validator(mode="after")
    def _check_protocol(self) -> AppConfig:
        """Reject any generated design that cannot satisfy the train/test protocol."""
        ev = self.evaluation
        problems: list[str] = []
        for n in self.generation.dataset_lengths:
            train = int(n * ev.train_fraction)
            test = n - train
            for m in self.generation.seasonal_periods:
                horizon = ev.forecast_horizon or m
                if train < MIN_TRAIN_CYCLES * m:
                    problems.append(f"length {n}, period {m}: training size {train} < {MIN_TRAIN_CYCLES} cycles ({MIN_TRAIN_CYCLES * m})")
                if horizon > test:
                    problems.append(f"length {n}, period {m}: horizon {horizon} exceeds test size {test}")
        if problems:
            raise ValueError("train/test protocol cannot be satisfied: " + "; ".join(problems))
        return self
