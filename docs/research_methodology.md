# Research methodology (fixed decisions)

## Synthetic generator (version 1.0, all constants in `generation.params`, recorded in every dataset)

Time index `t = 0..n-1`, base level `B = 100`.

* **Trend**: none → `B`; linear → `B + 0.3 t`.
* **Seasonal profile** `p(t)` with period `m`: one cycle `sin(2πc/m + φ) + 0.4 sin(4πc/m + 2φ)`, mean-removed,
  scaled to `max|p| = 1`, tiled exactly; `φ ~ U(0, 2π)` drawn per replication.
* **Additive**: `Y = Trend + 15 p(t) + ε`, `ε ~ N(0, σ·B)`.
* **Multiplicative**: `Y = Trend × (1 + 0.15 p(t)) × (1 + e)`, `e ~ N(0, σ)` clipped to ±0.5 so every factor is positive.
* **Noise σ**: low 0.02, medium 0.05, high 0.10.
* **Ground truth** stored per dataset: trend, seasonal component, noise (additive values or multiplicative factors,
  labelled by `noise_kind`).
* **Seeds**: `SeedSequence([base_seed, blake2b32("seas|period|trend|noise|length"), replication])` → one 64-bit integer.
  A combination's data depends only on its identity, so editing the design lists does not change existing datasets.
* **Dataset id**: `ds_` + 12 hex characters of SHA-256 over canonical parameters, seed, generator params and version.

## Protocol

Chronological split, `train = floor(0.8 n)`; horizon = one seasonal cycle `m` unless `forecast_horizon` overrides it;
validation rejects designs with fewer than 2 training cycles or a horizon longer than the test segment. Only the first
`horizon` test observations are evaluated. All methods get identical training data, horizon and actuals.
Models receive only the seasonal period and the seasonal *form* (`seasonal_mode: matched` = the true form;
`additive`/`multiplicative` force a deliberate mis-specification). They never see trend type, noise level or ground truth.

## Methods

**Classical decomposition** (train only): centred moving average (`m` odd; 2×m for even `m`) → detrended values (difference
or ratio) → mean per phase → indices normalised to sum 0 (additive) or mean 1 (multiplicative) → straight-line OLS on the
deseasonalised series → forecast `T̂(t) + S` or `T̂(t) × S`, repeating the seasonal cycle.

**Holt-Winters**: `ExponentialSmoothing`, seasonal form as above, `initialization_method="estimated"`, no damping, no Box-Cox;
trend ∈ {none, additive} chosen by AICc on training data.

**SARIMA** (all steps on training data): seasonal differencing `D=1` if seasonal strength `F_s = 1 − Var(R)/Var(S+R)` exceeds
0.64; regular `d=1` if KPSS (level, α = 0.05) rejects on the (seasonally differenced) series; grid over `(p,q)`, `p+q ≤ 2`
(6) × `(P,Q)`, `P+Q ≤ 1` (3) = 18 candidates, simplest first, each with ≤100 iterations; AIC selection; non-converged
candidates skipped; a constant is included when `d + D ≤ 1`; soft 300 s budget checked between candidates (search stops
early, flagged `search_truncated`; with no usable candidate the evaluation fails as `TIMEOUT`). `log_transform_multiplicative`
(off by default) fits SARIMA on log data when the seasonal form is multiplicative. Search limits, every candidate
and its score, the selected orders and fitting warnings are stored with the evaluation.

## Metrics

MAE, MSE, RMSE, MAPE (percent). MAPE is undefined (`NULL`, `UNDEFINED_NEAR_ZERO`) if `mean|actual| = 0` or any
`|actual| < 1e-3 · mean|actual|`.

## Aggregation policy

Statistics (mean, std, median, min, max, n_valid) over COMPLETED evaluations with defined values; failures and undefined
values are always counted and reported. Best method = lowest chosen statistic per group and metric (ties by model name);
`complete_coverage` flags groups where failures/undefined values make methods non-comparable.

## Retry policy

Resume automatically re-runs `PENDING`, `INTERRUPTED`, stale `RUNNING`, and `COMPLETED`-with-bad-artifacts evaluations, plus
evaluations that failed with `PERSISTENCE_ERROR` (infrastructure, not model, failures). Other `FAILED` evaluations are retried
only with `--retry-failed [--retry-category ...]`. A persistence error stops the experiment.
