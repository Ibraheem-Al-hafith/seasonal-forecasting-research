# SARIMA methodology (method version 2)

This document records what changed in the SARIMA method, why, which parts of the external technical review were adopted,
changed or declined, and the measurements behind each decision. Every SARIMA evaluation stores `method_version: 2` and the
full set of diagnostics described below.

## 1. What changed relative to method v1

| Topic | v1 | v2 |
|---|---|---|
| Seasonal differencing `D` | `D = 1` if a seasonal-strength threshold was exceeded | configurable strategy, default `fixed_one`; strength kept as a stored diagnostic |
| Seasonal strength for multiplicative data | additive decomposition of the raw series | additive decomposition of `log(y)`; reliability flag for few observations per phase |
| KPSS | level-stationarity only | `auto`: level test, then trend test; trend-stationary series keep `d = 0` and get a deterministic trend |
| Constant / drift | implicit rule `d + D <= 1` | explicit, recorded (`intercept`, `drift`, `seasonal_drift`, `linear_time_trend`, `none`) |
| Criterion | AIC or BIC (`aicc` was rejected by the config) | `aic`, `aicc`, `bic` with explicit mapping; default `aicc` |
| Convergence | missing metadata was treated as converged | missing metadata is **not** selectable (`no_convergence_info`) |
| Search | exhaustive grid (18 candidates by default) | two-stage bounded search (9 + up to 9), exhaustive grid still available |
| Diagnostics | orders, AIC, candidate list | per-candidate status/scores/fit time, stage decisions and reasons, Ljung-Box, KPSS tests, strength, counts |
| Log transform | unchecked | validated before the transform and after the inverse |

Experiments created with v1 cannot be resumed if they still have SARIMA evaluations to run (that would mix two methods in one
experiment); `resume` says so. Completed v1 results stay available for analysis (`method_versions` shows `1`).

## 2. The external review: verdicts

| ID | Review finding | Verdict | What was done |
|---|---|---|---|
| S01 | seasonal strength must not decide `D` | **Confirmed** (strength is prominence, not a unit-root test) | Decoupled. **But** the review's replacement, `D = 0`, was *not* adopted as the default (section 3). |
| S02 | multiplicative data decomposed additively | **Confirmed** | log-domain strength; tested against the additive-only value |
| S03 | "AICc silently selects BIC" | **Partly**: in this code `aicc` was *rejected* by validation, not silently mapped | explicit `aic`/`aicc`/`bic`; default `aicc`; test that the requested criterion is the one read |
| S04 | level KPSS on deterministic-trend data | **Confirmed**; the proposed "provide the trend structure to the model" design was **declined** | Giving models the generator's trend type would leak ground truth and break the fair-comparison invariants. Replaced by a data-driven `auto` KPSS (level, then trend) on training data. |
| S05 | implicit constant/drift | **Confirmed** | explicit and recorded |
| S06 | a running fit cannot be interrupted | **Confirmed, not changed** | still a soft budget; per-fit durations are now recorded; a hard limit needs one killable process per fit (not implemented) |
| S07 | log transform not validated | **Confirmed** | validated both ways |
| S08 | weak strength estimates for m = 24, 30 | **Confirmed** | `reliable` flag, warning, reported in `research analyze --view search` |
| S09 | no candidate counts | **Confirmed** | counts of selectable / non-converged / no-info / errors |
| Two-stage | 9 first-order candidates, then the remaining ones | **Implemented**; the review's savings estimate (47.5 %) holds only when the base model is adequate (section 4) | |
| pmdarima | benchmark, do not replace | **Agreed**; OCSB/CH offered as optional `D` strategies; benchmark script included (section 5) | |

Not implemented (deliberately): a cross-experiment result cache (`resume` already never repeats a completed evaluation inside an
experiment), MASE, hard per-fit timeouts, local-neighbourhood expansion ("strategy B").

The review assumed 36 candidates (`max_seasonal_pq_sum: 2`). The shipped default is `max_seasonal_pq_sum: 1` (18 candidates);
the two-stage search works for any limits (with the review's limits: 9 + 27 = 36, covered by a test). Your search bounds were not changed.

## 3. Why `fixed_one` and not the review's `fixed_zero`

The review argues that the generator creates *deterministic* seasonality, which has no seasonal unit root, so `D = 0` is the
theoretically clean choice. The premise is right (verified: the generator has no seasonal stochastic trend). The conclusion is
not what the measurements support. A SARIMA without seasonal regressors can represent a *fixed* seasonal pattern only through
seasonal differencing combined with a seasonal MA term (or a seasonal AR coefficient forced below 1 by the stationarity
constraint, which damps the pattern). With `D = 0` the fitted models are misspecified.

Measured on **independent** synthetic data (generator seed 777, different from the experiment's seed; training-only selection):
`rmse_ratio` = test RMSE / standard deviation of the generator's noise in the test window (1.0 = irreducible error; computable
only because the data are synthetic).

| Design | n | classical | Holt-Winters | legacy v1 rule | SARIMA `D=0` | SARIMA `ocsb` | SARIMA `D=1` | `D=1` better than `D=0` on |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| periods 4/7/12, length 200 | 24 | 1.14 | 1.13 | 1.30 | **1.59** | 1.40 | **1.16** | 22 of 24 |
| periods 7/24/30, length 150 | 22 | 1.17 | 1.15 | 1.26 | **1.74** | n/a | **1.19** | 19 of 22 |

Independent evidence from training data only: after `D = 0`, the Ljung-Box test found remaining residual autocorrelation (so
stage 2 was triggered) in 88-96 % of fits, against 0-42 % with `D = 1`.

Reading: with `D = 1`, SARIMA is on par with the other two methods; with `D = 0` it is handicapped by about 20-45 %. Reporting
the latter as "SARIMA performance" would bias the research conclusion against SARIMA. `fixed_zero` remains available and should
be run as an explicitly labelled **sensitivity experiment** (`configs/examples/sarima_D0_sensitivity.yaml`); the data-driven
alternatives are `ocsb` / `ch` (need `pip install pmdarima`; OCSB chose `D = 1` for only one third of these deterministic
series and scored in between).

Caveats: 22-24 datasets per design, one replication, noise levels low/high only, one generator. The conclusion is a measured
property of *this* generator and *this* model class, not a general statement about SARIMA. Reproduce with
`python scripts/sarima_study.py --base-seed 777 --baselines`.

Disclosure: the first pass of this study used the experiment's own seed (so its datasets overlapped the experiment's data); the
numbers above are from the repeat on independent seeds. Both passes gave the same ranking.

## 4. Two-stage search: measured behaviour

Stage 1 = `p+q <= 1`, `P+Q <= 1` (9 candidates). Stage 2 = the rest of the full grid, run only if (a) the best stage-1 model sits on a
stage-1 limit that is below the final limit, or (b) the Ljung-Box test on its *training* residuals rejects (`alpha = 0.05`). The
test set can never influence this decision (a test perturbs every test observation and requires identical decisions).

| Setting (D = 1) | stage 2 triggered | fits saved vs full grid | time saved | same selected order | mean test RMSE ratio (full grid -> two-stage) |
|---|---:|---:|---:|---:|---|
| periods 4/7/12, `boundary_or_ljung_box` | 21 % | 40 % | 57 % | 88 % | 1.16 -> 1.15 |
| periods 7/24/30, `boundary_or_ljung_box` | 23 % | 39 % | 46 % | 91 % | 1.19 -> 1.19 |
| periods 4/7/12, `ljung_box` only | 0-13 % | 44-50 % | 56-68 % | 79-88 % | 1.16 -> 1.15-1.16 |

* The review's illustrative "70 % stop after stage 1" only materialises when the base model is adequate. With `D = 0` the
  residual diagnostic correctly reports inadequacy and stage 2 runs in 88-96 % of cases (almost no saving).
* Individual datasets can differ: the largest single-dataset RMSE change between two-stage and the full grid was 0.12 noise
  units on the long-period design and 0.8 on one short-period dataset. If exact equivalence with the exhaustive grid matters for a
  final run, use `search_strategy: full_grid` (cost: about 1.7x the fits).
* `boundary_or_ljung_box` (default) is the most conservative trigger; `ljung_box` saves more and was equally accurate on average here.

## 5. Benchmark against pmdarima (`auto_arima`, stepwise, AICc)

16 independent datasets (periods 4 and 12, length 200, low/high noise). `scripts/benchmark_pmdarima.py`.

| Comparison | implementation | mean RMSE | median RMSE | mean seconds |
|---|---|---:|---:|---:|
| same `d`, `D` fixed | custom two-stage | 6.79 | 5.70 | 0.64 |
| | pmdarima stepwise | 6.64 | 4.93 | 1.57 |
| each chooses its differencing | custom (`fixed_one`) | 6.79 | 5.70 | 0.64 |
| | pmdarima (KPSS + OCSB) | 6.99 | 5.49 | 2.31 |

Custom was faster on 14 of 16 datasets (median ratio 2.3x); accuracy was comparable (the custom search had the lower RMSE on 9
of 16). Non-seasonal orders agreed on 81 % of datasets, seasonal orders on only 19 %, because the search spaces differ
(pmdarima allowed `P=1, Q=1` together; the custom bound is `P+Q <= 1`). AICc values are not directly comparable (different
constant handling). No extreme or non-finite forecasts occurred in either. **Conclusion: no reason to replace the statsmodels
implementation; the benchmark is a sanity check, not a proof.**

## 6. Runtime planning

Measured per SARIMA evaluation on one core (training length 120-160; D = 1; independent data):

| seasonal period | full grid (18 fits) | two-stage | time saved |
|---:|---:|---:|---:|
| 7 | 1.0 s | 0.4 s | 60 % |
| 24 | 5.3 s | 2.7 s | 49 % |
| 30 | 7.2 s | 4.2 s | 42 % |

(periods 4-12 at length 200: about 1.9 s full grid, 0.8 s two-stage.) Cost grows steeply with the seasonal period; a design with periods 24
and 30 costs about 5-7x more per evaluation than one with periods 4-12, and the stage-2 saving shrinks a little because more
searches expand. The classical and Holt-Winters methods stay at ~0.03 s and ~0.1 s. Parallel execution divides the wall-clock
time roughly by the number of cores used (not measured here: the development sandbox has one core).

## 7. Reporting what the search did

```bash
research analyze <EXP> --view reliability --group-by period        # success %, extreme-forecast %, runtime per model
research analyze <EXP> --view search --group-by seasonality        # fits, stage-2 %, truncation %, convergence %, D/d choices
research export <EXP> --kind search
```

`success_pct` counts every evaluation (failures are never dropped); `extreme_forecast_pct` flags forecasts more than one full
training range outside the observed training range; `pct_strength_unreliable` flags groups where the seasonal-strength
diagnostic rests on fewer than 5 observations per phase.
