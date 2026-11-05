# Parallel processing update

Copy the files in this archive over your repository (same relative paths). No new dependencies, no database
migration, no change to existing experiments or workspaces.

## New files
| File | Purpose |
|---|---|
| `src/forecasting_research/execution/parallel.py` | `ParallelStrategy`: process pool, bounded in-flight work, crash/Ctrl-C/persistence-failure handling, parent-death watchdog |
| `src/forecasting_research/execution/factory.py` | `create_strategy(ExecutionCfg)` (replaces the old function in `sequential.py`) |
| `src/forecasting_research/experiments/work.py` | `run_work_item`: the pure, picklable computation of one evaluation (shared by both modes) |
| `configs/examples/parallel.yaml` | example configuration |
| `docs/parallel_execution.md` | design, settings, failure behaviour, practical notes |
| `tests/parallel_methods.py`, `tests/system/test_parallel.py`, `tests/unit/test_execution.py` | 23 new tests |

## Modified files
| File | Change |
|---|---|
| `config/schemas.py` | `ExecutionCfg`: `strategy: sequential\|parallel`, `workers`, `start_method`, `worker_modules`, `max_tasks_per_worker`, `limit_blas_threads`, `max_worker_restarts` |
| `execution/base.py`, `execution/sequential.py` | strategy contract takes a `TaskExecutor`; `create_strategy` moved to `factory.py` |
| `experiments/task_runner.py` | split into `begin / work_item / commit` (parent side) around the pure computation |
| `experiments/manager.py` | strategy chosen per run (`strategy_factory`); `resume(..., execution=...)` |
| `application/services.py` | uses the factory; `resume` takes execution settings from the loaded config |
| `monitoring/progress.py` | ETA from observed throughput (correct with any number of workers); "N evaluations running in parallel" |
| `cli/app.py` | `-j/--workers` on `run`, `evaluate`, `resume`; prints the execution mode |
| `configs/default.yaml` | `execution:` block now `parallel` / `workers: auto` (set `strategy: sequential` to go back) |
| `tests/unit/test_forecasters.py` | registry test now checks the built-ins are a *subset* (tests register extra methods) |

## Paste into README.md (e.g. after the "Commands" table)

```markdown
### Parallel execution

Evaluations are independent, so they can run on several CPU cores:

    research -c configs/default.yaml run -j 4      # 4 worker processes
    research -c configs/default.yaml run -j 0      # automatic (CPU cores - 1, max 8)
    research -c configs/default.yaml run -j 1      # sequential

or set `execution: {strategy: parallel, workers: auto}` in the YAML. Results are identical to sequential execution; workers never
write to the database (the main process is the single writer), Ctrl-C / crashes / failures remain safe and resumable, and
`resume` may use a different number of workers than the original run. About 215 MB RAM per worker. See `docs/parallel_execution.md`.
```

In the README "Runtime" section, add: "The figures above are per evaluation and sequential; with `-j N` on N physical cores
expect roughly N times less wall-clock time (not yet measured on a multi-core machine)."

`docs/architecture.md` ("Extending execution") is superseded by `docs/parallel_execution.md` (the seam is unchanged).

---

# SARIMA v2 update

Copy the files in this archive over your repository (same relative paths). No new required dependencies, no database
migration. `pmdarima` is **optional** (only for the `ocsb`/`ch` strategies and `scripts/benchmark_pmdarima.py`).

**I deliberately did not ship `configs/default.yaml`**: your design (periods, lengths, replications) differs from the one I
shipped, and I do not want to overwrite it. (If the previous update's `default.yaml` replaced your file, please restore your
design values; only the `execution:` block from that update was intended.) Paste the block below into your own YAML instead.

## Paste into your configuration (`forecasting.sarima`), these are the defaults, so omitting the block is equivalent
```yaml
forecasting:
  sarima:
    seasonal_differencing_strategy: fixed_one   # fixed_one | fixed_zero | ocsb | ch | seasonal_strength (legacy)
    kpss_regression: auto                       # c | ct | auto
    kpss_alpha: 0.05
    max_pq_sum: 2                               # keep your own bounds; nothing here changes them
    max_seasonal_pq_sum: 1
    selection_criterion: aicc                   # aic | aicc | bic
    max_iterations: 100
    time_budget_seconds: 300
    search_strategy: two_stage                  # two_stage | full_grid
    stage2_trigger: boundary_or_ljung_box       # boundary_or_ljung_box | boundary | ljung_box | always
    ljung_box_alpha: 0.05
    log_transform_multiplicative: false
```
Remove the old `seasonal_strength_threshold` line only if you do not use the legacy strategy (it is still accepted).

## What you must know before the full run
1. **Default `D` is `fixed_one`, not the review's `fixed_zero`.** On independent data `D=0` made SARIMA 20-45 % worse than
   `D=1` (and worse than Holt-Winters/classical) and left autocorrelated residuals in ~90 % of fits. Run
   `configs/examples/sarima_D0_sensitivity.yaml` as a labelled sensitivity experiment. See `docs/sarima_methodology.md` section 3.
2. Old (v1) experiments cannot be resumed if SARIMA work remains (would mix methods). Re-run SARIMA in a new experiment.
3. Two-stage search saved ~40 % of fits / ~45-57 % of time at equal mean accuracy, but individual datasets can differ from the
   exhaustive grid; use `search_strategy: full_grid` if you need exact equivalence with the reference search.
4. For periods 24/30, SARIMA costs ~5-7x more per evaluation than for periods 4-12 (about 3-5 s two-stage, 5-8 s full grid per evaluation, one core).

## New / modified files
| File | Change |
|---|---|
| `forecasting/sarima.py` | rewritten as a thin orchestrator (method v2) |
| `forecasting/sarima_components/` (new) | `candidates.py`, `differencing.py`, `fitting.py`, `search.py` |
| `forecasting/classical_decomposition.py` | input validation, near-zero/non-finite guards, unstable-seasonal-index warning |
| `config/schemas.py` | new `SarimaCfg` fields, `SARIMA_METHOD_VERSION` |
| `experiments/manager.py` | refuses to resume v1 experiments with pending SARIMA work |
| `analysis/reliability.py` (new), `application/services.py`, `cli/app.py`, `gui/app.py`, `persistence/repositories/evaluations.py` | `analyze --view reliability|search`, exports, GUI buttons |
| `scripts/sarima_study.py`, `scripts/benchmark_pmdarima.py` (new) | reproduce the evidence in `docs/sarima_methodology.md` |
| `docs/sarima_methodology.md`, `docs/decisions/0003-...md`, `docs/research_methodology.md`, `configs/examples/sarima_*.yaml` | documentation and examples |
| `tests/unit/test_sarima.py`, `tests/system/test_sarima_integration.py` (new); `tests/unit/test_forecasters.py`, `tests/research_validity/test_validity.py` | 54 new tests; two existing tests adjusted (timing fields excluded from the "parameters identical" check; SARIMA diagnostics assertions) |

## Paste into README.md (e.g. after "Interpreting results")
```markdown
### SARIMA search diagnostics and reliability

    research analyze <EXP> --view reliability --group-by period   # success %, extreme forecasts %, runtime per model
    research analyze <EXP> --view search --group-by seasonality   # fits, stage-2 %, truncation %, convergence %, D/d choices

SARIMA uses method v2 (two-stage bounded search, AICc, data-driven KPSS, strict convergence handling); see
`docs/sarima_methodology.md` for the decisions and the evidence behind them.
```
