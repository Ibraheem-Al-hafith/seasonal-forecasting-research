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
