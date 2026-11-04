# Parallel execution

Evaluations are independent and CPU-bound (SARIMA dominates), so they parallelise well across processes.
Enable it in the configuration or per command:

```yaml
execution:
  strategy: parallel      # sequential | parallel
  workers: auto           # auto = CPU cores - 1 (max 8), or an integer >= 1
```

```bash
research -c configs/default.yaml run            # uses the config
research -c configs/default.yaml run -j 4       # 4 worker processes
research -c configs/default.yaml run -j 1       # force sequential
research -c configs/default.yaml run -j 0       # parallel with automatic worker count
research -c configs/default.yaml resume <EXP_ID> -j 6
```

`-j/--workers` is available on `run`, `evaluate` and `resume`. Execution settings are machine-specific, not scientific:
**they are not part of an experiment's identity**, so an experiment started sequentially can be resumed in parallel (and
vice versa). On `resume` the settings come from the configuration you pass now (or `-j`), not from the original run.

## What to expect

* Evaluations are independent, so the expected speed-up is roughly proportional to the number of *physical cores* used
  (not yet measured on a multi-core machine - the development sandbox had a single core; please time your own run).
  Each worker pays a one-off ~3 s start-up (importing statsmodels), so tiny experiments (a few dozen evaluations) are not
  faster; the full 1,620-evaluation run is where it matters.
* Memory: about 215 MB per worker plus ~190 MB for the main process (measured). `workers: auto` caps at 8
  (~1.9 GB). On an 8 GB machine with 4 logical cores `auto` gives 3 workers.
* Results are **identical** to sequential execution (same data, same code: bit-identical metrics in our tests).
* ETA is computed from observed throughput, so it is correct for any number of workers.

## Design

```text
                 parent process (single writer)                          worker processes
 begin()  RUNNING + attempt row   ─┐
 work_item()  plain-data WorkItem  ├─ submit ───────────────────────────>  run_work_item():
 commit()  artifact -> COMPLETED  <┘ <───────── WorkResult ───────────────   load observed data, split,
        (one transaction)                                                    select parameters, fit, forecast, metrics
 progress events <───────────────── stage queue ───────────────────────   (no database, no file writes)
```

* **Single writer.** Workers never touch SQLite or the artifact store; the parent persists every result with exactly the
  same safe sequence as sequential mode (write + verify artifact, then `COMPLETED` + metrics in one transaction). All
  database/artifact invariants therefore hold unchanged.
* **Bounded in-flight work.** At most `workers` evaluations are submitted at a time, so `RUNNING` rows always correspond to
  work that is really running.
* **Ground truth stays out of reach.** A worker receives only the observed-data artifact path; the leakage guarantees of the
  evaluation protocol are untouched (the research-validity tests run against both modes' shared code path).

## Failure and interruption behaviour

| Event | Behaviour |
|---|---|
| A model fails (exception, NaN forecast, bad data) | Only that evaluation is `FAILED`; workers and the run continue |
| A worker process crashes (segfault, OOM kill) | The evaluations in flight are `FAILED` (`UNEXPECTED`, "worker process terminated"); the pool is rebuilt (up to `max_worker_restarts`, default 3, then the run stops resumable). Retry with `resume --retry-failed` |
| Stop button / `stop()` | No new evaluations start; in-flight ones finish and are recorded; status `INTERRUPTED` |
| Ctrl-C | Parent handles it (workers ignore it), kills the pool immediately; in-flight evaluations become `INTERRUPTED` and are re-run on resume |
| Database / artifact write failure | Pool is killed, experiment `FAILED/ABORTED`, in-flight evaluations `INTERRUPTED`; `resume` continues |
| Main process killed (`kill -9`, OOM) | Workers detect the dead parent within ~2 s and exit; stale `RUNNING` rows are recovered by `resume` |

## Practical notes

* **Run from a script or the `research` command.** `spawn` workers re-import the main program, which is impossible for
  `python -` (stdin). In that case the run falls back to sequential execution and logs a warning. Notebooks, scripts, the
  CLI and the GUI are fine.
* **Start method.** Default `spawn` everywhere: it is the only method available on Windows and the safest when the
  experiment runs from a thread (the GUI). `fork`/`forkserver` start faster on Linux but can misbehave with threads.
* **BLAS threads.** `limit_blas_threads: true` sets `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_*_THREADS=1` for the workers (unless
  you set them yourself), preventing `workers x BLAS threads` oversubscription.
* **Custom forecasting methods.** Workers start with a fresh interpreter, so a method registered at runtime in your own
  script is unknown to them. Register it at *import time* of a module and list that module:

  ```yaml
  execution:
    worker_modules: [my_package.my_methods]
  ```

  The module is imported in the main process (so the name validates) and in every worker. A bad module name is rejected
  before anything runs. Built-in methods need nothing.
* **Long-lived workers.** `max_tasks_per_worker: 200` recycles workers (Python 3.11+) if you ever observe memory growth.
* **Do not run two experiments against one workspace at the same time** with different processes unless you must: SQLite
  serialises writers (it works, but progress will be slower). One experiment with more workers is better.
* The GUI picks the strategy from the configuration you apply in its Configuration tab, per run.

## Extending

`ExecutionStrategy.execute(tasks, runner, should_stop)` is unchanged as the seam. `runner` now exposes
`begin / work_item / commit / stage` (the parent-side steps) besides being callable for a whole in-process task; see
`execution/base.py::TaskExecutor`. A different parallel backend (threads, a job queue) only has to honour the same
contract: compute through `run_work_item`, persist through `commit` from a single thread, poll `should_stop`.
