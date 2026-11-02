# Architecture

Modular monolith: one Python process, direct calls, no internal services.

```text
CLI (Typer) ─┐                                   ┌─ Progress bus ─> Rich bar / Gradio timer
             ├─> ResearchApp (application.services) ─> ExperimentManager ─> ExecutionStrategy (sequential)
GUI (Gradio)─┘          │                                   │                        │
                        │                                   │                   TaskRunner (13-step protocol)
                        ├─ Analysis engine                  ├─ Dataset generator      ├─ Evaluator (split, metrics)
                        └─ Repository / ArtifactStore       └─ Recovery / lifecycle   └─ Forecasting engine (registry)
                                  │                    │
                               SQLite              NPZ artifacts
```

| Package | Responsibility |
|---|---|
| `config` | YAML loading, Pydantic validation (incl. train/test protocol feasibility) |
| `domain` | enums, contracts (dataclasses), exception hierarchy mapped to failure categories |
| `generation` | seed policy, component mathematics, dataset ids, factorial design iterator (lazy) |
| `forecasting` | `Forecaster` interface, registry, the three methods |
| `evaluation` | chronological splitter, metrics, evaluator (pure computation) |
| `execution` | `ExecutionStrategy` contract + sequential implementation |
| `experiments` | planner, lifecycle policy, recovery policy, task runner, manager |
| `persistence` | SQLite (`database`, `repositories`), NPZ `artifacts`, `backup`/integrity |
| `analysis` | aggregation, best-method comparison, CSV export, Matplotlib figures |
| `monitoring` | event bus, progress tracker, logging |
| `application` | `ResearchApp`: the only API the CLI and GUI use |

## Data model (SQLite, schema v1, migrations via `PRAGMA user_version`)

`dataset_batches` 1-n `datasets` (n-n through `batch_datasets`, so re-generating an identical design re-uses
datasets); `experiments` n-n `datasets` (`experiment_datasets`); `experiments` 1-n `model_evaluations`
(unique per experiment+dataset+model) 1-n `evaluation_attempts`; `model_evaluations` 1-n `evaluation_metrics`
(one row per metric with value and status); `progress_events` (durable state transitions only).
WAL mode and foreign keys are enabled; every multi-row change is a `BEGIN IMMEDIATE` transaction.

## Safe persistence sequence

1. write artifact to a temp file in the target directory, 2. read it back and compare, 3. `os.replace`
(atomic on POSIX and Windows), 4. compute SHA-256, 5. commit the database reference **and** status/metrics in one
transaction. An evaluation becomes `COMPLETED` only in step 5. A crash between 3 and 5 leaves an identifiable
orphan file (`research workspace check`), never a database row pointing at a missing artifact. Dataset artifacts are
never silently overwritten (identical content is accepted, different content raises).

## State machines

Evaluation: `PENDING -> RUNNING -> COMPLETED | FAILED | INTERRUPTED`; `INTERRUPTED/FAILED/COMPLETED -> RUNNING`
only through resume (retry / integrity re-run). Illegal transitions raise. Experiment status and outcome are derived in
one place (`experiments/lifecycle.py`):

| Situation | Status / outcome |
|---|---|
| all evaluations completed | `COMPLETED / SUCCESS` |
| all processed, some failed | `COMPLETED / PARTIAL` (shown as "COMPLETED (partial: n failed)") |
| every evaluation failed | `FAILED / ALL_FAILED` |
| stopped by user or crash, work remains | `INTERRUPTED / INCOMPLETE` |
| persistence failure | `FAILED / ABORTED` |

## Leakage defence in depth

* `ForecastRequest` has exactly four fields (train, period, horizon, seasonal form).
* The train array handed to a method is a read-only copy.
* Observed data and ground truth are separate files; the task runner only ever reads the observed file.
* Parameter selection happens inside the method, on the training array only.
* Tests perturb the future values and require bit-identical forecasts and parameters, and spy on artifact reads.

## Extending execution

`ExecutionStrategy.execute(tasks, runner, should_stop)` is the only seam. A parallel strategy must call
`runner` (which converts model failures to `FAILED` outcomes), honour `should_stop`, add explicit resource limits, and
serialise SQLite writers (each repository call already uses its own short transaction).
