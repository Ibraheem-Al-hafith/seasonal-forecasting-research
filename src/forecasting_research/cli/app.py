"""Typer command-line interface (primary interface). Thin: every workflow calls ResearchApp."""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .. import __version__
from ..application.services import ResearchApp
from ..config.loader import dump_config_yaml, load_config
from ..config.schemas import AppConfig
from ..domain.contracts import ProgressEvent
from ..domain.exceptions import ResearchError
from ..execution.factory import create_strategy
from ..monitoring import events as E
from ..monitoring.progress import format_seconds

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(name="research", no_args_is_help=True, add_completion=False, rich_markup_mode=None,
                  help="Seasonal forecasting research system: generate data, run experiments, analyse results.")
experiments_app = typer.Typer(no_args_is_help=True, help="Inspect experiments.")
datasets_app = typer.Typer(no_args_is_help=True, help="Inspect generated datasets.")
workspace_app = typer.Typer(no_args_is_help=True, help="Workspace maintenance.")
app.add_typer(experiments_app, name="experiments")
app.add_typer(datasets_app, name="datasets")
app.add_typer(workspace_app, name="workspace")


@dataclass
class Globals:
    config_path: Path | None
    workspace: Path | None
    verbose: bool
    _app: ResearchApp | None = None
    _cfg: AppConfig | None = None

    def cfg(self) -> AppConfig:
        if self._cfg is None:
            self._cfg = load_config(self.config_path)
        return self._cfg

    def app(self) -> ResearchApp:
        if self._app is None:
            self._app = ResearchApp(self.workspace, self.cfg(), console_log_level="INFO" if self.verbose else "WARNING")
        return self._app


def _g(ctx: typer.Context) -> Globals:
    return ctx.obj


def guarded(fn):
    """Convert expected errors into a clean message and non-zero exit code."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ResearchError as exc:
            err_console.print(f"[red]Error:[/red] {exc}", markup=True, highlight=False)
            raise typer.Exit(1) from exc

    return wrapper


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", "-c", envvar="RESEARCH_CONFIG", help="YAML configuration file.")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", "-w", envvar="RESEARCH_WORKSPACE", help="Workspace directory (overrides config).")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show INFO logs on the console.")] = False,
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.", is_eager=True)] = False,
) -> None:
    if version:
        console.print(f"forecasting-research {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:  # bare `research`: show help (needed so --version works without a command)
        console.print(ctx.get_help())
        raise typer.Exit()
    ctx.obj = Globals(config, workspace, verbose)


# ------------------------------------------------------------------------------------------------
# rendering helpers
# ------------------------------------------------------------------------------------------------
def print_df(df: pd.DataFrame, title: str = "", max_rows: int | None = 60, float_fmt: str = "{:.3f}") -> None:
    if df is None or df.empty:
        console.print(f"[yellow]{title or 'Result'}: no rows[/yellow]")
        return
    t = Table(title=title or None, show_lines=False)
    for c in df.columns:
        is_id = str(c).endswith("_id")  # ids must stay intact so they can be copy-pasted
        width = int(df[c].astype(str).str.len().max()) if is_id else None
        t.add_column(str(c), justify="right" if pd.api.types.is_numeric_dtype(df[c]) else "left",
                     overflow="fold", no_wrap=is_id, min_width=width)
    shown = df if max_rows is None else df.head(max_rows)
    for _, row in shown.iterrows():
        t.add_row(*[("" if pd.isna(v) else float_fmt.format(v) if isinstance(v, float) else str(v)) for v in row])
    console.print(t)
    if max_rows is not None and len(df) > max_rows:
        console.print(f"[dim]... {len(df) - max_rows} more rows (use `research export` for the full table)[/dim]")


@contextmanager
def progress_display(ra: ResearchApp, enabled: bool) -> Iterator[None]:
    """Render shared progress events with Rich (stage, counters, ETA only when meaningful)."""
    if not enabled:
        yield
        return
    prog = Progress(SpinnerColumn(), BarColumn(bar_width=30), TaskProgressColumn(),
                    TextColumn("{task.fields[counts]}"), TimeElapsedColumn(), TextColumn("ETA {task.fields[eta]}"),
                    TextColumn("{task.description}"), console=console, transient=False)
    task = prog.add_task("starting", total=None, counts="", eta="-")

    def on_event(ev: ProgressEvent) -> None:
        s = ra.tracker.snapshot()
        if ev.event_type == E.EVALUATION_FAILED:
            prog.console.print(f"[red]  FAILED[/red] {ev.model_name} on {ev.dataset_id}: {ev.message.split(': ', 1)[-1]}", highlight=False)
        if s["status"] == "GENERATING":
            g = s["generation"]
            prog.update(task, total=g["total"] or None, completed=g["done"], description=s["activity"], counts="", eta="-")
        else:
            prog.update(task, total=s["total"] or None, completed=s["completed"] + s["failed"], description=s["activity"],
                        counts=f"ok {s['completed']} fail {s['failed']} run {s['running']} pend {s['pending']}",
                        eta=format_seconds(s["eta_seconds"]))

    unsub = ra.bus.subscribe(on_event)
    with prog:
        try:
            yield
        finally:
            unsub()


def _summary_panel(ra: ResearchApp, experiment_id: str) -> int:
    d = ra.experiment_details(experiment_id)
    c = d["counts"]
    color = {"COMPLETED": "green", "FAILED": "red", "INTERRUPTED": "yellow"}.get(d["status"], "white")
    if d["outcome"] == "PARTIAL":
        color = "yellow"
    body = (f"Experiment : {experiment_id}\nStatus     : [{color}]{d['status_display']}[/{color}]  (outcome {d['outcome']})\n"
            f"Evaluations: {c['COMPLETED']} completed, {c['FAILED']} failed, {c['PENDING'] + c['INTERRUPTED'] + c['RUNNING']} unfinished, {c['TOTAL']} total\n"
            f"Workspace  : {ra.workspace}")
    console.print(Panel(body, title="Experiment summary"))
    if c["FAILED"]:
        console.print(f"[yellow]{c['FAILED']} evaluation(s) failed.[/yellow] Inspect: research experiments show {experiment_id}")
    if d["status"] == "INTERRUPTED":
        console.print(f"Resume with: research resume {experiment_id}")
    elif d["status"] == "COMPLETED":
        console.print(f"Analyse with: research analyze {experiment_id}")
    return {"SUCCESS": 0, "PARTIAL": 2}.get(d["outcome"], 1)


def _override_methods(cfg: AppConfig, methods: str | None) -> AppConfig:
    if not methods:
        return cfg
    data = cfg.model_dump(mode="json")
    data["forecasting"]["methods"] = [m.strip() for m in methods.split(",") if m.strip()]
    return AppConfig.model_validate(data)


def _override_execution(cfg: AppConfig, workers: int | None) -> AppConfig:
    """--workers: 1 = sequential, N >= 2 = parallel with N processes, 0 = parallel with automatic worker count."""
    if workers is None:
        return cfg
    data = cfg.model_dump(mode="json")
    if workers == 1:
        data["execution"]["strategy"] = "sequential"
    else:
        data["execution"]["strategy"] = "parallel"
        data["execution"]["workers"] = "auto" if workers == 0 else workers
    return AppConfig.model_validate(data)


MethodsOpt = Annotated[str | None, typer.Option("--methods", help="Comma-separated methods overriding the config.")]
WorkersOpt = Annotated[int | None, typer.Option("--workers", "-j", min=0, help="Worker processes: 1 = sequential, N = parallel, 0 = automatic. Default: from config.")]
NoProgressOpt = Annotated[bool, typer.Option("--no-progress", help="Disable the live progress display.")]


# ------------------------------------------------------------------------------------------------
# workflows
# ------------------------------------------------------------------------------------------------
@app.command()
@guarded
def generate(ctx: typer.Context, no_progress: NoProgressOpt = False) -> None:
    """Generate the synthetic dataset batch described by the configuration (no model training)."""
    g = _g(ctx)
    ra = g.app()
    n = g.cfg().generation.dataset_count()
    console.print(f"Generating {n} dataset(s) in {ra.workspace}")
    with progress_display(ra, not no_progress):
        batch = ra.generate()
    console.print(f"[green]Batch complete:[/green] {batch}")


@app.command()
@guarded
def run(ctx: typer.Context, methods: MethodsOpt = None, workers: WorkersOpt = None, no_progress: NoProgressOpt = False) -> None:
    """Generate datasets AND evaluate every configured method on them (Mode B)."""
    g = _g(ctx)
    cfg = _override_execution(_override_methods(g.cfg(), methods), workers)
    ra = g.app()
    gen = cfg.generation
    console.print(f"Design: {gen.combination_count()} combinations x {gen.replications} replications = {gen.dataset_count()} datasets; "
                  f"methods: {', '.join(cfg.forecasting.methods)}; execution: {create_strategy(cfg.execution).describe()}")
    exp_id = None
    with progress_display(ra, not no_progress):
        try:
            exp_id = ra.run(cfg)
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted.[/yellow]")
    if exp_id is None:
        exp_id = ra.tracker.snapshot()["experiment_id"]
    if exp_id is None:
        raise typer.Exit(130)
    raise typer.Exit(_summary_panel(ra, exp_id))


@app.command()
@guarded
def evaluate(
    ctx: typer.Context,
    batch: Annotated[str | None, typer.Option("--batch", help="Dataset batch id.")] = None,
    dataset: Annotated[list[str] | None, typer.Option("--dataset", "-d", help="Dataset id (repeatable).")] = None,
    seasonality: Annotated[str | None, typer.Option(help="additive | multiplicative")] = None,
    period: Annotated[int | None, typer.Option(help="Seasonal period filter.")] = None,
    trend: Annotated[str | None, typer.Option(help="none | linear")] = None,
    noise: Annotated[str | None, typer.Option(help="low | medium | high")] = None,
    length: Annotated[int | None, typer.Option(help="Dataset length filter.")] = None,
    methods: MethodsOpt = None,
    workers: WorkersOpt = None,
    no_progress: NoProgressOpt = False,
) -> None:
    """Evaluate methods on previously generated datasets (Mode C)."""
    g = _g(ctx)
    cfg = _override_execution(_override_methods(g.cfg(), methods), workers)
    ra = g.app()
    filters = {"seasonality_type": seasonality, "seasonal_period": period, "trend_type": trend, "noise_level": noise, "length": length}
    exp_id = None
    with progress_display(ra, not no_progress):
        try:
            exp_id = ra.evaluate(cfg, batch_id=batch, dataset_ids=dataset, filters=filters)
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted.[/yellow]")
    exp_id = exp_id or ra.tracker.snapshot()["experiment_id"]
    if exp_id is None:
        raise typer.Exit(130)
    raise typer.Exit(_summary_panel(ra, exp_id))


@app.command()
@guarded
def resume(
    ctx: typer.Context,
    experiment_id: Annotated[str, typer.Argument(help="Experiment id.")],
    retry_failed: Annotated[bool, typer.Option("--retry-failed", help="Also retry FAILED evaluations.")] = False,
    retry_category: Annotated[list[str] | None, typer.Option("--retry-category", help="Only retry these failure categories (repeatable).")] = None,
    workers: WorkersOpt = None,
    no_progress: NoProgressOpt = False,
) -> None:
    """Resume an interrupted experiment; valid completed evaluations are skipped.

    Execution settings come from the loaded configuration (or --workers), not from the original run.
    """
    g = _g(ctx)
    g._cfg = _override_execution(g.cfg(), workers)
    ra = g.app()
    console.print(f"Execution: {create_strategy(ra.config.execution).describe()}")
    with progress_display(ra, not no_progress):
        try:
            ra.resume(experiment_id, retry_failed or bool(retry_category), retry_category)
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted.[/yellow]")
    raise typer.Exit(_summary_panel(ra, experiment_id))


# ------------------------------------------------------------------------------------------------
# experiments / datasets
# ------------------------------------------------------------------------------------------------
@experiments_app.command("list")
@guarded
def experiments_list(ctx: typer.Context) -> None:
    """List all experiments."""
    df = _g(ctx).app().list_experiments()
    if not df.empty:
        df = df[["experiment_id", "name", "status_display", "completed", "failed", "total"]]
    print_df(df, "Experiments", max_rows=None)


@experiments_app.command("show")
@guarded
def experiments_show(ctx: typer.Context, experiment_id: str, failures_only: Annotated[bool, typer.Option("--failures-only")] = False,
                     max_rows: Annotated[int, typer.Option(help="Rows of the evaluation table to print.")] = 40) -> None:
    """Show an experiment and its evaluation table."""
    ra = _g(ctx).app()
    d = ra.experiment_details(experiment_id)
    console.print(f"[bold]{experiment_id}[/bold]  {d['status_display']}  outcome={d['outcome']}  datasets={d['n_datasets']}  {d['counts']}")
    console.print(f"started={d['started_at']}  completed={d['completed_at']}  python={d['environment'].get('python')}")
    t = ra.evaluations_table(experiment_id)
    if failures_only:
        t = t[t["status"] == "FAILED"]
    print_df(t.drop(columns=["evaluation_id", "started"]), "Evaluations", max_rows=max_rows)


@experiments_app.command("status")
@guarded
def experiments_status(ctx: typer.Context, experiment_id: str) -> None:
    """One-screen status (Level A overview) from the database."""
    d = _g(ctx).app().experiment_details(experiment_id)
    c = d["counts"]
    t = Table(show_header=False)
    for k, v in [("Experiment", experiment_id), ("Status", d["status_display"]), ("Progress", f"{d['percent']}%"),
                 ("Completed", c["COMPLETED"]), ("Failed", c["FAILED"]), ("Running", c["RUNNING"]),
                 ("Pending", c["PENDING"]), ("Interrupted", c["INTERRUPTED"]), ("Total", c["TOTAL"])]:
        t.add_row(k, str(v))
    console.print(t)


@datasets_app.command("list")
@guarded
def datasets_list(
    ctx: typer.Context,
    batch: Annotated[str | None, typer.Option("--batch")] = None,
    seasonality: Annotated[str | None, typer.Option()] = None,
    period: Annotated[int | None, typer.Option()] = None,
    trend: Annotated[str | None, typer.Option()] = None,
    noise: Annotated[str | None, typer.Option()] = None,
    length: Annotated[int | None, typer.Option()] = None,
    max_rows: Annotated[int, typer.Option()] = 40,
) -> None:
    """List generated datasets (optionally filtered)."""
    df = _g(ctx).app().list_datasets(batch_id=batch, seasonality_type=seasonality, seasonal_period=period,
                                     trend_type=trend, noise_level=noise, length=length)
    print_df(df, f"Datasets ({len(df)})", max_rows=max_rows)


@datasets_app.command("show")
@guarded
def datasets_show(ctx: typer.Context, dataset_id: str) -> None:
    """Show one dataset's metadata and artifact references."""
    d = _g(ctx).app().dataset_details(dataset_id)
    t = Table(show_header=False)
    for k, v in d.items():
        t.add_row(k, str(v) if k != "metadata" else "\n".join(f"{a}: {b}" for a, b in v.items() if a != "generator_params"))
    console.print(t)


# ------------------------------------------------------------------------------------------------
# analysis / export
# ------------------------------------------------------------------------------------------------
@app.command()
@guarded
def analyze(
    ctx: typer.Context,
    experiment_ids: Annotated[list[str] | None, typer.Argument(help="Experiment id(s); default: all.")] = None,
    group_by: Annotated[str | None, typer.Option("--group-by", help="Comma list of: seasonality,period,trend,noise,length (default: all five).")] = None,
    metric: Annotated[list[str] | None, typer.Option("--metric", "-m", help="MAE, MSE, RMSE, MAPE (repeatable).")] = None,
    stat: Annotated[str, typer.Option(help="Statistic used to rank methods: mean | median | min | max | std")] = "mean",
    view: Annotated[str, typer.Option(help="best | aggregate | wins")] = "best",
    max_rows: Annotated[int, typer.Option()] = 60,
) -> None:
    """Grouped comparison of methods from stored results."""
    ra = _g(ctx).app()
    ids = experiment_ids or None
    dims = group_by.split(",") if group_by else None
    res = ra.results(ids)
    if not res.empty:
        n_f = int((res["status"] == "FAILED").sum())
        n_u = int((~res["status"].isin(["COMPLETED", "FAILED"])).sum())
        console.print(f"Evaluations: {len(res)} total, {int((res['status'] == 'COMPLETED').sum())} completed, {n_f} failed, {n_u} unfinished "
                      f"(failures are excluded from metric statistics but always counted)")
    if view == "aggregate":
        print_df(ra.aggregate(ids, dims, metric), "Aggregated metrics", max_rows)
    else:
        best = ra.best_methods(ids, dims, metric, stat)
        if view == "wins":
            from ..analysis.comparison import win_counts
            print_df(win_counts(best), "Groups won per method and metric", None)
        else:
            if not best.empty:
                best = best.drop(columns=["stat", "n_methods"])
            print_df(best, f"Best method per group ({stat})", max_rows)
            if not best.empty and not best["complete_coverage"].all():
                console.print("[yellow]Some groups have incomplete coverage (failures or undefined metrics); see complete_coverage.[/yellow]")


@app.command()
@guarded
def export(
    ctx: typer.Context,
    experiment_id: str,
    kind: Annotated[str, typer.Option(help="evaluations | aggregate | best | forecasts")] = "evaluations",
    group_by: Annotated[str | None, typer.Option("--group-by")] = None,
    metric: Annotated[str, typer.Option("--metric", "-m")] = "RMSE",
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output CSV path (default: workspace/exports).")] = None,
) -> None:
    """Export a result table to CSV."""
    path = _g(ctx).app().export_results(experiment_id, kind, group_by.split(",") if group_by else None, metric, out)
    console.print(f"[green]Wrote[/green] {path}")


@app.command()
@guarded
def plot(
    ctx: typer.Context,
    experiment_id: str,
    kind: Annotated[str, typer.Option(help="forecast | metric")] = "metric",
    dataset: Annotated[str | None, typer.Option("--dataset", "-d", help="Dataset id (kind=forecast).")] = None,
    dimension: Annotated[str, typer.Option(help="seasonality|period|trend|noise|length (kind=metric)")] = "noise",
    metric: Annotated[str, typer.Option("--metric", "-m")] = "RMSE",
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
) -> None:
    """Save a figure (PNG) generated from stored results."""
    ra = _g(ctx).app()
    if kind == "forecast":
        if not dataset:
            raise ResearchError("--dataset is required for kind=forecast")
        fig, name = ra.forecast_figure(experiment_id, dataset), f"{experiment_id}_{dataset}_forecast.png"
    else:
        fig, name = ra.metric_figure([experiment_id], dimension, metric), f"{experiment_id}_{metric}_by_{dimension}.png"
    path = out or ra.workspace / "exports" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    console.print(f"[green]Wrote[/green] {path}")


# ------------------------------------------------------------------------------------------------
# workspace / gui
# ------------------------------------------------------------------------------------------------
@workspace_app.command("init")
@guarded
def workspace_init(ctx: typer.Context, show_config: Annotated[bool, typer.Option("--show-config", help="Print the effective configuration.")] = False) -> None:
    """Create the workspace layout and database (idempotent)."""
    g = _g(ctx)
    ra = g.app()
    console.print(f"[green]Workspace ready:[/green] {ra.workspace} (schema v{ra.db.schema_version()})")
    if show_config:
        console.print(dump_config_yaml(g.cfg()), highlight=False)


@workspace_app.command("check")
@guarded
def workspace_check(ctx: typer.Context) -> None:
    """Verify database integrity, artifact checksums, missing and orphan files."""
    rep = _g(ctx).app().check_workspace()
    console.print(f"Artifacts checked: {rep.artifacts_checked}")
    for label, items in [("Database problems", rep.database_problems), ("Missing artifacts", rep.missing_artifacts),
                         ("Corrupt artifacts", rep.corrupt_artifacts), ("Orphan artifacts (not deleted)", rep.orphan_artifacts),
                         ("Leftover temp files (not deleted)", rep.temp_files)]:
        if items:
            console.print(f"[yellow]{label}: {len(items)}[/yellow]")
            for i in items[:10]:
                console.print(f"   {i}", highlight=False)
    if rep.ok:
        console.print("[green]Workspace OK[/green]")
    else:
        raise typer.Exit(1)


@workspace_app.command("backup")
@guarded
def workspace_backup(ctx: typer.Context) -> None:
    """Back up the SQLite database with SQLite's online backup API."""
    console.print(f"[green]Backup written:[/green] {_g(ctx).app().backup()}")


@app.command()
@guarded
def gui(ctx: typer.Context, port: Annotated[int, typer.Option()] = 7860, host: Annotated[str, typer.Option()] = "127.0.0.1") -> None:
    """Launch the optional local Gradio GUI."""
    try:
        from ..gui.app import launch
    except ImportError as exc:
        raise ResearchError("The GUI needs Gradio: pip install 'forecasting-research[gui]'  (or: uv sync --extra gui)") from exc
    launch(_g(ctx).app(), host=host, port=port)


def main() -> None:  # console-script entry point
    app()


if __name__ == "__main__":
    main()
