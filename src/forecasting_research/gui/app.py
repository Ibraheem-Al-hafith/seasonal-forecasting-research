"""Optional local Gradio GUI.

A thin skin over the SAME ResearchApp services the CLI uses. Long-running workflows execute in a
background thread; a gr.Timer polls the shared ProgressTracker (live) and SQLite (evaluation table)
every second, so the interface never looks frozen.
"""

from __future__ import annotations

from typing import Any

import gradio as gr
import pandas as pd

from ..analysis.aggregation import ALL_DIMS
from ..analysis.comparison import win_counts
from ..application.services import ResearchApp
from ..config.loader import config_from_yaml_text, dump_config_yaml
from ..domain.exceptions import ResearchError
from ..monitoring.progress import format_seconds

METRICS = ["RMSE", "MAE", "MSE", "MAPE"]
DIM_CHOICES = {"seasonality": "seasonality_type", "period": "seasonal_period", "trend": "trend_type",
               "noise": "noise_level", "length": "length"}
STATUS_FILTERS = ["All", "RUNNING", "FAILED", "COMPLETED", "PENDING", "INTERRUPTED"]


class GuiController:
    """All GUI behaviour as plain methods (unit-testable without launching a server)."""

    def __init__(self, ra: ResearchApp):
        self.ra = ra

    # ---- helpers -----------------------------------------------------------------------
    def experiment_ids(self) -> list[str]:
        df = self.ra.list_experiments()
        return list(reversed(df["experiment_id"].tolist())) if not df.empty else []

    def batch_ids(self) -> list[str]:
        df = self.ra.list_batches()
        return list(reversed(df["batch_id"].tolist())) if not df.empty else []

    def _guard(self, fn, *a, **kw) -> str:
        try:
            fn(*a, **kw)
            return ""
        except ResearchError as exc:
            return f"**Error:** {exc}"

    # ---- configuration -----------------------------------------------------------------
    def apply_config(self, yaml_text: str) -> str:
        try:
            cfg = config_from_yaml_text(yaml_text)
        except ResearchError as exc:
            return f"**Invalid configuration** \n```\n{exc}\n```"
        self.ra.config = cfg
        g = cfg.generation
        return (f"**Configuration applied.** {g.combination_count()} combinations x {g.replications} replications = "
                f"{g.dataset_count()} datasets x {len(cfg.forecasting.methods)} methods = "
                f"{g.dataset_count() * len(cfg.forecasting.methods)} evaluations. SARIMA candidates per fit: "
                f"{cfg.forecasting.sarima.candidate_count()}.")

    # ---- workflows (background) ----------------------------------------------------------
    def _start(self, label: str, fn, *a, **kw) -> str:
        if not self.ra.start_background(fn, *a, **kw):
            return "A job is already running. Wait for it to finish or press **Stop**."
        return f"Started: {label}. Watch the **Live progress** tab."

    def start_generate(self) -> str:
        return self._start("dataset generation", self.ra.generate)

    def start_run(self) -> str:
        return self._start("generate + evaluate", self.ra.run)

    def start_evaluate(self, batch_id: str | None) -> str:
        return self._start("evaluate existing datasets", self.ra.evaluate, batch_id=batch_id or None)

    def start_resume(self, experiment_id: str | None, retry_failed: bool) -> str:
        if not experiment_id:
            return "Select an experiment to resume."
        return self._start(f"resume {experiment_id}", self.ra.resume, experiment_id, retry_failed)

    def stop(self) -> str:
        self.ra.stop()
        return "Stop requested: the current evaluation will finish, then the run stops safely (resumable)."

    # ---- live progress -------------------------------------------------------------------------
    def poll(self, status_filter: str = "All") -> tuple[str, pd.DataFrame, str]:
        s = self.ra.tracker.snapshot()
        filled = int(s["percent"] / 100 * 30)
        bar = "█" * filled + "░" * (30 - filled)
        if s["status"] == "GENERATING":
            g = s["generation"]
            head = f"### Generating datasets: {g['done']}/{g['total']}"
        elif s["experiment_id"] is None:
            head = "### Idle"
        else:
            head = (f"### {s['experiment_id']} - {s['status']}" + (f" ({s['outcome']})" if s["outcome"] else "") +
                    f"\n`{bar}` **{s['percent']}%**  \n"
                    f"completed **{s['completed']}** · failed **{s['failed']}** · running **{s['running']}** · pending **{s['pending']}** "
                    f"(total {s['total']})  \n"
                    f"elapsed {format_seconds(s['elapsed_seconds'])} · ETA {format_seconds(s['eta_seconds'])}  \n"
                    f"current: {s['activity']}")
        if self.ra.last_error:
            head += f"\n\n**Job failed:** {self.ra.last_error}"
        table = pd.DataFrame()
        if s["experiment_id"]:
            try:
                table = self.ra.evaluations_table(s["experiment_id"]).drop(columns=["evaluation_id", "started"])
                if status_filter != "All":
                    table = table[table["status"] == status_filter]
                order = {"RUNNING": 0, "FAILED": 1, "INTERRUPTED": 2, "PENDING": 3, "COMPLETED": 4}
                table = table.sort_values("status", key=lambda c: c.map(order), kind="stable").head(500)
            except ResearchError:
                pass
        return head, table, "\n".join(s["log"])

    # ---- results ---------------------------------------------------------------------------------
    def _dims(self, dims: list[str]) -> list[str]:
        return [DIM_CHOICES.get(d, d) for d in (dims or ALL_DIMS)] or ALL_DIMS

    def best(self, experiment_id: str | None, dims: list[str], metrics: list[str], stat: str = "mean") -> pd.DataFrame:
        if not experiment_id:
            return pd.DataFrame()
        return self.ra.best_methods([experiment_id], self._dims(dims), metrics or ["RMSE"], stat).round(4)

    def wins(self, experiment_id: str | None, dims: list[str], metrics: list[str]) -> pd.DataFrame:
        return win_counts(self.best(experiment_id, dims, metrics)) if experiment_id else pd.DataFrame()

    def aggregate(self, experiment_id: str | None, dims: list[str]) -> pd.DataFrame:
        if not experiment_id:
            return pd.DataFrame()
        return self.ra.aggregate([experiment_id], self._dims(dims)).round(4)

    def dataset_choices(self, experiment_id: str | None) -> list[str]:
        if not experiment_id:
            return []
        return self.ra.repo.experiments.dataset_ids(experiment_id)

    def forecast_plot(self, experiment_id: str | None, dataset_id: str | None):
        if not experiment_id or not dataset_id:
            return None
        return self.ra.forecast_figure(experiment_id, dataset_id)

    def metric_plot(self, experiment_id: str | None, dimension: str, metric: str):
        if not experiment_id:
            return None
        return self.ra.metric_figure([experiment_id], DIM_CHOICES.get(dimension, dimension), metric)

    def export(self, experiment_id: str | None, kind: str, metric: str) -> tuple[str, Any]:
        if not experiment_id:
            return "Select an experiment first.", None
        try:
            p = self.ra.export_results(experiment_id, kind, None, metric)
            return f"Wrote `{p}`", str(p)
        except ResearchError as exc:
            return f"**Error:** {exc}", None


def build_ui(ra: ResearchApp) -> gr.Blocks:
    c = GuiController(ra)
    with gr.Blocks(title="Seasonal Forecasting Research") as demo:
        gr.Markdown(f"# Seasonal Forecasting Research\nWorkspace: `{ra.workspace}`")

        with gr.Tab("Configuration"):
            yaml_box = gr.Code(value=dump_config_yaml(ra.config), language="yaml", label="Experiment configuration (YAML)", lines=24)
            apply_btn = gr.Button("Validate & apply", variant="primary")
            cfg_msg = gr.Markdown()
            apply_btn.click(c.apply_config, yaml_box, cfg_msg)

        with gr.Tab("Run"):
            gr.Markdown("All workflows call the same application services as the CLI.")
            with gr.Row():
                gen_btn = gr.Button("1. Generate datasets only")
                run_btn = gr.Button("2. Generate + evaluate", variant="primary")
                stop_btn = gr.Button("Stop (safe, resumable)", variant="stop")
            with gr.Row():
                batch_dd = gr.Dropdown(choices=c.batch_ids(), label="Existing dataset batch (blank = all datasets)", allow_custom_value=False)
                eval_btn = gr.Button("3. Evaluate existing datasets")
            with gr.Row():
                exp_resume = gr.Dropdown(choices=c.experiment_ids(), label="Experiment to resume")
                retry_cb = gr.Checkbox(label="Also retry FAILED evaluations")
                resume_btn = gr.Button("4. Resume")
            run_msg = gr.Markdown()
            gen_btn.click(c.start_generate, None, run_msg)
            run_btn.click(c.start_run, None, run_msg)
            eval_btn.click(c.start_evaluate, batch_dd, run_msg)
            resume_btn.click(c.start_resume, [exp_resume, retry_cb], run_msg)
            stop_btn.click(c.stop, None, run_msg)

        with gr.Tab("Live progress"):
            status_md = gr.Markdown()
            flt = gr.Dropdown(STATUS_FILTERS, value="All", label="Show evaluations with status")
            table = gr.Dataframe(label="Evaluations (Level B) - stage shows the current operation (Level C)", interactive=False, wrap=True)
            log_box = gr.Textbox(label="Recent events", lines=8, interactive=False)
            timer = gr.Timer(1.0)
            timer.tick(c.poll, flt, [status_md, table, log_box])

        with gr.Tab("Results"):
            with gr.Row():
                res_exp = gr.Dropdown(choices=c.experiment_ids(), label="Experiment")
                refresh = gr.Button("Refresh lists")
            dims = gr.CheckboxGroup(list(DIM_CHOICES), value=list(DIM_CHOICES), label="Group by")
            mets = gr.CheckboxGroup(METRICS, value=["RMSE", "MAE", "MAPE"], label="Metrics for best-method ranking")
            with gr.Row():
                b1 = gr.Button("Best method per group", variant="primary")
                b2 = gr.Button("Win counts")
                b3 = gr.Button("Full aggregates")
            res_tbl = gr.Dataframe(interactive=False, wrap=True)
            b1.click(c.best, [res_exp, dims, mets], res_tbl)
            b2.click(c.wins, [res_exp, dims, mets], res_tbl)
            b3.click(c.aggregate, [res_exp, dims], res_tbl)

        with gr.Tab("Visualizations"):
            viz_exp = gr.Dropdown(choices=c.experiment_ids(), label="Experiment")
            with gr.Row():
                ds_dd = gr.Dropdown(choices=[], label="Dataset")
                fplot_btn = gr.Button("Actual vs forecast")
            fplot = gr.Plot()
            with gr.Row():
                dim_dd = gr.Dropdown(list(DIM_CHOICES), value="noise", label="Dimension")
                met_dd = gr.Dropdown(METRICS, value="RMSE", label="Metric")
                mplot_btn = gr.Button("Metric by dimension")
            mplot = gr.Plot()
            viz_exp.change(lambda e: gr.update(choices=c.dataset_choices(e), value=None), viz_exp, ds_dd)
            fplot_btn.click(c.forecast_plot, [viz_exp, ds_dd], fplot)
            mplot_btn.click(c.metric_plot, [viz_exp, dim_dd, met_dd], mplot)

        with gr.Tab("Export"):
            exp_exp = gr.Dropdown(choices=c.experiment_ids(), label="Experiment")
            kind = gr.Radio(["evaluations", "aggregate", "best", "forecasts"], value="evaluations", label="Table")
            exp_met = gr.Dropdown(METRICS, value="RMSE", label="Metric (for 'best')")
            exp_btn = gr.Button("Export CSV", variant="primary")
            exp_msg = gr.Markdown()
            exp_file = gr.File(label="Download")
            exp_btn.click(c.export, [exp_exp, kind, exp_met], [exp_msg, exp_file])

        def refresh_lists():
            ids, batches = c.experiment_ids(), c.batch_ids()
            return [gr.update(choices=ids)] * 4 + [gr.update(choices=batches)]

        outs = [res_exp, viz_exp, exp_exp, exp_resume, batch_dd]
        refresh.click(refresh_lists, None, outs)
        demo.load(refresh_lists, None, outs)
    return demo


def launch(ra: ResearchApp, host: str = "127.0.0.1", port: int = 7860, inbrowser: bool = False) -> None:
    demo = build_ui(ra)
    demo.queue()
    demo.launch(server_name=host, server_port=port, inbrowser=inbrowser, allowed_paths=[str(ra.workspace / "exports")])
