"""Standard-library logging: console + application log + per-experiment log, with identifiers."""

from __future__ import annotations

import logging
import logging.handlers
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = "forecasting_research"
_FORMAT = (
    "%(asctime)s %(levelname)-8s %(name)s "
    "[exp=%(experiment_id)s ev=%(evaluation_id)s ds=%(dataset_id)s model=%(model_name)s] %(message)s"
)
_FIELDS = ("experiment_id", "evaluation_id", "dataset_id", "model_name")
_MARK = "_fr_handler"


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for f in _FIELDS:
            if not hasattr(record, f):
                setattr(record, f, "-")
        return True


def log_ctx(**kw: str | None) -> dict:
    """`logger.info("msg", extra=log_ctx(experiment_id=..., model_name=...))`"""
    return {k: v for k, v in kw.items() if k in _FIELDS and v is not None}


def _make(handler: logging.Handler, level: int) -> logging.Handler:
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(_ContextFilter())
    setattr(handler, _MARK, True)
    return handler


def setup_logging(workspace: Path | None, level: str = "INFO", console_level: str = "WARNING") -> None:
    """Idempotent: replaces handlers installed by a previous call."""
    logger = logging.getLogger(ROOT)
    for h in [h for h in logger.handlers if getattr(h, _MARK, False)]:
        logger.removeHandler(h)
        h.close()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(_make(logging.StreamHandler(), getattr(logging, console_level)))
    if workspace is not None:
        log_dir = Path(workspace) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "application.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        logger.addHandler(_make(fh, getattr(logging, level)))


@contextmanager
def experiment_log(workspace: Path, experiment_id: str, level: str = "INFO") -> Iterator[None]:
    """Attach a per-experiment log file for the duration of a run."""
    logger = logging.getLogger(ROOT)
    d = Path(workspace) / "logs" / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    fh = _make(logging.FileHandler(d / f"{experiment_id}.log", encoding="utf-8"), getattr(logging, level))
    logger.addHandler(fh)
    try:
        yield
    finally:
        logger.removeHandler(fh)
        fh.close()
