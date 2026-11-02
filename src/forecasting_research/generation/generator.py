"""Dataset generator: specs -> reproducible datasets with preserved ground truth."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterator

from ..config.defaults import GENERATOR_VERSION
from ..config.schemas import GenerationCfg
from ..domain.contracts import DatasetSpec, GeneratedDataset
from .components import build_series
from .seeds import derive_seed


def make_dataset_id(spec: DatasetSpec, cfg: GenerationCfg) -> str:
    """ds_ + 12 hex chars of a hash of canonical parameters, seed and generator version."""
    canonical = json.dumps(
        {
            "spec": spec.canonical_key(),
            "replication": spec.replication,
            "seed": spec.seed,
            "params": cfg.params.model_dump(mode="json"),
            "version": GENERATOR_VERSION,
        },
        sort_keys=True,
    )
    return "ds_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


class DatasetGenerator:
    def __init__(self, cfg: GenerationCfg):
        self.cfg = cfg

    def iter_specs(self) -> Iterator[DatasetSpec]:
        """Deterministic lazy iteration over the full factorial design (never materialised)."""
        c = self.cfg
        for s, m, t, nl, n in itertools.product(
            c.seasonality_types, c.seasonal_periods, c.trends, c.noise_levels, c.dataset_lengths
        ):
            for rep in range(c.replications):
                spec = DatasetSpec(s.value, m, t.value, nl.value, n, rep, 0)
                seed = derive_seed(c.base_seed, spec.canonical_key(), rep)
                yield DatasetSpec(s.value, m, t.value, nl.value, n, rep, seed)

    def generate(self, spec: DatasetSpec) -> GeneratedDataset:
        observed, truth, extra = build_series(
            spec.seasonality_type,
            spec.seasonal_period,
            spec.trend_type,
            spec.noise_level,
            spec.length,
            spec.seed,
            self.cfg.params,
        )
        dataset_id = make_dataset_id(spec, self.cfg)
        metadata = {
            "dataset_id": dataset_id,
            "seasonality_type": spec.seasonality_type,
            "seasonal_period": spec.seasonal_period,
            "trend_type": spec.trend_type,
            "noise_level": spec.noise_level,
            "length": spec.length,
            "replication": spec.replication,
            "seed": str(spec.seed),
            "base_seed": self.cfg.base_seed,
            "generator_version": GENERATOR_VERSION,
            "generator_params": self.cfg.params.model_dump(mode="json"),
            "noise_kind": truth.noise_kind,
            **extra,
        }
        return GeneratedDataset(dataset_id, spec, observed, truth, metadata)
