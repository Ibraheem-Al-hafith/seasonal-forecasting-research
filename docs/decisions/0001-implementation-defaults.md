# ADR 0001 - Section 24 implementation defaults

Status: accepted (confirmed by the project owner). Details and formulas: `docs/research_methodology.md`.

| Item | Decision |
|---|---|
| Python | 3.11 pinned; supports `>=3.10,<3.13` |
| Generator | base 100; linear slope 0.3; additive amplitude 15; multiplicative 1 + 0.15 p; sinusoid + 0.4 second harmonic with random phase |
| Noise | σ = 0.02 / 0.05 / 0.10; additive `N(0, σ·100)`; multiplicative factor `1 + e`, clipped to ±0.5 |
| Seeds | `SeedSequence([base, blake2b32(canonical key), replication])` |
| Dataset id | `ds_` + 12 hex of hash(params, seed, generator version) |
| Classical decomposition | centred MA, per-phase means, OLS extrapolation, ≥2 cycles |
| Metadata for models | seasonal period and seasonal form only (`seasonal_mode`) |
| Holt-Winters | statsmodels, estimated init, trend none/add by AICc |
| SARIMA | KPSS + seasonal strength for d/D; p+q≤2, P+Q≤1; AIC; 100 iterations; 300 s soft budget |
| Horizon | default `m`; integer override; validated |
| MAPE | percent; `NULL` + `UNDEFINED_NEAR_ZERO` near zero |
| Retry | see methodology; every attempt stored |
| Final status | status + derived outcome (SUCCESS / PARTIAL / ALL_FAILED / ABORTED / INCOMPLETE) |
| Schema | SQLite WAL, FK on, numbered migrations (`user_version`) |
| Artifacts | `np.savez_compressed`, no pickle, separate observed/truth files, schema_version, SHA-256 in DB |
| CLI | `research` entry point as documented in the README |
| GUI refresh | background thread + `gr.Timer(1 s)` polling tracker + SQLite |

## Deviations from the blueprint wording (all in the direction of safety)

1. Observed data and ground truth are two files per dataset (structural leakage barrier).
2. Metrics are plain NumPy; scikit-learn is a test-only cross-check.
3. Re-generation compares regenerated arrays with the stored arrays (checksummed on read) instead of comparing file hashes,
   because compressed NPZ bytes embed timestamps.
4. `PERSISTENCE_ERROR` failures are retried automatically on resume (they are infrastructure failures).
5. `batch_datasets` association table: identical datasets are shared between batches rather than duplicated.
6. `application/use_cases/` was not created; `ResearchApp` in `application/services.py` is a single cohesive service class.
