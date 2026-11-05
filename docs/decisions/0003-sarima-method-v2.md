# ADR 0003 - SARIMA method version 2

Status: accepted. Details and measurements: `docs/sarima_methodology.md`.

Decisions
1. Seasonal strength is a stored diagnostic; it no longer decides `D`.
2. `D` strategy default is `fixed_one` (**deviation from the external review, which proposed `fixed_zero`**): `D = 0` made SARIMA
   20-45 % less accurate on independent synthetic data and left autocorrelated residuals in ~90 % of fits. `fixed_zero`, `ocsb`
   and `ch` are available; run `fixed_zero` as a labelled sensitivity experiment.
3. Data-driven KPSS (`auto`: level, then trend) on training data. Models are never told the generator's trend type.
4. Information criterion `aicc` by default with explicit `aic | aicc | bic` mapping.
5. Two-stage bounded search (stage 1: p+q<=1, P+Q<=1; stage 2: the remaining full-grid candidates when the best stage-1 model is
   on a boundary or its training residuals are autocorrelated). Exhaustive grid remains available as the reference.
6. A candidate is selectable only if it fitted, reported convergence and has a finite criterion.
7. Experiments created with method v1 cannot resume SARIMA work under v2 (no mixing of methods inside an experiment).

Consequences: SARIMA results from v1 and v2 are not comparable; re-run SARIMA in new experiments. The stored `method_version`
and the per-evaluation diagnostics make this auditable.
