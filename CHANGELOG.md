# Changelog

## 0.4.0 — 2026-09-12
- `experiments`: symbol-day stratified A/B assignment; difference in mean cost with CR0 cluster-robust SE and t(G−1) inference; CUPED adjustment with the 1 − ρ² identity tested; minimum detectable effect and required n checked against simulation.

## 0.3.0 — 2026-09-12
- `schedule`: Almgren-Chriss (2000) closed-form trajectory for risk aversion λ (expm1-stable, λ→0 = TWAP), expected cost and variance, efficient frontier, TWAP / VWAP / POV baselines, cost-to-completion ratio (2/3 rule). Tests: the eq. 15 difference equation, closed-form TWAP cost/variance, optimality under random perturbation, frontier monotonicity.

## 0.2.0 — 2026-09-11
- `impact.fit_almgren2005` (permanent + temporary) and `impact.fit_istar` through one NLS harness; hold-out **by day**; bootstrap CIs on exponents; exact recovery tests
- `scorecards.scorecard` / `difficulty_adjusted` with bootstrap CIs and a small-n note instead of a silent fit
- `options.net_mid`, `options.contract_costs` — $/contract, fraction of half-spread, delta-adjusted bps (NaN unless delta+spot given)
- `report` — markdown sections that state unit, sign convention, n, and the n-gate
- `adapters.tcalogger` — per-child JSONL of multi-leg option combos → canonical frames; reproduces the source's own IS cents exactly
- `examples/tcalogger_report.py` — end-to-end post-trade report
- CI (3.11/3.12, ruff, pytest with coverage), MIT licence

## 0.1.0 — 2026-09-11
- schema, synthetic market, implementation shortfall, benchmarks, square-root-law fit; 27 tests
