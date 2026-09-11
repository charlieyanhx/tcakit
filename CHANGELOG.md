# Changelog

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
