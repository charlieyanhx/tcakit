# tcakit

[![ci](https://github.com/charlieyanhx/tcakit/actions/workflows/ci.yml/badge.svg)](https://github.com/charlieyanhx/tcakit/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Transaction cost analysis and market-impact calibration for equities and listed options.
A small, tested library that answers the three questions execution desks ask:

| Question | Module | What you get |
|---|---|---|
| **What did this order cost, and why?** | `shortfall`, `benchmarks`, `options` | Perold implementation-shortfall decomposition (delay / spread / timing / opportunity / fees) with an exact identity; arrival, interval VWAP/TWAP, close, participation, post-trade reversion; option costs in $/contract, fraction of half-spread, and delta-adjusted bps |
| **Which broker / venue / algo / tier is cheaper, after controlling for difficulty?** | `scorecards` | Grouped costs with bootstrap CIs, then mean *residuals* of a difficulty regression — the number that survives "your orders were just harder" |
| **What will a new order cost?** | `impact` | Square-root law, Almgren et al. (2005), and Kissell I-star, every fit reporting held-out-by-day RMSE and bootstrap CIs, not just R² |
| **Did the new algo actually help?** | `experiments` | Symbol-day-stratified A/B assignment, difference in cost with symbol-day clustered standard errors, CUPED variance reduction (1 − ρ², tested as an identity), minimum detectable effect and required n |
| **How should it be scheduled?** | `schedule` | Almgren-Chriss (2000) closed-form trajectories and the efficient frontier for a risk aversion λ; TWAP / VWAP / POV baselines; the 2/3-rule cost-to-completion ratio for any schedule |

Design rules, stated once and tested:

- Sign convention everywhere is `side · (paid − benchmark)`, positive = cost to the trader.
- `Σ components == total` to 1e-9; a sell flips every sign; an unfilled order is pure opportunity + fees.
- Missing mids give NaN, never zero. Components the data cannot identify (e.g. delay without a decision timestamp) are reported as *0 by construction* in the report, not hidden.
- Every `FitResult` carries `n_in, n_out, rmse_out`. `holdout=0` is allowed but visible.
- Estimators are tested for **parameter recovery** on a seeded synthetic market with known impact, not just for running.

## Install

```bash
pip install -e ".[dev]"
pytest -q            # 65 tests
python examples/quickstart.py
```

## Quickstart

```python
from tcakit import implementation_shortfall, benchmark_slippage, fit_sqrt_law, fit_almgren2005, realized_impact
from tcakit.synth import simulate

sim = simulate(n_days=120, orders_per_day=8, seed=42, sigma_daily=0.02, spread_bps=5.0, sqrt_law_y=0.8)

sf = implementation_shortfall(sim.orders, sim.fills, sim.market)   # one row per parent order
bm = benchmark_slippage(sim.orders, sim.fills, sim.market)          # arrival / vwap / twap / close / reversion
df = realized_impact(sim.orders, sim.fills, sim.market).merge(sim.orders[["order_id", "arrival_ts"]])
print(fit_sqrt_law(df).summary())                 # one free parameter, exponent fixed at 0.5
print(fit_almgren2005(df)["permanent"].summary())  # free exponent
# sqrt_law on impact_end: Y=0.898 [0.617, 1.198] | R²_in=0.027 RMSE_out=0.00415 (n_in=665, n_out=285)
# almgren2005_permanent:  gamma=1.72 [0.19, 51.4], alpha=0.654 [0.246, 1.389] | RMSE_out=0.0044 (n_in=665, n_out=285)
```

Read those two lines together: with 950 orders at realistic noise the square-root law's
single coefficient is identified (true Y = 0.8 sits inside the CI) while a free exponent is
not — the CI on `alpha` spans the whole literature range. That is why desks fix the
exponent, and the library shows it instead of printing a point estimate.

### Your own fills

Three long-format `pandas` frames — `orders`, `fills`, `market` — validated by
`tcakit.schema` (tz-aware timestamps, `side ∈ {+1, −1}`, fills must reference a parent).
See [docs/DESIGN.md](docs/DESIGN.md) for the columns. An adapter for a per-child JSONL
execution log of multi-leg option combos (`tcakit.adapters.tcalogger`) is included and
produces a full post-trade report:

```bash
python examples/tcalogger_report.py tests/fixtures/tcalogger_sample.jsonl
```

which prints implementation shortfall per parent, option cost units, scorecards by leg /
escalator tier / urgency, and an **n-gate** stating whether the sample is large enough for
any impact-model number to be quoted (it is not, until ≥ 50 parents).

## What is where

```
src/tcakit/
  schema.py        canonical frames, validation, asof quote lookup
  synth.py         seeded synthetic market with known impact parameters
  benchmarks.py    arrival / VWAP / TWAP / close / participation / reversion(k)
  shortfall.py     Perold IS decomposition, exact identity
  impact.py        realized_impact; fit_sqrt_law, fit_almgren2005, fit_istar → FitResult
  schedule.py      Almgren-Chriss closed forms, efficient frontier, TWAP / VWAP / POV, cost/completion ratio
  experiments.py   A/B: stratified assignment, clustered SE, CUPED, MDE / required n
  scorecards.py    grouped costs + difficulty-adjusted residuals, bootstrap CIs
  options.py       per-leg NBBO net mid, $/contract, fraction of half-spread, delta-adj bps
  report.py        markdown report; every table states unit, sign, n
  adapters/        one file per data source
docs/DESIGN.md     the spec           docs/PLAN.md   roadmap, references, pre-registered bars
```

## Roadmap

v0.3 (shipped): Almgren-Chriss scheduler, efficient frontier, VWAP/TWAP/POV baselines ·
v0.4 (shipped): A/B module (symbol-day stratified assignment, clustered SE, CUPED, power) ·
v0.5: per-leg NBBO market frame for options, so spread and timing can be split and `bps_underlying` computed · a
private descriptive pass once ≥ 50 live parent orders exist (the impact-model n-gate) ·
optional: a public order-book validation on Nasdaq's free ITCH days (`docs/PLAN.md` §2).

## Data and privacy

Synthetic data from `tcakit.synth` with known parameters drives every recovery test. The
`tcalogger` adapter is additionally exercised on six of the author's IBKR paper-trading
combo fills from one day; they are gitignored (`tests/fixtures/private_*`), and the public
fixture `tests/fixtures/tcalogger_sample.jsonl` has the same schema so the same report runs
in CI. No live fills, positions or broker configuration are in the repo.

## Companion repos

[pricers](https://github.com/charlieyanhx/pricers) — option pricers validated against closed forms and QuantLib ·
[riskkit](https://github.com/charlieyanhx/riskkit) — VaR/ES, backtests with known size and power, SPAN margin ·
[volsurf](https://github.com/charlieyanhx/volsurf) — implied-vol surfaces from option chains with static-arbitrage
checks reported, not repaired ·
[quotesim](https://github.com/charlieyanhx/quotesim) — options quoting simulator with synthetic flow and an exact
P&L attribution ·
[deskboard](https://github.com/charlieyanhx/deskboard) — options risk and P&L dashboard
with deterministic replay; its planned execution page would consume tcakit's per-order
costs ·
[quant-research-agent](https://github.com/charlieyanhx/quant-research-agent) — a backtest
review agent and the evals that measure it ·
[tickq](https://github.com/charlieyanhx/tickq) — DuckDB market-data SQL: partitioned Parquet lake, ASOF
joins with the tie rule stated, quality checks with recall and precision ·
[lobcore](https://github.com/charlieyanhx/lobcore) — bounded-array limit order book in Rust with a
reference-book differential test, ITCH 5.0 replay and PyO3 bindings.

## References

Perold (1988) · Almgren & Chriss (2000) · Almgren, Thum, Hauptmann, Li (2005) · Kissell,
*The Science of Algorithmic Trading and Portfolio Management* · Bouchaud, Bonart, Donier,
Gould, *Trades, Quotes and Prices* · Zarinelli, Treccani, Farmer, Lillo (2015) · Bucci,
Benzaquen, Lillo, Bouchaud (2019) · Cont, Kukanov, Stoikov (2014) · Muravyev & Pearson
(2020) · Webster, *Handbook of Price Impact Modeling* (2023). Metric naming follows
[cuemacro/tcapy](https://github.com/cuemacro/tcapy).

MIT © Hanxiong (Charlie) Yan
