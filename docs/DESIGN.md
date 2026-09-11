# tcakit — design

## Why this exists

Transaction cost analysis is the most-cited requirement across execution-research seats
(Tower Central Execution, Wellington Trading Research & Analytics, AQR Electronic Trading,
Citi Algo, Point72 Trading Research, Acadian, Voleon, Maven), and there is no maintained,
general-purpose open-source TCA library for equities and listed options. `cuemacro/tcapy`
is FX-spot only and carries a database/dashboard stack; the rest are notebooks.

`tcakit` is a small, tested library that answers the questions those desks actually ask:

1. **What did this order cost, and why?** — implementation-shortfall decomposition
   (delay / spread / impact-and-timing / opportunity / fees), benchmark slippage
   (arrival, interval VWAP, TWAP, close), participation, post-trade reversion.
2. **Which broker / venue / algo is cheaper, after controlling for difficulty?** —
   scorecards with difficulty-adjusted cost (regress cost on size/ADV, spread, volatility;
   compare residuals, not raw averages).
3. **What is the pre-trade cost estimate for a new order?** — market-impact models
   (square-root law, Almgren-Chriss temporary/permanent, I-star) calibrated on the
   firm's own fills, with out-of-sample error reported.

Non-goals (v0.x): order routing, live connectivity, dashboards, FX-specific conventions.

## Method rules (same as the working papers)

- Every metric has a sign convention stated once, in code, as `side * (paid - benchmark)`,
  positive = cost to the trader. No metric is reported without its unit (bps of benchmark
  notional) and its denominator.
- Decomposition identities are tested exactly (`sum(components) == total` to 1e-12).
- Calibrations report the pre-registered fit statistic **and** out-of-sample error on a
  held-out window. A fit with no held-out error is not a result.
- Synthetic data is generated with known parameters so that every estimator is tested for
  parameter recovery, not just for running without error.
- Real-data validation uses LOBSTER sample files (public) with a data-availability note.

## Canonical data model

Three long-format `pandas.DataFrame`s, validated by `schema.validate_*`. Timestamps are
tz-aware UTC `datetime64[ns, UTC]`; prices are floats in quote currency; quantities are
floats (shares or contracts); `side` is `+1` buy / `-1` sell.

| frame | one row per | required columns |
|-------|-------------|------------------|
| `orders` | parent order | `order_id, symbol, side, qty, decision_ts, arrival_ts, end_ts, decision_px, arrival_px` |
| `fills` | child fill | `fill_id, order_id, ts, px, qty, venue, fee` (+ optional `mid`, `bid`, `ask`, `algo`, `broker`) |
| `market` | quote/trade snapshot | `symbol, ts, bid, ask, last, volume` (interval-aggregated is fine) |

`decision_px` is the mid at the investment decision; `arrival_px` is the mid when the order
reaches the desk/algo. If the caller cannot distinguish them, set them equal and the delay
component is zero by construction — it is not silently dropped.

## Implementation shortfall (Perold 1988 / Almgren-Chriss 2000 accounting)

For a parent order with total `Q`, executed `Q_x = Σ q_i`, average fill price `P̄`,
decision `P_d`, arrival `P_0`, end-of-horizon mid `P_T`, fees `F`, side `s`:

```
total     = s·(Q_x·P̄ − Q_x·P_d) + s·(Q − Q_x)·(P_T − P_d) + F        # cash, +ve = cost
delay     = s·Q_x·(P_0 − P_d)
spread    = Σ s·q_i·(p_i − mid_i)                                   # crossing cost
timing    = Σ s·q_i·(mid_i − P_0)                                   # impact + drift during execution
opportunity = s·(Q − Q_x)·(P_T − P_d)
fees      = F
identity: total == delay + spread + timing + opportunity + fees
```

All five are also returned in bps of decision notional `Q·P_d`. If `mid_i` is absent from
fills, it is looked up from `market` as the prevailing quote at `ts` (asof-join, backward),
and `spread` is reported as `NaN` if no quote is available rather than assumed zero.

## Benchmarks (per order)

| name | definition | slippage |
|------|------------|----------|
| arrival | `P_0` | `s·(P̄ − P_0)/P_0` |
| interval VWAP | market VWAP over `[first_fill_ts, last_fill_ts]` | `s·(P̄ − VWAP)/VWAP` |
| interval TWAP | mean mid over the same interval | `s·(P̄ − TWAP)/TWAP` |
| close | last mid at `end_ts` | `s·(P̄ − P_T)/P_T` |
| participation | `Q_x / market volume in interval` | — |
| reversion(k) | `s·(P̄ − mid(t_last + k))/P̄` | +ve = temporary impact paid back |

Interval VWAP uses market trades **excluding the order's own fills** when `market`
carries our fills flagged; otherwise it is documented as inclusive.

## Impact models (`impact.py`)

Realized impact per parent order `I = s·(P̄ − P_0)/P_0` (or mid-to-mid `s·(P_T − P_0)/P_0`
for permanent impact). Explanatory variables: participation `π = Q_x / V_interval`,
size ratio `Q/ADV`, daily vol `σ`, spread.

1. **Square-root law** — `I = Y·σ·sqrt(Q/ADV) + ε`; fit `Y` by OLS through the origin;
   report `Y`, R², and held-out RMSE. Literature range `Y ∈ [0.5, 1.5]`.
2. **Almgren et al. (2005)** — temporary `η·σ·π^β` (β≈0.6) and permanent `γ·σ·(Q/ADV)^α`
   (α≈1); nonlinear least squares with bootstrap CIs on exponents.
3. **I-star (Kissell)** — `I* = a1·(Q/ADV)^a2·σ^a3`, temporary share `b1·π^a4`; same
   fitting harness.
4. **Propagator** (Bouchaud) — deferred; needs signed-trade series.

Every fit is done through one `fit(model, orders, market, holdout=0.3)` entry point that
returns a `FitResult` with `params, param_ci, r2_in, rmse_in, rmse_out, n_in, n_out`.

## Scorecards (`scorecards.py`)

Group orders by `broker`/`venue`/`algo`. For each group: n, notional, mean and median
arrival slippage, VWAP slippage, spread cost, reversion. Then **difficulty adjustment**:
fit `slippage ~ σ·sqrt(Q/ADV) + spread_bps + log(notional)` on all orders, and report each
group's mean residual with a bootstrap CI — the number that survives a "your orders were
just harder" objection.

## Synthetic market (`synth.py`)

Seeded generator: mid follows arithmetic Brownian motion with daily vol `σ`; a U-shaped
intraday volume profile; permanent impact `γ·σ·(q/ADV)` and temporary impact
`η·σ·(π)^0.6` applied to our own child orders; spread `S` constant or a function of time;
child fills cross the spread with probability `p_cross` else fill at mid. Outputs the three
canonical frames plus the true parameter dict so tests can assert recovery.

## Real-data validation (`docs/validation.md`, later)

LOBSTER sample (AMZN/AAPL/GOOG/INTC/MSFT, 2012-06-21, levels 1–10): replay a synthetic
parent order against the real book to produce fills, then run the full pipeline. The
data-availability note states exactly which file and hash was used.

## Package layout

```
src/tcakit/
  __init__.py       public API re-exports
  schema.py         frame validation, asof quote lookup
  synth.py          seeded synthetic market + fills
  benchmarks.py     arrival / VWAP / TWAP / close / participation / reversion
  shortfall.py      IS decomposition
  impact.py         model definitions + fit harness
  scorecards.py     grouped stats + difficulty adjustment
  report.py         markdown report for one order or one scorecard
tests/              pytest; one file per module; identities + recovery tests
docs/DESIGN.md      this file
```

## Roadmap

- v0.1 — schema, synth, benchmarks, shortfall, tests (this week)
- v0.2 — impact fits with held-out error, scorecards, markdown report
- v0.3 — LOBSTER replay + validation note; options-specific benchmarks (mid-of-NBBO,
  delta-adjusted cost for multi-leg); PyPI release
- later — propagator model, A/B framework for algo variants (separate package)
