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

## Impact models (`impact/`)

Two families, kept apart because they are identified by different data (see
`docs/PLAN.md` §2). A historical book cannot react to orders that were not in it, so
replaying synthetic parents against public data measures **mechanical** cost only; the
square-root law and its relatives need real parent orders.

**Public-identifiable (fit on ITCH / any L2 tape):**

1. **OFI** (Cont, Kukanov, Stoikov 2014) — `Δmid = λ·OFI + ε` per symbol-day at 10 s;
   `λ ∝ 1/depth`. Literature: R² ≥ 0.50 on 44/50 stocks.
2. **Trade response / propagator** (Bouchaud) — `R(ℓ) = ⟨ε_t·(m_{t+ℓ} − m_t)⟩` on signed
   executions; slow power-law decay; `R(1) < 0` is a sign bug.
3. **Book-walk cost** — instantaneous cost of sweeping a snapshot to size `q`; equals the
   half-spread at `q = 1` share to 1e-9 and is monotone in `q`.

**Metaorder (fit on real parent orders; synthetic for recovery tests):**

Realized impact per parent `I = s·(P̄ − P_0)/P_0` (permanent: `s·(P_T − P_0)/P_0`).
Explanatory variables: participation `π = Q_x / V_interval`, `Q/ADV`, daily `σ`, spread.

4. **Square-root law** — `I = Y·σ·sqrt(Q/ADV)`; OLS through the origin. `Y ∈ [0.5, 1.5]`.
5. **Almgren et al. (2005)** — temporary `η·σ·π^β` (β≈0.6), permanent `γ·σ·(Q/ADV)^α`
   (α≈0.9–1); NLS with bootstrap CIs on exponents.
6. **I-star (Kissell)** — `a1·(Q/ADV)^a2·σ^a3`, temporary share `b1·π^a4`.
7. **Log form** (Zarinelli et al. 2015) — pre-registered alternative to 4; fits five decades
   of `Q/ADV` where sqrt fits two. The Bucci et al. (2019) linear→sqrt crossover at
   `Q/ADV ≈ 1e-3` is the test run when `n` allows.

Every fit goes through one `fit(model, ..., holdout=...)` entry point returning a
`FitResult(params, param_ci, r2_in, rmse_in, rmse_out, n_in, n_out)`. Hold-out is **by
day**, never by row. Private metaorder fits print an n-gate (≥ 50 parents AND CI on `Y`
narrower than `[0.5, 1.5]`) instead of a number until it is met.

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

LOBSTER's free samples are no longer downloadable (request form since 2025). Public data
is Nasdaq's own TotalView-ITCH 5.0 samples: 15 full-market days 2018–2020 at
`emi.nasdaq.com/ITCH/Nasdaq ITCH/`, ~4.8 GB gz each. `book/itch.py` streams one day at a
time (curl → gunzip → parser, early filter on stock-locate, per-symbol parquet, gz never
written to disk) for a 10-name panel; the data-availability note records file names and
SHA-256 of the gz streams. Replay of synthetic parents against the rebuilt book yields
the mechanical cost; OFI and response fits yield the public-identifiable impact; the gap
to the metaorder literature is reported, not filled in.

Private validation uses the live bot's `TCALogger` JSONL (`adapters/tcalogger.py`,
gitignored fixture) and is gated on `n` as above. `decision_ts` is absent from that
schema, so the delay component is zero by construction on private data and the report
says so.

## Options conventions (`options.py`)

Per-leg mid-of-NBBO at fill time; a multi-leg parent's benchmark is the net of leg mids.
Costs are reported in three units, never in bps of premium notional (a 3¢ miss on a
−$1.00 spread is "300 bps"): **$ per contract** (primary; equals the bot's
`implementation_shortfall_cents`), **fraction of net half-spread** (comparable across
escalator tiers — Muravyev & Pearson 2020 show timed option executions pay < 40 % of the
quoted spread, which is what the tiers measure), and **bps of delta-adjusted underlying
notional** (comparable with equity scorecards). Vol points per leg only when vega is
available.

## Package layout

```
src/tcakit/
  __init__.py       public API re-exports
  schema.py         frame validation, asof quote lookup
  synth.py          seeded synthetic market + fills
  benchmarks.py     arrival / VWAP / TWAP / close / participation / reversion
  shortfall.py      IS decomposition
  impact/           metaorder.py, ofi.py, response.py, bookwalk.py, fitresult.py
  book/             itch.py (streaming parser), lob.py (builder), replay.py
  scorecards.py     grouped stats + difficulty adjustment
  schedule.py       Almgren-Chriss, frontier, VWAP/TWAP/POV, cvx VWAP tracking
  experiments.py    A/B: parent-level randomization, symbol-day clusters, CUPED, power
  options.py        NBBO mid per leg, multi-leg parents, three unit conventions
  report.py         markdown report; every number carries unit + basis + n
  adapters/         one file per data source; private ones gitignored
tests/              pytest; one file per module; identities + recovery tests
docs/DESIGN.md      this file (the spec)
docs/PLAN.md        milestones, lit review, pre-registered bars, identification map
```

## Roadmap

See `docs/PLAN.md` §5. Order: public first (ITCH panel, public-identifiable impact,
paper #8), then scorecards / private adapter / options / PyPI, then scheduler + A/B. The
private descriptive pass runs when ≥ 50 live parent orders have accrued.
