# tcakit — plan v2 (2026-09-11)

**Scope (owner's call, 2026-09-11): this project is a tool, tested on our own fills.** The
public-data study (ITCH panel, paper #8) is optional later work, kept below because the
identification facts it rests on also govern what the tool may claim.

Supersedes the v1 plan pasted from the job-listing analysis. Two facts changed it:
the public dataset it named no longer exists in downloadable form, and the identification
strategy it assumed does not work. Both are documented below so the paper's "what this data
can and cannot identify" section is the plan, not an afterthought.

## 0. What changed and why

| v1 assumption | What is actually true (checked 2026-09-11) | Consequence |
|---|---|---|
| LOBSTER sample files (5 names, 2012-06-21, L10) are a free download | lobsterdata.com is now a React app; samples are behind a "book a sample" request form; the old direct URLs return the SPA shell | Public data = **Nasdaq's own TotalView-ITCH 5.0 samples**: 15 full-market days 2018–2020 at `emi.nasdaq.com/ITCH/Nasdaq ITCH/`, ~4.8 GB gz each, no account. 15× the data, and it is the source LOBSTER derived from. |
| "Synthetic parent orders are replayed against the real book to produce fills, then the full pipeline runs" → calibrate the square-root law | A historical book does not react to orders that were not there. Replay identifies the **mechanical** cost (spread crossing + walking the book) only. The metaorder impact literature (Almgren 2005; Zarinelli 2015; Bucci 2019) was all fit on *proprietary* metaorders. | `impact.py` is split into **public-identifiable** models (OFI λ, trade response/propagator, book-walk cost) and **metaorder** models (sqrt / Almgren / I-star) that need real parent orders. Paper #8's title changes accordingly. |
| Private validation: "fills/executions with timestamps … order tickets" | In the research repo today: **6 real IBKR paper combo fills** (`src/livetester/logs/tca_20260424.jsonl`, right schema), 30 livetester entries with `decision_to_submit_ms: null`, and 100 `reports/live_day_*/chn_bot_trades.jsonl` that are **byte-identical copies** of one file. Live small-cap fills accrue on the AWS box, pulled nightly. | Private adapter targets the `TCALogger` JSONL schema and is validated on the 6 records; no private number is quoted before **≥ 50 parent orders** (n-gate below). The descriptive pass moves from v0.2 to "when the gate is met". |
| Delay component measurable | `ExecutionIntent` carries `arrival_mid` and `arrival_mid_at_child`, no decision timestamp | On private data **delay = 0 by construction** and the report says so. Adding `decision_ts/decision_mid` to the live bot is a user-owned change to an armed deployment; not made here. |
| Options cost in bps of notional | A 2-leg credit spread with net mid −$1.00 and 3¢ slippage is "300 bps" | Units: **$ per contract** (primary; matches `implementation_shortfall_cents`), **fraction of net half-spread** (tier-comparable), **bps of delta-adjusted underlying notional** (comparable to equity scorecards). Vol points per leg only. |
| Hardware unstated | This Mac: 8.6 GB RAM, 9.4 GiB free disk | ITCH must be **streamed** (curl → gunzip → parser), one day at a time, per-symbol parquet out, gz deleted. Never decompress to disk. |

## 1. Lit review — what each source contributes

| Source | Use in tcakit | Number to check against |
|---|---|---|
| Perold (1988) | IS decomposition identity (`shortfall.py`, done) | Σ components == total, 1e-9 |
| Almgren & Chriss (2000) | scheduler, efficient frontier (`schedule.py`) | linear-schedule cost/completion ∈ (2/3, 0.72) — tested |
| Almgren, Thum, Hauptmann, Li (2005) | temporary η·σ·π^β, permanent γ·σ·(Q/ADV)^α; fit harness with bootstrap CI on exponents | β ≈ 0.6, α ≈ 0.9–1.0 |
| Kissell — I-star | a1·(Q/ADV)^a2·σ^a3 with temporary share b1·π^a4 | same harness, compare held-out RMSE with Almgren-2005 |
| Tóth et al. (2011); Bouchaud et al. *Trades, Quotes and Prices* | square-root law Y·σ·√(Q/ADV); 2/3 rule; propagator/response function | Y ∈ [0.5, 1.5]; response decays as a slow power law |
| Zarinelli, Treccani, Farmer, Lillo (2015) | log form fits 5 decades vs 2 for sqrt; impact *surface* in (duration, participation) | pre-registered alternative to sqrt in the metaorder harness |
| Bucci, Benzaquen, Lillo, Bouchaud (2019, PRL) | linear→sqrt crossover: linear for Q/ADV ∈ [1e-5, 1e-3], sqrt for [1e-3, 1e-1] | the crossover location is the test the metaorder harness runs when n allows |
| Cont, Kukanov, Stoikov (2014) | **OFI** at the touch → Δmid linear, slope ∝ 1/depth; the public-data impact model | R² ≥ 50% for 44/50 stocks at 10 s |
| Kyle & Obizhaeva (2016) invariance | bet-size covariate W = P·V·σ for the scorecard difficulty regression | compare against σ·√(Q/ADV) on held-out residual RMSE |
| Webster, *Handbook of Price Impact Modeling* (2023) | A/B testing execution algos without bias; Obizhaeva-Wang class | randomize at parent level, cluster by symbol-day |
| Deng, Xu, Kohavi, Walker (2013) — CUPED | variance reduction in `experiments.py`; covariate = pre-trade cost estimate | report variance reduction ratio |
| Muravyev & Pearson (2020, RFS) | options: effective spread of *timed* executions < 40% of quoted; our escalator tiers are exactly that timing | report realized fraction of half-spread by tier hit |
| cuemacro/tcapy; AshJha0/electronic-trading; cvxgrp/vwap_opt_exec | metric naming / report layout; golden-value tests; VWAP tracking as a convex program | — |

Not reused: tcapy's DB/dashboard stack; hftbacktest (queue-position modelling deferred — replay
already shows the mechanical cost and queue position is a second-order refinement on it).

## 2. Identification map (the spine of paper #8)

```
                              public ITCH (15 days)      private fills (TCALogger)
mechanical cost (spread+walk)  YES — replay               YES — realized
OFI impact λ (Cont et al.)     YES — per symbol-day        n/a
trade response / propagator    YES — signed executions     n/a
intraday σ, depth, volume      YES — pre-trade inputs      n/a
sqrt-law Y, Almgren η γ β α    NO  — book cannot react     YES — needs n (see gate)
temporary vs permanent split   NO                          YES — needs post-trade mids
delay component                n/a                         NO  — decision_ts absent
```

The pre-trade model in v0.2 is therefore: **mechanical cost from the replayed book +
OFI-implied impact of the order's own flow**, and the paper's honest claim is that this is a
*lower bound* on metaorder impact, with the gap to the literature's Y reported, not filled in.

## 3. Pre-registered bars (state before running)

- ITCH parse: reconstructed book at 16:00 for each of the 10 panel names must match the
  ITCH cross/NOII closing print to the tick; message counts tie to the day's total.
- OFI fit: per symbol-day OLS of Δmid on OFI at 10 s; bar = R² ≥ 0.50 on ≥ 70 % of
  symbol-days (Cont et al.: 44/50). Below that, the extractor is wrong before the model is.
- Response function: R(ℓ) = ⟨ε_t · (m_{t+ℓ} − m_t)⟩ for ℓ = 1..1000 trades; bar = positive,
  monotone-increasing on ℓ ≤ 100, slow decay thereafter. A negative R(1) is a sign bug.
- Book-walk cost: monotone in size; at 1 share equals the half-spread to 1e-9.
- Held-out by **day**, never by row: fit on 10 days, test on 5. Report `n_in, n_out,
  rmse_out` on every `FitResult`.
- Metaorder harness (synthetic + private): noise-free generator → Y = 1.000000; noisy →
  true Y inside the bootstrap CI ≥ 90 % of seeds.
- **n-gate for private numbers**: quote nothing until ≥ 50 parent orders AND the bootstrap
  CI on Y is narrower than the literature range [0.5, 1.5]. With n = 6 the CI is the whole
  real line; the report prints the gate status instead of a number.

## 4. Architecture (delta from v1)

```
tcakit/
  schema.py        (done) orders / fills / market; asof quote lookup
  synth.py         (done) seeded market + known impact params
  benchmarks.py    (done) arrival / VWAP / TWAP / close / participation / reversion(k)
  shortfall.py     (done) Perold IS, exact identity
  impact/
    metaorder.py   sqrt-law (done, moved), Almgren-2005, I-star, log-form; one fit() harness
    ofi.py         Cont-Kukanov-Stoikov OFI λ per symbol-day
    response.py    signed-trade response function + propagator fit
    bookwalk.py    instantaneous cost of sweeping a snapshot to size q
    fitresult.py   FitResult(params, ci, r2_in, rmse_in, rmse_out, n_in, n_out)
  book/
    itch.py        streaming ITCH 5.0 parser (struct.unpack; early filter on locate)
    lob.py         order-by-order book builder → L10 snapshots + signed trades parquet
    replay.py      synthetic parent → child fills against snapshots (mechanical cost)
  scorecards.py    groups + difficulty residuals (σ√(Q/ADV), spread, log notional | KO invariance)
  schedule.py      Almgren-Chriss, frontier, VWAP/TWAP/POV baselines, cvx VWAP tracking
  experiments.py   parent-level randomization, symbol-day clusters, CUPED, power
  options.py       NBBO mid per leg, multi-leg parents, the three unit conventions
  report.py        markdown; every number carries unit + basis + n
  adapters/
    tcalogger.py   private JSONL → canonical frames (gitignored fixture of the 6 records)
```

## 5. Milestones (tool first; public study optional)

| Version | Scope | Status |
|---|---|---|
| v0.1 | schema, synth, IS, benchmarks, sqrt-law | done — 27 tests |
| v0.2 | Almgren-2005 + I-star fits (hold-out by day); scorecards + difficulty adjustment; options units; markdown report; `adapters/tcalogger` on our fills; CI, licence, README | **done 2026-09-11 — 48 tests; report runs on the 6 real fills** |
| v0.3 | Almgren-Chriss scheduler + efficient frontier; VWAP/TWAP/POV baselines; A/B module (parent-level randomisation, symbol-day clusters, CUPED, power) | next |
| v0.4 | market frame for options (per-leg NBBO at fill from the chain recorder) → real spread/timing split, reversion(k), `bps_underlying`; add `decision_ts` to the live logger (user-owned) | when the recorder runs |
| private pass | descriptive report on ≥ 50 live parents from the AWS box | gated (§3) |
| optional | ITCH public panel, OFI / response / book-walk, paper #8 | not scheduled |

## 6. Open items owned by the user

1. `decision_ts` / `decision_mid` in `ExecutionIntent` on the live bot (armed deployment —
   not touched from here).
2. Whether the 6 paper e2e fills may ship as a gitignored private fixture (they are paper
   fills from a test harness; still kept private by default).
3. Databento $125 signup credit would add recent (2024–26) days to the panel; optional.

## 7. Engineering constraints

- 8.6 GB RAM / 9.4 GiB disk: stream, filter on stock-locate before decoding payloads, write
  per-symbol parquet, delete gz. Budget: < 60 min per ITCH day in pure Python; otherwise
  bind a Rust parser (itchy / bbalouki/itch) — decision deferred until measured.
- Every fit through one entry point; every result carries `n_in, n_out, rmse_out`.
- Sign convention once, in code: `side · (paid − benchmark)`, positive = cost.
- Fork experiments, not accounting: no change to `shortfall.py` / `benchmarks.py` while
  adding models.
