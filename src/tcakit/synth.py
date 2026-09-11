"""Seeded synthetic market + parent orders + child fills with known impact parameters.

Model (one symbol "SYN", 1-minute bars, `MINUTES_PER_DAY` per session):

* mid is geometric Brownian motion with daily vol `sigma_daily` (per-minute
  `sigma_daily / sqrt(MINUTES_PER_DAY)`), plus the **permanent** impact of our own orders;
* each parent order of size Q executes linearly (one child per minute) at a target
  participation rate drawn from `pov_range`, so `duration = Q / (pov · ADV / MINUTES_PER_DAY)`
  clipped to `duration_range`; after a fraction φ of Q is done the cumulative impact on mid is
  `side · Y · sigma_daily · sqrt(φ · Q / ADV)` (square-root law, coefficient `Y`), applied
  multiplicatively to the arrival mid and never reverting;
* a child fills at the post-impact mid, plus `side · half_spread` with probability
  `p_cross` (else at mid); fees are `fee_per_share · qty`;
* market volume follows a U-shaped intraday profile that sums to `adv` per day and is
  inclusive of our own prints;
* `decision_ts` precedes `arrival_ts` by a random 0–5 minutes; `end_ts` is `reversion_min`
  after the last fill; `decision_px`/`arrival_px`/`end_px` are the mids at those times.

Orders never overlap in time, so each order's realized impact is its own. The true
parameters are returned so estimators can be tested for recovery.

A day may hold fewer than `orders_per_day` orders when the windows do not fit; the count
is therefore an upper bound. Note that our own permanent impact adds variance to the mid
path, so vol measured from `market` is *higher* than `sigma_daily` when `sqrt_law_y > 0` —
this is a real effect (impact contaminates realized-vol estimates), not a bug, and the
recovery tests pass the true `sigma_daily` explicitly to isolate the estimator.
`noise_scale=0` switches the Brownian noise off (impact still uses `sigma_daily`) for
exact identity checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 390
SYMBOL = "SYN"


@dataclass
class SynthResult:
    orders: pd.DataFrame
    fills: pd.DataFrame
    market: pd.DataFrame
    params: dict = field(default_factory=dict)


def _u_shape(n: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, n)
    w = 1.0 + 2.0 * x**2
    return w / w.sum()


def simulate(
    n_days: int = 20,
    orders_per_day: int = 10,
    seed: int = 0,
    mid0: float = 100.0,
    sigma_daily: float = 0.02,
    adv: float = 1_000_000.0,
    spread_bps: float = 5.0,
    sqrt_law_y: float = 1.0,
    p_cross: float = 0.7,
    fee_per_share: float = 0.002,
    size_adv_range: tuple[float, float] = (0.0005, 0.02),
    pov_range: tuple[float, float] = (0.05, 0.25),
    duration_range: tuple[int, int] = (5, 90),
    reversion_min: int = 15,
    noise_scale: float = 1.0,
    start: str = "2026-01-05 14:30:00",
) -> SynthResult:
    rng = np.random.default_rng(seed)
    n_min = n_days * MINUTES_PER_DAY
    sig_min = sigma_daily / np.sqrt(MINUTES_PER_DAY)

    # session timestamps: n_days consecutive business days of MINUTES_PER_DAY bars
    day0 = pd.Timestamp(start, tz="UTC")
    days = pd.bdate_range(day0.normalize(), periods=n_days)
    ts = np.concatenate(
        [pd.date_range(d + (day0 - day0.normalize()), periods=MINUTES_PER_DAY, freq="1min").to_numpy()
         for d in days]
    )
    ts = pd.DatetimeIndex(ts).tz_localize("UTC") if pd.DatetimeIndex(ts).tz is None else pd.DatetimeIndex(ts)

    logmid = np.log(mid0) + np.cumsum(noise_scale * sig_min * rng.standard_normal(n_min))
    # permanent impact is applied in log space from each child fill onward
    volume = np.tile(_u_shape(MINUTES_PER_DAY) * adv, n_days)
    half_frac = spread_bps / 2 / 1e4  # half-spread as a fraction of the prevailing mid

    orders, fills = [], []
    oid = 0
    for d in range(n_days):
        base = d * MINUTES_PER_DAY
        # carve non-overlapping windows: [decision, arrival .. last fill .. end]
        cursor = 6
        placed = 0
        while placed < orders_per_day:
            delay = int(rng.integers(0, 6))
            q_adv = float(np.exp(rng.uniform(np.log(size_adv_range[0]), np.log(size_adv_range[1]))))
            pov = float(rng.uniform(*pov_range))
            dur = int(np.clip(round(q_adv * MINUTES_PER_DAY / pov), *duration_range))
            if cursor + delay + dur + reversion_min >= MINUTES_PER_DAY:
                break
            gap = int(rng.integers(0, 4))
            dec = base + cursor + gap
            arr = dec + delay
            side = int(rng.choice([1, -1]))
            qty = q_adv * adv
            child = qty / dur
            arrival_px = float(np.exp(logmid[arr]))
            for j in range(1, dur + 1):
                t = arr + j
                imp_new = sqrt_law_y * sigma_daily * np.sqrt(j / dur * q_adv)
                imp_old = sqrt_law_y * sigma_daily * np.sqrt((j - 1) / dur * q_adv)
                logmid[t:] += np.log1p(side * imp_new) - np.log1p(side * imp_old)
                mid_t = float(np.exp(logmid[t]))
                px = mid_t * (1 + side * half_frac * (rng.random() < p_cross))
                fills.append(
                    {
                        "fill_id": f"f{oid}_{j}",
                        "order_id": f"o{oid}",
                        "ts": ts[t],
                        "px": float(px),
                        "qty": float(child),
                        "mid": mid_t,
                        "venue": "SYN-X",
                        "fee": float(fee_per_share * child),
                    }
                )
            last = arr + dur
            end = last + reversion_min
            orders.append(
                {
                    "order_id": f"o{oid}",
                    "symbol": SYMBOL,
                    "side": side,
                    "qty": float(qty),
                    "decision_ts": ts[dec],
                    "arrival_ts": ts[arr],
                    "end_ts": ts[end],
                    "decision_px": float(np.exp(logmid[dec])),
                    "arrival_px": arrival_px,
                    "end_px": float(np.exp(logmid[end])),
                    "duration_min": dur,
                }
            )
            oid += 1
            placed += 1
            cursor = end - base + 1

    mid = np.exp(logmid)
    market = pd.DataFrame(
        {
            "symbol": SYMBOL,
            "ts": ts,
            "bid": mid * (1 - half_frac),
            "ask": mid * (1 + half_frac),
            "last": mid,
            "volume": volume,
        }
    )
    params = {
        "n_days": n_days, "orders_per_day": orders_per_day, "seed": seed, "mid0": mid0,
        "sigma_daily": sigma_daily, "adv": adv, "spread_bps": spread_bps,
        "sqrt_law_y": sqrt_law_y, "p_cross": p_cross, "fee_per_share": fee_per_share,
        "reversion_min": reversion_min, "pov_range": pov_range, "noise_scale": noise_scale,
    }
    return SynthResult(pd.DataFrame(orders), pd.DataFrame(fills), market, params)
