"""Listed-option conventions: per-leg NBBO mids, multi-leg net prices, and cost units.

Costs on option combos are never reported in bps of premium notional — a 3¢ miss on a
−$1.00 credit spread would read as "300 bps". Three units are used instead:

* `usd_per_contract`   — cash cost per contract (quote-unit cost × multiplier);
* `frac_half_spread`   — cost per contract in quote units ÷ the net half-spread at arrival;
                         1.0 = paid the full half-spread, 0 = filled at mid, <0 = better
                         than mid (the number an escalator tier is judged on);
* `bps_underlying`     — cost per contract ÷ (|delta| · spot · multiplier) × 1e4, the
                         delta-adjusted notional basis comparable with equity scorecards.
                         NaN unless `delta` and `spot` are supplied — never assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BPS = 1e4


def net_mid(leg_quotes: pd.DataFrame) -> tuple[float, float]:
    """Net mid and net half-spread of a combo from per-leg quotes.

    `leg_quotes` columns: bid, ask, ratio (signed: +1 buy leg, −1 sell leg, or ±n).
    Net mid = Σ ratio·mid; net half-spread = Σ |ratio|·(ask−bid)/2 (the cost of crossing
    every leg). Any NaN quote makes both NaN.
    """
    mid = (leg_quotes["bid"] + leg_quotes["ask"]) / 2.0
    half = (leg_quotes["ask"] - leg_quotes["bid"]) / 2.0
    if mid.isna().any() or half.isna().any():
        return np.nan, np.nan
    return float((leg_quotes["ratio"] * mid).sum()), float((leg_quotes["ratio"].abs() * half).sum())


def contract_costs(
    shortfall: pd.DataFrame,
    orders: pd.DataFrame,
    cost_col: str = "execution",
) -> pd.DataFrame:
    """Per-order option cost units from a shortfall frame and the orders frame.

    Requires `orders.multiplier` and `orders.half_spread` (quote units). Optional
    `orders.delta` (per contract, signed or not) and `orders.spot` enable `bps_underlying`.
    """
    o = orders.set_index("order_id")
    s = shortfall.set_index("order_id")
    mult = o["multiplier"].astype(float).reindex(s.index)
    filled = s["filled_qty"].astype(float)
    per_contract_quote = (s[cost_col] / filled).where(filled > 0, np.nan)
    out = pd.DataFrame(index=s.index)
    out["cost_quote_per_contract"] = per_contract_quote
    out["usd_per_contract"] = per_contract_quote * mult
    out["usd_total"] = s[cost_col] * mult
    hs = o["half_spread"].astype(float).reindex(s.index)
    out["frac_half_spread"] = (per_contract_quote / hs).where(hs > 0, np.nan)
    if {"delta", "spot"} <= set(o.columns):
        notional = o["delta"].abs() * o["spot"] * mult
        out["bps_underlying"] = out["usd_per_contract"] / notional.reindex(s.index) * BPS
    else:
        out["bps_underlying"] = np.nan
    return out.reset_index()
