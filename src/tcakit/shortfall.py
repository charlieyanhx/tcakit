"""Implementation shortfall (Perold 1988) decomposition per parent order.

Cash convention: every component is `side * (price paid − reference) * qty`, so positive is
a cost to the trader regardless of side. The identity

    total == delay + spread + timing + opportunity + fees

is exact by construction; `execution == spread + timing` is always defined even when the
per-fill mid is unavailable (then spread and timing are NaN, never assumed zero).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import asof_mid, validate_fills, validate_orders

COMPONENTS = ["delay", "spread", "timing", "opportunity", "fees"]
BPS = 1e4


def _end_price(orders: pd.DataFrame, market: pd.DataFrame | None) -> pd.Series:
    """End-of-horizon mid: `orders.end_px` if given, else asof mid from market at end_ts."""
    if "end_px" in orders.columns:
        return orders["end_px"].astype(float)
    if market is None:
        raise ValueError("orders lacks end_px and no market frame was supplied")
    out = pd.Series(np.nan, index=orders.index, dtype=float)
    for sym, grp in orders.groupby("symbol"):
        out.loc[grp.index] = asof_mid(market, sym, grp["end_ts"]).to_numpy()
    return out


def _fill_mids(fills: pd.DataFrame, orders: pd.DataFrame, market: pd.DataFrame | None) -> pd.Series:
    if "mid" in fills.columns:
        return fills["mid"].astype(float)
    if market is None:
        return pd.Series(np.nan, index=fills.index, dtype=float)
    sym = fills["order_id"].map(orders.set_index("order_id")["symbol"])
    out = pd.Series(np.nan, index=fills.index, dtype=float)
    for s, grp in fills.groupby(sym):
        out.loc[grp.index] = asof_mid(market, s, grp["ts"]).to_numpy()
    return out


def implementation_shortfall(
    orders: pd.DataFrame,
    fills: pd.DataFrame,
    market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per parent order with cash components, the total, and bps of decision notional.

    Columns: order_id, side, qty, filled_qty, avg_px, decision_px, arrival_px, end_px,
    delay, spread, timing, execution, opportunity, fees, total, and *_bps for each.
    """
    validate_orders(orders)
    validate_fills(fills, orders)
    o = orders.set_index("order_id")
    side = o["side"].astype(float)

    f = fills.copy()
    f["mid"] = _fill_mids(fills, orders, market)
    f["notional"] = f["px"] * f["qty"]
    f["side"] = f["order_id"].map(side)
    f["arrival_px"] = f["order_id"].map(o["arrival_px"])
    f["spread_cash"] = f["side"] * (f["px"] - f["mid"]) * f["qty"]
    f["timing_cash"] = f["side"] * (f["mid"] - f["arrival_px"]) * f["qty"]
    if "fee" not in f.columns:
        f["fee"] = 0.0

    g = f.groupby("order_id")
    filled_qty = g["qty"].sum().reindex(o.index).fillna(0.0)
    notional = g["notional"].sum().reindex(o.index).fillna(0.0)
    avg_px = (notional / filled_qty).where(filled_qty > 0, np.nan)
    fees = g["fee"].sum().reindex(o.index).fillna(0.0)
    # NaN mids propagate to NaN spread/timing; an order with no fills has zero, not NaN
    spread = g["spread_cash"].sum(min_count=1).reindex(o.index)
    timing = g["timing_cash"].sum(min_count=1).reindex(o.index)
    spread = spread.where(filled_qty > 0, 0.0)
    timing = timing.where(filled_qty > 0, 0.0)

    end_px = _end_price(orders, market).set_axis(o.index)
    delay = side * filled_qty * (o["arrival_px"] - o["decision_px"])
    execution = (side * (avg_px - o["arrival_px"]) * filled_qty).where(filled_qty > 0, 0.0)
    unfilled = o["qty"] - filled_qty
    opportunity = side * unfilled * (end_px - o["decision_px"])
    total = delay + execution + opportunity + fees

    out = pd.DataFrame(
        {
            "side": o["side"],
            "qty": o["qty"],
            "filled_qty": filled_qty,
            "avg_px": avg_px,
            "decision_px": o["decision_px"],
            "arrival_px": o["arrival_px"],
            "end_px": end_px,
            "delay": delay,
            "spread": spread,
            "timing": timing,
            "execution": execution,
            "opportunity": opportunity,
            "fees": fees,
            "total": total,
        }
    )
    denom = o["qty"] * o["decision_px"]
    for c in COMPONENTS + ["execution", "total"]:
        out[f"{c}_bps"] = out[c] / denom * BPS
    return out.reset_index()
