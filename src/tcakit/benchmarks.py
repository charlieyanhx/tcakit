"""Per-order benchmark slippage: arrival, interval VWAP/TWAP, close, participation, reversion.

Slippage is `side * (avg_px − benchmark) / benchmark * 1e4` (bps), positive = cost.
The execution interval is `[first_fill_ts, last_fill_ts]` inclusive on market bar timestamps.
Interval VWAP uses `market.last * market.volume` over that interval; it is inclusive of the
order's own prints unless the caller has removed them from `market`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import asof_mid, market_mid, validate_fills, validate_market, validate_orders

BPS = 1e4


def _interval(market: pd.DataFrame, symbol: str, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    m = market[market["symbol"] == symbol]
    return m[(m["ts"] >= t0) & (m["ts"] <= t1)]


def interval_vwap(market: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, symbol: str | None = None) -> float:
    symbol = symbol or market["symbol"].iloc[0]
    bars = _interval(market, symbol, t0, t1)
    vol = bars["volume"].sum()
    return float((bars["last"] * bars["volume"]).sum() / vol) if vol > 0 else np.nan


def interval_twap(market: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, symbol: str | None = None) -> float:
    symbol = symbol or market["symbol"].iloc[0]
    bars = _interval(market, symbol, t0, t1)
    return float(market_mid(bars).mean()) if len(bars) else np.nan


def interval_volume(market: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, symbol: str | None = None) -> float:
    symbol = symbol or market["symbol"].iloc[0]
    return float(_interval(market, symbol, t0, t1)["volume"].sum())


def _slip(side: float, paid: float, bench: float) -> float:
    if np.isnan(paid) or np.isnan(bench) or bench == 0:
        return np.nan
    return side * (paid - bench) / bench * BPS


def benchmark_slippage(
    orders: pd.DataFrame,
    fills: pd.DataFrame,
    market: pd.DataFrame,
    reversion_horizon: str | pd.Timedelta = "5min",
) -> pd.DataFrame:
    """One row per order: avg_px, first/last fill ts, arrival/vwap/twap/close bps,
    participation, reversion_bps at `reversion_horizon` after the last fill."""
    validate_orders(orders)
    validate_fills(fills, orders)
    validate_market(market)
    horizon = pd.Timedelta(reversion_horizon)

    f = fills.assign(notional=fills["px"] * fills["qty"])
    g = f.groupby("order_id")
    agg = pd.DataFrame(
        {
            "filled_qty": g["qty"].sum(),
            "notional": g["notional"].sum(),
            "first_ts": g["ts"].min(),
            "last_ts": g["ts"].max(),
        }
    ).reindex(orders["order_id"])
    agg["avg_px"] = agg["notional"] / agg["filled_qty"]

    rows = []
    for o, a in zip(orders.itertuples(index=False), agg.itertuples()):
        side = float(o.side)
        filled = 0.0 if pd.isna(a.filled_qty) else float(a.filled_qty)
        avg_px = np.nan if filled == 0 else float(a.avg_px)
        end_px = float(o.end_px) if "end_px" in orders.columns else float(
            asof_mid(market, o.symbol, pd.Series([o.end_ts])).iloc[0]
        )
        if filled > 0:
            vwap = interval_vwap(market, a.first_ts, a.last_ts, o.symbol)
            twap = interval_twap(market, a.first_ts, a.last_ts, o.symbol)
            mkt_vol = interval_volume(market, a.first_ts, a.last_ts, o.symbol)
            participation = filled / mkt_vol if mkt_vol > 0 else np.nan
            rev_mid = float(asof_mid(market, o.symbol, pd.Series([a.last_ts + horizon])).iloc[0])
            last_quote = market.loc[market["symbol"] == o.symbol, "ts"].max()
            if a.last_ts + horizon > last_quote:
                rev_mid = np.nan
            reversion = _slip(side, avg_px, rev_mid) if not np.isnan(rev_mid) else np.nan
            # reversion is quoted on avg_px, not on the later mid
            reversion = np.nan if np.isnan(rev_mid) else side * (avg_px - rev_mid) / avg_px * BPS
        else:
            vwap = twap = np.nan
            participation = 0.0
            reversion = np.nan
        rows.append(
            {
                "order_id": o.order_id,
                "side": o.side,
                "qty": o.qty,
                "filled_qty": filled,
                "avg_px": avg_px,
                "first_ts": a.first_ts,
                "last_ts": a.last_ts,
                "arrival_px": o.arrival_px,
                "vwap": vwap,
                "twap": twap,
                "end_px": end_px,
                "arrival_bps": _slip(side, avg_px, float(o.arrival_px)),
                "vwap_bps": _slip(side, avg_px, vwap),
                "twap_bps": _slip(side, avg_px, twap),
                "close_bps": _slip(side, avg_px, end_px),
                "participation": participation,
                "reversion_bps": reversion,
            }
        )
    return pd.DataFrame(rows)
