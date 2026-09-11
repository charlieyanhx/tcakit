"""Canonical frames and validation.

Three long-format DataFrames — orders, fills, market — see docs/DESIGN.md. Validation is
strict on the things that silently corrupt TCA numbers: naive timestamps, non-±1 sides,
fills that reference no parent order, negative quantities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ORDER_COLUMNS = [
    "order_id", "symbol", "side", "qty",
    "decision_ts", "arrival_ts", "end_ts", "decision_px", "arrival_px",
]
FILL_COLUMNS = ["fill_id", "order_id", "ts", "px", "qty"]
MARKET_COLUMNS = ["symbol", "ts", "bid", "ask", "last", "volume"]


class SchemaError(ValueError):
    """Raised when a frame does not satisfy the canonical contract."""


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"{name} frame is missing required columns: {missing}")


def _require_tz_aware(df: pd.DataFrame, cols: list[str], name: str) -> None:
    for c in cols:
        if not isinstance(df[c].dtype, pd.DatetimeTZDtype):
            raise SchemaError(f"{name}.{c} must be tz-aware datetime64 (got {df[c].dtype})")


def validate_orders(orders: pd.DataFrame) -> pd.DataFrame:
    _require_columns(orders, ORDER_COLUMNS, "orders")
    _require_tz_aware(orders, ["decision_ts", "arrival_ts", "end_ts"], "orders")
    if not orders["side"].isin([1, -1]).all():
        raise SchemaError("orders.side must be +1 (buy) or -1 (sell)")
    if (orders["qty"] <= 0).any():
        raise SchemaError("orders.qty must be positive")
    if orders["order_id"].duplicated().any():
        raise SchemaError("orders.order_id must be unique")
    return orders


def validate_fills(fills: pd.DataFrame, orders: pd.DataFrame | None = None) -> pd.DataFrame:
    _require_columns(fills, FILL_COLUMNS, "fills")
    if len(fills):
        _require_tz_aware(fills, ["ts"], "fills")
        if (fills["qty"] <= 0).any():
            raise SchemaError("fills.qty must be positive")
    if orders is not None:
        unknown = set(fills["order_id"]) - set(orders["order_id"])
        if unknown:
            raise SchemaError(f"fills reference unknown order_id values: {sorted(unknown)[:5]}")
    return fills


def validate_market(market: pd.DataFrame) -> pd.DataFrame:
    _require_columns(market, MARKET_COLUMNS, "market")
    _require_tz_aware(market, ["ts"], "market")
    if (market["ask"] < market["bid"]).any():
        raise SchemaError("market has crossed quotes (ask < bid)")
    return market


def market_mid(market: pd.DataFrame) -> pd.Series:
    return (market["bid"] + market["ask"]) / 2.0


def asof_mid(market: pd.DataFrame, symbol: str, ts: pd.Series) -> pd.Series:
    """Prevailing mid for each timestamp (backward asof). NaN before the first quote."""
    m = market.loc[market["symbol"] == symbol, ["ts"]].assign(mid=market_mid(market)).sort_values("ts")
    if m.empty:
        return pd.Series(np.nan, index=ts.index, dtype=float)
    probe = pd.DataFrame({"ts": ts.to_numpy()}, index=ts.index)
    order = probe["ts"].argsort(kind="stable")
    merged = pd.merge_asof(
        probe.iloc[order].reset_index(names="__idx"), m, on="ts", direction="backward"
    ).set_index("__idx")
    return merged["mid"].reindex(ts.index)
