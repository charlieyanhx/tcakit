"""Adapter for the `TCALogger` JSONL written by the live combo executor.

One JSONL line per child order of a multi-leg option parent ("intent"). Prices are
IBKR-signed net combo prices: negative = credit received, positive = debit paid. The
canonical frames use unsigned prices and an explicit side, so the mapping is

    side = sign(arrival_mid)      (+1 we pay a debit / buy the combo, −1 we receive a credit)
    px   = |price|                (per share; `orders.multiplier` = 100 converts to $/contract)

and `side · (px − mid)` reproduces the logger's `implementation_shortfall_cents / 100`
exactly (checked in tests). Timestamps: the logger writes `ts` when the parent completes, so
`arrival_ts = ts − parent_total_latency_ms` and each child's fill time is
`arrival_ts + fill_time_ms`.

Known limits of this source, surfaced rather than hidden:

* no decision timestamp → `decision_ts == arrival_ts`, delay component is 0 by construction;
* no post-trade quote → `end_px` is the last child's arrival mid (opportunity cost of any
  unfilled quantity is therefore measured to the last observed mid, not a later one);
* no fee field → `fee = fee_per_contract_leg · n_legs · qty` from the argument (default 0);
  a child with empty `per_leg_fills` takes its siblings' leg count, else the file mode
  (`orders.n_legs_inferred = True`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import validate_fills, validate_orders

MULTIPLIER = 100.0


def read_jsonl(paths: Iterable[str | Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with Path(p).open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _side(signed_mid: float) -> int:
    if signed_mid == 0 or not np.isfinite(signed_mid):
        raise ValueError(f"cannot infer side from signed arrival mid {signed_mid!r}")
    return 1 if signed_mid > 0 else -1


def to_frames(
    records: list[dict],
    fee_per_contract_leg: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonical (orders, fills) from TCALogger records. Extra columns are kept:
    orders: urgency, leg (open/close), half_spread, n_legs, multiplier, parent_status,
            latency_ms; fills: tier_hit, tier_offset_cents, latency_ms, per_leg_fills.
    """
    if not records:
        raise ValueError("no TCALogger records")
    df = pd.DataFrame(records).rename(columns={"side": "leg"})  # open/close, not ±1
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["arrival_ts"] = df["ts"] - pd.to_timedelta(df["parent_total_latency_ms"], unit="ms")
    n_legs = df["per_leg_fills"].map(lambda d: len(d) if isinstance(d, dict) and d else np.nan)
    # a child logged with empty per_leg_fills inherits the leg count of its siblings
    df["n_legs"] = n_legs.groupby(df["intent_id"]).transform("max")
    # still unknown (single child, empty per_leg_fills): use the file-wide mode and flag it
    df["n_legs_inferred"] = df["n_legs"].isna()
    if df["n_legs"].isna().all():
        raise ValueError("no record carries per_leg_fills; cannot infer leg count for fees")
    df["n_legs"] = df["n_legs"].fillna(df["n_legs"].mode().iloc[0])
    df["side"] = df["arrival_mid"].map(_side)
    df = df.sort_values(["intent_id", "child_idx"])

    orders_rows = []
    for intent_id, g in df.groupby("intent_id", sort=False):
        first, last = g.iloc[0], g.iloc[-1]
        arrival_px = abs(float(first["arrival_mid"]))
        orders_rows.append({
            "order_id": intent_id,
            "symbol": _symbol(first),
            "side": int(first["side"]),
            "qty": float(first["parent_qty"]),
            "decision_ts": first["arrival_ts"],
            "arrival_ts": first["arrival_ts"],
            "end_ts": first["arrival_ts"] + pd.to_timedelta(first["parent_total_latency_ms"], unit="ms"),
            "decision_px": arrival_px,
            "arrival_px": arrival_px,
            "end_px": abs(float(last["arrival_mid_at_child"])),
            "leg": first["leg"],
            "urgency": first.get("urgency"),
            "half_spread": float(first["half_spread"]),
            "n_legs": first["n_legs"],
            "n_legs_inferred": bool(first["n_legs_inferred"]),
            "multiplier": MULTIPLIER,
            "parent_status": first.get("parent_status"),
            "latency_ms": float(first["parent_total_latency_ms"]),
            "n_children": len(g),
        })
    orders = pd.DataFrame(orders_rows)

    filled = df[df["filled_qty"].fillna(0) > 0]
    fills = pd.DataFrame({
        "fill_id": filled["intent_id"] + "/" + filled["child_idx"].astype(str),
        "order_id": filled["intent_id"],
        "ts": filled["arrival_ts"] + pd.to_timedelta(filled["fill_time_ms"], unit="ms"),
        "px": filled["avg_fill_price"].abs().astype(float),
        "qty": filled["filled_qty"].astype(float),
        "mid": filled["arrival_mid_at_child"].abs().astype(float),
        "fee": fee_per_contract_leg * filled["n_legs"] * filled["filled_qty"],
        "tier_hit": filled["tier_hit"],
        "tier_offset_cents": filled["tier_offset_cents"],
        "latency_ms": filled["fill_time_ms"].astype(float),
        "venue": "IBKR-SMART",
        "per_leg_fills": filled["per_leg_fills"],
    }).reset_index(drop=True)

    validate_orders(orders)
    validate_fills(fills, orders)
    return orders, fills


def _symbol(rec: pd.Series) -> str:
    legs = rec.get("per_leg_fills")
    if isinstance(legs, dict) and legs:
        return "SPY:" + "+".join(sorted(legs))
    meta = rec.get("meta") or {}
    return str(meta.get("pair", "COMBO"))


def load(paths: Iterable[str | Path], fee_per_contract_leg: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    return to_frames(read_jsonl(paths), fee_per_contract_leg=fee_per_contract_leg)
