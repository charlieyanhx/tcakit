"""Shared fixtures: a tiny hand-built order with known costs."""

import pandas as pd
import pytest

UTC = "UTC"


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


@pytest.fixture
def buy_order():
    """One buy parent: Q=1000, decision 100.00, arrival 100.10, end mid 100.50.

    Fills: 400 @ 100.20 (mid 100.15), 400 @ 100.30 (mid 100.25). 200 unfilled. Fees 5.00.

    Hand computation (cash, +ve = cost):
      Q_x = 800, P̄ = 100.25
      delay       = 800 * (100.10 - 100.00) = 80.00
      spread      = 400*(100.20-100.15) + 400*(100.30-100.25) = 20 + 20 = 40.00
      timing      = 400*(100.15-100.10) + 400*(100.25-100.10) = 20 + 60 = 80.00
      opportunity = 200 * (100.50 - 100.00) = 100.00
      fees        = 5.00
      total       = 800*(100.25-100.00) + 100 + 5 = 200 + 100 + 5 = 305.00
    bps of decision notional (1000*100 = 100_000): total = 30.5 bps
    """
    orders = pd.DataFrame(
        {
            "order_id": ["o1"],
            "symbol": ["ABC"],
            "side": [1],
            "qty": [1000.0],
            "decision_ts": [ts("2026-01-05 14:30:00")],
            "arrival_ts": [ts("2026-01-05 14:31:00")],
            "end_ts": [ts("2026-01-05 15:00:00")],
            "decision_px": [100.00],
            "arrival_px": [100.10],
            "end_px": [100.50],
        }
    )
    fills = pd.DataFrame(
        {
            "fill_id": ["f1", "f2"],
            "order_id": ["o1", "o1"],
            "ts": [ts("2026-01-05 14:35:00"), ts("2026-01-05 14:45:00")],
            "px": [100.20, 100.30],
            "qty": [400.0, 400.0],
            "mid": [100.15, 100.25],
            "venue": ["X", "Y"],
            "fee": [2.5, 2.5],
        }
    )
    return orders, fills


@pytest.fixture
def market_abc():
    """Minute bars 14:30–15:00 for ABC; mid = 100 + 0.01*minute_index, volume 1000/min except
    one heavy minute so interval VWAP is not equal to TWAP."""
    idx = pd.date_range(ts("2026-01-05 14:30:00"), ts("2026-01-05 15:00:00"), freq="1min")
    mid = 100.0 + 0.01 * pd.RangeIndex(len(idx)).to_numpy()
    vol = pd.Series(1000.0, index=idx)
    vol.iloc[12] = 10_000.0  # 14:42
    return pd.DataFrame(
        {
            "symbol": "ABC",
            "ts": idx,
            "bid": mid - 0.02,
            "ask": mid + 0.02,
            "last": mid,
            "volume": vol.to_numpy(),
        }
    )
