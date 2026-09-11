import pandas as pd
import pytest

from tcakit.schema import SchemaError, asof_mid, validate_fills, validate_market, validate_orders


def test_validate_orders_accepts_fixture(buy_order):
    orders, _ = buy_order
    validate_orders(orders)


def test_validate_orders_rejects_missing_column(buy_order):
    orders, _ = buy_order
    with pytest.raises(SchemaError, match="decision_px"):
        validate_orders(orders.drop(columns=["decision_px"]))


def test_validate_orders_rejects_bad_side(buy_order):
    orders, _ = buy_order
    with pytest.raises(SchemaError, match="side"):
        validate_orders(orders.assign(side=2))


def test_validate_rejects_naive_timestamps(buy_order):
    orders, _ = buy_order
    naive = orders.assign(decision_ts=orders["decision_ts"].dt.tz_localize(None))
    with pytest.raises(SchemaError, match="tz"):
        validate_orders(naive)


def test_validate_fills_rejects_unknown_order(buy_order):
    orders, fills = buy_order
    with pytest.raises(SchemaError, match="order_id"):
        validate_fills(fills.assign(order_id="nope"), orders)


def test_validate_market_and_asof_mid(market_abc):
    validate_market(market_abc)
    t = pd.Timestamp("2026-01-05 14:35:30", tz="UTC")  # between 14:35 and 14:36 bars
    assert asof_mid(market_abc, "ABC", pd.Series([t])).iloc[0] == pytest.approx(100.05)
    before = pd.Timestamp("2026-01-05 14:00:00", tz="UTC")
    assert pd.isna(asof_mid(market_abc, "ABC", pd.Series([before])).iloc[0])
