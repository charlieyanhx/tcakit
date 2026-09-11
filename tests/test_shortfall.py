import numpy as np
import pandas as pd
import pytest

from tcakit.shortfall import COMPONENTS, implementation_shortfall


def test_components_match_hand_computation(buy_order):
    orders, fills = buy_order
    out = implementation_shortfall(orders, fills)
    row = out.set_index("order_id").loc["o1"]
    assert row["delay"] == pytest.approx(80.0)
    assert row["spread"] == pytest.approx(40.0)
    assert row["timing"] == pytest.approx(80.0)
    assert row["opportunity"] == pytest.approx(100.0)
    assert row["fees"] == pytest.approx(5.0)
    assert row["total"] == pytest.approx(305.0)
    assert row["total_bps"] == pytest.approx(30.5)
    assert row["filled_qty"] == pytest.approx(800.0)
    assert row["avg_px"] == pytest.approx(100.25)


def test_identity_holds_exactly(buy_order):
    orders, fills = buy_order
    row = implementation_shortfall(orders, fills).iloc[0]
    assert abs(sum(row[c] for c in COMPONENTS) - row["total"]) < 1e-9


def test_sell_order_flips_sign(buy_order):
    orders, fills = buy_order
    orders = orders.assign(side=-1)
    row = implementation_shortfall(orders, fills).iloc[0]
    # selling at 100.25 avg when decision was 100.00 is a *gain* on the executed part,
    # but the unfilled 200 missed a rise to 100.50 → opportunity is a gain too for a seller.
    assert row["delay"] == pytest.approx(-80.0)
    assert row["spread"] == pytest.approx(-40.0)
    assert row["timing"] == pytest.approx(-80.0)
    assert row["opportunity"] == pytest.approx(-100.0)
    assert row["fees"] == pytest.approx(5.0)
    assert row["total"] == pytest.approx(-295.0)


def test_no_decision_arrival_gap_gives_zero_delay(buy_order):
    orders, fills = buy_order
    orders = orders.assign(arrival_px=orders["decision_px"])
    row = implementation_shortfall(orders, fills).iloc[0]
    assert row["delay"] == 0.0


def test_unfilled_order_is_pure_opportunity_and_fees(buy_order):
    orders, fills = buy_order
    row = implementation_shortfall(orders, fills.iloc[0:0]).iloc[0]
    assert row["filled_qty"] == 0.0
    assert np.isnan(row["avg_px"])
    assert row["delay"] == 0.0 and row["spread"] == 0.0 and row["timing"] == 0.0
    assert row["opportunity"] == pytest.approx(1000.0 * 0.5)
    assert row["total"] == pytest.approx(500.0)


def test_missing_mid_yields_nan_spread_not_zero(buy_order):
    orders, fills = buy_order
    fills = fills.drop(columns=["mid"])
    row = implementation_shortfall(orders, fills).iloc[0]
    assert np.isnan(row["spread"]) and np.isnan(row["timing"])
    # but the executed cost (spread + timing) is still known as a lump
    assert row["execution"] == pytest.approx(120.0)
    assert row["total"] == pytest.approx(305.0)


def test_mid_looked_up_from_market_when_absent(buy_order, market_abc):
    orders, fills = buy_order
    fills = fills.drop(columns=["mid"])
    row = implementation_shortfall(orders, fills, market=market_abc).iloc[0]
    # market mid at 14:35 = 100.05, at 14:45 = 100.15
    spread = 400 * (100.20 - 100.05) + 400 * (100.30 - 100.15)
    assert row["spread"] == pytest.approx(spread)
    assert row["spread"] + row["timing"] == pytest.approx(120.0)


def test_multiple_orders_are_independent(buy_order):
    orders, fills = buy_order
    o2 = orders.assign(order_id="o2", side=-1)
    f2 = fills.assign(order_id="o2", fill_id=["g1", "g2"])
    out = implementation_shortfall(pd.concat([orders, o2]), pd.concat([fills, f2]))
    assert list(out["order_id"]) == ["o1", "o2"]
    assert out.set_index("order_id").loc["o2", "total"] == pytest.approx(-295.0)
