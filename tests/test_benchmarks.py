import numpy as np
import pytest

from tcakit.benchmarks import benchmark_slippage, interval_twap, interval_vwap

BPS = 1e4


def test_interval_vwap_and_twap_hand_computed(market_abc, buy_order):
    orders, fills = buy_order
    t0, t1 = fills["ts"].min(), fills["ts"].max()
    bars = market_abc[(market_abc["ts"] >= t0) & (market_abc["ts"] <= t1)]
    # bars i=5..15, vol 1000 each except i=12 -> 10000
    assert len(bars) == 11
    assert interval_twap(market_abc, t0, t1) == pytest.approx(100.10)
    expected_vwap = (1_101_100 + 9000 * 100.12) / 20_000
    assert interval_vwap(market_abc, t0, t1) == pytest.approx(expected_vwap)


def test_benchmark_slippage_columns_and_values(market_abc, buy_order):
    orders, fills = buy_order
    row = benchmark_slippage(orders, fills, market_abc, reversion_horizon="5min").iloc[0]
    assert row["avg_px"] == pytest.approx(100.25)
    assert row["arrival_bps"] == pytest.approx((100.25 - 100.10) / 100.10 * BPS)
    vwap = (1_101_100 + 9000 * 100.12) / 20_000
    assert row["vwap_bps"] == pytest.approx((100.25 - vwap) / vwap * BPS)
    assert row["twap_bps"] == pytest.approx((100.25 - 100.10) / 100.10 * BPS)
    assert row["close_bps"] == pytest.approx((100.25 - 100.50) / 100.50 * BPS)
    assert row["participation"] == pytest.approx(800 / 20_000)
    # mid 5 min after last fill (14:50, i=20) = 100.20
    assert row["reversion_bps"] == pytest.approx((100.25 - 100.20) / 100.25 * BPS)


def test_sell_side_flips_benchmark_sign(market_abc, buy_order):
    orders, fills = buy_order
    buy = benchmark_slippage(orders, fills, market_abc).iloc[0]
    sell = benchmark_slippage(orders.assign(side=-1), fills, market_abc).iloc[0]
    for c in ["arrival_bps", "vwap_bps", "twap_bps", "close_bps", "reversion_bps"]:
        assert sell[c] == pytest.approx(-buy[c])
    assert sell["participation"] == pytest.approx(buy["participation"])


def test_unfilled_order_gives_nan_slippage(market_abc, buy_order):
    orders, fills = buy_order
    row = benchmark_slippage(orders, fills.iloc[0:0], market_abc).iloc[0]
    assert np.isnan(row["avg_px"]) and np.isnan(row["arrival_bps"]) and np.isnan(row["vwap_bps"])
    assert row["participation"] == 0.0


def test_reversion_nan_when_horizon_beyond_data(market_abc, buy_order):
    orders, fills = buy_order
    row = benchmark_slippage(orders, fills, market_abc, reversion_horizon="3h").iloc[0]
    assert np.isnan(row["reversion_bps"])
