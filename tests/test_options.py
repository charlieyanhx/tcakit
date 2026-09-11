import numpy as np
import pandas as pd
import pytest

from tcakit.options import contract_costs, net_mid


def test_net_mid_and_half_spread():
    legs = pd.DataFrame({"bid": [8.10, 7.10], "ask": [8.30, 7.20], "ratio": [-1, 1]})
    mid, half = net_mid(legs)
    assert mid == pytest.approx(-8.20 + 7.15)
    assert half == pytest.approx(0.10 + 0.05)


def test_net_mid_nan_propagates():
    legs = pd.DataFrame({"bid": [8.10, np.nan], "ask": [8.30, 7.20], "ratio": [-1, 1]})
    assert all(np.isnan(v) for v in net_mid(legs))


def test_contract_cost_units():
    orders = pd.DataFrame({"order_id": ["a"], "multiplier": [100.0], "half_spread": [0.07],
                           "delta": [0.20], "spot": [700.0]})
    sf = pd.DataFrame({"order_id": ["a"], "execution": [0.07], "filled_qty": [1.0]})
    c = contract_costs(sf, orders).set_index("order_id").loc["a"]
    assert c["usd_per_contract"] == pytest.approx(7.0)
    assert c["frac_half_spread"] == pytest.approx(1.0)
    assert c["bps_underlying"] == pytest.approx(7.0 / (0.2 * 700 * 100) * 1e4)


def test_bps_underlying_nan_without_delta():
    orders = pd.DataFrame({"order_id": ["a"], "multiplier": [100.0], "half_spread": [0.07]})
    sf = pd.DataFrame({"order_id": ["a"], "execution": [0.07], "filled_qty": [1.0]})
    assert np.isnan(contract_costs(sf, orders)["bps_underlying"].iloc[0])


def test_unfilled_order_is_nan_not_zero():
    orders = pd.DataFrame({"order_id": ["a"], "multiplier": [100.0], "half_spread": [0.07]})
    sf = pd.DataFrame({"order_id": ["a"], "execution": [0.0], "filled_qty": [0.0]})
    assert np.isnan(contract_costs(sf, orders)["usd_per_contract"].iloc[0])
