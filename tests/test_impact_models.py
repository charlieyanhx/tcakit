"""Almgren-2005 and I-star recover the generator's parameters; hold-out is by day."""

import numpy as np
import pandas as pd
import pytest

from tcakit.impact import fit_almgren2005, fit_istar, realized_impact
from tcakit.synth import simulate


def _impact_frame(noise_scale, seed=7, n_days=120):
    s = simulate(n_days=n_days, orders_per_day=8, seed=seed, sigma_daily=0.02, adv=1_000_000,
                 spread_bps=0.0, sqrt_law_y=1.0, p_cross=0.0, noise_scale=noise_scale)
    st = pd.DataFrame({"symbol": ["SYN"], "adv": [s.params["adv"]], "sigma_daily": [s.params["sigma_daily"]]})
    df = realized_impact(s.orders, s.fills, s.market, stats=st)
    return df.merge(s.orders[["order_id", "arrival_ts"]], on="order_id")


def test_almgren_permanent_exact_without_noise():
    r = fit_almgren2005(_impact_frame(0.0), n_boot=10)["permanent"]
    assert r.params["gamma"] == pytest.approx(1.0, abs=1e-6)
    assert r.params["alpha"] == pytest.approx(0.5, abs=1e-6)
    assert r.rmse_out < 1e-10


def test_istar_exact_without_noise():
    r = fit_istar(_impact_frame(0.0), target="impact_end", n_boot=10)
    assert r.params["a1"] == pytest.approx(1.0, abs=1e-4)
    assert r.params["a2"] == pytest.approx(0.5, abs=1e-4)


def test_almgren_permanent_ci_covers_truth_with_noise():
    r = fit_almgren2005(_impact_frame(1.0, n_days=200), n_boot=100)["permanent"]
    lo, hi = r.param_ci["alpha"]
    assert lo < 0.5 < hi
    assert r.rmse_out > 0 and np.isfinite(r.r2_in)


def test_holdout_is_by_day():
    df = _impact_frame(0.0)
    r = fit_istar(df, target="impact_end", holdout=0.3, n_boot=5)
    assert r.n_in + r.n_out == len(df)
    days = df["arrival_ts"].dt.normalize().nunique()
    assert r.n_out > 0 and r.notes["holdout"] == "by day"
    # 30% of days held out, so n_out/n is near 0.3 but not exactly (day sizes vary)
    assert 0.2 < r.n_out / len(df) < 0.4 and days > 10
