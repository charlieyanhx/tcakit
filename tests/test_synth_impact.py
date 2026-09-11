"""Parameter-recovery tests: the estimator must get back what the generator put in."""

import numpy as np
import pytest

from tcakit.impact import fit_sqrt_law, market_stats, realized_impact
from tcakit.schema import validate_fills, validate_market, validate_orders
from tcakit.shortfall import implementation_shortfall
from tcakit.synth import simulate


@pytest.fixture(scope="module")
def sim():
    return simulate(n_days=150, orders_per_day=8, seed=7, sigma_daily=0.02, adv=1_000_000,
                    spread_bps=0.0, sqrt_law_y=1.0, p_cross=0.0)


def test_frames_validate_and_sizes(sim):
    validate_orders(sim.orders)
    validate_fills(sim.fills, sim.orders)
    validate_market(sim.market)
    assert 0.95 * 150 * 8 <= len(sim.orders) <= 150 * 8
    assert sim.fills["order_id"].nunique() == len(sim.orders)
    assert set(sim.params) >= {"sigma_daily", "adv", "sqrt_law_y", "spread_bps"}


def test_orders_do_not_overlap_in_time(sim):
    o = sim.orders.sort_values("arrival_ts")
    assert (o["arrival_ts"].to_numpy()[1:] >= o["end_ts"].to_numpy()[:-1]).all()


def test_market_stats_recover_generator_inputs_without_impact():
    s = simulate(n_days=60, orders_per_day=8, seed=5, sqrt_law_y=0.0)
    st = market_stats(s.market).set_index("symbol").loc["SYN"]
    assert st["adv"] == pytest.approx(s.params["adv"], rel=0.02)
    assert st["sigma_daily"] == pytest.approx(s.params["sigma_daily"], rel=0.10)


def test_own_impact_inflates_measured_vol(sim):
    """Permanent impact adds variance to mid; measured σ exceeds the generator's σ.
    Documented in synth.py — a real-world caveat for impact regressions."""
    st = market_stats(sim.market).set_index("symbol").loc["SYN"]
    assert st["sigma_daily"] > sim.params["sigma_daily"]
    assert st["sigma_daily"] < 1.5 * sim.params["sigma_daily"]


def _true_stats(sim):
    import pandas as pd
    return pd.DataFrame({"symbol": ["SYN"], "adv": [sim.params["adv"]],
                         "sigma_daily": [sim.params["sigma_daily"]]})


def test_sqrt_law_recovers_y_on_completion_impact(sim):
    df = realized_impact(sim.orders, sim.fills, sim.market, stats=_true_stats(sim))
    res = fit_sqrt_law(df, target="impact_end", holdout=0.3, seed=1)
    assert res.params["Y"] == pytest.approx(1.0, abs=0.12)
    assert res.param_ci["Y"][0] < 1.0 < res.param_ci["Y"][1]
    assert res.n_in + res.n_out == len(df)
    assert res.rmse_out > 0 and np.isfinite(res.r2_in)


def test_average_cost_is_two_thirds_of_completion_impact(sim):
    """Linear schedule under the square-root law: mean paid impact = 2/3 of peak (Bouchaud)."""
    df = realized_impact(sim.orders, sim.fills, sim.market, stats=_true_stats(sim))
    y_end = fit_sqrt_law(df, target="impact_end", holdout=0.0).params["Y"]
    y_cost = fit_sqrt_law(df, target="cost", holdout=0.0).params["Y"]
    assert y_cost / y_end == pytest.approx(2 / 3, abs=0.06)


def test_spread_shows_up_in_shortfall_spread_component():
    s = simulate(n_days=5, orders_per_day=10, seed=3, spread_bps=10.0, p_cross=1.0, sqrt_law_y=0.0)
    is_ = implementation_shortfall(s.orders, s.fills, s.market)
    # every fill crossed a 10 bps-wide spread → paid half-spread = 5 bps of fill notional
    paid = is_["spread"].sum() / (s.fills["px"] * s.fills["qty"]).sum() * 1e4
    assert paid == pytest.approx(5.0, abs=0.05)


def test_zero_impact_zero_spread_gives_zero_mean_cost():
    s = simulate(n_days=20, orders_per_day=20, seed=11, spread_bps=0.0, p_cross=0.0, sqrt_law_y=0.0)
    df = realized_impact(s.orders, s.fills, s.market)
    assert abs(df["cost"].mean()) < 3 * df["cost"].std() / np.sqrt(len(df))
