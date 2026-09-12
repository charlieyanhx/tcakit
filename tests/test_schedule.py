"""Almgren-Chriss closed forms: limits, identities, optimality, frontier monotonicity, baselines."""

import numpy as np
import pytest

from tcakit.schedule import (
    ACParams,
    almgren_chriss,
    cost_completion_ratio,
    efficient_frontier,
    expected_cost,
    pov,
    twap,
    variance,
    vwap,
)

P = ACParams(sigma=0.95, gamma=2.5e-7, eta=2.5e-6, epsilon=0.0625)   # Almgren-Chriss (2000) §4 example scale
X, T, N = 1_000_000.0, 5.0, 5


def test_lambda_zero_is_twap_and_tiny_lambda_is_numerically_stable():
    ac0 = almgren_chriss(X, T, N, P, lam=0.0)
    tw = twap(X, T, N, P)
    assert np.allclose(ac0.x, tw.x, atol=1e-9)
    ac_tiny = almgren_chriss(X, T, N, P, lam=1e-12)
    assert np.allclose(ac_tiny.x, tw.x, atol=1e-3 * X)      # expm1 form: no catastrophic cancellation


@pytest.mark.parametrize("lam", [1e-7, 2e-6, 1e-5])
def test_trajectory_shape_and_difference_equation(lam):
    s = almgren_chriss(X, T, 20, P, lam)
    assert s.x[0] == X and s.x[-1] == 0.0 and s.n.sum() == pytest.approx(X)
    assert (np.diff(s.x) <= 1e-9).all()                       # monotone liquidation
    # eq. (15): x_{j-1} - 2 x_j + x_{j+1} = kappa_tilde^2 tau^2 x_j  with kappa_tilde^2 tau^2 = 2 (cosh(kappa tau) - 1)
    tau = s.tau
    k2t2 = 2 * (np.cosh(s.notes["kappa"] * tau) - 1)
    lhs = s.x[:-2] - 2 * s.x[1:-1] + s.x[2:]
    assert np.allclose(lhs, k2t2 * s.x[1:-1], rtol=1e-9, atol=1e-6)


def test_twap_cost_and_variance_closed_forms():
    tw = twap(X, T, N, P)
    tau = T / N
    eta_t = P.eta - 0.5 * P.gamma * tau
    assert tw.expected_cost == pytest.approx(0.5 * P.gamma * X**2 + P.epsilon * X + eta_t * X**2 / T, rel=1e-12)
    assert tw.variance == pytest.approx(P.sigma**2 * tau * X**2 * (N - 1) * (2 * N - 1) / (6 * N), rel=1e-12)


def test_almgren_chriss_minimises_expected_cost_plus_lambda_variance():
    """Perturb the optimal path (endpoints fixed): the objective never decreases."""
    lam = 2e-6
    s = almgren_chriss(X, T, 20, P, lam)
    tau = s.tau
    base = expected_cost(s.x, X, tau, P) + lam * variance(s.x, tau, P)
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = rng.normal(0, 0.02 * X, len(s.x))
        d[0] = d[-1] = 0.0
        alt = s.x + d
        assert expected_cost(alt, X, tau, P) + lam * variance(alt, tau, P) >= base - 1e-6 * abs(base)


def test_efficient_frontier_is_monotone():
    lams = np.logspace(-8, -4, 12)
    fr = efficient_frontier(X, T, 20, P, lams)
    costs = [f.expected_cost for f in fr]
    vars_ = [f.variance for f in fr]
    assert all(np.diff(costs) >= -1e-9) and all(np.diff(vars_) <= 1e-9)
    assert fr[-1].notes["half_life"] < fr[0].notes["half_life"]


def test_cost_completion_ratio_two_thirds_rule():
    tw = twap(X, T, 1000, P)
    r = cost_completion_ratio(tw, exponent=0.5)
    assert 2 / 3 < r < 0.67
    assert cost_completion_ratio(tw, exponent=1.0) == pytest.approx((1000 + 1) / 2000, rel=1e-12)


def test_vwap_follows_profile_and_pov_caps_participation():
    prof = np.array([3.0, 1.0, 1.0, 1.0, 4.0])
    v = vwap(X, T, prof, P)
    assert np.allclose(v.n, X * prof / prof.sum()) and v.x[-1] == 0.0
    vol = np.array([200_000.0] * 5)
    p10 = pov(X, T, vol, 0.10, P)
    assert (p10.n[:-1] <= 0.10 * vol[:-1] + 1e-9).all() and p10.n.sum() == pytest.approx(X)
    assert p10.notes["forced"] == pytest.approx(X - 0.10 * vol.sum())   # 10% of 1M volume is only 100k shares
    p100 = pov(X, T, vol, 1.0, P)
    assert p100.notes["forced"] == 0.0
    with pytest.raises(ValueError):
        pov(X, T, vol, 0.0, P)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        almgren_chriss(X, T, N, P, lam=-1.0)
    with pytest.raises(ValueError):
        almgren_chriss(X, T, 1, ACParams(sigma=1, gamma=1.0, eta=0.1), lam=1e-6)   # eta_tilde <= 0
    with pytest.raises(ValueError):
        vwap(X, T, np.array([0.0, 0.0]), P)
