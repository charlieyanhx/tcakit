"""A/B: balanced assignment; CUPED identity (1 − ρ²) and unbiasedness; clustered SE grows with
within-cluster correlation; power formula matches simulation; a known effect is recovered."""

import numpy as np
import pandas as pd
import pytest

from tcakit.experiments import ab_test, assign, cuped_adjust, min_detectable_effect, required_n


def _orders(n_days=40, per_day=10, symbols=("AAA", "BBB", "CCC"), effect=0.0, rho_cluster=0.0, seed=0):
    """Per-order arrival slippage = symbol-day shock (shared) + idiosyncratic + covariate signal + effect·1[B]."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in symbols:
        for day in range(n_days):
            shock = rng.normal(0, np.sqrt(rho_cluster) * 10)
            for _ in range(per_day):
                x = rng.normal(0, 1)                                  # pre-trade difficulty covariate
                eps = rng.normal(0, np.sqrt(1 - rho_cluster) * 10)
                rows.append({"symbol": s, "day": day, "difficulty": x, "base": shock + 6 * x + eps})
    df = pd.DataFrame(rows)
    df["arm"] = assign(df, seed=seed)
    df["arrival_bps"] = df["base"] + np.where(df["arm"] == "B", effect, 0.0)
    return df


def test_assignment_is_balanced_within_every_symbol_day():
    df = _orders(per_day=10)
    counts = df.groupby(["symbol", "day"])["arm"].value_counts().unstack()
    assert (counts["A"] == counts["B"]).all()
    odd = _orders(per_day=7)
    c = odd.groupby(["symbol", "day"])["arm"].value_counts().unstack()
    assert ((c["A"] - c["B"]).abs() == 1).all()


def test_cuped_variance_reduction_equals_one_minus_rho_squared_and_keeps_the_difference():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 50_000)
    y = 3 * x + rng.normal(0, 4, 50_000)
    adj, theta, vr = cuped_adjust(y, x)
    rho2 = np.corrcoef(x, y)[0, 1] ** 2
    assert theta == pytest.approx(3.0, abs=0.05)
    assert vr == pytest.approx(rho2, abs=1e-6)                          # identity: 1 − var_adj/var = ρ²
    assert adj.mean() == pytest.approx(y.mean(), abs=1e-9)              # centred covariate: mean unchanged


def test_cuped_recovers_a_known_effect_with_a_tighter_interval():
    df = _orders(effect=-1.5, seed=3)
    raw = ab_test(df)
    cu = ab_test(df, covariate_col="difficulty")
    assert cu.effect == pytest.approx(-1.5, abs=3 * cu.se)
    assert cu.se < raw.se and cu.variance_reduction > 0.2
    assert cu.ci_lo < -1.5 < cu.ci_hi


def test_clustered_se_exceeds_naive_when_symbol_days_share_shocks():
    df = _orders(rho_cluster=0.5, seed=5)
    r = ab_test(df)
    # within-symbol-day randomisation removes the shared shock from the *difference*; the clustered SE
    # is then close to the naive one — the inflation appears when the arms sit on different days
    assert r.n_clusters == 120 and r.se > 0
    unbalanced = df.copy()
    unbalanced["arm"] = np.where(unbalanced["day"] % 2 == 0, "A", "B")   # arms on different days: shocks bite
    ru = ab_test(unbalanced)
    naive_u = np.sqrt(unbalanced[unbalanced.arm == "A"].arrival_bps.var() / (unbalanced.arm == "A").sum()
                      + unbalanced[unbalanced.arm == "B"].arrival_bps.var() / (unbalanced.arm == "B").sum())
    assert ru.se > 1.5 * naive_u                                        # the naive SE would overstate certainty


def test_null_effect_has_nominal_false_positive_rate():
    rejects = sum(ab_test(_orders(seed=s, n_days=20)).p_value < 0.05 for s in range(60))
    assert rejects <= 8                                                 # ≤ ~13% on 60 draws at nominal 5%


def test_power_formula_matches_simulation():
    sigma, effect, n = 10.0, 2.0, required_n(2.0, 10.0)
    assert n == 393                                                     # 2·(2.80·10/2)² rounded up
    assert min_detectable_effect(n, sigma) == pytest.approx(effect, rel=0.01)
    rng = np.random.default_rng(7)
    hits = 0
    for _ in range(300):
        a, b = rng.normal(0, sigma, n), rng.normal(effect, sigma, n)
        t = (b.mean() - a.mean()) / np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
        hits += abs(t) > 1.96
    assert 0.72 < hits / 300 < 0.88                                     # nominal power 0.80


def test_bad_arms_raise():
    df = _orders(n_days=2)
    df["arm"] = "A"
    with pytest.raises(ValueError):
        ab_test(df)
