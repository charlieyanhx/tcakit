import numpy as np
import pandas as pd

from tcakit.scorecards import difficulty_adjusted, scorecard


def _df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    diff = rng.uniform(0.5, 2.0, n)
    group = np.where(np.arange(n) % 2 == 0, "A", "B")
    # B is cheaper by 1.0 after difficulty; raw means are confounded by giving B harder orders
    diff = np.where(group == "B", diff + 1.0, diff)
    cost = 3.0 * diff + np.where(group == "B", -1.0, 0.0) + rng.normal(0, 0.1, n)
    return pd.DataFrame({"group": group, "difficulty": diff, "cost": cost})


def test_scorecard_basic():
    sc = scorecard(_df(), "group", "cost").set_index("group")
    assert sc.loc["A", "n"] == 30 and sc.loc["B", "n"] == 30
    assert sc.loc["A", "ci_lo"] <= sc.loc["A", "mean"] <= sc.loc["A", "ci_hi"]


def test_difficulty_adjustment_flips_the_ranking():
    df = _df()
    raw = scorecard(df, "group", "cost").set_index("group")
    adj = difficulty_adjusted(df, "group", "cost", ["difficulty"]).set_index("group")
    assert raw.loc["B", "mean"] > raw.loc["A", "mean"]          # B looks worse raw
    assert adj.loc["B", "resid_mean"] < adj.loc["A", "resid_mean"]  # B is better adjusted
    assert adj.loc["B", "resid_ci_hi"] < adj.loc["A", "resid_ci_lo"]
    assert adj["r2_difficulty"].iloc[0] > 0.9


def test_small_n_gives_nan_and_note():
    df = _df(n=3)
    adj = difficulty_adjusted(df, "group", "cost", ["difficulty"])
    assert adj["resid_mean"].isna().all()
    assert "needs >=" in adj["note"].iloc[0]


def test_single_order_group_has_nan_ci():
    df = pd.DataFrame({"group": ["A"], "cost": [1.0]})
    sc = scorecard(df, "group", "cost")
    assert np.isnan(sc["ci_lo"].iloc[0]) and sc["mean"].iloc[0] == 1.0
