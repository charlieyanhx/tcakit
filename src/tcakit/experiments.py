"""A/B experiments on execution: randomise parent orders between two algos / brokers / venues,
estimate the cost difference with the noise it actually has, and size the experiment first.

Three things go wrong in practice and each is handled explicitly:

* **Interference** — two arms trading the same name on the same day share the same market
  move, so per-order observations are not independent. Randomisation is stratified by
  symbol-day, and standard errors are clustered by symbol-day (Liang-Zeger / CR0).
* **Noise** — arrival slippage has a standard deviation many times the effect a desk is
  looking for. CUPED (Deng, Xu, Kohavi, Walker 2013) subtracts the part explained by a
  pre-treatment covariate (here a pre-trade cost estimate or the order's difficulty), which
  cuts the variance by 1 − ρ² without biasing the difference.
* **Power** — the minimum detectable effect for n orders per arm at a given σ, or the n
  needed for an effect, so the desk knows before running whether the answer can exist.

Units: `cost` in bps of arrival notional (or any per-order cost); positive = cost. The
treatment effect is `mean(cost | B) − mean(cost | A)`; negative = B is cheaper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ABResult:
    n_a: int
    n_b: int
    effect: float            # mean_B − mean_A of the (CUPED-adjusted) cost
    se: float                # cluster-robust standard error of the effect
    ci_lo: float
    ci_hi: float
    p_value: float
    n_clusters: int
    cuped_theta: float | None
    variance_reduction: float | None   # 1 − var(adjusted)/var(raw) of the pooled cost
    notes: dict

    def summary(self) -> str:
        cu = f", CUPED θ={self.cuped_theta:.3g} (−{self.variance_reduction:.0%} var)" if self.cuped_theta is not None else ""
        return (f"B − A = {self.effect:+.3g} ± {1.96 * self.se:.3g} (95%), p={self.p_value:.3g}, "
                f"n={self.n_a}/{self.n_b}, {self.n_clusters} symbol-day clusters{cu}")


def assign(orders: pd.DataFrame, seed: int = 0, strata: tuple[str, ...] = ("symbol", "day")) -> pd.Series:
    """Randomise parent orders to 'A'/'B', balanced within each stratum (symbol-day by
    default) so the two arms see the same market on the same names."""
    rng = np.random.default_rng(seed)
    arm = pd.Series(index=orders.index, dtype=object)
    for idx in orders.groupby(list(strata)).groups.values():
        idx = list(idx)
        k = len(idx)
        labels = np.array(["A", "B"] * (k // 2) + (["A"] if k % 2 and rng.random() < 0.5 else ["B"] if k % 2 else []))
        rng.shuffle(labels)
        arm.loc[idx] = labels
    return arm


def cuped_adjust(cost: np.ndarray, covariate: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Y_adj = Y − θ (X − mean X) with θ = cov(Y, X)/var(X); returns (Y_adj, θ, 1 − var(Y_adj)/var(Y)).
    Same θ for both arms, so the difference in means is unbiased when X is pre-treatment."""
    x = covariate - covariate.mean()
    vx = float((x**2).mean())
    if vx == 0:
        return cost.copy(), 0.0, 0.0
    theta = float((x * (cost - cost.mean())).mean() / vx)
    adj = cost - theta * x
    return adj, theta, 1.0 - float(adj.var()) / float(cost.var()) if cost.var() > 0 else 0.0


def _cluster_se(y: np.ndarray, d: np.ndarray, clusters: np.ndarray) -> tuple[float, int]:
    """CR0 cluster-robust SE of the OLS slope in y = a + b·d (d ∈ {0,1}); returns (se, G)."""
    X = np.column_stack([np.ones(len(d)), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    meat = np.zeros((2, 2))
    labels = pd.factorize(clusters)[0]
    for g in np.unique(labels):
        Xg, ug = X[labels == g], resid[labels == g]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    return float(np.sqrt(V[1, 1])), len(np.unique(labels))


def ab_test(
    df: pd.DataFrame,
    arm_col: str = "arm",
    cost_col: str = "arrival_bps",
    cluster_cols: tuple[str, ...] = ("symbol", "day"),
    covariate_col: str | None = None,
) -> ABResult:
    """Difference in mean cost (B − A) with symbol-day clustered SE; CUPED if `covariate_col`."""
    d = df.dropna(subset=[cost_col] + ([covariate_col] if covariate_col else [])).copy()
    if set(d[arm_col].unique()) - {"A", "B"} or d[arm_col].nunique() < 2:
        raise ValueError("arm_col must contain both 'A' and 'B' and nothing else")
    y = d[cost_col].to_numpy(dtype=float)
    theta = vr = None
    if covariate_col:
        y, theta, vr = cuped_adjust(y, d[covariate_col].to_numpy(dtype=float))
    treat = (d[arm_col] == "B").to_numpy(dtype=float)
    effect = float(y[treat == 1].mean() - y[treat == 0].mean())
    clusters = d[list(cluster_cols)].astype(str).agg("|".join, axis=1).to_numpy()
    se, G = _cluster_se(y, treat, clusters)
    df_t = max(G - 1, 1)
    tcrit = stats.t.ppf(0.975, df_t)
    p = float(2 * stats.t.sf(abs(effect / se), df_t)) if se > 0 else float("nan")
    return ABResult(int((treat == 0).sum()), int((treat == 1).sum()), effect, se, effect - tcrit * se, effect + tcrit * se,
                    p, G, theta, vr, {"cost": cost_col, "clusters": list(cluster_cols), "t_df": df_t})


def min_detectable_effect(n_per_arm: int, sigma: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Two-sample, equal n, known σ: MDE = (z_{1−α/2} + z_power) · σ · sqrt(2/n)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sigma * np.sqrt(2.0 / n_per_arm))


def required_n(effect: float, sigma: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Orders per arm to detect `effect` with the given σ (before any CUPED reduction)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2.0 * (z * sigma / effect) ** 2))
