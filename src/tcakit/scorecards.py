"""Broker / venue / algo / tier scorecards with difficulty adjustment.

`scorecard` groups a per-order cost frame and reports n, mean, median, and a bootstrap CI
on the mean. `difficulty_adjusted` first regresses cost on difficulty covariates across
*all* orders, then reports each group's mean residual with a bootstrap CI — the number that
survives "your orders were just harder". With too few orders for the regression the
adjusted columns are NaN and `note` says why; nothing is silently dropped to make it fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BOOT_N = 2


def _boot_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator, q=(2.5, 97.5)) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < MIN_BOOT_N:
        return (np.nan, np.nan)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, q)
    return float(lo), float(hi)


def scorecard(
    df: pd.DataFrame,
    by: str | list[str],
    cost_col: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per group: n, n_valid, mean, median, ci_lo, ci_hi of `cost_col`."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby(by, dropna=False, sort=True):
        x = g[cost_col].to_numpy(dtype=float)
        lo, hi = _boot_ci(x, n_boot, rng)
        row = dict(zip([by] if isinstance(by, str) else by, key if isinstance(key, tuple) else (key,)))
        row.update({
            "n": len(g),
            "n_valid": int(np.isfinite(x).sum()),
            "mean": float(np.nanmean(x)) if np.isfinite(x).any() else np.nan,
            "median": float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan,
            "ci_lo": lo,
            "ci_hi": hi,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def difficulty_adjusted(
    df: pd.DataFrame,
    by: str | list[str],
    cost_col: str,
    covariates: list[str],
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Scorecard on OLS residuals of `cost_col ~ 1 + covariates` fit across all orders.

    Adds `resid_mean, resid_ci_lo, resid_ci_hi, r2_difficulty, note`. Needs at least
    `len(covariates) + 3` orders with finite cost and covariates; otherwise the residual
    columns are NaN and the raw scorecard is returned with a note.
    """
    raw = scorecard(df, by, cost_col, n_boot=n_boot, seed=seed)
    cols = [cost_col] + covariates
    d = df.dropna(subset=cols)
    need = len(covariates) + 3
    if len(d) < need:
        raw["resid_mean"] = np.nan
        raw["resid_ci_lo"] = np.nan
        raw["resid_ci_hi"] = np.nan
        raw["r2_difficulty"] = np.nan
        raw["note"] = f"difficulty regression needs >= {need} orders, have {len(d)}"
        return raw
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(dtype=float) for c in covariates])
    y = d[cost_col].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else np.nan
    d = d.assign(_resid=resid)
    adj = scorecard(d, by, "_resid", n_boot=n_boot, seed=seed).rename(columns={
        "mean": "resid_mean", "ci_lo": "resid_ci_lo", "ci_hi": "resid_ci_hi"
    }).drop(columns=["n", "n_valid", "median"])
    keys = [by] if isinstance(by, str) else by
    out = raw.merge(adj, on=keys, how="left")
    out["r2_difficulty"] = r2
    out["note"] = f"residuals of {cost_col} ~ 1 + {' + '.join(covariates)} (n={len(d)})"
    return out
