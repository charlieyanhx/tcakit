"""Market-impact measurement and model calibration with held-out error.

`realized_impact` turns orders+fills+market into one row per parent order with the
explanatory variables (`q_adv`, `sigma`, `participation`, `duration_min`) and two targets:

* `cost`       — `side · (avg_px − arrival_px) / arrival_px`, what was actually paid
                 (includes spread and timing);
* `impact_end` — `side · (mid_last_fill − arrival_px) / arrival_px`, the price move over the
                 execution (peak/permanent impact proxy, spread-free).

`fit_sqrt_law` fits `target = Y · sigma · sqrt(q_adv)` by OLS through the origin on a
random in-sample split and reports the out-of-sample RMSE alongside in-sample R². A fit
with `holdout=0` reports `rmse_out = nan` and `n_out = 0` — it is not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .benchmarks import interval_volume
from .schema import asof_mid, market_mid, validate_fills, validate_market, validate_orders
from .synth import MINUTES_PER_DAY


@dataclass
class FitResult:
    model: str
    target: str
    params: dict[str, float]
    param_ci: dict[str, tuple[float, float]]
    r2_in: float
    rmse_in: float
    rmse_out: float
    n_in: int
    n_out: int
    notes: dict = field(default_factory=dict)

    def summary(self) -> str:
        p = ", ".join(f"{k}={v:.4g} [{lo:.4g}, {hi:.4g}]" for k, (v, (lo, hi)) in
                      ((k, (self.params[k], self.param_ci[k])) for k in self.params))
        return (f"{self.model} on {self.target}: {p} | R²_in={self.r2_in:.3f} "
                f"RMSE_in={self.rmse_in:.3g} RMSE_out={self.rmse_out:.3g} "
                f"(n_in={self.n_in}, n_out={self.n_out})")


def market_stats(market: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol ADV (mean daily volume) and daily vol (std of 1-bar log-mid returns,
    scaled by sqrt(bars per day observed))."""
    validate_market(market)
    rows = []
    for sym, m in market.groupby("symbol"):
        m = m.sort_values("ts")
        day = m["ts"].dt.normalize()
        adv = m.groupby(day)["volume"].sum().mean()
        mid = market_mid(m)
        r = np.log(mid).diff().dropna()
        bars_per_day = m.groupby(day).size().median()
        rows.append({"symbol": sym, "adv": float(adv),
                     "sigma_daily": float(r.std() * np.sqrt(bars_per_day)),
                     "bars_per_day": float(bars_per_day)})
    return pd.DataFrame(rows)


def realized_impact(
    orders: pd.DataFrame,
    fills: pd.DataFrame,
    market: pd.DataFrame,
    stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    validate_orders(orders)
    validate_fills(fills, orders)
    validate_market(market)
    stats = (stats if stats is not None else market_stats(market)).set_index("symbol")

    f = fills.assign(notional=fills["px"] * fills["qty"]).sort_values("ts")
    g = f.groupby("order_id")
    agg = pd.DataFrame({
        "filled_qty": g["qty"].sum(),
        "notional": g["notional"].sum(),
        "first_ts": g["ts"].min(),
        "last_ts": g["ts"].max(),
    })
    o = orders.set_index("order_id").join(agg, how="inner")
    o = o[o["filled_qty"] > 0]
    o["avg_px"] = o["notional"] / o["filled_qty"]

    mid_last = pd.Series(np.nan, index=o.index)
    part = pd.Series(np.nan, index=o.index)
    for sym, grp in o.groupby("symbol"):
        mid_last.loc[grp.index] = asof_mid(market, sym, grp["last_ts"]).to_numpy()
        part.loc[grp.index] = [
            grp.at[i, "filled_qty"] / max(interval_volume(market, a, b, sym), 1e-12)
            for i, a, b in zip(grp.index, grp["first_ts"], grp["last_ts"])
        ]
    side = o["side"].astype(float)
    out = pd.DataFrame({
        "symbol": o["symbol"],
        "side": o["side"],
        "qty": o["qty"],
        "filled_qty": o["filled_qty"],
        "adv": o["symbol"].map(stats["adv"]),
        "sigma": o["symbol"].map(stats["sigma_daily"]),
        "duration_min": (o["last_ts"] - o["first_ts"]).dt.total_seconds() / 60.0,
        "participation": part,
        "cost": side * (o["avg_px"] - o["arrival_px"]) / o["arrival_px"],
        "impact_end": side * (mid_last - o["arrival_px"]) / o["arrival_px"],
    })
    out["q_adv"] = out["filled_qty"] / out["adv"]
    return out.reset_index()


def _split(n: int, holdout: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(n)
    k = round(n * holdout)
    return idx[k:], idx[:k]


def fit_sqrt_law(
    df: pd.DataFrame,
    target: str = "impact_end",
    holdout: float = 0.3,
    seed: int = 0,
    n_boot: int = 500,
) -> FitResult:
    """`target = Y · sigma · sqrt(q_adv)`; OLS through the origin; bootstrap CI on Y."""
    d = df.dropna(subset=[target, "sigma", "q_adv"])
    x = (d["sigma"] * np.sqrt(d["q_adv"])).to_numpy()
    y = d[target].to_numpy()
    rng = np.random.default_rng(seed)
    tr, te = _split(len(d), holdout, rng)

    def ols0(xx, yy):
        return float(xx @ yy / (xx @ xx))

    Y = ols0(x[tr], y[tr])
    resid = y[tr] - Y * x[tr]
    ss_tot = float(((y[tr] - y[tr].mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else np.nan
    rmse_in = float(np.sqrt((resid**2).mean()))
    rmse_out = float(np.sqrt(((y[te] - Y * x[te]) ** 2).mean())) if len(te) else np.nan
    boots = np.array([
        ols0(x[tr][b], y[tr][b]) for b in (rng.integers(0, len(tr), len(tr)) for _ in range(n_boot))
    ])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return FitResult(
        model="sqrt_law", target=target, params={"Y": Y},
        param_ci={"Y": (float(lo), float(hi))},
        r2_in=r2, rmse_in=rmse_in, rmse_out=rmse_out, n_in=len(tr), n_out=len(te),
        notes={"form": "target = Y * sigma_daily * sqrt(qty/ADV)", "bars_per_day": MINUTES_PER_DAY},
    )
