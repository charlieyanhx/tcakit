"""Optimal execution schedules: Almgren-Chriss (2000) closed forms, the efficient frontier,
and the TWAP / VWAP / POV baselines every schedule is compared against.

Model (Almgren & Chriss, "Optimal execution of portfolio transactions", J. Risk 2000):
liquidate X shares over T with N equal intervals τ = T/N; price follows an arithmetic random
walk with volatility σ (per √time); permanent impact γ per share traded; temporary impact
ε·sign(v) + η·v per share for trade rate v = n_k/τ (n_k shares in interval k).

    E[cost]  = ½ γ X² + ε X + (η̃/τ) Σ n_k²            with η̃ = η − ½ γ τ
    Var[cost] = σ² τ Σ x_k²                             x_k = shares remaining after interval k
    minimise  E + λ Var  →  x_j = X · sinh(κ (T − t_j)) / sinh(κ T)
    κ = arccosh(½ κ̃² τ² + 1) / τ,   κ̃² = λ σ² / η̃

λ → 0 gives the straight line (TWAP, minimum expected cost); λ → ∞ front-loads everything
into the first interval (minimum variance). Costs are per share² units of γ/η as given —
pass γ, η in price per share per share, ε in price per share, σ in price per share per √T.

All schedules return a `Schedule` with the holdings path `x` (length N+1, x₀ = X, x_N = 0),
the trade list `n` (length N, Σ n = X), and the model's expected cost and variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ACParams:
    sigma: float          # volatility, price per share per sqrt(time unit of T)
    gamma: float          # permanent impact, price per share per share
    eta: float            # temporary impact, price per share per (share / time)
    epsilon: float = 0.0  # fixed cost per share (half-spread + fees)

    def eta_tilde(self, tau: float) -> float:
        return self.eta - 0.5 * self.gamma * tau


@dataclass(frozen=True)
class Schedule:
    name: str
    X: float
    T: float
    x: np.ndarray            # holdings after each interval, x[0] = X, x[-1] = 0
    n: np.ndarray            # shares traded in each interval, sum = X
    expected_cost: float
    variance: float
    notes: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return len(self.n)

    @property
    def tau(self) -> float:
        return self.T / self.N


def expected_cost(x: np.ndarray, X: float, tau: float, p: ACParams) -> float:
    """E[cost] of a holdings path under the AC model (discrete, Almgren-Chriss eq. 8)."""
    n = -np.diff(x)
    return 0.5 * p.gamma * X**2 + p.epsilon * np.abs(n).sum() + p.eta_tilde(tau) / tau * (n**2).sum()


def variance(x: np.ndarray, tau: float, p: ACParams) -> float:
    """Var[cost] of a holdings path: σ² τ Σ x_k² over k = 1..N (eq. 9)."""
    return p.sigma**2 * tau * float((x[1:] ** 2).sum())


def _make(name: str, X: float, T: float, x: np.ndarray, p: ACParams, **notes) -> Schedule:
    x = np.asarray(x, dtype=float)
    tau = T / (len(x) - 1)
    return Schedule(name, X, T, x, -np.diff(x), float(expected_cost(x, X, tau, p)), float(variance(x, tau, p)), notes)


def almgren_chriss(X: float, T: float, N: int, p: ACParams, lam: float) -> Schedule:
    """Risk-averse optimal trajectory for risk aversion `lam` (λ ≥ 0), closed form.

    Uses `arccosh` and `sinh` directly; for κT small the ratio sinh(κ(T−t))/sinh(κT) is
    evaluated with `expm1`-stable forms so λ → 0 reproduces the straight line to 1e-12.
    """
    if X == 0:
        raise ValueError("X must be nonzero")
    if lam < 0 or N < 1 or T <= 0:
        raise ValueError("lam >= 0, N >= 1, T > 0")
    tau = T / N
    eta_t = p.eta_tilde(tau)
    if eta_t <= 0:
        raise ValueError("eta - gamma*tau/2 must be positive (eq. 10); use a finer grid or smaller gamma")
    t = np.linspace(0.0, T, N + 1)
    if lam == 0:
        x = X * (1 - t / T)
        kappa = 0.0
    else:
        k2 = lam * p.sigma**2 / eta_t
        kappa = np.arccosh(0.5 * k2 * tau**2 + 1.0) / tau
        # sinh(a)/sinh(b) = (e^{a} - e^{-a}) / (e^{b} - e^{-b}) = e^{a-b} (1 - e^{-2a}) / (1 - e^{-2b})
        a, b = kappa * (T - t), kappa * T
        x = X * np.exp(a - b) * (-np.expm1(-2 * a)) / (-np.expm1(-2 * b))
        x[-1] = 0.0
    return _make("almgren-chriss", X, T, x, p, lam=lam, kappa=float(kappa), half_life=float(np.log(2) / kappa) if kappa else np.inf)


def twap(X: float, T: float, N: int, p: ACParams) -> Schedule:
    """Straight line: the λ = 0 (risk-neutral) optimum and the usual reference schedule."""
    x = X * (1 - np.linspace(0.0, T, N + 1) / T)
    return _make("twap", X, T, x, p)


def vwap(X: float, T: float, profile: np.ndarray, p: ACParams) -> Schedule:
    """Trade in proportion to an expected volume profile (length N, any positive scale)."""
    w = np.asarray(profile, dtype=float)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("profile must be non-negative with positive total")
    n = X * w / w.sum()
    x = np.concatenate([[X], X - np.cumsum(n)])
    x[-1] = 0.0
    return _make("vwap", X, T, x, p)


def pov(X: float, T: float, profile: np.ndarray, rate: float, p: ACParams) -> Schedule:
    """Participate at `rate` of expected market volume until done; the remainder, if any,
    is forced in the last interval (reported in notes['forced'])."""
    if not 0 < rate <= 1:
        raise ValueError("rate in (0, 1]")
    v = np.asarray(profile, dtype=float)
    n = np.zeros(len(v))
    remaining = X
    for k, vk in enumerate(v):
        n[k] = min(rate * vk, remaining)
        remaining -= n[k]
    forced = remaining
    if forced > 1e-12:
        n[-1] += forced
    x = np.concatenate([[X], X - np.cumsum(n)])
    x[-1] = 0.0
    return _make("pov", X, T, x, p, rate=rate, forced=float(max(forced, 0.0)))


def efficient_frontier(X: float, T: float, N: int, p: ACParams, lams: np.ndarray) -> list[Schedule]:
    """Schedules for a grid of λ; expected cost rises and variance falls monotonically in λ."""
    return [almgren_chriss(X, T, N, p, float(l)) for l in np.asarray(lams, dtype=float)]


def cost_completion_ratio(schedule: Schedule, exponent: float = 0.5) -> float:
    """Mean paid impact ÷ completion impact for a power-law impact I(φ) ∝ φ^exponent along a
    schedule; equals 2/3 for a linear schedule under the square-root law (Bouchaud's 2/3 rule)."""
    phi = 1 - schedule.x[1:] / schedule.X            # fraction done after each interval
    w = schedule.n / schedule.X
    return float((w * phi**exponent).sum() / phi[-1] ** exponent)
