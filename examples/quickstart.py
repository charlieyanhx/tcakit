"""Quickstart: simulate, decompose shortfall, benchmark, and calibrate the square-root law."""

import pandas as pd

from tcakit import benchmark_slippage, implementation_shortfall
from tcakit.impact import fit_sqrt_law, market_stats, realized_impact
from tcakit.synth import simulate

sim = simulate(n_days=120, orders_per_day=8, seed=42, sigma_daily=0.02, spread_bps=5.0, sqrt_law_y=0.8)

is_ = implementation_shortfall(sim.orders, sim.fills, sim.market)
print("Implementation shortfall, mean bps of decision notional:")
print(is_[["delay_bps", "spread_bps", "timing_bps", "opportunity_bps", "fees_bps", "total_bps"]].mean().round(2))

bm = benchmark_slippage(sim.orders, sim.fills, sim.market, reversion_horizon="10min")
print("\nBenchmark slippage, mean bps:")
print(bm[["arrival_bps", "vwap_bps", "twap_bps", "close_bps", "reversion_bps"]].mean().round(2))
print(f"mean participation: {bm['participation'].mean():.3%}")

print("\nMarket stats measured from bars (note: own impact inflates σ):")
print(market_stats(sim.market).round(5).to_string(index=False))

df = realized_impact(sim.orders, sim.fills, sim.market)
print("\nSquare-root law fits (true Y = 0.8; expect cost ≈ 2/3 of completion impact):")
print(fit_sqrt_law(df, target="impact_end", seed=1).summary())
print(fit_sqrt_law(df, target="cost", seed=1).summary())
