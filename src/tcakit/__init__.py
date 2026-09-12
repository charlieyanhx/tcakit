"""tcakit — transaction cost analysis and market-impact calibration."""

from .benchmarks import benchmark_slippage, interval_twap, interval_vwap
from .impact import (
    FitResult,
    fit_almgren2005,
    fit_istar,
    fit_sqrt_law,
    market_stats,
    realized_impact,
)
from .options import contract_costs, net_mid
from .schedule import ACParams, Schedule, almgren_chriss, efficient_frontier, pov, twap, vwap
from .schema import SchemaError, validate_fills, validate_market, validate_orders
from .scorecards import difficulty_adjusted, scorecard
from .shortfall import COMPONENTS, implementation_shortfall

__all__ = [
    "COMPONENTS",
    "ACParams",
    "FitResult",
    "Schedule",
    "SchemaError",
    "almgren_chriss",
    "benchmark_slippage",
    "contract_costs",
    "difficulty_adjusted",
    "efficient_frontier",
    "fit_almgren2005",
    "fit_istar",
    "fit_sqrt_law",
    "implementation_shortfall",
    "interval_twap",
    "interval_vwap",
    "market_stats",
    "net_mid",
    "pov",
    "realized_impact",
    "scorecard",
    "twap",
    "validate_fills",
    "validate_market",
    "validate_orders",
    "vwap",
]
__version__ = "0.3.0"
