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
from .schema import SchemaError, validate_fills, validate_market, validate_orders
from .scorecards import difficulty_adjusted, scorecard
from .shortfall import COMPONENTS, implementation_shortfall

__all__ = [
    "COMPONENTS",
    "FitResult",
    "SchemaError",
    "benchmark_slippage",
    "contract_costs",
    "difficulty_adjusted",
    "fit_almgren2005",
    "fit_istar",
    "fit_sqrt_law",
    "implementation_shortfall",
    "interval_twap",
    "interval_vwap",
    "market_stats",
    "net_mid",
    "realized_impact",
    "scorecard",
    "validate_fills",
    "validate_market",
    "validate_orders",
]
__version__ = "0.2.0"
