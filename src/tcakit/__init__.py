"""tcakit — transaction cost analysis and market-impact calibration."""

from .benchmarks import benchmark_slippage, interval_twap, interval_vwap
from .schema import SchemaError, validate_fills, validate_market, validate_orders
from .shortfall import COMPONENTS, implementation_shortfall

__all__ = [
    "COMPONENTS",
    "SchemaError",
    "benchmark_slippage",
    "implementation_shortfall",
    "interval_twap",
    "interval_vwap",
    "validate_fills",
    "validate_market",
    "validate_orders",
]
__version__ = "0.1.0"
