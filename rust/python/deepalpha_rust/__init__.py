"""DeepAlpha Rust Performance Modules

This package provides high-performance Rust implementations of critical
components for the DeepAlpha quantitative trading system.
"""

from . import _core

try:
    # Import the main classes from the Rust core
    from ._core import TechnicalIndicators
    from ._core import WebSocketManager
    from ._core import MarketDataStream
    from ._core import ExecutionEngine
except ImportError:
    # Fallback when Rust module is not compiled
    import warnings
    warnings.warn(
        "Rust module not compiled. Please run 'maturin develop' to build the Rust extensions.",
        ImportWarning
    )

__version__ = "0.1.0"
__all__ = [
    "TechnicalIndicators",
    "WebSocketManager",
    "MarketDataStream",
    "ExecutionEngine",
]