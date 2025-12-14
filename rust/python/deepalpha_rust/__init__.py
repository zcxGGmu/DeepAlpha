"""DeepAlpha Rust Performance Modules

This package provides high-performance Rust implementations of critical
components for the DeepAlpha quantitative trading system.
"""

from . import _core

try:
    # Import the main classes from the Rust core
    from ._core import TechnicalIndicators

    # WebSocket module
    from ._core.websocket import WebSocketManager, ConnectionStats

    # Message types
    from ._core.websocket import (
        TradeSignal,
        MarketData,
        CandlestickData,
        MessageType
    )

    # Stream module
    from ._core.stream import MarketDataStream, StreamStats

    # Execution module (placeholder)
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
    "ConnectionStats",
    "TradeSignal",
    "MarketData",
    "CandlestickData",
    "MessageType",
    "MarketDataStream",
    "StreamStats",
    "ExecutionEngine",
]