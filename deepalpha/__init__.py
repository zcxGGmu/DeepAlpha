"""
DeepAlpha - AI驱动的多智能体量化交易系统

一个基于Python的高性能量化交易系统，采用多智能体架构和AI决策引擎，
专注于加密货币市场的自动化交易。
"""

__version__ = "1.0.0"
__author__ = "DeepAlpha Team"
__email__ = "team@deepalpha.com"

from deepalpha.core.engine import DeepAlphaEngine
from deepalpha.core.models import Signal, Decision, Position, Order
from deepalpha.config.settings import Settings

__all__ = [
    "DeepAlphaEngine",
    "Signal",
    "Decision",
    "Position",
    "Order",
    "Settings",
]