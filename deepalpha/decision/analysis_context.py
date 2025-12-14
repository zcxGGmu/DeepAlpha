"""分析上下文，对应 Go 版本的 AnalysisContext"""

import json
from typing import Any, Dict, Optional

from deepalpha.market.data import Candle
from deepalpha.market.format import format_candle_csv


class AnalysisContext:
    """分析上下文，对应 Go 版本的 AnalysisContext"""

    def __init__(
        self,
        symbol: str,
        interval: str,
        forecast_horizon: str,
        candles: Optional[list[Candle]] = None,
        indicator_report: Optional[Dict[str, Any]] = None,
        pattern_report: Optional[str] = None,
        trend_report: Optional[str] = None,
        image_note: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.symbol = symbol
        self.interval = interval
        self.forecast_horizon = forecast_horizon
        self.candles = candles or []
        self.indicator_report = indicator_report or {}
        self.pattern_report = pattern_report or ""
        self.trend_report = trend_report or ""
        self.image_note = image_note or ""
        self.metadata = metadata or {}

    @property
    def kline_csv(self) -> str:
        """获取K线CSV格式数据"""
        if not self.candles:
            return ""
        return format_candle_csv(self.candles)

    @property
    def kline_json(self) -> str:
        """获取K线JSON格式数据"""
        if not self.candles:
            return ""

        # 转换为字典列表
        data = []
        for candle in self.candles:
            data.append({
                "open_time": candle.open_time,
                "close_time": candle.close_time,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "trades": candle.trades
            })

        return json.dumps(data, ensure_ascii=False)

    @property
    def indicator_json(self) -> str:
        """获取指标JSON格式数据"""
        if not self.indicator_report:
            return ""
        return json.dumps(self.indicator_report, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "forecast_horizon": self.forecast_horizon,
            "candles": [
                {
                    "open_time": c.open_time,
                    "close_time": c.close_time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "trades": c.trades
                }
                for c in self.candles
            ],
            "indicator_report": self.indicator_report,
            "pattern_report": self.pattern_report,
            "trend_report": self.trend_report,
            "image_note": self.image_note,
            "metadata": self.metadata
        }