"""市场数据模型"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field


class Candle(BaseModel):
    """K线数据模型，对应 Go 版本的 Candle"""
    open_time: int = Field(..., description="开盘时间戳（毫秒）")
    close_time: int = Field(..., description="收盘时间戳（毫秒）")
    open: float = Field(..., description="开盘价")
    high: float = Field(..., description="最高价")
    low: float = Field(..., description="最低价")
    close: float = Field(..., description="收盘价")
    volume: float = Field(..., description="成交量")
    trades: int = Field(0, description="成交笔数")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
        }


class CandleEvent(BaseModel):
    """K线事件，对应 Go 版本的 CandleEvent"""
    symbol: str = Field(..., description="交易对符号")
    interval: str = Field(..., description="时间间隔")
    candle: Candle = Field(..., description="K线数据")


class TradeEvent(BaseModel):
    """实时成交价事件，对应 Go 版本的 TradeEvent"""
    symbol: str = Field(..., description="交易对符号")
    price: float = Field(..., description="成交价格")
    quantity: float = Field(..., description="成交数量")
    event_time: int = Field(..., description="事件时间戳（毫秒）")
    trade_time: int = Field(..., description="交易时间戳（毫秒）")


class OpenInterestPoint(BaseModel):
    """持仓量数据点，对应 Go 版本的 OpenInterestPoint"""
    symbol: str = Field(..., description="交易对符号")
    sum_open_interest: float = Field(..., description="总持仓量")
    sum_open_interest_value: float = Field(..., description="总持仓价值")
    timestamp: int = Field(..., description="时间戳（毫秒）")


class SubscribeOptions(BaseModel):
    """订阅选项，对应 Go 版本的 SubscribeOptions"""
    batch_size: int = Field(100, description="批处理大小")
    buffer: int = Field(512, description="缓冲区大小")
    on_connect: Optional[Callable[[], None]] = Field(None, description="连接成功回调")
    on_disconnect: Optional[Callable[[Exception], None]] = Field(None, description="连接断开回调")


class SourceStats(BaseModel):
    """数据源统计，对应 Go 版本的 SourceStats"""
    reconnects: int = Field(0, description="重连次数")
    subscribe_errors: int = Field(0, description="订阅错误次数")
    last_error: str = Field("", description="最后错误信息")