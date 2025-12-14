"""市场数据源接口"""

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional

from deepalpha.market.data import Candle, CandleEvent, OpenInterestPoint, SubscribeOptions, TradeEvent, SourceStats


class Source(ABC):
    """市场数据源接口，对应 Go 版本的 Source"""

    @abstractmethod
    async def fetch_history(
        self,
        symbol: str,
        interval: str,
        limit: int
    ) -> List[Candle]:
        """获取历史K线数据"""
        pass

    @abstractmethod
    async def subscribe(
        self,
        symbols: List[str],
        intervals: List[str],
        opts: SubscribeOptions
    ) -> AsyncGenerator[CandleEvent, None]:
        """订阅实时K线数据"""
        pass

    @abstractmethod
    async def subscribe_trades(
        self,
        symbols: List[str],
        opts: SubscribeOptions
    ) -> AsyncGenerator[TradeEvent, None]:
        """订阅实时成交数据"""
        pass

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> float:
        """获取资金费率"""
        pass

    @abstractmethod
    async def get_open_interest_history(
        self,
        symbol: str,
        period: str,
        limit: int
    ) -> List[OpenInterestPoint]:
        """获取持仓量历史"""
        pass

    @abstractmethod
    def stats(self) -> SourceStats:
        """获取数据源统计信息"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭数据源，释放资源"""
        pass


class BaseSource(Source):
    """数据源基类实现"""

    def __init__(self):
        self._stats = SourceStats()
        self._closed = False

    def stats(self) -> SourceStats:
        """获取统计信息"""
        return self._stats

    async def close(self) -> None:
        """关闭数据源"""
        self._closed = True

    def _record_reconnect(self):
        """记录重连"""
        self._stats.reconnects += 1

    def _record_subscribe_error(self, error: Exception):
        """记录订阅错误"""
        self._stats.subscribe_errors += 1
        self._stats.last_error = str(error)

    async def _ensure_not_closed(self):
        """确保数据源未关闭"""
        if self._closed:
            raise ConnectionError("Data source is closed")