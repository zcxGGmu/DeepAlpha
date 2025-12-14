"""K线数据存储"""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from deepalpha.core.exceptions import DataError
from deepalpha.market.data import Candle
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class KlineStore:
    """K线存储接口，对应 Go 版本的 KlineStore"""

    async def get(self, symbol: str, interval: str) -> List[Candle]:
        """获取K线数据"""
        raise NotImplementedError

    async def set(self, symbol: str, interval: str, klines: List[Candle]) -> None:
        """设置K线数据"""
        raise NotImplementedError

    async def put(self, symbol: str, interval: str, klines: List[Candle], max_count: int) -> None:
        """添加K线数据并限制最大数量"""
        raise NotImplementedError


class MemoryKlineStore(KlineStore):
    """内存K线存储实现"""

    def __init__(self):
        self._data: Dict[str, Dict[str, List[Candle]]] = defaultdict(lambda: defaultdict(list))
        self._lock = asyncio.Lock()

    async def get(self, symbol: str, interval: str) -> List[Candle]:
        """获取K线数据"""
        async with self._lock:
            return list(self._data[symbol.upper()][interval])

    async def set(self, symbol: str, interval: str, klines: List[Candle]) -> None:
        """设置K线数据，会替换现有数据"""
        if not klines:
            logger.warning(f"尝试设置空的K线数据: {symbol} {interval}")
            return

        async with self._lock:
            self._data[symbol.upper()][interval] = klines
            logger.debug(f"存储K线数据: {symbol} {interval} 数量: {len(klines)}")

    async def put(self, symbol: str, interval: str, klines: List[Candle], max_count: int) -> None:
        """添加K线数据并限制最大数量"""
        if not klines:
            return

        symbol = symbol.upper()

        async with self._lock:
            existing = self._data[symbol][interval]

            # 合并新数据，避免重复
            for kline in klines:
                # 查找是否已存在相同时间戳的K线
                found = False
                for i, existing_kline in enumerate(existing):
                    if existing_kline.open_time == kline.open_time:
                        # 更新现有K线
                        existing[i] = kline
                        found = True
                        break

                if not found:
                    # 添加新K线
                    existing.append(kline)

            # 按时间排序
            existing.sort(key=lambda x: x.open_time)

            # 限制最大数量
            if len(existing) > max_count:
                self._data[symbol][interval] = existing[-max_count:]
            else:
                self._data[symbol][interval] = existing

            logger.debug(f"更新K线数据: {symbol} {interval} 当前数量: {len(self._data[symbol][interval])}")

    async def get_latest(self, symbol: str, interval: str) -> Optional[Candle]:
        """获取最新K线"""
        klines = await self.get(symbol, interval)
        return klines[-1] if klines else None

    async def clear(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> None:
        """清除数据"""
        async with self._lock:
            if symbol and interval:
                self._data[symbol.upper()].pop(interval, None)
            elif symbol:
                self._data.pop(symbol.upper(), None)
            else:
                self._data.clear()

    async def get_symbols(self) -> List[str]:
        """获取所有交易对"""
        async with self._lock:
            return list(self._data.keys())

    async def get_intervals(self, symbol: str) -> List[str]:
        """获取指定交易对的所有时间间隔"""
        async with self._lock:
            return list(self._data[symbol.upper()].keys())


class DatabaseKlineStore(KlineStore):
    """数据库K线存储实现"""

    def __init__(self, db_session_factory):
        self.session_factory = db_session_factory

    async def get(self, symbol: str, interval: str) -> List[Candle]:
        """从数据库获取K线数据"""
        # TODO: 实现数据库查询
        raise DataError("DatabaseKlineStore not implemented yet")

    async def set(self, symbol: str, interval: str, klines: List[Candle]) -> None:
        """设置K线数据到数据库"""
        # TODO: 实现数据库存储
        raise DataError("DatabaseKlineStore not implemented yet")

    async def put(self, symbol: str, interval: str, klines: List[Candle], max_count: int) -> None:
        """添加K线数据到数据库"""
        # TODO: 实现数据库存储
        raise DataError("DatabaseKlineStore not implemented yet")