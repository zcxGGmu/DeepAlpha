"""市场数据WebSocket推送"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from deepalpha.market.data import Candle, CandleEvent
from deepalpha.transport.websocket.manager import manager
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class MarketDataBroadcaster:
    """市场数据广播器"""

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._subscriptions: Dict[str, List[str]] = {}  # symbol -> client_ids

    async def start(self):
        """启动广播器"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._broadcast_loop())
        logger.info("市场数据广播器已启动")

    async def stop(self):
        """停止广播器"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("市场数据广播器已停止")

    async def subscribe_symbol(self, client_id: str, symbol: str):
        """订阅交易对"""
        await manager.subscribe(client_id, f"market_data:{symbol}")

        if symbol not in self._subscriptions:
            self._subscriptions[symbol] = []
        if client_id not in self._subscriptions[symbol]:
            self._subscriptions[symbol].append(client_id)

        logger.info(f"客户端 {client_id} 订阅交易对: {symbol}")

    async def unsubscribe_symbol(self, client_id: str, symbol: str):
        """取消订阅交易对"""
        await manager.unsubscribe(client_id, f"market_data:{symbol}")

        if symbol in self._subscriptions:
            if client_id in self._subscriptions[symbol]:
                self._subscriptions[symbol].remove(client_id)
                if not self._subscriptions[symbol]:
                    del self._subscriptions[symbol]

        logger.info(f"客户端 {client_id} 取消订阅交易对: {symbol}")

    async def push_candle(self, candle: Candle):
        """推送K线数据"""
        message = {
            "type": "candle",
            "data": {
                "symbol": candle.symbol,
                "exchange": candle.exchange,
                "interval": candle.interval,
                "timestamp": candle.timestamp.isoformat(),
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": float(candle.volume),
            }
        }

        await manager.broadcast(f"market_data:{candle.symbol}", message)

    async def push_price_update(self, symbol: str, price: float):
        """推送价格更新"""
        message = {
            "type": "price_update",
            "data": {
                "symbol": symbol,
                "price": price,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        await manager.broadcast(f"market_data:{symbol}", message)

    async def push_orderbook_update(self, symbol: str, orderbook: Dict):
        """推送订单簿更新"""
        message = {
            "type": "orderbook",
            "data": {
                "symbol": symbol,
                "bids": orderbook.get("bids", []),
                "asks": orderbook.get("asks", []),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        await manager.broadcast(f"market_data:{symbol}", message)

    async def push_trades(self, symbol: str, trades: List[Dict]):
        """推送交易记录"""
        message = {
            "type": "trades",
            "data": {
                "symbol": symbol,
                "trades": trades,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        await manager.broadcast(f"market_data:{symbol}", message)

    async def _broadcast_loop(self):
        """广播循环（模拟数据推送）"""
        while self._running:
            try:
                # 模拟价格更新
                for symbol in self._subscriptions:
                    # TODO: 实际应该从交易所API获取实时数据
                    # 这里只是模拟
                    if self._subscriptions[symbol]:  # 有订阅者
                        # 生成随机价格
                        import random
                        base_price = 50000.0 if "BTC" in symbol else 3000.0
                        price = base_price * (1 + random.uniform(-0.001, 0.001))

                        await self.push_price_update(symbol, price)

                await asyncio.sleep(1)  # 每秒推送一次

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"广播循环错误: {e}")
                await asyncio.sleep(5)


# 全局广播器实例
broadcaster = MarketDataBroadcaster()


# 导出
__all__ = ["MarketDataBroadcaster", "broadcaster"]