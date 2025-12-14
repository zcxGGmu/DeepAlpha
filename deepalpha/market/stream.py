"""WebSocket数据流更新器"""

import asyncio
from typing import Callable, List, Optional

from deepalpha.market.data import CandleEvent, SubscribeOptions
from deepalpha.market.source import Source
from deepalpha.market.store import KlineStore
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class WSUpdater:
    """WebSocket数据更新器，对应 Go 版本的 WSUpdater"""

    def __init__(
        self,
        store: KlineStore,
        max_count: int,
        source: Source,
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnected: Optional[Callable[[Exception], None]] = None,
        on_event: Optional[Callable[[CandleEvent], None]] = None,
    ):
        self.store = store
        self.max_count = max_count
        self.source = source
        self.on_connected = on_connected
        self.on_disconnected = on_disconnected
        self.on_event = on_event

        self._started = False
        self._task: Optional[asyncio.Task] = None

    async def update(self, symbol: str, interval: str, kline: "Candle") -> None:
        """更新单根K线到存储"""
        try:
            await self.store.put(symbol.upper(), interval, [kline], self.max_count)
        except Exception as e:
            logger.error(f"更新K线失败 {symbol} {interval}: {e}")

    async def start(self, symbols: List[str], intervals: List[str]) -> None:
        """启动订阅实时行情"""
        if not self.source:
            raise ValueError("WSUpdater missing source")

        if not symbols or not intervals:
            raise ValueError("WSUpdater requires symbols & intervals")

        if self._started:
            logger.warning("WSUpdater already started")
            return

        options = SubscribeOptions(
            on_connect=self.on_connected,
            on_disconnect=self.on_disconnected,
            buffer=512,
        )

        try:
            events = await self.source.subscribe(symbols, intervals, options)
            self._started = True

            # 启动消费任务
            self._task = asyncio.create_task(self._consume(events))

            logger.info(f"订阅已启动 symbols={symbols} intervals={intervals}")
        except Exception as e:
            logger.error(f"启动订阅失败: {e}")
            raise

    async def _consume(self, events_channel) -> None:
        """消费事件"""
        try:
            async for event in events_channel:
                try:
                    # 写入存储
                    await self.update(
                        event.symbol.upper(),
                        event.interval,
                        event.candle
                    )

                    # 触发事件回调
                    if self.on_event:
                        self.on_event(event)

                except Exception as e:
                    logger.error(f"处理事件失败 {event.symbol} {event.interval}: {e}")
        except Exception as e:
            logger.error(f"事件消费循环错误: {e}")
            if self.on_disconnected:
                self.on_disconnected(e)

    async def stop(self) -> None:
        """停止更新器"""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._started = False

        if self.source:
            await self.source.close()

        logger.info("WSUpdater 已停止")

    def stats(self) -> dict:
        """获取数据源统计"""
        if not self.source:
            return {}
        return self.source.stats()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()