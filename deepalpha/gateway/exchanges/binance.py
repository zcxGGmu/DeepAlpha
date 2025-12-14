"""Binance数据源实现"""

import asyncio
import json
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

from deepalpha.config.settings import ExchangeSettings
from deepalpha.market.data import Candle, CandleEvent, OpenInterestPoint, SubscribeOptions, TradeEvent, SourceStats
from deepalpha.market.source import BaseSource
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)

MAX_HISTORY_LIMIT = 1500


class BinanceSource(BaseSource):
    """Binance数据源，基于 go-binance SDK 的 Python 实现"""

    def __init__(self, config: ExchangeSettings):
        super().__init__()
        self.config = config
        self.base_url = "https://fapi.binance.com"
        self.ws_url = "wss://fstream.binance.com/ws"

        # 用于代理的session
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connections: List[websockets.WebSocketServerProtocol] = []

        # 取消标记
        self._candle_cancel: Optional[asyncio.Event] = None
        self._trade_cancel: Optional[asyncio.Event] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def fetch_history(
        self,
        symbol: str,
        interval: str,
        limit: int
    ) -> List[Candle]:
        """获取历史K线数据"""
        await self._ensure_not_closed()

        if limit <= 0:
            limit = 100
        if limit > MAX_HISTORY_LIMIT:
            limit = MAX_HISTORY_LIMIT

        symbol = symbol.upper().strip()
        if not symbol:
            raise ValueError("symbol is required")

        interval = interval.lower().strip()
        if not interval:
            raise ValueError("interval is required")

        url = f"{self.base_url}/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        try:
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                klines = []
                for item in data:
                    if not item:
                        continue

                    kline = Candle(
                        open_time=item[0],
                        close_time=item[6],
                        open=float(item[1]),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[5]),
                        trades=int(item[8]) if len(item) > 8 else 0
                    )
                    klines.append(kline)

                return klines

        except Exception as e:
            logger.error(f"获取历史K线失败 {symbol} {interval}: {e}")
            raise

    async def subscribe(
        self,
        symbols: List[str],
        intervals: List[str],
        opts: SubscribeOptions
    ) -> AsyncGenerator[CandleEvent, None]:
        """订阅实时K线数据"""
        await self._ensure_not_closed()

        # 构建订阅映射
        mapping = self._build_symbol_intervals(symbols, intervals)
        if not mapping:
            raise ValueError("no valid symbols or intervals for subscription")

        # 创建取消事件
        self._candle_cancel = asyncio.Event()

        try:
            async for event in self._run_kline_loop(mapping, opts):
                yield event
        finally:
            self._candle_cancel = None

    async def subscribe_trades(
        self,
        symbols: List[str],
        opts: SubscribeOptions
    ) -> AsyncGenerator[TradeEvent, None]:
        """订阅实时成交数据"""
        await self._ensure_not_closed()

        # 清理和验证符号
        targets = []
        for sym in symbols:
            upper = sym.upper().strip()
            if upper:
                targets.append(upper)

        if not targets:
            raise ValueError("no valid symbols for trade subscription")

        # 创建取消事件
        self._trade_cancel = asyncio.Event()

        try:
            async for event in self._run_trade_loop(targets, opts):
                yield event
        finally:
            self._trade_cancel = None

    async def _run_kline_loop(
        self,
        mapping: Dict[str, List[str]],
        opts: SubscribeOptions
    ) -> AsyncGenerator[CandleEvent, None]:
        """运行K线订阅循环"""
        streams = []
        # 构建流URL
        for symbol, intervals in mapping.items():
            for interval in intervals:
                streams.append(f"{symbol.lower()}@kline_{interval}")

        stream_url = f"{self.ws_url}/stream?streams={'/'.join(streams)}"

        while not self._candle_cancel.is_set():
            try:
                async with websockets.connect(stream_url) as websocket:
                    self._ws_connections.append(websocket)
                    logger.info(f"Binance WebSocket已连接: {len(streams)} streams")

                    if opts.on_connect:
                        opts.on_connect()

                    # 处理消息
                    async for message in websocket:
                        if self._candle_cancel.is_set():
                            break

                        try:
                            data = json.loads(message)
                            event = self._convert_kline_event(data)
                            if event:
                                yield event
                        except Exception as e:
                            logger.error(f"处理K线消息失败: {e}")

            except ConnectionClosed as e:
                logger.warning(f"WebSocket连接断开: {e}")
                self._record_reconnect()
                if opts.on_disconnect:
                    opts.on_disconnect(e)

                # 等待后重试
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"K线订阅错误: {e}")
                self._record_subscribe_error(e)
                if opts.on_disconnect:
                    opts.on_disconnect(e)
                await asyncio.sleep(5)

    async def _run_trade_loop(
        self,
        symbols: List[str],
        opts: SubscribeOptions
    ) -> AsyncGenerator[TradeEvent, None]:
        """运行成交订阅循环"""
        streams = [f"{sym.lower()}@aggTrade" for sym in symbols]
        stream_url = f"{self.ws_url}/stream?streams={'/'.join(streams)}"

        while not self._trade_cancel.is_set():
            try:
                async with websockets.connect(stream_url) as websocket:
                    self._ws_connections.append(websocket)
                    logger.info(f"Binance成交WebSocket已连接: {len(streams)} streams")

                    if opts.on_connect:
                        opts.on_connect()

                    # 处理消息
                    async for message in websocket:
                        if self._trade_cancel.is_set():
                            break

                        try:
                            data = json.loads(message)
                            event = self._convert_trade_event(data)
                            if event:
                                yield event
                        except Exception as e:
                            logger.error(f"处理成交消息失败: {e}")

            except ConnectionClosed as e:
                logger.warning(f"成交WebSocket连接断开: {e}")
                self._record_reconnect()
                if opts.on_disconnect:
                    opts.on_disconnect(e)
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"成交订阅错误: {e}")
                self._record_subscribe_error(e)
                if opts.on_disconnect:
                    opts.on_disconnect(e)
                await asyncio.sleep(5)

    def _build_symbol_intervals(
        self,
        symbols: List[str],
        intervals: List[str]
    ) -> Dict[str, List[str]]:
        """构建交易对到时间间隔的映射"""
        mapping = {}
        for symbol in symbols:
            upper = symbol.upper().strip()
            if upper:
                mapping[upper] = []
                for interval in intervals:
                    interval = interval.lower().strip()
                    if interval:
                        mapping[upper].append(interval)
        return mapping

    def _convert_kline_event(self, data: dict) -> Optional[CandleEvent]:
        """转换K线事件"""
        try:
            if "data" not in data or "k" not in data["data"]:
                return None

            k = data["data"]["k"]
            if not k["x"]:  # 只处理已完成的K线
                return None

            candle = Candle(
                open_time=k["t"],
                close_time=k["T"],
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
                trades=int(k["n"])
            )

            return CandleEvent(
                symbol=data["data"]["s"],
                interval=k["i"],
                candle=candle
            )
        except Exception as e:
            logger.error(f"转换K线事件失败: {e}")
            return None

    def _convert_trade_event(self, data: dict) -> Optional[TradeEvent]:
        """转换成交事件"""
        try:
            if "data" not in data:
                return None

            d = data["data"]
            return TradeEvent(
                symbol=d["s"],
                price=float(d["p"]),
                quantity=float(d["q"]),
                event_time=d["E"],
                trade_time=d["T"]
            )
        except Exception as e:
            logger.error(f"转换成交事件失败: {e}")
            return None

    async def get_funding_rate(self, symbol: str) -> float:
        """获取资金费率"""
        await self._ensure_not_closed()

        url = f"{self.base_url}/fapi/v1/premiumIndex"
        params = {"symbol": symbol.upper()}

        try:
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return float(data["lastFundingRate"])
        except Exception as e:
            logger.error(f"获取资金费率失败 {symbol}: {e}")
            raise

    async def get_open_interest_history(
        self,
        symbol: str,
        period: str,
        limit: int
    ) -> List[OpenInterestPoint]:
        """获取持仓量历史"""
        await self._ensure_not_closed()

        url = f"{self.base_url}/fapi/v1/openInterestHist"
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": limit
        }

        try:
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                points = []
                for item in data:
                    point = OpenInterestPoint(
                        symbol=item["symbol"],
                        sum_open_interest=float(item["sumOpenInterest"]),
                        sum_open_interest_value=float(item["sumOpenInterestValue"]),
                        timestamp=item["timestamp"]
                    )
                    points.append(point)

                return points
        except Exception as e:
            logger.error(f"获取持仓量历史失败 {symbol}: {e}")
            raise

    async def close(self) -> None:
        """关闭数据源"""
        await super().close()

        # 关闭WebSocket连接
        for ws in self._ws_connections:
            try:
                await ws.close()
            except Exception as e:
                logger.error(f"关闭WebSocket失败: {e}")
        self._ws_connections.clear()

        # 取消订阅
        if self._candle_cancel:
            self._candle_cancel.set()
        if self._trade_cancel:
            self._trade_cancel.set()

        # 关闭HTTP会话
        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Binance数据源已关闭")