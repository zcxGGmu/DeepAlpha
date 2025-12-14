"""价格处理
对应Go版本的 price.go
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import statistics

from deepalpha.executor.freqtrade.client import FreqtradeClient
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class PriceProcessor:
    """价格处理器"""

    def __init__(self, client: FreqtradeClient):
        self.client = client
        self.price_cache: Dict[str, Tuple[Decimal, datetime]] = {}
        self.cache_duration = timedelta(seconds=5)  # 缓存5秒

    async def get_current_price(self, symbol: str) -> Decimal:
        """
        获取当前价格

        Args:
            symbol: 交易对

        Returns:
            当前价格
        """
        # 检查缓存
        if symbol in self.price_cache:
            price, timestamp = self.price_cache[symbol]
            if datetime.utcnow() - timestamp < self.cache_duration:
                return price

        # 从交易所获取
        try:
            ticker = await self.client.get_ticker(symbol)
            price = Decimal(str(ticker.last))

            # 更新缓存
            self.price_cache[symbol] = (price, datetime.utcnow())

            return price
        except Exception as e:
            logger.error(f"获取价格失败 {symbol}: {e}")
            raise

    async def get_historical_prices(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> List[Tuple[datetime, Decimal]]:
        """
        获取历史价格

        Args:
            symbol: 交易对
            timeframe: 时间框架
            limit: 数量限制

        Returns:
            价格列表 (timestamp, price)
        """
        try:
            candles = await self.client.get_candles(symbol, timeframe, limit)
            return [
                (datetime.fromtimestamp(c.timestamp / 1000), Decimal(str(c.close)))
                for c in candles
            ]
        except Exception as e:
            logger.error(f"获取历史价格失败 {symbol}: {e}")
            raise

    def calculate_average_price(
        self,
        symbol: str,
        amount: float,
        order_type: str = "market"
    ) -> Decimal:
        """
        计算平均价格（模拟）

        Args:
            symbol: 交易对
            amount: 数量
            order_type: 订单类型

        Returns:
            平均价格
        """
        # 这里应该使用订单簿数据计算
        # 简化实现，返回当前价格
        if symbol in self.price_cache:
            return self.price_cache[symbol][0]

        logger.warning(f"价格缓存中没有 {symbol}，返回模拟价格")
        return Decimal("0")

    def round_price(self, price: Decimal, symbol: str, side: str = "buy") -> Decimal:
        """
        根据交易对精度调整价格

        Args:
            price: 原始价格
            symbol: 交易对
            side: 买卖方向

        Returns:
            调整后的价格
        """
        # 获取交易对的最小价格变动
        tick_size = self._get_tick_size(symbol)

        if side == "buy":
            # 买单向下取整
            return price.quantize(tick_size, rounding=ROUND_DOWN)
        else:
            # 卖单向上取整
            return price.quantize(tick_size, rounding=ROUND_UP)

    def round_amount(self, amount: Decimal, symbol: str) -> Decimal:
        """
        根据交易对精度调整数量

        Args:
            amount: 原始数量
            symbol: 交易对

        Returns:
            调整后的数量
        """
        # 获取交易对的最小数量变动
        step_size = self._get_step_size(symbol)
        return amount.quantize(step_size, rounding=ROUND_DOWN)

    def _get_tick_size(self, symbol: str) -> Decimal:
        """
        获取交易对的最小价格变动

        Args:
            symbol: 交易对

        Returns:
            最小价格变动
        """
        # 这里应该从交易所信息获取
        # 简化实现，返回固定值
        if "USDT" in symbol:
            return Decimal("0.01")
        else:
            return Decimal("0.00000001")

    def _get_step_size(self, symbol: str) -> Decimal:
        """
        获取交易对的最小数量变动

        Args:
            symbol: 交易对

        Returns:
            最小数量变动
        """
        # 这里应该从交易所信息获取
        # 简化实现
        if "BTC" in symbol:
            return Decimal("0.00001")
        elif "ETH" in symbol:
            return Decimal("0.001")
        else:
            return Decimal("0.01")

    async def calculate_slippage(
        self,
        symbol: str,
        amount: float,
        order_type: str = "market"
    ) -> Dict[str, Decimal]:
        """
        计算滑点

        Args:
            symbol: 交易对
            amount: 数量
            order_type: 订单类型

        Returns:
            滑点信息
        """
        try:
            ticker = await self.client.get_ticker(symbol)
            current_price = Decimal(str(ticker.last))

            # 模拟滑点计算（实际应该使用订单簿）
            slippage_rate = self._estimate_slippage_rate(symbol, amount, order_type)

            slippage_amount = current_price * slippage_rate

            return {
                "current_price": current_price,
                "slippage_rate": slippage_rate,
                "slippage_amount": slippage_amount,
                "expected_price": current_price - slippage_amount if order_type == "buy" else current_price + slippage_amount
            }
        except Exception as e:
            logger.error(f"计算滑点失败 {symbol}: {e}")
            return {}

    def _estimate_slippage_rate(
        self,
        symbol: str,
        amount: float,
        order_type: str
    ) -> Decimal:
        """
        估算滑点率

        Args:
            symbol: 交易对
            amount: 数量
            order_type: 订单类型

        Returns:
            滑点率
        """
        # 简化的滑点估算
        # 实际应该基于订单簿深度
        base_slippage = Decimal("0.001")  # 0.1%

        # 根据数量调整
        if amount > 10:  # 大额订单
            slippage = base_slippage * Decimal("3")
        elif amount > 1:
            slippage = base_slippage * Decimal("1.5")
        else:
            slippage = base_slippage

        return slippage

    async def get_price_change(self, symbol: str, period: str = "24h") -> Dict[str, Decimal]:
        """
        获取价格变化

        Args:
            symbol: 交易对
            period: 时间周期

        Returns:
            价格变化信息
        """
        try:
            ticker = await self.client.get_ticker(symbol)

            return {
                "current": Decimal(str(ticker.last)),
                "change": Decimal(str(ticker.change)),
                "change_percent": Decimal(str(ticker.changePercentage)),
                "high": Decimal(str(ticker.high)),
                "low": Decimal(str(ticker.low)),
                "volume": Decimal(str(ticker.baseVolume))
            }
        except Exception as e:
            logger.error(f"获取价格变化失败 {symbol}: {e}")
            return {}

    def calculate_position_value(
        self,
        symbol: str,
        amount: float,
        price: Optional[Decimal] = None
    ) -> Decimal:
        """
        计算持仓价值

        Args:
            symbol: 交易对
            amount: 数量
            price: 价格（可选）

        Returns:
            持仓价值
        """
        if price is None:
            if symbol in self.price_cache:
                price = self.price_cache[symbol][0]
            else:
                logger.warning(f"没有 {symbol} 的价格信息")
                return Decimal("0")

        return Decimal(str(amount)) * price

    def calculate_unrealized_pnl(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        amount: float,
        side: str
    ) -> Decimal:
        """
        计算未实现盈亏

        Args:
            entry_price: 开仓价格
            current_price: 当前价格
            amount: 数量
            side: 方向

        Returns:
            未实现盈亏
        """
        amount_decimal = Decimal(str(amount))

        if side in ["long", "buy"]:
            # 多头：(当前价格 - 开仓价格) * 数量
            return (current_price - entry_price) * amount_decimal
        else:
            # 空头：(开仓价格 - 当前价格) * 数量
            return (entry_price - current_price) * amount_decimal

    def calculate_pnl_percent(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        side: str
    ) -> Decimal:
        """
        计算盈亏百分比

        Args:
            entry_price: 开仓价格
            current_price: 当前价格
            side: 方向

        Returns:
            盈亏百分比
        """
        if side in ["long", "buy"]:
            return ((current_price - entry_price) / entry_price) * Decimal("100")
        else:
            return ((entry_price - current_price) / entry_price) * Decimal("100")

    async def get_order_book_summary(self, symbol: str) -> Dict[str, Any]:
        """
        获取订单簿摘要（模拟）

        Args:
            symbol: 交易对

        Returns:
            订单簿摘要
        """
        try:
            ticker = await self.client.get_ticker(symbol)

            return {
                "symbol": symbol,
                "best_bid": Decimal(str(ticker.bid)) if hasattr(ticker, 'bid') else None,
                "best_ask": Decimal(str(ticker.ask)) if hasattr(ticker, 'ask') else None,
                "spread": None,  # 需要计算ask - bid
                "spread_percent": None,
                "bid_volume": None,
                "ask_volume": None
            }
        except Exception as e:
            logger.error(f"获取订单簿摘要失败 {symbol}: {e}")
            return {}