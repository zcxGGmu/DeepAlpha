"""风险管理模块，对应 Go 版本中的风险控制逻辑"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from deepalpha.core.models import Position, Order, OrderSide, OrderStatus
from deepalpha.decision.types import Decision, DecisionTiers
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class RiskManager:
    """风险管理器"""

    def __init__(
        self,
        max_position_size: float = 0.02,  # 最大仓位比例
        max_total_position: float = 0.5,  # 总仓位上限
        stop_loss: float = 0.02,  # 止损比例
        take_profit: float = 0.06,  # 止盈比例
        max_drawdown: float = 0.10,  # 最大回撤
        max_trades_per_day: int = 10,  # 每日最大交易数
        min_profit_ratio: float = 1.5,  # 最小盈亏比
    ):
        self.max_position_size = max_position_size
        self.max_total_position = max_total_position
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_drawdown = max_drawdown
        self.max_trades_per_day = max_trades_per_day
        self.min_profit_ratio = min_profit_ratio

        # 记录当日交易
        self._daily_trades: List[datetime] = []
        self._daily_pnl: Decimal = Decimal('0')
        self._peak_balance: Optional[Decimal] = None

    async def check_decision_risk(
        self,
        decision: Decision,
        current_price: float,
        positions: Dict[str, Position],
        current_balance: float
    ) -> Tuple[bool, str]:
        """检查决策风险"""
        # 检查日交易限制
        if not await self._check_daily_limit():
            return False, "超过每日最大交易数限制"

        # 检查仓位大小
        if decision.position_size_usd:
            required_size = decision.position_size_usd / current_balance
            if required_size > self.max_position_size:
                return False, f"仓位大小超限: {required_size:.2%} > {self.max_position_size:.2%}"

        # 检查总仓位
        total_exposure = sum(
            abs(pos.size) * (pos.current_price or pos.entry_price)
            for pos in positions.values()
        )
        total_ratio = total_exposure / current_balance if current_balance > 0 else 0

        if decision.action in ["enter_long", "enter_short"] and decision.position_size_usd:
            new_total = total_exposure + decision.position_size_usd
            new_ratio = new_total / current_balance
            if new_ratio > self.max_total_position:
                return False, f"总仓位超限: {new_ratio:.2%} > {self.max_total_position:.2%}"

        # 检查止损止盈设置
        if decision.action in ["enter_long", "enter_short"]:
            if not decision.stop_loss and not decision.take_profit:
                return False, "必须设置止损或止盈"

            if decision.stop_loss:
                expected_loss = abs(decision.stop_loss - current_price) / current_price
                if expected_loss < self.stop_loss * 0.5:  # 至少设置最小止损的一半
                    return False, f"止损设置过小: {expected_loss:.2%}"

            if decision.take_profit and decision.stop_loss:
                profit_ratio = abs(decision.take_profit - decision.stop_loss) / abs(decision.stop_loss - current_price)
                if profit_ratio < self.min_profit_ratio:
                    return False, f"盈亏比过低: {profit_ratio:.2f} < {self.min_profit_ratio:.2f}"

        # 检查回撤
        drawdown_check = await self._check_drawdown(current_balance)
        if not drawdown_check[0]:
            return False, f"超过最大回撤限制: {drawdown_check[1]}"

        return True, "风险检查通过"

    async def check_order_risk(
        self,
        order: Order,
        current_price: float,
        positions: Dict[str, Position],
        current_balance: float
    ) -> Tuple[bool, str]:
        """检查订单风险"""
        # 检查订单大小
        if order.amount * order.price > current_balance * 0.95:  # 使用95%的余额
            return False, "订单金额过大"

        # 检查流动性
        symbol_positions = [pos for pos in positions.values() if pos.symbol == order.symbol]
        if symbol_positions:
            current_exposure = sum(pos.size for pos in symbol_positions)
            # 简单的流动性检查
            if (order.side == OrderSide.BUY and current_exposure > 0) or \
               (order.side == OrderSide.SELL and current_exposure < 0):
                return False, "增加现有方向的风险敞口"

        return True, "订单风险检查通过"

    def calculate_position_size(
        self,
        symbol: str,
        confidence: int,
        current_price: float,
        balance: float,
        atr: Optional[float] = None
    ) -> float:
        """计算仓位大小"""
        # 基于置信度的仓位调整
        confidence_factor = confidence / 100.0
        base_size = balance * self.max_position_size

        # 根据ATR调整（如果有）
        if atr:
            # 使用ATR作为波动率指标调整仓位
            volatility_factor = min(1.0, 0.01 / (atr / current_price))
            base_size *= volatility_factor

        # 应用置信度因子
        position_size = base_size * confidence_factor

        # 确保不超过最大限制
        max_size = balance * self.max_position_size * 0.95  # 留5%缓冲
        position_size = min(position_size, max_size)

        return max(0, position_size)

    def calculate_stop_loss(
        self,
        side: OrderSide,
        entry_price: float,
        atr: Optional[float] = None,
        lookback_periods: int = 14
    ) -> float:
        """计算止损价格"""
        if atr:
            # 使用ATR的2倍作为止损距离
            stop_distance = atr * 2
        else:
            # 使用固定百分比
            stop_distance = entry_price * self.stop_loss

        if side == OrderSide.BUY:
            return entry_price - stop_distance
        else:  # SELL
            return entry_price + stop_distance

    def calculate_take_profit(
        self,
        side: OrderSide,
        entry_price: float,
        stop_loss: float,
        tiers: Optional[DecisionTiers] = None
    ) -> float:
        """计算止盈价格"""
        # 检查层级设置
        if tiers and tiers.tier1_target and tiers.tier1_target > 0:
            return tiers.tier1_target

        # 根据止损计算止盈，保证最小盈亏比
        min_profit = abs(entry_price - stop_loss) * self.min_profit_ratio

        if side == OrderSide.BUY:
            return entry_price + min_profit
        else:  # SELL
            return entry_price - min_profit

    def check_tier_executability(
        self,
        position: Position,
        current_price: float,
        tier_info: Dict[str, Any]
    ) -> bool:
        """检查层级是否可以执行"""
        if "target" not in tier_info or "ratio" not in tier_info:
            return False

        target = tier_info["target"]
        ratio = tier_info["ratio"]
        done = tier_info.get("done", False)

        if done:
            return False

        # 检查是否达到目标价格
        if position.side == OrderSide.BUY:
            if current_price >= target:
                return True
        else:  # SELL
            if current_price <= target:
                return True

        return False

    async def _check_daily_limit(self) -> bool:
        """检查每日交易限制"""
        today = datetime.utcnow().date()

        # 清理旧记录
        self._daily_trades = [
            trade_time for trade_time in self._daily_trades
            if trade_time.date() == today
        ]

        return len(self._daily_trades) < self.max_trades_per_day

    async def _check_drawdown(self, current_balance: float) -> Tuple[bool, str]:
        """检查回撤"""
        if self._peak_balance is None:
            self._peak_balance = Decimal(str(current_balance))
            return True, ""

        if current_balance > self._peak_balance:
            self._peak_balance = Decimal(str(current_balance))
            return True, ""

        drawdown = (self._peak_balance - Decimal(str(current_balance))) / self._peak_balance
        if drawdown > self.max_drawdown:
            return False, f"回撤: {drawdown:.2%} > {self.max_drawdown:.2%}"

        return True, ""

    def record_trade(self, pnl: float):
        """记录交易"""
        self._daily_trades.append(datetime.utcnow())
        self._daily_pnl += Decimal(str(pnl))

    async def get_risk_metrics(self) -> Dict[str, Any]:
        """获取风险指标"""
        today = datetime.utcnow().date()

        # 清理今日记录
        self._daily_trades = [
            trade_time for trade_time in self._daily_trades
            if trade_time.date() == today
        ]

        return {
            "max_position_size": self.max_position_size,
            "max_total_position": self.max_total_position,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "max_drawdown": self.max_drawdown,
            "max_trades_per_day": self.max_trades_per_day,
            "min_profit_ratio": self.min_profit_ratio,
            "daily_trades": len(self._daily_trades),
            "daily_pnl": float(self._daily_pnl),
            "peak_balance": float(self._peak_balance) if self._peak_balance else 0,
            "current_drawdown": (
                float((self._peak_balance - Decimal('0')) / self._peak_balance)
                if self._peak_balance and self._peak_balance > 0
                else 0
            )
        }