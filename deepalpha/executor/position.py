"""仓位管理，对应 Go 版本中持仓相关的管理逻辑"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from deepalpha.core.models import Order, OrderSide, OrderStatus, Position
from deepalpha.executor.freqtrade.types import APIPosition, TierInfo
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class PositionManager:
    """仓位管理器"""

    def __init__(self):
        self._positions: Dict[str, Position] = {}  # symbol -> Position
        self._trade_id_map: Dict[int, str] = {}  # trade_id -> symbol
        self._lock = asyncio.Lock()

    async def update_from_api(self, api_positions: List[APIPosition]) -> None:
        """从API更新持仓，对应 Go 版本的持仓同步逻辑"""
        async with self._lock:
            # 清除旧映射
            self._positions.clear()
            self._trade_id_map.clear()

            # 转换API持仓
            for api_pos in api_positions:
                if api_pos.status == "closed":
                    continue  # 跳过已平仓的

                # 转换为Position模型
                position = Position(
                    symbol=api_pos.symbol,
                    side=OrderSide(api_pos.side),
                    size=Decimal(str(api_pos.amount)),
                    entry_price=Decimal(str(api_pos.entry_price)),
                    current_price=Decimal(str(api_pos.current_price)) if api_pos.current_price else None,
                    unrealized_pnl=Decimal(str(api_pos.unrealized_pnl_usd)) if api_pos.unrealized_pnl_usd else Decimal('0'),
                    realized_pnl=Decimal(str(api_pos.realized_pnl_usd)) if api_pos.realized_pnl_usd else Decimal('0'),
                )

                # 设置层级信息
                if api_pos.tier1.target > 0 or api_pos.tier1.ratio > 0:
                    if not position.metadata:
                        position.metadata = {}
                    position.metadata["tier1"] = {
                        "target": api_pos.tier1.target,
                        "ratio": api_pos.tier1.ratio,
                        "done": api_pos.tier1.done
                    }

                if api_pos.tier2.target > 0 or api_pos.tier2.ratio > 0:
                    if not position.metadata:
                        position.metadata = {}
                    position.metadata["tier2"] = {
                        "target": api_pos.tier2.target,
                        "ratio": api_pos.tier2.ratio,
                        "done": api_pos.tier2.done
                    }

                if api_pos.tier3.target > 0 or api_pos.tier3.ratio > 0:
                    if not position.metadata:
                        position.metadata = {}
                    position.metadata["tier3"] = {
                        "target": api_pos.tier3.target,
                        "ratio": api_pos.tier3.ratio,
                        "done": api_pos.tier3.done
                    }

                self._positions[api_pos.symbol] = position
                self._trade_id_map[api_pos.trade_id] = api_pos.symbol

            logger.info(f"更新持仓数量: {len(self._positions)}")

    async def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定符号的持仓"""
        async with self._lock:
            return self._positions.get(symbol)

    async def get_position_by_trade_id(self, trade_id: int) -> Optional[Position]:
        """通过trade_id获取持仓"""
        async with self._lock:
            symbol = self._trade_id_map.get(trade_id)
            if symbol:
                return self._positions.get(symbol)
            return None

    async def get_all_positions(self) -> Dict[str, Position]:
        """获取所有持仓"""
        async with self._lock:
            return self._positions.copy()

    async def update_position(
        self,
        symbol: str,
        current_price: Optional[float] = None,
        unrealized_pnl: Optional[float] = None
    ) -> None:
        """更新持仓信息"""
        async with self._lock:
            position = self._positions.get(symbol)
            if position:
                if current_price is not None:
                    position.current_price = Decimal(str(current_price))
                if unrealized_pnl is not None:
                    position.unrealized_pnl = Decimal(str(unrealized_pnl))
                position.timestamp = datetime.utcnow()

    async def close_position(
        self,
        symbol: str,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
        realized_pnl: Optional[float] = None
    ) -> None:
        """平仓"""
        async with self._lock:
            position = self._positions.get(symbol)
            if position:
                if exit_price is not None:
                    position.current_price = Decimal(str(exit_price))
                if realized_pnl is not None:
                    position.realized_pnl = Decimal(str(realized_pnl))
                if exit_reason:
                    position.metadata = position.metadata or {}
                    position.metadata["exit_reason"] = exit_reason

                # 从持仓列表中移除
                del self._positions[symbol]

                # 从trade_id映射中移除
                to_remove = [
                    trade_id for trade_id, sym in self._trade_id_map.items()
                    if sym == symbol
                ]
                for trade_id in to_remove:
                    del self._trade_id_map[trade_id]

                logger.info(f"平仓: {symbol}, 原因: {exit_reason}")

    async def update_tier(
        self,
        trade_id: int,
        tier: int,
        target: Optional[float] = None,
        ratio: Optional[float] = None,
        done: Optional[bool] = None
    ) -> None:
        """更新层级信息"""
        async with self._lock:
            position = await self.get_position_by_trade_id(trade_id)
            if not position:
                return

            if not position.metadata:
                position.metadata = {}

            tier_key = f"tier{tier}"
            tier_info = position.metadata.get(tier_key, {})

            if target is not None:
                tier_info["target"] = target
            if ratio is not None:
                tier_info["ratio"] = ratio
            if done is not None:
                tier_info["done"] = done

            position.metadata[tier_key] = tier_info
            logger.info(f"更新层级: trade_id={trade_id}, tier={tier}")

    async def get_total_exposure(self) -> Decimal:
        """计算总敞口"""
        async with self._lock:
            total = Decimal('0')
            for position in self._positions.values():
                total += position.size * (position.current_price or position.entry_price)
            return total

    async def get_pnl_summary(self) -> Dict[str, Any]:
        """获取盈亏汇总"""
        async with self._lock:
            total_unrealized = Decimal('0')
            total_realized = Decimal('0')
            profit_count = 0
            loss_count = 0

            for position in self._positions.values():
                if position.unrealized_pnl > 0:
                    profit_count += 1
                elif position.unrealized_pnl < 0:
                    loss_count += 1
                total_unrealized += position.unrealized_pnl
                total_realized += position.realized_pnl

            return {
                "total_unrealized": float(total_unrealized),
                "total_realized": float(total_realized),
                "total_pnl": float(total_unrealized + total_realized),
                "profit_positions": profit_count,
                "loss_positions": loss_count,
                "total_positions": len(self._positions)
            }

    async def get_symbol_pnl(self, symbol: str) -> Dict[str, Any]:
        """获取指定符号的盈亏信息"""
        position = await self.get_position(symbol)
        if not position:
            return {}

        return {
            "symbol": symbol,
            "unrealized_pnl": float(position.unrealized_pnl),
            "realized_pnl": float(position.realized_pnl),
            "total_pnl": float(position.unrealized_pnl + position.realized_pnl),
            "pnl_percent": float(
                ((position.current_price - position.entry_price) / position.entry_price * 100)
                if position.entry_price > 0 else 0
            )
        }

    async def check_risk_limits(
        self,
        max_exposure: float,
        max_drawdown: float,
        current_balance: float
    ) -> Dict[str, Any]:
        """检查风险限制"""
        total_exposure = await self.get_total_exposure()
        exposure_ratio = float(total_exposure / current_balance) if current_balance > 0 else 0

        # 计算当前回撤（简化实现）
        pnl_summary = await self.get_pnl_summary()
        drawdown = abs(min(pnl_summary["total_pnl"], 0)) / current_balance if current_balance > 0 else 0

        return {
            "total_exposure": float(total_exposure),
            "exposure_ratio": exposure_ratio,
            "max_exceeded": total_exposure > max_exposure,
            "drawdown": drawdown,
            "drawdown_exceeded": drawdown > max_drawdown,
            "risk_level": "HIGH" if exposure_ratio > 0.5 or drawdown > 0.1 else "NORMAL"
        }