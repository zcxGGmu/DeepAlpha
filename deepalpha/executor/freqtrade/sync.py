"""Freqtrade同步逻辑
对应Go版本的 sync.go
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_

from deepalpha.executor.freqtrade.client import FreqtradeClient
from deepalpha.executor.freqtrade.types import (
    FreqtradeStatus, FreqtradeTrade, FreqtradePosition,
    FreqtradeOrder, FreqtradeBalance
)
from deepalpha.db.models import (
    LiveOrder, LiveOrderLog, OrderStatus, OrderSide, OrderType
)
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class FreqtradeSync:
    """Freqtrade同步管理器"""

    def __init__(self, client: FreqtradeClient, db: Session):
        self.client = client
        self.db = db
        self.last_sync_time = None
        self.sync_interval = 30  # 30秒同步一次

    async def start_sync(self):
        """启动同步任务"""
        logger.info("启动Freqtrade同步任务")

        while True:
            try:
                await self.sync_all()
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"同步失败: {e}")
                await asyncio.sleep(5)  # 错误后等待5秒再重试

    async def sync_all(self):
        """同步所有数据"""
        logger.debug("开始同步Freqtrade数据")

        # 同步订单状态
        await self.sync_orders()

        # 同步持仓
        await self.sync_positions()

        # 同步交易记录
        await self.sync_trades()

        # 同步余额
        await self.sync_balance()

        self.last_sync_time = datetime.utcnow()
        logger.debug("Freqtrade数据同步完成")

    async def sync_orders(self):
        """同步订单状态"""
        try:
            # 获取Freqtrade订单
            ft_orders = await self.client.get_orders()

            # 获取本地订单
            local_orders = self.db.query(LiveOrder).filter(
                LiveOrder.ft_order_id.isnot(None)
            ).all()

            # 创建订单ID映射
            local_orders_map = {order.ft_order_id: order for order in local_orders}

            for ft_order in ft_orders:
                local_order = local_orders_map.get(ft_order.id)

                if local_order:
                    # 检查状态是否需要更新
                    new_status = self._convert_order_status(ft_order.status)
                    if local_order.status != new_status:
                        # 记录状态变更
                        self._create_order_log(
                            local_order,
                            "status_update",
                            f"状态从 {local_order.status.value} 更新为 {new_status.value}",
                            from_status=local_order.status,
                            to_status=new_status
                        )

                        # 更新订单信息
                        local_order.status = new_status
                        local_order.filled_amount = ft_order.filled or 0
                        local_order.filled_price = ft_order.average or ft_order.price
                        local_order.updated_at = datetime.utcnow()

                        # 如果订单完成，设置关闭时间
                        if new_status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                            local_order.closed_at = datetime.utcnow()

                        logger.info(f"更新订单状态: {ft_order.id} -> {new_status.value}")

            self.db.commit()

        except Exception as e:
            self.db.rollback()
            logger.error(f"同步订单失败: {e}")
            raise

    async def sync_positions(self):
        """同步持仓信息"""
        try:
            # 获取Freqtrade持仓
            ft_positions = await self.client.get_positions()

            # 获取本地持仓（已成交的订单）
            local_positions = self.db.query(LiveOrder).filter(
                LiveOrder.status == OrderStatus.FILLED,
                LiveOrder.closed_at.is_(None)
            ).all()

            # 更新持仓信息
            for ft_pos in ft_positions:
                # 查找对应的本地订单
                local_order = self.db.query(LiveOrder).filter(
                    LiveOrder.symbol == ft_pos.symbol,
                    LiveOrder.side == self._convert_position_side(ft_pos.side),
                    LiveOrder.status == OrderStatus.FILLED,
                    LiveOrder.closed_at.is_(None)
                ).first()

                if local_order:
                    # 计算未实现盈亏
                    unrealized_pnl = self._calculate_unrealized_pnl(local_order, ft_pos)

                    # 更新订单的盈亏信息
                    local_order.realized_pnl = unrealized_pnl
                    local_order.updated_at = datetime.utcnow()

                    logger.debug(f"更新持仓盈亏: {ft_pos.symbol} = ${unrealized_pnl:.2f}")

            self.db.commit()

        except Exception as e:
            self.db.rollback()
            logger.error(f"同步持仓失败: {e}")
            raise

    async def sync_trades(self):
        """同步交易记录"""
        try:
            # 获取最近的交易记录
            since = self.last_sync_time - timedelta(hours=1) if self.last_sync_time else None
            ft_trades = await self.client.get_trades(limit=100, since=since)

            # 这里可以将交易记录保存到本地
            # 实际实现可能需要专门的TradeLog表

            logger.debug(f"同步交易记录: {len(ft_trades)} 条")

        except Exception as e:
            logger.error(f"同步交易记录失败: {e}")

    async def sync_balance(self):
        """同步账户余额"""
        try:
            # 获取余额信息
            balances = await self.client.get_balance()

            # 这里可以保存余额信息到本地
            # 用于监控和风险控制

            total_usd = sum(
                b.free + b.used
                for b in balances
                if b.currency in ["USDT", "USD", "BUSD"]
            )

            logger.debug(f"账户总余额: ${total_usd:.2f}")

        except Exception as e:
            logger.error(f"同步余额失败: {e}")

    def _convert_order_status(self, ft_status: str) -> OrderStatus:
        """转换Freqtrade订单状态到本地状态"""
        status_map = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
        }
        return status_map.get(ft_status.lower(), OrderStatus.OPEN)

    def _convert_position_side(self, ft_side: str) -> OrderSide:
        """转换Freqtrade持仓方向到本地方向"""
        if ft_side == "long":
            return OrderSide.LONG
        elif ft_side == "short":
            return OrderSide.SHORT
        else:
            return OrderSide.BUY if ft_side == "buy" else OrderSide.SELL

    def _calculate_unrealized_pnl(self, order: LiveOrder, position: FreqtradePosition) -> float:
        """计算未实现盈亏"""
        if not position.mark_price:
            return order.realized_pnl or 0

        entry_price = order.entry_price or order.price
        current_price = position.mark_price
        filled_amount = order.filled_amount

        if order.side in [OrderSide.BUY, OrderSide.LONG]:
            # 多头: (当前价格 - 入场价格) * 数量
            return (current_price - entry_price) * filled_amount
        else:
            # 空头: (入场价格 - 当前价格) * 数量
            return (entry_price - current_price) * filled_amount

    def _create_order_log(
        self,
        order: LiveOrder,
        action: str,
        message: str,
        from_status: Optional[OrderStatus] = None,
        to_status: Optional[OrderStatus] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """创建订单日志"""
        log = LiveOrderLog(
            order_id=order.id,
            action=action,
            message=message,
            from_status=from_status,
            to_status=to_status,
            metadata=metadata or {}
        )
        self.db.add(log)

    async def force_sync(self):
        """强制同步"""
        logger.info("强制同步Freqtrade数据")
        await self.sync_all()

    async def get_sync_status(self) -> Dict[str, Any]:
        """获取同步状态"""
        return {
            "last_sync": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "sync_interval": self.sync_interval,
            "freqtrade_status": await self._check_freqtrade_status()
        }

    async def _check_freqtrade_status(self) -> Dict[str, Any]:
        """检查Freqtrade状态"""
        try:
            status = await self.client.get_status()
            return {
                "state": status.state,
                "running": status.state == "running",
                "strategy": getattr(status, "strategy", None),
                "bot_name": getattr(status, "bot_name", None)
            }
        except Exception as e:
            return {
                "state": "error",
                "running": False,
                "error": str(e)
            }