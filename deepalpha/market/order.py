"""市场订单管理，对应 Go 版本的 order.go"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from deepalpha.core.models import Order, OrderSide, OrderStatus, OrderType
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class OrderManager:
    """订单管理器"""

    def __init__(self):
        self._orders: Dict[str, Order] = {}
        self._client_order_ids: Dict[str, str] = {}  # client_order_id -> order_id
        self._lock = asyncio.Lock()

    async def add_order(self, order: Order) -> bool:
        """添加订单"""
        async with self._lock:
            if order.id and order.id in self._orders:
                logger.warning(f"订单已存在: {order.id}")
                return False

            if order.client_order_id and order.client_order_id in self._client_order_ids:
                logger.warning(f"客户端订单ID已存在: {order.client_order_id}")
                return False

            if order.id:
                self._orders[order.id] = order

            if order.client_order_id:
                self._client_order_ids[order.client_order_id] = order.id or ""

            logger.info(f"添加订单: {order.id or order.client_order_id}")
            return True

    async def update_order(self, order: Order) -> bool:
        """更新订单"""
        async with self._lock:
            if not order.id and not order.client_order_id:
                logger.error("更新订单需要提供ID或客户端订单ID")
                return False

            # 查找订单
            existing_order = None
            if order.id and order.id in self._orders:
                existing_order = self._orders[order.id]
            elif order.client_order_id and order.client_order_id in self._client_order_ids:
                order_id = self._client_order_ids[order.client_order_id]
                if order_id:
                    existing_order = self._orders.get(order_id)

            if not existing_order:
                logger.warning(f"订单不存在: {order.id or order.client_order_id}")
                return False

            # 更新字段
            if order.status != existing_order.status:
                logger.info(f"订单状态更新: {existing_order.id} {existing_order.status} -> {order.status}")

            update_fields = [
                'status', 'filled', 'remaining', 'average_price',
                'fee', 'fee_currency', 'trades'
            ]

            for field in update_fields:
                if getattr(order, field) is not None:
                    setattr(existing_order, field, getattr(order, field))

            # 更新时间戳
            existing_order.timestamp = datetime.utcnow()

            return True

    async def get_order(self, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> Optional[Order]:
        """获取订单"""
        async with self._lock:
            if order_id:
                return self._orders.get(order_id)
            elif client_order_id:
                order_id = self._client_order_ids.get(client_order_id)
                if order_id:
                    return self._orders.get(order_id)
            return None

    async def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """获取指定交易对的订单"""
        async with self._lock:
            symbol = symbol.upper()
            return [order for order in self._orders.values() if order.symbol == symbol]

    async def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """获取指定状态的订单"""
        async with self._lock:
            return [order for order in self._orders.values() if order.status == status]

    async def get_active_orders(self) -> List[Order]:
        """获取活跃订单"""
        async with self._lock:
            return [
                order for order in self._orders.values()
                if order.is_active
            ]

    async def get_all_orders(self) -> List[Order]:
        """获取所有订单"""
        async with self._lock:
            return list(self._orders.values())

    async def cancel_order(self, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> bool:
        """取消订单"""
        async with self._lock:
            order = await self.get_order(order_id, client_order_id)
            if not order:
                logger.warning(f"取消订单失败，订单不存在: {order_id or client_order_id}")
                return False

            if order.status in [OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED]:
                logger.warning(f"订单无法取消，当前状态: {order.status}")
                return False

            order.status = OrderStatus.CANCELLED
            order.timestamp = datetime.utcnow()

            logger.info(f"订单已取消: {order.id}")
            return True

    async def remove_order(self, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> bool:
        """移除订单记录"""
        async with self._lock:
            order = await self.get_order(order_id, client_order_id)
            if not order:
                return False

            # 从字典中移除
            if order.id and order.id in self._orders:
                del self._orders[order.id]

            if order.client_order_id and order.client_order_id in self._client_order_ids:
                del self._client_order_ids[order.client_order_id]

            logger.info(f"订单记录已移除: {order.id or client_order_id}")
            return True

    async def cleanup_completed_orders(self, days: int = 30) -> int:
        """清理已完成的旧订单"""
        async with self._lock:
            cutoff_time = datetime.utcnow().timestamp() - (days * 24 * 3600 * 1000)  # 毫秒
            to_remove = []

            for order_id, order in self._orders.items():
                if (order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED] and
                    order.timestamp.timestamp() * 1000 < cutoff_time):
                    to_remove.append(order_id)

            for order_id in to_remove:
                order = self._orders[order_id]
                if order.client_order_id and order.client_order_id in self._client_order_ids:
                    del self._client_order_ids[order.client_order_id]
                del self._orders[order_id]

            logger.info(f"清理了 {len(to_remove)} 个旧订单")
            return len(to_remove)

    async def get_statistics(self) -> Dict[str, int]:
        """获取订单统计信息"""
        async with self._lock:
            stats = {
                'total': len(self._orders),
                'pending': 0,
                'open': 0,
                'filled': 0,
                'cancelled': 0,
                'rejected': 0
            }

            for order in self._orders.values():
                stats[order.status.value] += 1

            return stats