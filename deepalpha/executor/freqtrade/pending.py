"""待处理订单管理
对应Go版本的 pending.go
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass
import uuid

from deepalpha.executor.freqtrade.types import FreqtradeOrder
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class PendingOrderStatus(Enum):
    """待处理订单状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PendingOrder:
    """待处理订单"""
    id: str
    pair: str
    ordertype: str
    side: str
    amount: float
    price: Optional[float]
    status: PendingOrderStatus
    created_at: datetime
    updated_at: datetime
    retries: int = 0
    max_retries: int = 3
    timeout: int = 30  # 秒
    metadata: Dict[str, Any] = None
    callback: Optional[Callable] = None
    error: Optional[str] = None
    ft_order_id: Optional[str] = None


class PendingOrderManager:
    """待处理订单管理器"""

    def __init__(self):
        self.orders: Dict[str, PendingOrder] = {}
        self.processing = False
        self.callbacks: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()

    async def add_order(
        self,
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        timeout: int = 30,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """
        添加待处理订单

        Args:
            pair: 交易对
            ordertype: 订单类型
            side: 方向
            amount: 数量
            price: 价格
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
            metadata: 元数据
            callback: 回调函数

        Returns:
            订单ID
        """
        order_id = str(uuid.uuid4())
        order = PendingOrder(
            id=order_id,
            pair=pair,
            ordertype=ordertype,
            side=side,
            amount=amount,
            price=price,
            status=PendingOrderStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            timeout=timeout,
            max_retries=max_retries,
            metadata=metadata or {},
            callback=callback
        )

        async with self._lock:
            self.orders[order_id] = order

        logger.info(f"添加待处理订单: {order_id} - {pair} {side} {amount}")
        return order_id

    async def get_order(self, order_id: str) -> Optional[PendingOrder]:
        """获取待处理订单"""
        async with self._lock:
            return self.orders.get(order_id)

    async def get_orders_by_status(self, status: PendingOrderStatus) -> List[PendingOrder]:
        """根据状态获取订单列表"""
        async with self._lock:
            return [order for order in self.orders.values() if order.status == status]

    async def get_pending_orders(self) -> List[PendingOrder]:
        """获取所有待处理的订单"""
        return await self.get_orders_by_status(PendingOrderStatus.PENDING)

    async def update_order_status(
        self,
        order_id: str,
        status: PendingOrderStatus,
        ft_order_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        更新订单状态

        Args:
            order_id: 订单ID
            status: 新状态
            ft_order_id: Freqtrade订单ID
            error: 错误信息

        Returns:
            是否成功
        """
        async with self._lock:
            order = self.orders.get(order_id)
            if not order:
                logger.warning(f"订单不存在: {order_id}")
                return False

            old_status = order.status
            order.status = status
            order.updated_at = datetime.utcnow()

            if ft_order_id:
                order.ft_order_id = ft_order_id

            if error:
                order.error = error

            # 调用状态变更回调
            await self._call_status_callbacks(order_id, old_status, status)

            # 执行订单回调
            if status in [PendingOrderStatus.COMPLETED, PendingOrderStatus.FAILED] and order.callback:
                try:
                    if asyncio.iscoroutinefunction(order.callback):
                        await order.callback(order)
                    else:
                        order.callback(order)
                except Exception as e:
                    logger.error(f"执行订单回调失败 {order_id}: {e}")

            logger.info(f"更新订单状态: {order_id} {old_status.value} -> {status.value}")
            return True

    async def remove_order(self, order_id: str) -> bool:
        """移除订单"""
        async with self._lock:
            if order_id in self.orders:
                del self.orders[order_id]
                logger.info(f"移除订单: {order_id}")
                return True
            return False

    async def cancel_order(self, order_id: str, reason: str = "User cancelled") -> bool:
        """取消订单"""
        return await self.update_order_status(
            order_id,
            PendingOrderStatus.CANCELLED,
            error=reason
        )

    async def start_processing(self, processor: Callable[[PendingOrder], FreqtradeOrder]):
        """
        开始处理订单

        Args:
            processor: 处理函数
        """
        if self.processing:
            logger.warning("订单处理器已在运行")
            return

        self.processing = True
        logger.info("开始处理待处理订单")

        try:
            while self.processing:
                # 获取待处理的订单
                pending_orders = await self.get_pending_orders()

                if not pending_orders:
                    await asyncio.sleep(1)
                    continue

                # 处理每个订单
                for order in pending_orders:
                    try:
                        # 检查超时
                        if datetime.utcnow() - order.updated_at > timedelta(seconds=order.timeout):
                            await self.update_order_status(
                                order.id,
                                PendingOrderStatus.FAILED,
                                error="Order timeout"
                            )
                            continue

                        # 标记为处理中
                        await self.update_order_status(order.id, PendingOrderStatus.PROCESSING)

                        # 调用处理函数
                        ft_order = await self._process_with_retry(order, processor)

                        if ft_order and ft_order.get("id"):
                            # 处理成功
                            await self.update_order_status(
                                order.id,
                                PendingOrderStatus.COMPLETED,
                                ft_order_id=ft_order["id"]
                            )
                        else:
                            # 处理失败
                            await self.update_order_status(
                                order.id,
                                PendingOrderStatus.FAILED,
                                error="Failed to create order"
                            )

                    except Exception as e:
                        logger.error(f"处理订单失败 {order.id}: {e}")
                        await self.update_order_status(
                            order.id,
                            PendingOrderStatus.FAILED,
                            error=str(e)
                        )

                await asyncio.sleep(0.1)  # 避免忙等待

        except Exception as e:
            logger.error(f"订单处理器异常: {e}")
        finally:
            self.processing = False
            logger.info("订单处理器停止")

    async def _process_with_retry(
        self,
        order: PendingOrder,
        processor: Callable[[PendingOrder], FreqtradeOrder]
    ) -> Optional[FreqtradeOrder]:
        """带重试的订单处理"""
        for attempt in range(order.max_retries):
            try:
                result = await processor(order)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"订单处理失败 (尝试 {attempt + 1}/{order.max_retries}): {e}")

                if attempt < order.max_retries - 1:
                    # 指数退避
                    await asyncio.sleep(2 ** attempt)
                    order.retries += 1
                    order.updated_at = datetime.utcnow()
                    await self.update_order_status(
                        order.id,
                        PendingOrderStatus.PENDING,
                        error=f"Retry {attempt + 1}: {str(e)}"
                    )

        return None

    def stop_processing(self):
        """停止处理"""
        self.processing = False

    async def add_status_callback(self, order_id: str, callback: Callable):
        """添加状态变更回调"""
        if order_id not in self.callbacks:
            self.callbacks[order_id] = []
        self.callbacks[order_id].append(callback)

    async def remove_status_callback(self, order_id: str, callback: Callable):
        """移除状态变更回调"""
        if order_id in self.callbacks:
            try:
                self.callbacks[order_id].remove(callback)
            except ValueError:
                pass

    async def _call_status_callbacks(self, order_id: str, old_status: PendingOrderStatus, new_status: PendingOrderStatus):
        """调用状态变更回调"""
        if order_id in self.callbacks:
            for callback in self.callbacks[order_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(order_id, old_status, new_status)
                    else:
                        callback(order_id, old_status, new_status)
                except Exception as e:
                    logger.error(f"状态回调失败 {order_id}: {e}")

    async def cleanup_old_orders(self, hours: int = 24):
        """清理旧订单"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        async with self._lock:
            to_remove = []
            for order_id, order in self.orders.items():
                if order.status in [PendingOrderStatus.COMPLETED, PendingOrderStatus.FAILED, PendingOrderStatus.CANCELLED]:
                    if order.updated_at < cutoff_time:
                        to_remove.append(order_id)

            for order_id in to_remove:
                del self.orders[order_id]

        if to_remove:
            logger.info(f"清理了 {len(to_remove)} 个旧订单")

    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        async with self._lock:
            stats = {
                "total": len(self.orders),
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "processing_running": self.processing
            }

            for order in self.orders.values():
                if order.status == PendingOrderStatus.PENDING:
                    stats["pending"] += 1
                elif order.status == PendingOrderStatus.PROCESSING:
                    stats["processing"] += 1
                elif order.status == PendingOrderStatus.COMPLETED:
                    stats["completed"] += 1
                elif order.status == PendingOrderStatus.FAILED:
                    stats["failed"] += 1
                elif order.status == PendingOrderStatus.CANCELLED:
                    stats["cancelled"] += 1

            return stats

    async def export_orders(self, status: Optional[PendingOrderStatus] = None) -> List[Dict]:
        """导出订单信息"""
        async with self._lock:
            orders = []
            for order in self.orders.values():
                if status is None or order.status == status:
                    orders.append({
                        "id": order.id,
                        "pair": order.pair,
                        "ordertype": order.ordertype,
                        "side": order.side,
                        "amount": order.amount,
                        "price": order.price,
                        "status": order.status.value,
                        "created_at": order.created_at.isoformat(),
                        "updated_at": order.updated_at.isoformat(),
                        "retries": order.retries,
                        "ft_order_id": order.ft_order_id,
                        "error": order.error,
                        "metadata": order.metadata
                    })
            return orders