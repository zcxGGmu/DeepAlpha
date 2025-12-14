"""Webhook处理
对应Go版本的 webhook.go
"""

from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException
from sqlalchemy.orm import Session

from deepalpha.db.models import LiveOrder, LiveOrderLog, OrderStatus, OrderSide, OrderType
from deepalpha.gateway.database import DecisionLogStore
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class FreqtradeWebhookHandler:
    """Freqtrade Webhook处理器"""

    def __init__(self, db: Session):
        self.db = db
        self.decision_store = DecisionLogStore(db)

    async def handle_webhook(self, request: Request) -> Dict[str, Any]:
        """处理Webhook请求"""
        try:
            # 获取请求数据
            data = await request.json()
            logger.info(f"收到Webhook: {data}")

            # 验证必要字段
            if not self._validate_webhook_data(data):
                raise HTTPException(status_code=400, detail="Invalid webhook data")

            # 处理不同类型的Webhook
            webhook_type = data.get("type", "unknown")

            if webhook_type == "order_fill":
                return await self._handle_order_fill(data)
            elif webhook_type == "order_canceled":
                return await self._handle_order_canceled(data)
            elif webhook_type == "position_closed":
                return await self._handle_position_closed(data)
            else:
                return await self._handle_generic_webhook(data)

        except Exception as e:
            logger.error(f"处理Webhook失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _validate_webhook_data(self, data: Dict[str, Any]) -> bool:
        """验证Webhook数据"""
        required_fields = ["timestamp", "exchange", "pair"]
        return all(field in data for field in required_fields)

    async def _handle_order_fill(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理订单成交"""
        order_id = data.get("order_id")
        if not order_id:
            raise HTTPException(status_code=400, detail="Missing order_id")

        # 查找本地订单
        order = self.db.query(LiveOrder).filter(
            LiveOrder.ft_order_id == order_id
        ).first()

        if not order:
            logger.warning(f"未找到订单: {order_id}")
            return {"status": "order_not_found"}

        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_amount = data.get("filled", order.filled_amount)
        order.filled_price = data.get("price", order.filled_price)
        order.updated_at = datetime.utcnow()

        # 记录日志
        self._create_order_log(
            order,
            "fill",
            f"订单成交: 价格={data.get('price')}, 数量={data.get('filled')}",
            metadata=data
        )

        # 如果是开仓订单，更新决策状态
        if order.decision_id:
            await self._update_decision_status(order.decision_id, "executed")

        self.db.commit()

        logger.info(f"订单成交处理完成: {order_id}")
        return {"status": "success", "order_id": order_id}

    async def _handle_order_canceled(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理订单取消"""
        order_id = data.get("order_id")
        if not order_id:
            raise HTTPException(status_code=400, detail="Missing order_id")

        # 查找本地订单
        order = self.db.query(LiveOrder).filter(
            LiveOrder.ft_order_id == order_id
        ).first()

        if not order:
            logger.warning(f"未找到订单: {order_id}")
            return {"status": "order_not_found"}

        # 更新订单状态
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        order.closed_at = datetime.utcnow()

        # 记录日志
        self._create_order_log(
            order,
            "cancel",
            f"订单取消: {data.get('reason', 'Unknown')}",
            metadata=data
        )

        # 如果是开仓订单，更新决策状态
        if order.decision_id:
            await self._update_decision_status(order.decision_id, "cancelled")

        self.db.commit()

        logger.info(f"订单取消处理完成: {order_id}")
        return {"status": "success", "order_id": order_id}

    async def _handle_position_closed(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理持仓关闭"""
        pair = data.get("pair")
        pnl = data.get("pnl", 0)
        close_price = data.get("close_price")

        if not pair:
            raise HTTPException(status_code=400, detail="Missing pair")

        # 查找持仓订单
        position = self.db.query(LiveOrder).filter(
            LiveOrder.symbol == pair,
            LiveOrder.status == OrderStatus.FILLED,
            LiveOrder.closed_at.is_(None)
        ).first()

        if not position:
            logger.warning(f"未找到持仓: {pair}")
            return {"status": "position_not_found"}

        # 更新持仓信息
        position.status = OrderStatus.CLOSED
        position.exit_price = close_price
        position.realized_pnl = pnl
        position.closed_at = datetime.utcnow()
        position.updated_at = datetime.utcnow()

        # 记录日志
        self._create_order_log(
            position,
            "position_close",
            f"持仓关闭: PnL={pnl}, 价格={close_price}",
            metadata=data
        )

        self.db.commit()

        logger.info(f"持仓关闭处理完成: {pair}, PnL={pnl}")
        return {"status": "success", "pair": pair, "pnl": pnl}

    async def _handle_generic_webhook(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理通用Webhook"""
        # 记录通用日志
        logger.info(f"通用Webhook: {data}")

        # 可以根据需要处理其他类型的Webhook
        return {"status": "received", "type": data.get("type")}

    async def _update_decision_status(self, decision_id: str, status: str) -> None:
        """更新决策状态"""
        try:
            status_map = {
                "executed": "executed",
                "cancelled": "cancelled",
                "failed": "failed"
            }

            new_status = status_map.get(status)
            if new_status:
                from deepalpha.db.models import DecisionStatus
                self.decision_store.update_status(
                    decision_id,
                    DecisionStatus(new_status),
                    f"通过Webhook更新: {status}"
                )
        except Exception as e:
            logger.error(f"更新决策状态失败: {e}")

    def _create_order_log(
        self,
        order: LiveOrder,
        action: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """创建订单日志"""
        log = LiveOrderLog(
            order_id=order.id,
            action=action,
            message=message,
            from_status=order.status,
            to_status=order.status,
            metadata=metadata or {}
        )
        self.db.add(log)

    def process_test_webhook(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理测试Webhook（用于调试）"""
        logger.info(f"测试Webhook: {test_data}")

        # 模拟处理流程
        return {
            "status": "test_received",
            "processed_at": datetime.utcnow().isoformat(),
            "data": test_data
        }


# Webhook路由处理器
async def webhook_handler(request: Request, db: Session) -> Dict[str, Any]:
    """FastAPI路由的Webhook处理器"""
    handler = FreqtradeWebhookHandler(db)
    return await handler.handle_webhook(request)