"""通知WebSocket推送"""

from datetime import datetime
from typing import Dict, List, Optional

from deepalpha.transport.websocket.manager import manager
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class NotificationBroadcaster:
    """通知广播器"""

    async def push_notification(
        self,
        level: str,  # info, warning, error, success
        title: str,
        message: str,
        data: Optional[Dict] = None
    ):
        """推送通知"""
        notification = {
            "type": "notification",
            "data": {
                "level": level,
                "title": title,
                "message": message,
                "data": data or {},
                "timestamp": datetime.utcnow().isoformat(),
            }
        }

        await manager.broadcast("notifications", notification)
        logger.info(f"推送通知: [{level}] {title} - {message}")

    async def push_decision_created(self, decision_data: Dict):
        """推送决策创建通知"""
        await self.push_notification(
            level="info",
            title="新决策",
            message=f"{decision_data.get('symbol')} {decision_data.get('action')} 置信度: {decision_data.get('confidence')}%",
            data=decision_data
        )

    async def push_decision_executed(self, decision_data: Dict):
        """推送决策执行通知"""
        await self.push_notification(
            level="success",
            title="决策执行",
            message=f"已执行 {decision_data.get('symbol')} {decision_data.get('action')}",
            data=decision_data
        )

    async def push_decision_failed(self, decision_data: Dict, error: str):
        """推送决策执行失败通知"""
        await self.push_notification(
            level="error",
            title="决策执行失败",
            message=f"{decision_data.get('symbol')} {decision_data.get('action')}: {error}",
            data=decision_data
        )

    async def push_order_created(self, order_data: Dict):
        """推送订单创建通知"""
        await self.push_notification(
            level="info",
            title="订单创建",
            message=f"{order_data.get('symbol')} {order_data.get('side')} {order_data.get('amount')}",
            data=order_data
        )

    async def push_order_filled(self, order_data: Dict):
        """推送订单成交通知"""
        await self.push_notification(
            level="success",
            title="订单成交",
            message=f"{order_data.get('symbol')} {order_data.get('side')} {order_data.get('amount')} @ {order_data.get('price')}",
            data=order_data
        )

    async def push_order_cancelled(self, order_data: Dict):
        """推送订单取消通知"""
        await self.push_notification(
            level="warning",
            title="订单取消",
            message=f"{order_data.get('symbol')} {order_data.get('side')}",
            data=order_data
        )

    async def push_system_alert(self, alert_type: str, message: str, data: Optional[Dict] = None):
        """推送系统告警"""
        await self.push_notification(
            level="error" if "error" in alert_type.lower() else "warning",
            title="系统告警",
            message=message,
            data={
                "alert_type": alert_type,
                **(data or {})
            }
        )

    async def push_agent_status(self, agent_name: str, status: str, details: Optional[Dict] = None):
        """推送代理状态更新"""
        await self.push_notification(
            level="info",
            title="代理状态",
            message=f"{agent_name}: {status}",
            data={
                "agent_name": agent_name,
                "status": status,
                **(details or {})
            }
        )

    async def push_market_alert(self, symbol: str, alert_type: str, message: str, data: Optional[Dict] = None):
        """推送市场告警"""
        await self.push_notification(
            level="warning",
            title="市场告警",
            message=f"{symbol}: {message}",
            data={
                "symbol": symbol,
                "alert_type": alert_type,
                **(data or {})
            }
        )

    async def broadcast_system_message(self, message: str, level: str = "info"):
        """广播系统消息"""
        system_message = {
            "type": "system_message",
            "data": {
                "level": level,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }

        await manager.broadcast("notifications", system_message)

    async def push_batch_notifications(self, notifications: List[Dict]):
        """批量推送通知"""
        for notif in notifications:
            await self.push_notification(
                level=notif.get("level", "info"),
                title=notif.get("title", ""),
                message=notif.get("message", ""),
                data=notif.get("data")
            )

    async def push_daily_summary(self, summary: Dict):
        """推送每日汇总"""
        await self.push_notification(
            level="info",
            title="每日汇总",
            message=f"今日交易: {summary.get('total_trades', 0)}笔, 盈亏: {summary.get('total_pnl', 0)}",
            data=summary
        )


# 全局广播器实例
broadcaster = NotificationBroadcaster()


# 导出
__all__ = ["NotificationBroadcaster", "broadcaster"]