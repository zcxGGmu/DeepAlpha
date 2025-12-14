"""持仓WebSocket推送"""

from datetime import datetime
from typing import Dict, Optional

from deepalpha.core.models import Position
from deepalpha.transport.websocket.manager import manager
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class PositionBroadcaster:
    """持仓广播器"""

    async def push_position_update(self, position: Position):
        """推送持仓更新"""
        message = {
            "type": "position_update",
            "data": {
                "symbol": position.symbol,
                "side": position.side.value,
                "size": float(position.size),
                "entry_price": float(position.entry_price),
                "current_price": float(position.current_price) if position.current_price else None,
                "unrealized_pnl": float(position.unrealized_pnl),
                "realized_pnl": float(position.realized_pnl),
                "timestamp": datetime.utcnow().isoformat(),
            }
        }

        await manager.broadcast("positions", message)

    async def push_position_opened(self, position: Position):
        """推送开仓通知"""
        message = {
            "type": "position_opened",
            "data": {
                "symbol": position.symbol,
                "side": position.side.value,
                "size": float(position.size),
                "entry_price": float(position.entry_price),
                "timestamp": datetime.utcnow().isoformat(),
            }
        }

        await manager.broadcast("positions", message)
        await manager.broadcast("notifications", {
            "type": "notification",
            "level": "info",
            "title": "开仓通知",
            "message": f"已开仓 {position.symbol} {position.side.value} {position.size}",
            "data": message["data"]
        })

    async def push_position_closed(self, position: Position, exit_price: float, reason: str):
        """推送平仓通知"""
        message = {
            "type": "position_closed",
            "data": {
                "symbol": position.symbol,
                "side": position.side.value,
                "size": float(position.size),
                "entry_price": float(position.entry_price),
                "exit_price": exit_price,
                "pnl": float(position.realized_pnl),
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }

        await manager.broadcast("positions", message)
        await manager.broadcast("notifications", {
            "type": "notification",
            "level": "info",
            "title": "平仓通知",
            "message": f"已平仓 {position.symbol}，盈亏: {position.realized_pnl}",
            "data": message["data"]
        })

    async def push_position_pnl_update(self, symbol: str, pnl: float, price: float):
        """推送持仓盈亏更新"""
        message = {
            "type": "pnl_update",
            "data": {
                "symbol": symbol,
                "pnl": pnl,
                "price": price,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }

        await manager.broadcast("positions", message)

    async def push_tier_update(self, symbol: str, tier: int, target: float, done: bool):
        """推送层级更新"""
        message = {
            "type": "tier_update",
            "data": {
                "symbol": symbol,
                "tier": tier,
                "target": target,
                "done": done,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }

        await manager.broadcast("positions", message)
        await manager.broadcast("notifications", {
            "type": "notification",
            "level": "info",
            "title": "层级更新",
            "message": f"{symbol} 层级{tier} {'已触发' if done else f'目标: {target}'}",
            "data": message["data"]
        })

    async def push_risk_alert(self, symbol: str, alert_type: str, message: str, data: Optional[Dict] = None):
        """推送风险告警"""
        alert_message = {
            "type": "risk_alert",
            "data": {
                "symbol": symbol,
                "alert_type": alert_type,
                "message": message,
                "data": data or {},
                "timestamp": datetime.utcnow().isoformat(),
            }
        }

        await manager.broadcast("positions", alert_message)
        await manager.broadcast("notifications", {
            "type": "notification",
            "level": "warning",
            "title": "风险告警",
            "message": message,
            "data": alert_message["data"]
        })


# 全局广播器实例
broadcaster = PositionBroadcaster()


# 导出
__all__ = ["PositionBroadcaster", "broadcaster"]