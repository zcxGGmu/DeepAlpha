"""WebSocket连接管理器，对应 Go 版本的 websocket manager"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Set, Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class ConnectionInfo(BaseModel):
    """连接信息"""
    websocket: WebSocket
    client_id: str
    connected_at: datetime
    subscriptions: Set[str]
    metadata: Dict[str, Any]


class WebSocketManager:
    """WebSocket连接管理器"""

    def __init__(self):
        # 活跃连接 {client_id: ConnectionInfo}
        self._connections: Dict[str, ConnectionInfo] = {}
        # 订阅映射 {topic: Set[client_id]}
        self._subscriptions: Dict[str, Set[str]] = {}
        # 锁
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """建立WebSocket连接"""
        try:
            await websocket.accept()

            async with self._lock:
                # 检查是否已存在连接
                if client_id in self._connections:
                    logger.warning(f"客户端已存在，关闭旧连接: {client_id}")
                    await self.disconnect(client_id)

                # 创建连接信息
                connection_info = ConnectionInfo(
                    websocket=websocket,
                    client_id=client_id,
                    connected_at=datetime.utcnow(),
                    subscriptions=set(),
                    metadata={}
                )

                self._connections[client_id] = connection_info
                logger.info(f"WebSocket连接已建立: {client_id}")

                return True

        except Exception as e:
            logger.error(f"建立WebSocket连接失败: {e}")
            return False

    async def disconnect(self, client_id: str):
        """断开WebSocket连接"""
        async with self._lock:
            connection = self._connections.pop(client_id, None)
            if connection:
                # 从所有订阅中移除
                for topic in connection.subscriptions:
                    if topic in self._subscriptions:
                        self._subscriptions[topic].discard(client_id)
                        if not self._subscriptions[topic]:
                            del self._subscriptions[topic]

                logger.info(f"WebSocket连接已断开: {client_id}")

    async def subscribe(self, client_id: str, topic: str) -> bool:
        """订阅主题"""
        async with self._lock:
            connection = self._connections.get(client_id)
            if not connection:
                logger.warning(f"客户端未连接: {client_id}")
                return False

            # 添加到连接的订阅列表
            connection.subscriptions.add(topic)

            # 添加到主题订阅映射
            if topic not in self._subscriptions:
                self._subscriptions[topic] = set()
            self._subscriptions[topic].add(client_id)

            logger.info(f"客户端 {client_id} 订阅主题: {topic}")
            return True

    async def unsubscribe(self, client_id: str, topic: str) -> bool:
        """取消订阅主题"""
        async with self._lock:
            connection = self._connections.get(client_id)
            if not connection:
                return False

            # 从连接的订阅列表移除
            connection.subscriptions.discard(topic)

            # 从主题订阅映射移除
            if topic in self._subscriptions:
                self._subscriptions[topic].discard(client_id)
                if not self._subscriptions[topic]:
                    del self._subscriptions[topic]

            logger.info(f"客户端 {client_id} 取消订阅主题: {topic}")
            return True

    async def send_message(self, client_id: str, message: Dict[str, Any]) -> bool:
        """发送消息给指定客户端"""
        async with self._lock:
            connection = self._connections.get(client_id)
            if not connection:
                return False

        try:
            await connection.websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"发送消息失败 {client_id}: {e}")
            # 连接可能已断开，尝试清理
            await self.disconnect(client_id)
            return False

    async def broadcast(self, topic: str, message: Dict[str, Any]) -> int:
        """广播消息给订阅主题的所有客户端"""
        async with self._lock:
            subscribers = self._subscriptions.get(topic, set()).copy()

        if not subscribers:
            logger.debug(f"主题 {topic} 没有订阅者")
            return 0

        # 添加主题和时间戳
        message["topic"] = topic
        message["timestamp"] = datetime.utcnow().isoformat()

        # 并发发送
        tasks = []
        for client_id in subscribers:
            task = self.send_message(client_id, message)
            tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # 统计成功发送的数量
            success_count = sum(1 for r in results if r is True)
            logger.info(f"广播消息到主题 {topic}: {success_count}/{len(subscribers)} 成功")
            return success_count

        return 0

    async def get_connections(self) -> List[Dict[str, Any]]:
        """获取所有连接信息"""
        async with self._lock:
            connections = []
            for client_id, conn in self._connections.items():
                connections.append({
                    "client_id": client_id,
                    "connected_at": conn.connected_at.isoformat(),
                    "subscriptions": list(conn.subscriptions),
                    "metadata": conn.metadata,
                })
            return connections

    async def get_subscriptions(self) -> Dict[str, int]:
        """获取所有主题的订阅数"""
        async with self._lock:
            return {topic: len(subscribers) for topic, subscribers in self._subscriptions.items()}

    async def ping_all(self):
        """向所有连接发送ping"""
        message = {
            "type": "ping",
            "timestamp": datetime.utcnow().isoformat()
        }

        async with self._lock:
            client_ids = list(self._connections.keys())

        for client_id in client_ids:
            await self.send_message(client_id, message)

    async def cleanup(self):
        """清理所有连接"""
        async with self._lock:
            client_ids = list(self._connections.keys())

        for client_id in client_ids:
            await self.disconnect(client_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_connections": len(self._connections),
            "total_subscriptions": len(self._subscriptions),
            "topics": list(self._subscriptions.keys()),
        }


# 全局WebSocket管理器实例
manager = WebSocketManager()

# 预定义的主题
TOPICS = {
    "market_data": "市场数据更新",
    "positions": "持仓更新",
    "orders": "订单更新",
    "decisions": "决策更新",
    "alerts": "告警通知",
    "notifications": "系统通知",
    "trades": "交易事件",
}

# 导出
__all__ = ["WebSocketManager", "manager", "TOPICS"]