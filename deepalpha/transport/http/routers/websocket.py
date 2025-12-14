"""WebSocket路由处理器"""

import uuid
from typing import Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel

from deepalpha.transport.websocket import manager, TOPICS, broadcaster as market_broadcaster
from deepalpha.transport.websocket.handlers import (
    market_data,
    notifications,
    positions,
)
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# WebSocket消息模型
class WSMessage(BaseModel):
    """WebSocket消息"""
    type: str
    data: Dict


class SubscribeRequest(BaseModel):
    """订阅请求"""
    topic: str
    params: Dict = {}


class UnsubscribeRequest(BaseModel):
    """取消订阅请求"""
    topic: str


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket连接端点"""
    logger.info(f"WebSocket连接请求: {client_id}")

    # 建立连接
    connected = await manager.connect(websocket, client_id)
    if not connected:
        await websocket.close(code=1000)
        return

    # 启动市场数据广播器（如果尚未启动）
    await market_broadcaster.start()

    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()

            try:
                # 解析消息
                import json
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "subscribe":
                    # 订阅主题
                    topic = message.get("topic")
                    await handle_subscribe(client_id, topic, message.get("params", {}))

                elif msg_type == "unsubscribe":
                    # 取消订阅
                    topic = message.get("topic")
                    await handle_unsubscribe(client_id, topic)

                elif msg_type == "ping":
                    # 响应ping
                    await manager.send_message(client_id, {
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    })

                else:
                    logger.warning(f"未知的消息类型: {msg_type}")

            except json.JSONDecodeError:
                logger.error(f"无法解析消息: {data}")
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": "无效的JSON格式"
                })
            except Exception as e:
                logger.error(f"处理WebSocket消息错误: {e}")
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": str(e)
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        await manager.disconnect(client_id)


async def handle_subscribe(client_id: str, topic: str, params: Dict):
    """处理订阅请求"""
    if not topic:
        await manager.send_message(client_id, {
            "type": "error",
            "message": "主题不能为空"
        })
        return

    # 检查是否为有效主题
    if topic in TOPICS:
        # 订阅系统主题
        success = await manager.subscribe(client_id, topic)
        if success:
            await manager.send_message(client_id, {
                "type": "subscribed",
                "topic": topic
            })
        else:
            await manager.send_message(client_id, {
                "type": "error",
                "message": "订阅失败"
            })

    elif topic.startswith("market_data:"):
        # 订阅市场数据
        symbol = topic.split(":", 1)[1]
        await market_broadcaster.subscribe_symbol(client_id, symbol)
        await manager.send_message(client_id, {
            "type": "subscribed",
            "topic": topic
        })

    else:
        await manager.send_message(client_id, {
            "type": "error",
            "message": f"未知的主题: {topic}"
        })


async def handle_unsubscribe(client_id: str, topic: str):
    """处理取消订阅请求"""
    if not topic:
        await manager.send_message(client_id, {
            "type": "error",
            "message": "主题不能为空"
        })
        return

    if topic.startswith("market_data:"):
        # 取消订阅市场数据
        symbol = topic.split(":", 1)[1]
        await market_broadcaster.unsubscribe_symbol(client_id, symbol)
    else:
        # 取消订阅系统主题
        await manager.unsubscribe(client_id, topic)

    await manager.send_message(client_id, {
        "type": "unsubscribed",
        "topic": topic
    })


@router.get("/ws/status")
async def websocket_status():
    """获取WebSocket状态"""
    stats = manager.get_stats()
    return {
        "websocket": {
            "total_connections": stats["total_connections"],
            "total_subscriptions": stats["total_subscriptions"],
            "topics": stats["topics"],
            "available_topics": TOPICS
        }
    }


@router.get("/ws/connections")
async def get_connections():
    """获取所有连接信息"""
    connections = await manager.get_connections()
    return {
        "connections": connections,
        "total": len(connections)
    }


# 导出
__all__ = ["router"]