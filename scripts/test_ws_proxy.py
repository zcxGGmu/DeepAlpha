#!/usr/bin/env python3
"""WebSocket代理测试脚本
对应Go版本的 test_ws_proxy.py
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any

import websockets
from websockets.exceptions import ConnectionClosed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebSocketProxyTester:
    """WebSocket代理测试器"""

    def __init__(self, ws_url: str = "ws://localhost:9991/ws", reconnect: bool = True):
        self.ws_url = ws_url
        self.reconnect = reconnect
        self.websocket = None
        self.running = False
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "connections": 0,
            "errors": 0
        }

    async def connect(self) -> bool:
        """连接WebSocket"""
        try:
            logger.info(f"连接到 {self.ws_url}")
            self.websocket = await websockets.connect(self.ws_url)
            self.stats["connections"] += 1
            logger.info("WebSocket连接成功 ✓")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            self.stats["errors"] += 1
            return False

    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("WebSocket连接已断开")

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """发送消息"""
        if not self.websocket:
            logger.error("WebSocket未连接")
            return False

        try:
            await self.websocket.send(json.dumps(message))
            self.stats["messages_sent"] += 1
            logger.debug(f"发送消息: {message}")
            return True
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            self.stats["errors"] += 1
            return False

    async def receive_message(self) -> Dict[str, Any]:
        """接收消息"""
        if not self.websocket:
            logger.error("WebSocket未连接")
            return {}

        try:
            message = await self.websocket.recv()
            self.stats["messages_received"] += 1
            data = json.loads(message)
            logger.debug(f"接收消息: {data}")
            return data
        except ConnectionClosed:
            logger.warning("WebSocket连接已关闭")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            self.stats["errors"] += 1
            return {}
        except Exception as e:
            logger.error(f"接收消息失败: {e}")
            self.stats["errors"] += 1
            return {}

    async def subscribe(self, channels: list):
        """订阅频道"""
        message = {
            "type": "subscribe",
            "channels": channels
        }
        return await self.send_message(message)

    async def unsubscribe(self, channels: list):
        """取消订阅"""
        message = {
            "type": "unsubscribe",
            "channels": channels
        }
        return await self.send_message(message)

    async def ping(self):
        """发送ping"""
        return await self.send_message({"type": "ping"})

    async def print_stats(self):
        """打印统计信息"""
        print("\n===== 连接统计 =====")
        print(f"连接次数: {self.stats['connections']}")
        print(f"发送消息: {self.stats['messages_sent']}")
        print(f"接收消息: {self.stats['messages_received']}")
        print(f"错误次数: {self.stats['errors']}")
        print("==================")

    async def test_basic_connection(self):
        """测试基本连接"""
        logger.info("测试基本连接...")

        if not await self.connect():
            return False

        # 测试ping
        logger.info("发送ping...")
        if await self.ping():
            response = await self.receive_message()
            if response.get("type") == "pong":
                logger.info("Ping/Pong测试通过 ✓")
            else:
                logger.error("Ping/Pong测试失败 ✗")

        await self.disconnect()
        return True

    async def test_subscription(self):
        """测试订阅功能"""
        logger.info("测试订阅功能...")

        if not await self.connect():
            return False

        # 订阅频道
        channels = ["notifications", "market_data"]
        logger.info(f"订阅频道: {channels}")

        if await self.subscribe(channels):
            logger.info("订阅成功 ✓")

            # 等待消息
            logger.info("等待接收消息...")
            try:
                for i in range(5):  # 等待5条消息
                    message = await asyncio.wait_for(
                        self.receive_message(),
                        timeout=10.0
                    )
                    if message:
                        logger.info(f"收到消息 {i+1}: {message.get('type', 'unknown')}")
            except asyncio.TimeoutError:
                logger.warning("接收消息超时")

            # 取消订阅
            logger.info("取消订阅...")
            if await self.unsubscribe(channels):
                logger.info("取消订阅成功 ✓")

        await self.disconnect()
        return True

    async def test_reconnection(self):
        """测试重连机制"""
        logger.info("测试重连机制...")

        for i in range(3):
            logger.info(f"连接尝试 {i+1}/3")

            if await self.connect():
                await self.ping()
                await self.disconnect()
                logger.info(f"连接 {i+1} 成功 ✓")
            else:
                logger.error(f"连接 {i+1} 失败 ✗")

            if i < 2:
                await asyncio.sleep(2)

        return True

    async def monitor_mode(self):
        """监控模式"""
        logger.info("启动监控模式...")

        while not await self.connect():
            logger.error("连接失败，5秒后重试...")
            await asyncio.sleep(5)

        # 订阅所有频道
        await self.subscribe(["notifications", "market_data", "positions"])

        try:
            while self.running:
                message = await self.receive_message()
                if message:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] {json.dumps(message, indent=2)}")
        except KeyboardInterrupt:
            logger.info("监控停止")
        finally:
            await self.disconnect()
            await self.print_stats()

    async def stress_test(self, duration: int = 60):
        """压力测试"""
        logger.info(f"启动压力测试，持续 {duration} 秒...")

        if not await self.connect():
            return False

        start_time = time.time()
        message_count = 0

        try:
            while time.time() - start_time < duration:
                # 发送测试消息
                test_message = {
                    "type": "test",
                    "timestamp": datetime.now().isoformat(),
                    "counter": message_count
                }

                if await self.send_message(test_message):
                    message_count += 1

                # 接收响应
                response = await self.receive_message()
                if response:
                    logger.debug(f"响应: {response}")

                await asyncio.sleep(0.1)  # 100ms间隔

        except Exception as e:
            logger.error(f"压力测试异常: {e}")

        await self.disconnect()
        await self.print_stats()

        logger.info(f"压力测试完成，发送了 {message_count} 条消息")
        return True


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket代理测试工具")
    parser.add_argument("command", choices=[
        "basic", "subscribe", "reconnect", "monitor", "stress"
    ], help="测试命令")
    parser.add_argument("--url", default="ws://localhost:9991/ws", help="WebSocket URL")
    parser.add_argument("--duration", type=int, default=60, help="压力测试持续时间（秒）")

    args = parser.parse_args()

    tester = WebSocketProxyTester(args.url)

    try:
        if args.command == "basic":
            success = await tester.test_basic_connection()
        elif args.command == "subscribe":
            success = await tester.test_subscription()
        elif args.command == "reconnect":
            success = await tester.test_reconnection()
        elif args.command == "monitor":
            tester.running = True
            success = await tester.monitor_mode()
        elif args.command == "stress":
            success = await tester.stress_test(args.duration)
        else:
            logger.error(f"未知命令: {args.command}")
            success = False

        if not success:
            logger.error("测试失败")
            sys.exit(1)
        else:
            logger.info("测试成功")

    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
        await tester.disconnect()
    except Exception as e:
        logger.error(f"测试异常: {e}")
        await tester.disconnect()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())