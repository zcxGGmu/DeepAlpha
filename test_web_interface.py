#!/usr/bin/env python3
"""
Web接口测试脚本
测试FastAPI REST接口和WebSocket连接
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import requests
import websockets

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from deepalpha.transport.http.server import HTTPServer
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


# 配置
API_BASE = "http://localhost:9991/api/live"
WS_URL = "ws://localhost:9991/ws/test_client"


def test_rest_api():
    """测试REST API"""
    print("\n=== 测试REST API ===")

    # 测试健康检查
    try:
        response = requests.get(f"{API_BASE}/../healthz")
        if response.status_code == 200:
            print("✓ 健康检查通过")
        else:
            print(f"✗ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 健康检查异常: {e}")
        return False

    # 测试决策API
    try:
        response = requests.get(f"{API_BASE}/decisions")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 获取决策列表成功，共 {data.get('total', 0)} 条")
        else:
            print(f"✗ 获取决策列表失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 获取决策列表异常: {e}")

    # 测试持仓API
    try:
        response = requests.get(f"{API_BASE}/freqtrade/positions")
        if response.status_code == 200:
            positions = response.json()
            print(f"✓ 获取持仓列表成功，共 {len(positions)} 个")
        else:
            print(f"✗ 获取持仓列表失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 获取持仓列表异常: {e}")

    # 测试余额API
    try:
        response = requests.get(f"{API_BASE}/freqtrade/balance")
        if response.status_code == 200:
            balance = response.json()
            print(f"✓ 获取余额成功，USDT余额: {balance[0].get('total', 0) if balance else 0}")
        else:
            print(f"✗ 获取余额失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 获取余额异常: {e}")

    # 测试监控API
    try:
        response = requests.get(f"{API_BASE}/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✓ 获取系统状态成功，状态: {status.get('status')}")
        else:
            print(f"✗ 获取系统状态失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 获取系统状态异常: {e}")

    return True


async def test_websocket():
    """测试WebSocket连接"""
    print("\n=== 测试WebSocket ===")

    try:
        async with websockets.connect(WS_URL) as websocket:
            print("✓ WebSocket连接成功")

            # 测试订阅
            await websocket.send(json.dumps({
                "type": "subscribe",
                "topic": "notifications"
            }))
            print("✓ 发送订阅请求")

            # 等待响应
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            data = json.loads(response)
            if data.get("type") == "subscribed":
                print("✓ 订阅成功")
            else:
                print(f"✗ 订阅响应异常: {data}")

            # 测试ping
            await websocket.send(json.dumps({
                "type": "ping",
                "timestamp": time.time()
            }))

            # 等待pong响应
            pong_response = await asyncio.wait_for(websocket.recv(), timeout=5)
            pong_data = json.loads(pong_response)
            if pong_data.get("type") == "pong":
                print("✓ Ping/Pong测试成功")
            else:
                print(f"✗ Pong响应异常: {pong_data}")

            # 测试市场数据订阅
            await websocket.send(json.dumps({
                "type": "subscribe",
                "topic": "market_data:BTC/USDT"
            }))
            print("✓ 订阅BTC/USDT市场数据")

            return True

    except Exception as e:
        print(f"✗ WebSocket测试失败: {e}")
        return False


def test_web_ui():
    """测试Web UI"""
    print("\n=== 测试Web UI ===")

    try:
        # 测试主页
        response = requests.get("http://localhost:9991/")
        if response.status_code == 200:
            print("✓ 主页访问成功")
        else:
            print(f"✗ 主页访问失败: {response.status_code}")

        # 测试管理界面
        response = requests.get("http://localhost:9991/admin")
        if response.status_code == 200:
            if "DeepAlpha" in response.text:
                print("✓ 管理界面访问成功")
            else:
                print("✗ 管理界面内容异常")
        else:
            print(f"✗ 管理界面访问失败: {response.status_code}")

        # 测试API文档
        response = requests.get("http://localhost:9991/api/docs")
        if response.status_code == 200:
            print("✓ API文档访问成功")
        else:
            print(f"✗ API文档访问失败: {response.status_code}")

        return True

    except Exception as e:
        print(f"✗ Web UI测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("开始测试Web接口...")

    # 启动HTTP服务器（后台）
    server = HTTPServer(host="localhost", port=9991, debug=True)

    # 在后台线程启动服务器
    import threading
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(2)

    # 运行测试
    tests = [
        ("REST API", test_rest_api),
        ("WebSocket", test_websocket),
        ("Web UI", test_web_ui),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print(f"{'='*50}")

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                print(f"\n✅ {test_name} 测试通过")
            else:
                failed += 1
                print(f"\n❌ {test_name} 测试失败")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} 测试异常: {e}")

    print(f"\n{'='*50}")
    print("测试结果汇总")
    print(f"{'='*50}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"总计: {passed + failed}")

    if failed == 0:
        print("\n🎉 所有测试通过！")
        print("\n可以访问以下地址:")
        print("- 主页: http://localhost:9991/")
        print("- 管理界面: http://localhost:9991/admin")
        print("- API文档: http://localhost:9991/api/docs")
    else:
        print("\n⚠️ 部分测试失败")

    # 保持服务器运行一段时间以供手动测试
    print("\n按 Ctrl+C 退出...")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n测试结束")


if __name__ == "__main__":
    # 安装所需的包
    try:
        import requests
        import websockets
    except ImportError:
        print("请安装测试依赖: pip install requests websockets")
        sys.exit(1)

    asyncio.run(main())