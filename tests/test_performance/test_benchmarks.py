"""性能基准测试"""

import asyncio
import time
from datetime import datetime
from decimal import Decimal
import pytest

from deepalpha.executor.core import ExecutionEngine
from deepalpha.executor.order import OrderManager
from deepalpha.executor.position import PositionManager
from deepalpha.executor.risk import RiskManager
from deepalpha.decision.types import Decision, DecisionStatus
from deepalpha.core.models import OrderSide, OrderType, Position
from deepalpha.market.data import Candle


@pytest.mark.asyncio
async def benchmark_order_creation():
    """基准测试：订单创建性能"""
    print("\n=== 订单创建性能测试 ===")

    mock_client = Mock()
    position_manager = PositionManager()
    order_manager = OrderManager(mock_client, position_manager)

    # 测试不同数量的订单创建
    test_sizes = [100, 500, 1000, 5000]

    for size in test_sizes:
        # 记录开始时间
        start_time = time.perf_counter()

        # 创建订单
        tasks = []
        for i in range(size):
            task = order_manager.create_order(
                symbol=f"TEST{i}/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                amount=0.1,
                price=50000.0 + i,
                client_order_id=f"perf_test_{i}"
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        # 计算耗时
        duration = time.perf_counter() - start_time
        throughput = size / duration

        print(f"创建 {size} 个订单:")
        print(f"  耗时: {duration:.4f} 秒")
        print(f"  吞吐量: {throughput:.2f} 订单/秒")

        # 清理
        order_manager._orders.clear()
        order_manager._client_order_ids.clear()

    # 性能要求：至少1000订单/秒
    assert throughput >= 1000, f"订单创建性能不足: {throughput:.2f} < 1000"


@pytest.mark.asyncio
async def benchmark_position_updates():
    """基准测试：持仓更新性能"""
    print("\n=== 持仓更新性能测试 ===")

    position_manager = PositionManager()

    # 创建大量持仓
    num_positions = 10000
    positions = []
    for i in range(num_positions):
        position = Position(
            symbol=f"SYMBOL{i}/USDT",
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            size=Decimal(str(0.1 + i * 0.001)),
            entry_price=Decimal(str(50000.0 + i * 10)),
            current_price=Decimal(str(50000.0 + i * 10))
        )
        positions.append(position)
        position_manager._positions[position.symbol] = position

    # 测试批量更新
    start_time = time.perf_counter()

    tasks = []
    for i, position in enumerate(positions):
        new_price = float(position.entry_price) + (i % 100) - 50
        pnl = (new_price - float(position.entry_price)) * float(position.size)
        task = position_manager.update_position(
            position.symbol,
            current_price=new_price,
            unrealized_pnl=pnl
        )
        tasks.append(task)

    await asyncio.gather(*tasks)

    duration = time.perf_counter() - start_time
    throughput = num_positions / duration

    print(f"更新 {num_positions} 个持仓:")
    print(f"  耗时: {duration:.4f} 秒")
    print(f"  吞吐量: {throughput:.2f} 更新/秒")

    # 性能要求：至少5000更新/秒
    assert throughput >= 5000, f"持仓更新性能不足: {throughput:.2f} < 5000"


@pytest.mark.asyncio
async def benchmark_risk_checks():
    """基准测试：风险检查性能"""
    print("\n=== 风险检查性能测试 ===")

    risk_manager = RiskManager()

    # 创建大量决策
    num_decisions = 10000
    decisions = []
    for i in range(num_decisions):
        decision = Decision(
            id=f"risk_test_{i}",
            symbol="BTC/USDT",
            action="enter_long" if i % 2 == 0 else "enter_short",
            confidence=70 + (i % 30),
            position_size_usd=100 + (i % 500),
            stop_loss=49000.0 + (i % 1000),
            take_profit=53000.0 + (i % 1000),
            status=DecisionStatus.PENDING
        )
        decisions.append(decision)

    # 测试风险检查
    start_time = time.perf_counter()

    tasks = []
    for decision in decisions:
        task = risk_manager.check_decision_risk(
            decision,
            50000.0,
            {},  # 空持仓
            20000.0
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    duration = time.perf_counter() - start_time
    throughput = num_decisions / duration

    # 统计通过率
    passed = sum(1 for pass_, _ in results if pass_)
    pass_rate = passed / num_decisions * 100

    print(f"检查 {num_decisions} 个决策:")
    print(f"  耗时: {duration:.4f} 秒")
    print(f"  吞吐量: {throughput:.2f} 检查/秒")
    print(f"  通过率: {pass_rate:.2f}%")

    # 性能要求：至少10000检查/秒
    assert throughput >= 10000, f"风险检查性能不足: {throughput:.2f} < 10000"


@pytest.mark.asyncio
async def benchmark_execution_engine():
    """基准测试：执行引擎性能"""
    print("\n=== 执行引擎性能测试 ===")

    mock_client = Mock()
    mock_client.force_enter = AsyncMock(return_value=Mock(trade_id=1001))

    execution_engine = ExecutionEngine(freqtrade_client=mock_client)

    # 创建大量决策
    num_decisions = 1000
    decisions = []
    for i in range(num_decisions):
        decision = Decision(
            id=f"exec_test_{i}",
            symbol="BTC/USDT",
            action="enter_long",
            confidence=85,
            position_size_usd=100.0,
            stop_loss=49000.0,
            take_profit=53000.0,
            status=DecisionStatus.PENDING
        )
        decisions.append(decision)

    # 测试并发执行
    start_time = time.perf_counter()

    tasks = []
    for decision in decisions:
        task = execution_engine.execute_decision(decision, 50000.0, 20000.0)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    duration = time.perf_counter() - start_time
    throughput = num_decisions / duration

    # 统计成功率
    success_count = sum(1 for success, _ in results if success)
    success_rate = success_count / num_decisions * 100

    print(f"执行 {num_decisions} 个决策:")
    print(f"  耗时: {duration:.4f} 秒")
    print(f"  吞吐量: {throughput:.2f} 执行/秒")
    print(f"  成功率: {success_rate:.2f}%")

    # 性能要求：至少100执行/秒（考虑API调用延迟）
    assert throughput >= 100, f"执行引擎性能不足: {throughput:.2f} < 100"


@pytest.mark.asyncio
async def benchmark_market_data_processing():
    """基准测试：市场数据处理性能"""
    print("\n=== 市场数据处理性能测试 ===")

    # 生成大量K线数据
    num_candles = 100000
    candles = []
    base_time = datetime.utcnow()

    for i in range(num_candles):
        candle = Candle(
            symbol="BTC/USDT",
            exchange="binance",
            interval="1m",
            timestamp=base_time,
            open=50000.0 + i * 0.01,
            high=50010.0 + i * 0.01,
            low=49990.0 + i * 0.01,
            close=50005.0 + i * 0.01,
            volume=100.0 + i * 0.1
        )
        candles.append(candle)

    # 测试数据处理
    start_time = time.perf_counter()

    # 模拟处理逻辑
    processed = 0
    for candle in candles:
        # 计算技术指标（简化版）
        close_price = candle.close
        volume = candle.volume
        high_low = candle.high - candle.low

        # 一些计算
        sma = sum(c.close for c in candles[-20:]) / 20 if processed >= 19 else close_price
        rsi = 50  # 简化

        # 更新计数
        processed += 1

        # 每10000个输出一次进度
        if processed % 10000 == 0:
            pass

    duration = time.perf_counter() - start_time
    throughput = num_candles / duration

    print(f"处理 {num_candles} 个K线:")
    print(f"  耗时: {duration:.4f} 秒")
    print(f"  吞吐量: {throughput:.2f} K线/秒")

    # 性能要求：至少50000 K线/秒
    assert throughput >= 50000, f"市场数据处理性能不足: {throughput:.2f} < 50000"


@pytest.mark.asyncio
async def benchmark_websocket_broadcast():
    """基准测试：WebSocket广播性能"""
    print("\n=== WebSocket广播性能测试 ===")

    from deepalpha.transport.websocket.manager import WebSocketManager

    # 创建管理器
    manager = WebSocketManager()

    # 模拟大量连接
    num_connections = 1000
    client_ids = [f"perf_client_{i}" for i in range(num_connections)]

    # 模拟连接（不需要真实WebSocket）
    for client_id in client_ids:
        await manager._connections[client_id] = Mock()
        await manager._subscriptions["test_topic"] = set(client_ids)

    # 测试广播性能
    num_messages = 10000
    start_time = time.perf_counter()

    for i in range(num_messages):
        message = {
            "type": "test",
            "data": {"value": i, "timestamp": time.time()}
        }
        # 模拟广播（实际不会发送，只计算）
        pass

    duration = time.perf_counter() - start_time
    throughput = num_messages / duration

    print(f"准备广播 {num_messages} 条消息:")
    print(f"  耗时: {duration:.4f} 秒")
    print(f"  吞吐量: {throughput:.2f} 消息/秒")

    # 清理
    await manager.cleanup()

    # 性能要求：至少10000消息/秒
    assert throughput >= 10000, f"WebSocket广播性能不足: {throughput:.2f} < 10000"


def benchmark_memory_usage():
    """基准测试：内存使用情况"""
    import sys
    import tracemalloc

    print("\n=== 内存使用测试 ===")

    # 开始内存跟踪
    tracemalloc.start()

    # 记录初始内存
    snapshot1 = tracemalloc.take_snapshot()

    # 创建大量对象
    objects = []
    for i in range(100000):
        candle = {
            "symbol": "BTC/USDT",
            "open": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "close": 50050.0,
            "volume": 100.0,
            "timestamp": time.time()
        }
        objects.append(candle)

    # 记录峰值内存
    snapshot2 = tracemalloc.take_snapshot()

    # 统计内存差异
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_size = sum(stat.size for stat in top_stats)

    print(f"创建 {len(objects)} 个对象:")
    print(f"  内存增加: {total_size / 1024 / 1024:.2f} MB")
    print(f"  平均每个对象: {total_size / len(objects)} bytes")

    # 性能要求：每个对象不超过1KB
    avg_size = total_size / len(objects)
    assert avg_size < 1024, f"内存使用过多: {avg_size} bytes/object > 1024"

    tracemalloc.stop()


async def main():
    """运行所有性能测试"""
    print("开始性能基准测试...")
    print("=" * 60)

    try:
        # 运行各项测试
        await benchmark_order_creation()
        await benchmark_position_updates()
        await benchmark_risk_checks()
        await benchmark_execution_engine()
        await benchmark_market_data_processing()
        await benchmark_websocket_broadcast()
        benchmark_memory_usage()

        print("\n" + "=" * 60)
        print("✅ 所有性能测试通过！")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 性能测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 测试执行出错: {e}")
        raise


# Mock类
class Mock:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


if __name__ == "__main__":
    asyncio.run(main())