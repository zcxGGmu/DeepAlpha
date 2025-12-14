"""性能分析和优化测试"""

import asyncio
import cProfile
import io
import pstats
import time
from datetime import datetime
from decimal import Decimal

from deepalpha.executor.order import OrderManager
from deepalpha.executor.position import PositionManager
from deepalpha.executor.risk import RiskManager
from deepalpha.executor.core import ExecutionEngine
from deepalpha.core.models import OrderSide, OrderType, Position
from deepalpha.decision.types import Decision, DecisionStatus


def profile_order_manager():
    """分析订单管理器性能"""
    print("\n=== 订单管理器性能分析 ===")

    # 创建订单管理器
    mock_client = Mock()
    position_manager = PositionManager()
    order_manager = OrderManager(mock_client, position_manager)

    # 创建性能分析器
    profiler = cProfile.Profile()
    profiler.enable()

    # 执行操作
    num_operations = 10000

    # 创建订单
    for i in range(num_operations):
        order_manager.create_order(
            symbol=f"TEST{i}/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=0.1,
            price=50000.0 + i
        )

    # 获取订单
    for i in range(num_operations):
        order_manager.get_order(order_id=f"order_{int(time.time() * 1000) - num_operations + i + 1}")

    # 更新订单
    for i in range(num_operations // 2):
        order_manager.update_order(
            order_id=f"order_{int(time.time() * 1000) - num_operations + i + 1}",
            status="filled"
        )

    profiler.disable()

    # 输出结果
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(10)  # 显示前10个最耗时的函数

    print(f"执行 {num_operations} 个订单操作:")
    print(s.getvalue())


def profile_position_manager():
    """分析仓位管理器性能"""
    print("\n=== 仓位管理器性能分析 ===")

    position_manager = PositionManager()

    # 创建性能分析器
    profiler = cProfile.Profile()
    profiler.enable()

    # 执行操作
    num_positions = 5000

    # 创建持仓
    for i in range(num_positions):
        position = Position(
            symbol=f"POS{i}/USDT",
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            size=Decimal(str(0.1 + i * 0.001)),
            entry_price=Decimal(str(50000.0 + i * 10)),
            current_price=Decimal(str(50000.0 + i * 10))
        )
        position_manager._positions[position.symbol] = position

    # 获取持仓
    for i in range(num_positions):
        position_manager.get_position(f"POS{i}/USDT")

    # 更新持仓
    for i in range(num_positions):
        position_manager.update_position(
            f"POS{i}/USDT",
            current_price=50000.0 + i * 10 + 100,
            unrealized_pnl=10.0 + i
        )

    # 计算总敞口
    for _ in range(1000):
        asyncio.run(position_manager.get_total_exposure())

    profiler.disable()

    # 输出结果
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(10)

    print(f"处理 {num_positions} 个持仓:")
    print(s.getvalue())


def profile_risk_manager():
    """分析风险管理器性能"""
    print("\n=== 风险管理器性能分析 ===")

    risk_manager = RiskManager()

    # 创建性能分析器
    profiler = cProfile.Profile()
    profiler.enable()

    # 执行操作
    num_checks = 10000

    # 创建决策
    decisions = []
    for i in range(num_checks):
        decision = Decision(
            id=f"risk_{i}",
            symbol="BTC/USDT",
            action="enter_long" if i % 2 == 0 else "enter_short",
            confidence=70 + (i % 30),
            position_size_usd=100 + (i % 500),
            stop_loss=49000.0,
            take_profit=53000.0,
            status=DecisionStatus.PENDING
        )
        decisions.append(decision)

    # 风险检查
    for decision in decisions:
        asyncio.run(risk_manager.check_decision_risk(
            decision,
            50000.0,
            {},
            20000.0
        ))

    # 计算仓位大小
    for i in range(num_checks):
        risk_manager.calculate_position_size(
            "BTC/USDT",
            70 + (i % 30),
            50000.0,
            20000.0
        )

    # 计算止损止盈
    for i in range(num_checks):
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        risk_manager.calculate_stop_loss(side, 50000.0)
        risk_manager.calculate_take_profit(
            side,
            50000.0,
            49000.0
        )

    profiler.disable()

    # 输出结果
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(10)

    print(f"执行 {num_checks} 次风险检查:")
    print(s.getvalue())


def profile_execution_engine():
    """分析执行引擎性能"""
    print("\n=== 执行引擎性能分析 ===")

    mock_client = Mock()
    mock_client.force_enter = AsyncMock(return_value=Mock(trade_id=1001))

    execution_engine = ExecutionEngine(freqtrade_client=mock_client)

    # 创建性能分析器
    profiler = cProfile.Profile()
    profiler.enable()

    # 执行操作
    num_executions = 1000

    # 创建决策
    decisions = []
    for i in range(num_executions):
        decision = Decision(
            id=f"exec_{i}",
            symbol="BTC/USDT",
            action="enter_long",
            confidence=85,
            position_size_usd=100.0,
            stop_loss=49000.0,
            take_profit=53000.0,
            status=DecisionStatus.PENDING
        )
        decisions.append(decision)

    # 执行决策
    for decision in decisions:
        asyncio.run(execution_engine.execute_decision(
            decision,
            50000.0,
            20000.0
        ))

    profiler.disable()

    # 输出结果
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(10)

    print(f"执行 {num_executions} 个决策:")
    print(s.getvalue())


def analyze_memory_allocation():
    """分析内存分配模式"""
    print("\n=== 内存分配分析 ===")

    import tracemalloc

    # 开始跟踪
    tracemalloc.start()

    # 测试不同操作的内存使用
    operations = [
        ("创建订单", lambda: create_orders(1000)),
        ("创建持仓", lambda: create_positions(1000)),
        ("风险检查", lambda: risk_checks(1000)),
    ]

    for name, func in operations:
        # 记录前一个快照
        snapshot1 = tracemalloc.take_snapshot()

        # 执行操作
        func()

        # 记录后一个快照
        snapshot2 = tracemalloc.take_snapshot()

        # 比较差异
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_size = sum(stat.size for stat in top_stats[:10])

        print(f"\n{name}:")
        print(f"  内存增长: {total_size / 1024:.2f} KB")

        # 显示最大的内存分配
        for stat in top_stats[:5]:
            print(f"  {stat}")

    tracemalloc.stop()


def create_orders(count):
    """创建订单"""
    mock_client = Mock()
    position_manager = PositionManager()
    order_manager = OrderManager(mock_client, position_manager)

    for i in range(count):
        order_manager.create_order(
            symbol=f"TEST{i}/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=0.1,
            price=50000.0
        )


def create_positions(count):
    """创建持仓"""
    position_manager = PositionManager()

    for i in range(count):
        position = Position(
            symbol=f"POS{i}/USDT",
            side=OrderSide.BUY,
            size=Decimal('0.1'),
            entry_price=Decimal('50000.0')
        )
        position_manager._positions[position.symbol] = position


def risk_checks(count):
    """风险检查"""
    risk_manager = RiskManager()

    for i in range(count):
        decision = Decision(
            id=f"risk_{i}",
            symbol="BTC/USDT",
            action="enter_long",
            confidence=85,
            position_size_usd=100.0
        )
        asyncio.run(risk_manager.check_decision_risk(
            decision,
            50000.0,
            {},
            20000.0
        ))


def identify_bottlenecks():
    """识别性能瓶颈"""
    print("\n=== 性能瓶颈分析 ===")

    # 测试不同场景的性能
    scenarios = [
        ("高并发订单创建", test_concurrent_orders),
        ("大量持仓更新", test_bulk_position_updates),
        ("密集风险检查", test_intensive_risk_checks),
    ]

    for name, test_func in scenarios:
        print(f"\n测试场景: {name}")
        start_time = time.perf_counter()
        test_func()
        duration = time.perf_counter() - start_time
        print(f"耗时: {duration:.4f} 秒")

        # 如果超过阈值，标记为瓶颈
        if duration > 1.0:
            print("⚠️  检测到性能瓶颈")


def test_concurrent_orders():
    """测试并发订单创建"""
    import asyncio

    mock_client = Mock()
    position_manager = PositionManager()
    order_manager = OrderManager(mock_client, position_manager)

    async def create_order_batch():
        tasks = []
        for i in range(100):
            task = order_manager.create_order(
                symbol=f"TEST{i}/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                amount=0.1,
                price=50000.0
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    asyncio.run(create_order_batch())


def test_bulk_position_updates():
    """测试批量持仓更新"""
    position_manager = PositionManager()

    # 创建初始持仓
    for i in range(100):
        position = Position(
            symbol=f"POS{i}/USDT",
            side=OrderSide.BUY,
            size=Decimal('0.1'),
            entry_price=Decimal('50000.0')
        )
        position_manager._positions[position.symbol] = position

    # 批量更新
    async def update_all():
        tasks = []
        for i in range(100):
            task = position_manager.update_position(
                f"POS{i}/USDT",
                current_price=50100.0,
                unrealized_pnl=10.0
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    asyncio.run(update_all())


def test_intensive_risk_checks():
    """测试密集风险检查"""
    risk_manager = RiskManager()

    async def check_all():
        tasks = []
        for i in range(1000):
            decision = Decision(
                id=f"risk_{i}",
                symbol="BTC/USDT",
                action="enter_long",
                confidence=85,
                position_size_usd=100.0
            )
            task = risk_manager.check_decision_risk(
                decision,
                50000.0,
                {},
                20000.0
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    asyncio.run(check_all())


def main():
    """运行所有性能分析"""
    print("开始性能分析...")
    print("=" * 60)

    # 运行各项分析
    profile_order_manager()
    profile_position_manager()
    profile_risk_manager()
    profile_execution_engine()
    analyze_memory_allocation()
    identify_bottlenecks()

    print("\n" + "=" * 60)
    print("性能分析完成")
    print("=" * 60)


# Mock类
class Mock:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


if __name__ == "__main__":
    main()