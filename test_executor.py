#!/usr/bin/env python3
"""
执行器模块测试脚本
测试订单管理、仓位管理、风险管理和执行引擎
"""

import asyncio
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from deepalpha.executor.order import OrderManager
from deepalpha.executor.position import PositionManager
from deepalpha.executor.risk import RiskManager
from deepalpha.executor.core import ExecutionEngine
from deepalpha.core.models import Order, OrderSide, OrderType, OrderStatus
from deepalpha.decision.types import Decision, DecisionTiers
from deepalpha.executor.freqtrade.client import FreqtradeClient
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


async def test_order_manager():
    """测试订单管理器"""
    print("\n=== 测试订单管理器 ===")

    # 创建订单管理器
    order_manager = OrderManager(
        freqtrade_client=None,  # 测试时不使用真实客户端
        position_manager=PositionManager()
    )

    # 测试创建订单
    success, order = await order_manager.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        amount=0.1,
        price=50000.0,
        stop_loss=45000.0,
        take_profit=55000.0,
        client_order_id="test_order_001"
    )

    if success:
        print(f"✓ 创建订单成功: {order.id}")
    else:
        print("✗ 创建订单失败")
        return False

    # 测试获取订单
    retrieved_order = await order_manager.get_order(order_id=order.id)
    if retrieved_order and retrieved_order.id == order.id:
        print("✓ 获取订单成功")
    else:
        print("✗ 获取订单失败")
        return False

    # 测试更新订单
    await order_manager.update_order(
        order_id=order.id,
        status=OrderStatus.FILLED,
        price=50100.0
    )
    print("✓ 更新订单成功")

    # 测试获取活跃订单
    active_orders = await order_manager.get_active_orders()
    print(f"✓ 活跃订单数量: {len(active_orders)}")

    # 测试订单统计
    stats = order_manager.get_statistics()
    print(f"✓ 订单统计: {stats}")

    # 测试清理订单
    cleaned = await order_manager.cleanup_completed_orders(max_age_days=0)
    print(f"✓ 清理了 {cleaned} 个订单")

    return True


async def test_position_manager():
    """测试仓位管理器"""
    print("\n=== 测试仓位管理器 ===")

    from deepalpha.executor.freqtrade.types import APIPosition, TierInfo

    # 创建仓位管理器
    position_manager = PositionManager()

    # 创建模拟API持仓
    api_positions = [
        APIPosition(
            trade_id=1001,
            symbol="BTC/USDT",
            side="long",
            amount=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            unrealized_pnl_usd=100.0,
            realized_pnl_usd=0.0,
            status="open",
            tier1=TierInfo(target=55000.0, ratio=0.3, done=False),
            tier2=TierInfo(target=60000.0, ratio=0.5, done=False)
        ),
        APIPosition(
            trade_id=1002,
            symbol="ETH/USDT",
            side="short",
            amount=1.0,
            entry_price=3000.0,
            current_price=2950.0,
            unrealized_pnl_usd=50.0,
            realized_pnl_usd=0.0,
            status="open"
        )
    ]

    # 测试从API更新
    await position_manager.update_from_api(api_positions)
    print(f"✓ 更新了 {len(api_positions)} 个持仓")

    # 测试获取持仓
    btc_position = await position_manager.get_position("BTC/USDT")
    if btc_position:
        print(f"✓ 获取BTC持仓: {btc_position.size} @ {btc_position.entry_price}")
    else:
        print("✗ 获取BTC持仓失败")
        return False

    # 测试获取所有持仓
    all_positions = await position_manager.get_all_positions()
    print(f"✓ 总持仓数: {len(all_positions)}")

    # 测试总敞口
    total_exposure = await position_manager.get_total_exposure()
    print(f"✓ 总敞口: {total_exposure}")

    # 测试盈亏汇总
    pnl_summary = await position_manager.get_pnl_summary()
    print(f"✓ 盈亏汇总: {pnl_summary}")

    # 测试风险限制检查
    risk_check = await position_manager.check_risk_limits(
        max_exposure=10000.0,
        max_drawdown=0.1,
        current_balance=20000.0
    )
    print(f"✓ 风险检查: {risk_check}")

    # 测试更新层级
    await position_manager.update_tier(
        trade_id=1001,
        tier=1,
        target=56000.0,
        ratio=0.4
    )
    print("✓ 更新层级成功")

    return True


async def test_risk_manager():
    """测试风险管理器"""
    print("\n=== 测试风险管理器 ===")

    # 创建风险管理器
    risk_manager = RiskManager(
        max_position_size=0.05,
        max_total_position=0.8,
        stop_loss=0.02,
        take_profit=0.06,
        max_drawdown=0.15,
        max_trades_per_day=20
    )

    # 创建测试决策
    decision = Decision(
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        position_size_usd=1000.0,
        leverage=2.0,
        stop_loss=49000.0,
        take_profit=53000.0,
        tiers=DecisionTiers(
            tier1_target=53000.0,
            tier1_ratio=0.3,
            tier2_target=56000.0,
            tier2_ratio=0.4
        )
    )

    # 测试决策风险检查
    positions = {}  # 空持仓
    risk_pass, risk_reason = await risk_manager.check_decision_risk(
        decision, 50000.0, positions, 20000.0
    )

    if risk_pass:
        print("✓ 决策风险检查通过")
    else:
        print(f"✗ 决策风险检查失败: {risk_reason}")

    # 测试仓位大小计算
    position_size = risk_manager.calculate_position_size(
        "BTC/USDT", 85, 50000.0, 20000.0
    )
    print(f"✓ 计算仓位大小: {position_size}")

    # 测试止损计算
    stop_loss_price = risk_manager.calculate_stop_loss(
        OrderSide.BUY, 50000.0, atr=500.0
    )
    print(f"✓ 止损价格: {stop_loss_price}")

    # 测试止盈计算
    take_profit_price = risk_manager.calculate_take_profit(
        OrderSide.BUY, 50000.0, stop_loss_price
    )
    print(f"✓ 止盈价格: {take_profit_price}")

    # 测试获取风险指标
    risk_metrics = await risk_manager.get_risk_metrics()
    print(f"✓ 风险指标: {risk_metrics}")

    # 测试记录交易
    risk_manager.record_trade(100.0)
    print("✓ 记录交易成功")

    return True


async def test_execution_engine():
    """测试执行引擎"""
    print("\n=== 测试执行引擎 ===")

    # 创建模拟客户端
    class MockFreqtradeClient:
        async def force_enter(self, payload):
            print(f"模拟开仓: {payload.pair} {payload.side} {payload.stakeamount}")
            return type('Response', (), {'trade_id': 1001})()

        async def force_exit(self, payload):
            print(f"模拟平仓: trade_id={payload.tradeid}")

        async def close(self):
            pass

    # 创建执行引擎
    client = MockFreqtradeClient()
    execution_engine = ExecutionEngine(
        freqtrade_client=client,
        horizon="24h"
    )

    # 创建测试决策
    decision = Decision(
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        position_size_usd=1000.0,
        leverage=2.0,
        stop_loss=49000.0,
        take_profit=53000.0
    )

    # 测试执行决策
    success, message, trade_id = await execution_engine.execute_decision(
        decision, 50000.0, 20000.0
    )

    if success:
        print(f"✓ 决策执行成功: {message}, trade_id={trade_id}")
    else:
        print(f"✗ 决策执行失败: {message}")
        return False

    # 测试获取持仓
    positions = await execution_engine.get_positions()
    print(f"✓ 获取持仓: {len(positions)} 个")

    # 测试获取余额
    balance = await execution_engine.get_balance()
    print(f"✓ 获取余额: {balance}")

    # 测试同步持仓
    synced = await execution_engine.sync_positions()
    print(f"✓ 同步了 {synced} 个持仓")

    # 测试获取执行状态
    status = execution_engine.get_execution_status()
    print(f"✓ 执行状态: {status}")

    # 测试关闭
    await execution_engine.close()
    print("✓ 执行引擎已关闭")

    return True


async def main():
    """主测试函数"""
    print("开始测试执行器模块...")

    # 运行所有测试
    tests = [
        ("订单管理器", test_order_manager),
        ("仓位管理器", test_position_manager),
        ("风险管理器", test_risk_manager),
        ("执行引擎", test_execution_engine),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if await test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                failed += 1
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} 测试异常: {e}")

    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"总计: {passed + failed}")

    if failed == 0:
        print("\n🎉 所有测试通过！")
        return True
    else:
        print("\n⚠️ 部分测试失败")
        return False


if __name__ == "__main__":
    asyncio.run(main())