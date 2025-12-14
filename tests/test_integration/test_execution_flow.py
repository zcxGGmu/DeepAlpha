"""测试执行流程集成"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from deepalpha.executor.core import ExecutionEngine
from deepalpha.executor.order import OrderManager
from deepalpha.executor.position import PositionManager
from deepalpha.executor.risk import RiskManager
from deepalpha.executor.freqtrade.client import FreqtradeClient
from deepalpha.decision.types import Decision, DecisionStatus, DecisionTiers
from deepalpha.core.models import OrderSide, Position


@pytest.mark.asyncio
async def test_full_execution_flow():
    """测试完整的执行流程"""
    # 创建组件
    mock_client = Mock(spec=FreqtradeClient)
    position_manager = PositionManager()
    risk_manager = RiskManager(
        max_position_size=0.05,
        max_total_position=0.5,
        stop_loss=0.02,
        take_profit=0.06
    )

    # 设置模拟响应
    mock_client.force_enter = AsyncMock(return_value=Mock(trade_id=1001))
    mock_client.get_positions = AsyncMock(return_value=Mock(positions=[]))
    mock_client.get_balance = AsyncMock(return_value=Mock(
        total=20000.0,
        free=20000.0,
        used=0.0
    ))

    # 创建执行引擎
    execution_engine = ExecutionEngine(
        freqtrade_client=mock_client,
        horizon="24h",
        risk_manager=risk_manager
    )
    execution_engine.position_manager = position_manager

    # 创建决策
    decision = Decision(
        id="decision_001",
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        status=DecisionStatus.PENDING,
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

    # 执行决策
    success, message, trade_id = await execution_engine.execute_decision(
        decision, 50000.0, 20000.0
    )

    # 验证执行成功
    assert success is True
    assert message == "执行成功"
    assert trade_id == 1001

    # 验证调用参数
    mock_client.force_enter.assert_called_once()
    call_args = mock_client.force_enter.call_args[0][0]
    assert call_args.pair == "BTC/USDT"
    assert call_args.side == "long"
    assert call_args.stakeamount == 1000.0
    assert call_args.leverage == 2.0
    assert call_args.stop_loss == 49000.0
    assert call_args.take_profit == 53000.0

    # 验证映射关系
    assert 1001 in execution_engine._trade_to_decision.values()


@pytest.mark.asyncio
async def test_position_management_flow():
    """测试仓位管理流程"""
    # 创建组件
    mock_client = Mock(spec=FreqtradeClient)
    position_manager = PositionManager()
    risk_manager = RiskManager()
    order_manager = OrderManager(mock_client, position_manager)

    # 模拟API持仓数据
    from deepalpha.executor.freqtrade.types import APIPosition, TierInfo
    api_position = APIPosition(
        trade_id=1001,
        symbol="BTC/USDT",
        side="long",
        amount=0.1,
        entry_price=50000.0,
        current_price=51000.0,
        unrealized_pnl_usd=100.0,
        realized_pnl_usd=0.0,
        status="open",
        tier1=TierInfo(target=53000.0, ratio=0.3, done=False)
    )

    # 更新持仓
    await position_manager.update_from_api([api_position])

    # 验证持仓
    position = await position_manager.get_position("BTC/USDT")
    assert position is not None
    assert position.symbol == "BTC/USDT"
    assert position.side == OrderSide.BUY
    assert position.size == 0.1

    # 验证映射
    trade_id_position = await position_manager.get_position_by_trade_id(1001)
    assert trade_id_position.symbol == "BTC/USDT"

    # 更新层级
    await position_manager.update_tier(1001, 1, target=54000.0, ratio=0.4)
    updated_position = await position_manager.get_position("BTC/USDT")
    assert updated_position.metadata["tier1"]["target"] == 54000.0
    assert updated_position.metadata["tier1"]["ratio"] == 0.4

    # 平仓
    await position_manager.close_position(
        "BTC/USDT",
        exit_price=53000.0,
        exit_reason="止盈",
        realized_pnl=300.0
    )

    # 验证已平仓
    closed_position = await position_manager.get_position("BTC/USDT")
    assert closed_position is None


@pytest.mark.asyncio
async def test_risk_management_integration():
    """测试风险管理集成"""
    risk_manager = RiskManager(
        max_position_size=0.02,
        max_total_position=0.5,
        max_trades_per_day=3
    )

    # 测试决策风险
    decision = Decision(
        id="test_001",
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        position_size_usd=500.0,  # 在限制内
        stop_loss=49000.0,
        take_profit=53000.0
    )

    # 第一次决策应该通过
    risk_pass, reason = await risk_manager.check_decision_risk(
        decision, 50000.0, {}, 20000.0
    )
    assert risk_pass is True

    # 记录交易
    risk_manager.record_trade(100.0)

    # 再次记录
    risk_manager.record_trade(-50.0)

    # 第三次记录
    risk_manager.record_trade(200.0)

    # 达到每日限制
    risk_pass, reason = await risk_manager.check_decision_risk(
        decision, 50000.0, {}, 20000.0
    )
    assert risk_pass is False
    assert "超过每日最大交易数" in reason

    # 验证统计
    metrics = await risk_manager.get_risk_metrics()
    assert metrics["daily_trades"] == 3
    assert metrics["daily_pnl"] == 250.0


@pytest.mark.asyncio
async def test_order_lifecycle_integration():
    """测试订单生命周期集成"""
    mock_client = Mock(spec=FreqtradeClient)
    position_manager = PositionManager()
    order_manager = OrderManager(mock_client, position_manager)

    # 创建订单
    success, order = await order_manager.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        amount=0.1,
        price=50000.0,
        client_order_id="client_001"
    )

    assert success is True
    assert order is not None

    # 获取订单
    retrieved_order = await order_manager.get_order(order_id=order.id)
    assert retrieved_order.id == order.id

    # 更新订单状态
    await order_manager.update_order(
        order_id=order.id,
        status="partially_filled",
        amount_filled=0.05
    )

    # 部分成交
    await order_manager.update_order(
        order_id=order.id,
        status="filled",
        amount_filled=0.1,
        fill_price=50010.0
    )

    # 验证最终状态
    final_order = await order_manager.get_order(order_id=order.id)
    assert final_order.status == "filled"

    # 获取统计
    stats = order_manager.get_statistics()
    assert stats["total"] == 1
    assert stats["filled"] == 1


@pytest.mark.asyncio
async def test_webhook_integration():
    """测试Webhook集成"""
    from deepalpha.transport.http.routers.freqtrade import WebhookRequest
    import json

    # 模拟Webhook请求
    webhook_data = {
        "type": "enter",
        "exchange": "binance",
        "pair": "BTC/USDT",
        "side": "long",
        "amount": 0.1,
        "price": 50000.0,
        "status": "closed",
        "trade_id": 1001,
        "message": "Position opened"
    }

    # 这里应该测试实际的Webhook处理器
    # 由于是集成测试，我们需要模拟HTTP请求
    # 这个测试需要实际的FastAPI测试客户端

    # 暂时验证数据结构
    webhook_request = WebhookRequest(**webhook_data)
    assert webhook_request.type == "enter"
    assert webhook_request.pair == "BTC/USDT"
    assert webhook_request.trade_id == 1001


@pytest.mark.asyncio
async def test_market_data_integration():
    """测试市场数据集成"""
    from deepalpha.transport.websocket.handlers.market_data import MarketDataBroadcaster
    from deepalpha.market.data import Candle

    # 创建广播器
    broadcaster = MarketDataBroadcaster()

    # 启动广播器
    await broadcaster.start()

    # 订阅交易对
    client_id = "test_client_001"
    await broadcaster.subscribe_symbol(client_id, "BTC/USDT")

    # 推送K线数据
    candle = Candle(
        symbol="BTC/USDT",
        exchange="binance",
        interval="1m",
        timestamp=datetime.utcnow(),
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50050.0,
        volume=125.5
    )

    # 这里需要模拟WebSocket推送
    # 由于是集成测试，实际验证需要WebSocket客户端

    # 停止广播器
    await broadcaster.stop()


@pytest.mark.asyncio
async def test_error_handling_integration():
    """测试错误处理集成"""
    # 创建会失败的客户端
    mock_client = Mock(spec=FreqtradeClient)
    mock_client.force_enter = AsyncMock(side_effect=Exception("API Error"))

    execution_engine = ExecutionEngine(freqtrade_client=mock_client)

    decision = Decision(
        id="error_test",
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        position_size_usd=1000.0,
        stop_loss=49000.0,
        take_profit=53000.0
    )

    # 执行应该失败但不应崩溃
    success, message, trade_id = await execution_engine.execute_decision(
        decision, 50000.0, 20000.0
    )

    assert success is False
    assert "执行异常" in message
    assert trade_id is None


@pytest.mark.asyncio
async def test_concurrent_execution():
    """测试并发执行"""
    import asyncio

    # 创建共享组件
    mock_client = Mock(spec=FreqtradeClient)
    mock_client.force_enter = AsyncMock(return_value=Mock(trade_id=1001))
    position_manager = PositionManager()
    risk_manager = RiskManager()

    execution_engine = ExecutionEngine(
        freqtrade_client=mock_client,
        risk_manager=risk_manager
    )
    execution_engine.position_manager = position_manager

    # 创建多个决策
    decisions = []
    for i in range(5):
        decision = Decision(
            id=f"concurrent_{i}",
            symbol=f"BTC{i}/USDT",
            action="enter_long",
            confidence=85,
            position_size_usd=200.0,
            stop_loss=49000.0,
            take_profit=53000.0
        )
        decisions.append(decision)

    # 并发执行
    tasks = []
    for decision in decisions:
        task = execution_engine.execute_decision(decision, 50000.0, 20000.0)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # 验证所有决策都执行了
    assert len(results) == 5
    for success, message, trade_id in results:
        assert success is True
        assert trade_id is not None

    # 验证调用次数
    assert mock_client.force_enter.call_count == 5


@pytest.mark.asyncio
async def test_state_persistence_integration():
    """测试状态持久化集成"""
    # 这个测试需要实际的数据库集成
    # 暂时验证状态管理逻辑

    position_manager = PositionManager()
    order_manager = OrderManager(Mock(), position_manager)

    # 创建订单和持仓
    order = await order_manager.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        amount=0.1,
        price=50000.0
    )

    position = Position(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        size=Decimal('0.1'),
        entry_price=Decimal('50000.0')
    )
    await position_manager._positions["BTC/USDT"] = position

    # 验证状态
    all_orders = await order_manager.get_all_orders()
    all_positions = await position_manager.get_all_positions()

    assert len(all_orders) == 1
    assert len(all_positions) == 1
    assert list(all_positions.keys())[0] == "BTC/USDT"