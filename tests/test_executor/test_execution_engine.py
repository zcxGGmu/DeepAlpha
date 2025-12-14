"""测试执行引擎"""

import pytest
from unittest.mock import AsyncMock, patch, Mock

from deepalpha.executor.core import ExecutionEngine
from deepalpha.decision.types import Decision, DecisionStatus, DecisionTiers


@pytest.mark.asyncio
async def test_execute_decision_enter_long(execution_engine, sample_decision):
    """测试执行开多决策"""
    current_price = 50000.0
    balance = 20000.0

    # 模拟force_enter返回
    execution_engine.client.force_enter = AsyncMock(return_value=Mock(trade_id=1001))

    success, message, trade_id = await execution_engine.execute_decision(
        sample_decision, current_price, balance
    )

    assert success is True
    assert message == "执行成功"
    assert trade_id == 1001

    # 验证调用
    execution_engine.client.force_enter.assert_called_once()
    call_args = execution_engine.client.force_enter.call_args[0][0]
    assert call_args.pair == "BTC/USDT"
    assert call_args.side == "long"
    assert call_args.stakeamount == 1000.0
    assert call_args.leverage == 2.0


@pytest.mark.asyncio
async def test_execute_decision_enter_short(execution_engine):
    """测试执行开空决策"""
    decision = Decision(
        id="decision_002",
        symbol="BTC/USDT",
        action="enter_short",
        confidence=85,
        status=DecisionStatus.PENDING,
        position_size_usd=1000.0,
        leverage=3.0,
        stop_loss=51000.0,
        take_profit=47000.0
    )

    execution_engine.client.force_enter = AsyncMock(return_value=Mock(trade_id=1002))

    success, message, trade_id = await execution_engine.execute_decision(
        decision, 50000.0, 20000.0
    )

    assert success is True
    assert trade_id == 1002

    # 验证调用参数
    call_args = execution_engine.client.force_enter.call_args[0][0]
    assert call_args.side == "short"
    assert call_args.leverage == 3.0


@pytest.mark.asyncio
async def test_execute_decision_close_position(execution_engine):
    """测试执行平仓决策"""
    decision = Decision(
        id="decision_003",
        symbol="BTC/USDT",
        action="close_long",
        confidence=85,
        status=DecisionStatus.PENDING,
        close_ratio=0.5  # 部分平仓
    )

    # 设置持仓
    from deepalpha.core.models import OrderSide, Position
    position = Position(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        size=0.2,  # 0.2个BTC
        entry_price=50000.0
    )
    await execution_engine.position_manager._positions["BTC/USDT"] = position
    execution_engine._trade_to_decision[1001] = "BTC/USDT"  # 模拟映射关系

    execution_engine.client.force_exit = AsyncMock()

    success, message, trade_id = await execution_engine.execute_decision(
        decision, 50000.0, 20000.0
    )

    assert success is True
    assert message == "执行成功"
    assert trade_id == 1001  # 返回原有的trade_id

    # 验证平仓参数
    call_args = execution_engine.client.force_exit.call_args[0][0]
    assert call_args.tradeid == "1001"
    assert call_args.amount == 0.1  # 0.2 * 0.5


@pytest.mark.asyncio
async def test_execute_decision_hold(execution_engine):
    """测试持有决策"""
    decision = Decision(
        id="decision_004",
        symbol="BTC/USDT",
        action="hold",
        confidence=85,
        status=DecisionStatus.PENDING
    )

    success, message, trade_id = await execution_engine.execute_decision(
        decision, 50000.0, 20000.0
    )

    assert success is True
    assert message == "持有"
    assert trade_id is None


@pytest.mark.asyncio
async def test_execute_decision_unsupported_action(execution_engine):
    """测试不支持的行动类型"""
    decision = Decision(
        id="decision_005",
        symbol="BTC/USDT",
        action="unknown_action",
        confidence=85,
        status=DecisionStatus.PENDING
    )

    success, message, trade_id = await execution_engine.execute_decision(
        decision, 50000.0, 20000.0
    )

    assert success is False
    assert "不支持的行动" in message
    assert trade_id is None


@pytest.mark.asyncio
async def test_execute_decision_risk_failed(execution_engine):
    """测试风险检查失败"""
    decision = Decision(
        id="decision_006",
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        position_size_usd=20000.0,  # 过大的仓位
        status=DecisionStatus.PENDING
    )

    success, message, trade_id = await execution_engine.execute_decision(
        decision, 50000.0, 20000.0
    )

    assert success is False
    assert "风险检查失败" in message
    assert trade_id is None


@pytest.mark.asyncio
async def test_execute_decision_invalid_position_size(execution_engine):
    """测试无效的仓位大小"""
    decision = Decision(
        id="decision_007",
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        position_size_usd=0,  # 无效大小
        status=DecisionStatus.PENDING
    )

    success, message, trade_id = await execution_engine.execute_decision(
        decision, 50000.0, 20000.0
    )

    assert success is False
    assert "仓位大小必须大于0" in message
    assert trade_id is None


@pytest.mark.asyncio
async def test_execute_decision_close_without_position(execution_engine):
    """测试平仓但无持仓"""
    decision = Decision(
        id="decision_008",
        symbol="BTC/USDT",
        action="close_long",
        confidence=85,
        status=DecisionStatus.PENDING
    )

    success, message, trade_id = await execution_engine.execute_decision(
        decision, 50000.0, 20000.0
    )

    assert success is False
    assert "未找到持仓" in message
    assert trade_id is None


@pytest.mark.asyncio
async def test_update_tiers(execution_engine):
    """测试更新层级"""
    decision = Decision(
        id="decision_009",
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        status=DecisionStatus.PENDING,
        tiers=DecisionTiers(
            tier1_target=53000.0,
            tier1_ratio=0.3,
            tier2_target=56000.0,
            tier2_ratio=0.4,
            tier3_target=60000.0,
            tier3_ratio=0.3
        )
    )

    # 设置持仓
    from deepalpha.core.models import OrderSide, Position
    position = Position(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        size=0.1,
        entry_price=50000.0
    )
    await execution_engine.position_manager._positions["BTC/USDT"] = position
    execution_engine._trade_to_decision[1001] = "BTC/USDT"

    # Mock position_manager的update_tier方法
    execution_engine.position_manager.update_tier = AsyncMock()

    success = await execution_engine.update_tiers(1001, decision)

    assert success is True

    # 验证调用
    assert execution_engine.position_manager.update_tier.call_count == 3
    execution_engine.position_manager.update_tier.assert_any_call(1001, 1, 53000.0, 0.3)
    execution_engine.position_manager.update_tier.assert_any_call(1001, 2, 56000.0, 0.4)
    execution_engine.position_manager.update_tier.assert_any_call(1001, 3, 60000.0, 0.3)


@pytest.mark.asyncio
async def test_get_positions(execution_engine):
    """测试获取持仓"""
    # 添加测试持仓
    from deepalpha.core.models import OrderSide, Position
    positions = {
        "BTC/USDT": Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            size=0.1,
            entry_price=50000.0
        ),
        "ETH/USDT": Position(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            size=1.0,
            entry_price=3000.0
        )
    }
    execution_engine.position_manager._positions = positions

    result = await execution_engine.get_positions()

    assert len(result) == 2
    assert result[0]["symbol"] == "BTC/USDT"
    assert result[1]["symbol"] == "ETH/USDT"


@pytest.mark.asyncio
async def test_get_risk_metrics(execution_engine):
    """测试获取风险指标"""
    metrics = await execution_engine.get_risk_metrics()

    assert "max_position_size" in metrics
    assert "max_total_position" in metrics
    assert "stop_loss" in metrics
    assert "take_profit" in metrics


@pytest.mark.asyncio
async def test_sync_positions(execution_engine):
    """测试同步持仓"""
    # Mock API响应
    mock_position = Mock()
    mock_position.symbol = "BTC/USDT"
    mock_position.amount = 0.1
    mock_position.entry_price = 50000.0
    mock_position.current_price = 51000.0
    mock_position.unrealized_pnl_usd = 100.0
    mock_position.realized_pnl_usd = 0.0
    mock_position.status = "open"
    mock_position.trade_id = 1001

    execution_engine.client.get_positions = AsyncMock(return_value=Mock(
        positions=[mock_position]
    ))

    # Mock position_manager的update_from_api
    execution_engine.position_manager.update_from_api = AsyncMock()

    count = await execution_engine.sync_positions()

    assert count == 1
    execution_engine.client.get_positions.assert_called_once()
    execution_engine.position_manager.update_from_api.assert_called_once()


@pytest.mark.asyncio
async def test_get_balance(execution_engine):
    """测试获取余额"""
    # Mock API响应
    mock_balance = Mock()
    mock_balance.total = 20000.0
    mock_balance.free = 15000.0
    mock_balance.used = 5000.0
    mock_balance.dict.return_value = {
        "total": 20000.0,
        "free": 15000.0,
        "used": 5000.0
    }

    execution_engine.client.get_balance = AsyncMock(return_value=mock_balance)

    balance = await execution_engine.get_balance()

    assert balance["total"] == 20000.0
    assert balance["free"] == 15000.0
    assert balance["used"] == 5000.0


def test_get_execution_status(execution_engine):
    """测试获取执行状态"""
    # 添加执行中的决策
    from deepalpha.decision.types import Decision
    decision = Decision(
        id="test_decision",
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        status=DecisionStatus.PENDING
    )
    trace_id = "test_trace"
    execution_engine._executing_decisions[trace_id] = decision
    execution_engine._decision_to_trade[trace_id] = 1001
    execution_engine._trade_to_decision[1001] = trace_id

    status = execution_engine.get_execution_status()

    assert trace_id in status["executing_decisions"]
    assert status["decision_to_trade"][trace_id] == 1001
    assert status["trade_to_decision"][1001] == trace_id


@pytest.mark.asyncio
async def test_cleanup_completed_trades(execution_engine):
    """测试清理已完成交易"""
    # 这个方法目前是空实现，但应该能正常调用
    await execution_engine.cleanup_completed_trades()
    # 没有异常就算通过


@pytest.mark.asyncio
async def test_close(execution_engine):
    """测试关闭执行引擎"""
    # Mock client的close方法
    execution_engine.client.close = AsyncMock()

    await execution_engine.close()

    execution_engine.client.close.assert_called_once()