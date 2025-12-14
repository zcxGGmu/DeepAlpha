"""测试仓位管理器"""

import pytest
from datetime import datetime
from decimal import Decimal

from deepalpha.executor.position import PositionManager
from deepalpha.core.models import OrderSide, Position
from deepalpha.executor.freqtrade.types import APIPosition, TierInfo


@pytest.mark.asyncio
async def test_update_from_api(position_manager):
    """测试从API更新持仓"""
    # 准备API数据
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
        ),
        APIPosition(
            trade_id=1003,
            symbol="BNB/USDT",
            side="long",
            amount=10.0,
            entry_price=300.0,
            current_price=0.0,
            unrealized_pnl_usd=0.0,
            realized_pnl_usd=0.0,
            status="closed"  # 应该被跳过
        )
    ]

    # 更新
    await position_manager.update_from_api(api_positions)

    # 验证
    assert len(position_manager._positions) == 2  # 排除已平仓
    assert position_manager._trade_id_map[1001] == "BTC/USDT"
    assert position_manager._trade_id_map[1002] == "ETH/USDT"

    # 验证BTC持仓
    btc_pos = await position_manager.get_position("BTC/USDT")
    assert btc_pos is not None
    assert btc_pos.side == OrderSide.BUY
    assert btc_pos.size == Decimal('0.1')
    assert btc_pos.entry_price == Decimal('50000.0')
    assert btc_pos.current_price == Decimal('51000.0')
    assert btc_pos.unrealized_pnl == Decimal('100.0')

    # 验证层级信息
    assert "tier1" in btc_pos.metadata
    assert btc_pos.metadata["tier1"]["target"] == 55000.0


@pytest.mark.asyncio
async def test_get_position_by_trade_id(position_manager):
    """测试通过trade_id获取持仓"""
    # 添加测试持仓
    position = Position(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        size=Decimal('0.1'),
        entry_price=Decimal('50000.0')
    )
    await position_manager._positions["BTC/USDT"] = position
    await position_manager._trade_id_map[1001] = "BTC/USDT"

    # 获取持仓
    found_pos = await position_manager.get_position_by_trade_id(1001)
    assert found_pos is not None
    assert found_pos.symbol == "BTC/USDT"

    # 不存在的trade_id
    not_found = await position_manager.get_position_by_trade_id(9999)
    assert not_found is None


@pytest.mark.asyncio
async def test_update_position(position_manager):
    """测试更新持仓"""
    # 创建持仓
    position = Position(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        size=Decimal('0.1'),
        entry_price=Decimal('50000.0'),
        current_price=Decimal('50000.0'),
        unrealized_pnl=Decimal('0.0')
    )
    await position_manager._positions["BTC/USDT"] = position

    # 更新价格和盈亏
    await position_manager.update_position(
        symbol="BTC/USDT",
        current_price=51000.0,
        unrealized_pnl=100.0
    )

    # 验证更新
    updated_pos = await position_manager.get_position("BTC/USDT")
    assert updated_pos.current_price == Decimal('51000.0')
    assert updated_pos.unrealized_pnl == Decimal('100.0')


@pytest.mark.asyncio
async def test_close_position(position_manager):
    """测试平仓"""
    # 创建持仓
    position = Position(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        size=Decimal('0.1'),
        entry_price=Decimal('50000.0')
    )
    await position_manager._positions["BTC/USDT"] = position
    await position_manager._trade_id_map[1001] = "BTC/USDT"

    # 平仓
    await position_manager.close_position(
        symbol="BTC/USDT",
        exit_price=51000.0,
        exit_reason="止盈",
        realized_pnl=100.0
    )

    # 验证已移除
    pos = await position_manager.get_position("BTC/USDT")
    assert pos is None

    # 验证映射已清理
    assert "BTC/USDT" not in position_manager._trade_id_map.values()


@pytest.mark.asyncio
async def test_update_tier(position_manager):
    """测试更新层级"""
    # 创建持仓
    position = Position(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        size=Decimal('0.1'),
        entry_price=Decimal('50000.0')
    )
    await position_manager._positions["BTC/USDT"] = position
    await position_manager._trade_id_map[1001] = "BTC/USDT"

    # 更新层级1
    await position_manager.update_tier(
        trade_id=1001,
        tier=1,
        target=55000.0,
        ratio=0.3,
        done=False
    )

    # 验证更新
    pos = await position_manager.get_position("BTC/USDT")
    assert "tier1" in pos.metadata
    assert pos.metadata["tier1"]["target"] == 55000.0
    assert pos.metadata["tier1"]["ratio"] == 0.3
    assert pos.metadata["tier1"]["done"] is False

    # 更新层级状态
    await position_manager.update_tier(
        trade_id=1001,
        tier=1,
        done=True
    )
    pos = await position_manager.get_position("BTC/USDT")
    assert pos.metadata["tier1"]["done"] is True


@pytest.mark.asyncio
async def test_get_total_exposure(position_manager):
    """测试计算总敞口"""
    # 添加多个持仓
    positions_data = [
        ("BTC/USDT", 0.1, 50000.0),
        ("ETH/USDT", 1.0, 3000.0),
        ("BNB/USDT", 10.0, 300.0)
    ]

    for symbol, size, price in positions_data:
        position = Position(
            symbol=symbol,
            side=OrderSide.BUY,
            size=Decimal(str(size)),
            entry_price=Decimal(str(price))
        )
        await position_manager._positions[symbol] = position

    # 计算敞口
    exposure = await position_manager.get_total_exposure()
    expected = Decimal('0.1') * Decimal('50000.0') + \
                Decimal('1.0') * Decimal('3000.0') + \
                Decimal('10.0') * Decimal('300.0')
    assert exposure == expected


@pytest.mark.asyncio
async def test_get_pnl_summary(position_manager):
    """测试获取盈亏汇总"""
    # 添加测试持仓
    positions_data = [
        ("BTC/USDT", 100.0, 50.0),  # 盈利
        ("ETH/USDT", -30.0, 20.0),  # 亏损
        ("BNB/USDT", 0.0, 0.0),     # 持平
        ("ADA/USDT", 150.0, 0.0),   # 盈利
    ]

    for symbol, unrealized, realized in positions_data:
        position = Position(
            symbol=symbol,
            side=OrderSide.BUY,
            size=Decimal('0.1'),
            entry_price=Decimal('50000.0'),
            unrealized_pnl=Decimal(str(unrealized)),
            realized_pnl=Decimal(str(realized))
        )
        await position_manager._positions[symbol] = position

    # 获取汇总
    summary = await position_manager.get_pnl_summary()
    assert summary["total_positions"] == 4
    assert summary["total_unrealized"] == 220.0
    assert summary["total_realized"] == 70.0
    assert summary["total_pnl"] == 290.0
    assert summary["profit_positions"] == 2  # BTC和ADA
    assert summary["loss_positions"] == 1   # ETH


@pytest.mark.asyncio
async def test_get_symbol_pnl(position_manager):
    """测试获取指定交易对盈亏"""
    # 创建持仓
    position = Position(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        size=Decimal('0.1'),
        entry_price=Decimal('50000.0'),
        current_price=Decimal('51000.0'),
        unrealized_pnl=Decimal('100.0'),
        realized_pnl=Decimal('50.0')
    )
    await position_manager._positions["BTC/USDT"] = position

    # 获取盈亏信息
    pnl_info = await position_manager.get_symbol_pnl("BTC/USDT")
    assert pnl_info["symbol"] == "BTC/USDT"
    assert pnl_info["unrealized_pnl"] == 100.0
    assert pnl_info["realized_pnl"] == 50.0
    assert pnl_info["total_pnl"] == 150.0
    assert pnl_info["pnl_percent"] == 2.0  # (51000-50000)/50000 * 100


@pytest.mark.asyncio
async def test_check_risk_limits(position_manager):
    """测试检查风险限制"""
    # 添加持仓（总价值5000）
    position = Position(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        size=Decimal('0.1'),
        entry_price=Decimal('50000.0'),
        current_price=Decimal('50000.0')
    )
    await position_manager._positions["BTC/USDT"] = position

    # 检查风险
    risk_info = await position_manager.check_risk_limits(
        max_exposure=10000.0,
        max_drawdown=0.1,
        current_balance=20000.0
    )

    assert risk_info["total_exposure"] == 5000.0
    assert risk_info["exposure_ratio"] == 0.25  # 5000/20000
    assert risk_info["max_exceeded"] is False  # 5000 < 10000
    assert risk_info["drawdown"] >= 0  # 盈亏为0
    assert risk_info["drawdown_exceeded"] is False
    assert risk_info["risk_level"] == "NORMAL"

    # 测试超限情况
    risk_info = await position_manager.check_risk_limits(
        max_exposure=4000.0,
        max_drawdown=0.1,
        current_balance=20000.0
    )
    assert risk_info["max_exceeded"] is True  # 5000 > 4000
    assert risk_info["risk_level"] == "HIGH"