"""测试风险管理器"""

import pytest
from decimal import Decimal

from deepalpha.executor.risk import RiskManager
from deepalpha.core.models import OrderSide, Position
from deepalpha.decision.types import Decision, DecisionStatus, DecisionTiers


@pytest.mark.asyncio
async def test_check_decision_risk_pass(risk_manager, sample_decision):
    """测试决策风险检查通过"""
    positions = {}
    balance = 20000.0

    # 正常的决策应该通过
    risk_pass, risk_reason = await risk_manager.check_decision_risk(
        sample_decision, 50000.0, positions, balance
    )

    assert risk_pass is True
    assert risk_reason == "风险检查通过"


@pytest.mark.asyncio
async def test_check_decision_risk_position_size_exceeded(risk_manager):
    """测试仓位大小超限"""
    large_decision = Decision(
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        position_size_usd=5000.0,  # 超过2%限制(20000*0.02=400)
        status=DecisionStatus.PENDING
    )

    positions = {}
    balance = 20000.0

    risk_pass, risk_reason = await risk_manager.check_decision_risk(
        large_decision, 50000.0, positions, balance
    )

    assert risk_pass is False
    assert "仓位大小超限" in risk_reason


@pytest.mark.asyncio
async def test_check_decision_risk_total_position_exceeded(risk_manager):
    """测试总仓位超限"""
    # 创建已有持仓（总价值9000，接近50%限制）
    positions = {
        "ETH/USDT": Position(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            size=Decimal('3.0'),
            entry_price=Decimal('3000.0'),
            current_price=Decimal('3000.0')
        )
    }

    balance = 20000.0

    # 新决策再增加2000仓位
    new_decision = Decision(
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        position_size_usd=2000.0,
        status=DecisionStatus.PENDING
    )

    risk_pass, risk_reason = await risk_manager.check_decision_risk(
        new_decision, 50000.0, positions, balance
    )

    # 总仓位将是11000(9000+2000)，超过50%限制(20000*0.5=10000)
    assert risk_pass is False
    assert "总仓位超限" in risk_reason


@pytest.mark.asyncio
async def test_check_decision_risk_no_stop_loss(risk_manager):
    """测试未设置止损"""
    decision_no_sl = Decision(
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        position_size_usd=1000.0,
        stop_loss=None,  # 未设置止损
        status=DecisionStatus.PENDING
    )

    risk_pass, risk_reason = await risk_manager.check_decision_risk(
        decision_no_sl, 50000.0, {}, 20000.0
    )

    assert risk_pass is False
    assert "必须设置止损或止盈" in risk_reason


@pytest.mark.asyncio
async def test_check_decision_risk_stop_loss_too_small(risk_manager):
    """测试止损设置过小"""
    decision_small_sl = Decision(
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        position_size_usd=1000.0,
        stop_loss=49900.0,  # 仅0.2%的止损，小于最小1%
        status=DecisionStatus.PENDING
    )

    risk_pass, risk_reason = await risk_manager.check_decision_risk(
        decision_small_sl, 50000.0, {}, 20000.0
    )

    assert risk_pass is False
    assert "止损设置过小" in risk_reason


@pytest.mark.asyncio
async def test_check_decision_risk_profit_ratio_too_low(risk_manager):
    """测试盈亏比过低"""
    decision_low_ratio = Decision(
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        position_size_usd=1000.0,
        stop_loss=49000.0,  # 2%止损
        take_profit=50500.0,  # 仅1%止盈，盈亏比0.5:1
        status=DecisionStatus.PENDING
    )

    risk_pass, risk_reason = await risk_manager.check_decision_risk(
        decision_low_ratio, 50000.0, {}, 20000.0
    )

    assert risk_pass is False
    assert "盈亏比过低" in risk_reason


@pytest.mark.asyncio
async def test_check_order_risk(risk_manager):
    """测试订单风险检查"""
    order = Mock()
    order.amount = 0.1
    order.price = 50000.0
    order.side = OrderSide.BUY
    order.symbol = "BTC/USDT"

    positions = {}
    balance = 20000.0

    # 正常订单应该通过
    risk_pass, risk_reason = await risk_manager.check_order_risk(
        order, 50000.0, positions, balance
    )

    assert risk_pass is True
    assert "订单风险检查通过" in risk_reason


@pytest.mark.asyncio
async def test_check_order_risk_amount_too_large(risk_manager):
    """测试订单金额过大"""
    large_order = Mock()
    large_order.amount = 10.0  # 10个BTC
    large_order.price = 50000.0  # 总价值50万，超过95%余额
    large_order.side = OrderSide.BUY
    large_order.symbol = "BTC/USDT"

    positions = {}
    balance = 20000.0  # 只有2万余额

    risk_pass, risk_reason = await risk_manager.check_order_risk(
        large_order, 50000.0, positions, balance
    )

    assert risk_pass is False
    assert "订单金额过大" in risk_reason


def test_calculate_position_size(risk_manager):
    """测试计算仓位大小"""
    balance = 20000.0
    current_price = 50000.0

    # 100%置信度
    size = risk_manager.calculate_position_size("BTC/USDT", 100, current_price, balance)
    expected = balance * risk_manager.max_position_size  # 20000 * 0.02 = 400
    assert size == expected

    # 50%置信度
    size = risk_manager.calculate_position_size("BTC/USDT", 50, current_price, balance)
    expected = balance * risk_manager.max_position_size * 0.5  # 400 * 0.5 = 200
    assert size == expected

    # 带ATR调整
    atr = 1000.0  # ATR为1000，即2%的波动率
    size = risk_manager.calculate_position_size("BTC/USDT", 100, current_price, balance, atr)
    # 波动率因子 = 0.01 / (1000/50000) = 0.5
    # 最终仓位 = 400 * 0.5 = 200
    assert size == 200.0


def test_calculate_stop_loss(risk_manager):
    """测试计算止损价格"""
    entry_price = 50000.0

    # 买入止损
    stop_loss = risk_manager.calculate_stop_loss(OrderSide.BUY, entry_price)
    expected = entry_price * (1 - risk_manager.stop_loss)  # 50000 * 0.98 = 49000
    assert stop_loss == expected

    # 卖出止损
    stop_loss = risk_manager.calculate_stop_loss(OrderSide.SELL, entry_price)
    expected = entry_price * (1 + risk_manager.stop_loss)  # 50000 * 1.02 = 51000
    assert stop_loss == expected

    # 带ATR
    atr = 500.0
    stop_loss = risk_manager.calculate_stop_loss(OrderSide.BUY, entry_price, atr)
    expected = entry_price - atr * 2  # 50000 - 1000 = 49000
    assert stop_loss == expected


def test_calculate_take_profit(risk_manager):
    """测试计算止盈价格"""
    entry_price = 50000.0
    stop_loss = 49000.0

    # 买入止盈
    take_profit = risk_manager.calculate_take_profit(OrderSide.BUY, entry_price, stop_loss)
    # 最小盈亏 = (50000-49000) * 1.5 = 1500
    # 止盈价 = 50000 + 1500 = 51500
    expected = entry_price + abs(entry_price - stop_loss) * risk_manager.min_profit_ratio
    assert take_profit == expected

    # 卖出止盈
    take_profit = risk_manager.calculate_take_profit(OrderSide.SELL, entry_price, stop_loss)
    expected = entry_price - abs(entry_price - stop_loss) * risk_manager.min_profit_ratio
    assert take_profit == expected


def test_check_tier_executability(risk_manager):
    """测试层级是否可执行"""
    position = Mock()
    position.side = OrderSide.BUY

    # 未完成的层级，当前价格未达到
    tier_info = {
        "target": 55000.0,
        "ratio": 0.3,
        "done": False
    }
    current_price = 51000.0
    assert risk_manager.check_tier_executability(position, current_price, tier_info) is False

    # 达到目标价格
    current_price = 56000.0
    assert risk_manager.check_tier_executability(position, current_price, tier_info) is True

    # 已完成的层级
    tier_info["done"] = True
    assert risk_manager.check_tier_executability(position, current_price, tier_info) is False

    # 卖出方向的测试
    position.side = OrderSide.SELL
    tier_info["done"] = False
    current_price = 54000.0  # 低于目标价55000
    assert risk_manager.check_tier_executability(position, current_price, tier_info) is True


@pytest.mark.asyncio
async def test_daily_limit(risk_manager):
    """测试每日交易限制"""
    # 记录最大数量的交易
    for _ in range(risk_manager.max_trades_per_day):
        risk_manager.record_trade(100.0)

    # 下一个交易应该被拒绝
    risk_pass, risk_reason = await risk_manager._check_daily_limit()
    assert risk_pass is False
    assert "超过每日最大交易数限制" in risk_reason


@pytest.mark.asyncio
async def test_drawdown_check(risk_manager):
    """测试回撤检查"""
    # 初始余额
    balance = 20000.0

    # 正常情况
    risk_pass, _ = await risk_manager._check_drawdown(balance)
    assert risk_pass is True

    # 设置更高的峰值
    risk_manager._peak_balance = Decimal('25000.0')

    # 回撤10%刚好在限制内
    balance = 22500.0  # 回撤 (25000-22500)/25000 = 10%
    risk_pass, _ = await risk_manager._check_drawdown(balance)
    assert risk_pass is True

    # 超过回撤限制
    balance = 22000.0  # 回撤 (25000-22000)/25000 = 12%
    risk_pass, reason = await risk_manager._check_drawdown(balance)
    assert risk_pass is False
    assert "回撤" in reason


def test_record_trade(risk_manager):
    """测试记录交易"""
    initial_pnl = risk_manager._daily_pnl
    initial_count = len(risk_manager._daily_trades)

    risk_manager.record_trade(100.0)

    assert len(risk_manager._daily_trades) == initial_count + 1
    assert risk_manager._daily_pnl == initial_pnl + Decimal('100.0')


@pytest.mark.asyncio
async def test_get_risk_metrics(risk_manager):
    """测试获取风险指标"""
    # 记录一些交易
    risk_manager.record_trade(100.0)
    risk_manager.record_trade(-50.0)
    risk_manager.record_trade(200.0)

    # 设置峰值
    risk_manager._peak_balance = Decimal('25000.0')

    metrics = await risk_manager.get_risk_metrics()

    assert metrics["max_position_size"] == risk_manager.max_position_size
    assert metrics["max_total_position"] == risk_manager.max_total_position
    assert metrics["stop_loss"] == risk_manager.stop_loss
    assert metrics["take_profit"] == risk_manager.take_profit
    assert metrics["max_drawdown"] == risk_manager.max_drawdown
    assert metrics["max_trades_per_day"] == risk_manager.max_trades_per_day
    assert metrics["daily_trades"] == 3
    assert metrics["daily_pnl"] == 250.0
    assert metrics["peak_balance"] == 25000.0


# Mock类用于测试
class Mock:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)