"""pytest配置文件"""

import pytest
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, AsyncMock

from deepalpha.core.models import (
    Order, OrderSide, OrderType, OrderStatus,
    Position, Decision, DecisionStatus, DecisionTiers
)
from deepalpha.decision.types import Decision as DecisionType, DecisionContext
from deepalpha.executor.freqtrade.client import FreqtradeClient
from deepalpha.executor.position import PositionManager
from deepalpha.executor.order import OrderManager
from deepalpha.executor.risk import RiskManager
from deepalpha.executor.core import ExecutionEngine


@pytest.fixture
def event_loop():
    """创建事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_freqtrade_client():
    """模拟Freqtrade客户端"""
    client = Mock(spec=FreqtradeClient)

    # 模拟force_enter响应
    client.force_enter = AsyncMock(return_value=Mock(trade_id=1001))

    # 模拟force_exit
    client.force_exit = AsyncMock()

    # 模拟get_positions
    client.get_positions = AsyncMock(return_value=Mock(
        positions=[]
    ))

    # 模拟get_balance
    client.get_balance = AsyncMock(return_value=Mock(
        total=Decimal('10000'),
        free=Decimal('8000'),
        used=Decimal('2000')
    ))

    return client


@pytest.fixture
def position_manager():
    """仓位管理器实例"""
    return PositionManager()


@pytest.fixture
def order_manager(mock_freqtrade_client, position_manager):
    """订单管理器实例"""
    return OrderManager(
        freqtrade_client=mock_freqtrade_client,
        position_manager=position_manager
    )


@pytest.fixture
def risk_manager():
    """风险管理器实例"""
    return RiskManager(
        max_position_size=0.02,
        max_total_position=0.5,
        stop_loss=0.02,
        take_profit=0.06,
        max_drawdown=0.10,
        max_trades_per_day=10
    )


@pytest.fixture
def execution_engine(mock_freqtrade_client):
    """执行引擎实例"""
    return ExecutionEngine(
        freqtrade_client=mock_freqtrade_client,
        horizon="24h"
    )


@pytest.fixture
def sample_decision():
    """示例决策"""
    return DecisionType(
        id="decision_001",
        symbol="BTC/USDT",
        action="enter_long",
        confidence=85,
        status=DecisionStatus.PENDING,
        timestamp=datetime.utcnow(),
        position_size_usd=1000.0,
        leverage=2.0,
        stop_loss=49000.0,
        take_profit=53000.0,
        tiers=DecisionTiers(
            tier1_target=53000.0,
            tier1_ratio=0.3,
            tier2_target=56000.0,
            tier2_ratio=0.4
        ),
        reason="技术指标显示看涨",
        agent_name="technical_agent"
    )


@pytest.fixture
def sample_position():
    """示例持仓"""
    return Position(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        size=Decimal('0.1'),
        entry_price=Decimal('50000.0'),
        current_price=Decimal('51000.0'),
        unrealized_pnl=Decimal('100.0'),
        realized_pnl=Decimal('0.0'),
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def sample_order():
    """示例订单"""
    return Order(
        id="order_001",
        client_order_id="client_order_001",
        symbol="BTC/USDT",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        amount=0.1,
        price=50000.0,
        status=OrderStatus.OPEN,
        timestamp=datetime.utcnow()
    )


# 测试数据生成器
def generate_test_candles(count=10, symbol="BTC/USDT"):
    """生成测试K线数据"""
    candles = []
    base_price = 50000.0
    base_time = datetime.utcnow()

    for i in range(count):
        candles.append({
            "symbol": symbol,
            "exchange": "binance",
            "interval": "1m",
            "timestamp": base_time,
            "open": base_price + i * 10,
            "high": base_price + i * 10 + 50,
            "low": base_price + i * 10 - 30,
            "close": base_price + i * 10 + 20,
            "volume": 100.5 + i
        })

    return candles


def generate_test_decisions(count=5):
    """生成测试决策"""
    decisions = []
    actions = ["enter_long", "enter_short", "close_long", "close_short", "hold"]

    for i in range(count):
        decisions.append({
            "id": f"decision_{i:03d}",
            "symbol": "BTC/USDT",
            "action": actions[i % len(actions)],
            "confidence": 70 + i * 5,
            "status": "pending",
            "position_size_usd": 1000.0 * (i + 1),
            "leverage": 1 + i % 3,
            "timestamp": datetime.utcnow()
        })

    return decisions


# 测试工具函数
async def create_test_position(manager, symbol="BTC/USDT", side="long", size=0.1, price=50000.0):
    """创建测试持仓"""
    position = Position(
        symbol=symbol,
        side=OrderSide.BUY if side == "long" else OrderSide.SELL,
        size=Decimal(str(size)),
        entry_price=Decimal(str(price)),
        current_price=Decimal(str(price)),
        unrealized_pnl=Decimal('0'),
        realized_pnl=Decimal('0'),
        timestamp=datetime.utcnow()
    )

    # 添加到管理器
    async with manager._lock:
        manager._positions[symbol] = position
        manager._trade_id_map[1001] = symbol

    return position


def assert_decimal_equal(actual, expected, places=2):
    """断言两个Decimal值相等（保留指定位数）"""
    quantize = Decimal('1').scaleb(-places)
    assert actual.quantize(quantize) == expected.quantize(quantize)