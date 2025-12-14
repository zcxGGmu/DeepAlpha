"""测试订单管理器"""

import pytest
from datetime import datetime
from decimal import Decimal

from deepalpha.executor.order import OrderManager
from deepalpha.core.models import Order, OrderSide, OrderType, OrderStatus
from deepalpha.executor.position import PositionManager


@pytest.mark.asyncio
async def test_create_order(order_manager):
    """测试创建订单"""
    # 创建买单
    success, order = await order_manager.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        amount=0.1,
        price=50000.0,
        stop_loss=45000.0,
        take_profit=55000.0,
        client_order_id="test_001"
    )

    assert success is True
    assert order is not None
    assert order.symbol == "BTC/USDT"
    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.LIMIT
    assert order.amount == 0.1
    assert order.price == 50000.0
    assert order.status == OrderStatus.OPEN


@pytest.mark.asyncio
async def test_get_order(order_manager, sample_order):
    """测试获取订单"""
    # 先创建订单
    await order_manager._orders[sample_order.id] = sample_order

    # 通过order_id获取
    order = await order_manager.get_order(order_id=sample_order.id)
    assert order is not None
    assert order.id == sample_order.id

    # 通过client_order_id获取
    await order_manager._client_order_ids[sample_order.client_order_id] = sample_order.id
    order = await order_manager.get_order(client_order_id=sample_order.client_order_id)
    assert order is not None
    assert order.client_order_id == sample_order.client_order_id


@pytest.mark.asyncio
async def test_update_order(order_manager, sample_order):
    """测试更新订单"""
    # 添加订单
    await order_manager._orders[sample_order.id] = sample_order

    # 更新状态
    success = await order_manager.update_order(
        order_id=sample_order.id,
        status=OrderStatus.FILLED,
        price=50100.0
    )
    assert success is True

    # 验证更新
    order = await order_manager.get_order(order_id=sample_order.id)
    assert order.status == OrderStatus.FILLED
    assert order.price == 50100.0


@pytest.mark.asyncio
async def test_cancel_order(order_manager, sample_order):
    """测试取消订单"""
    # 添加订单
    sample_order.status = OrderStatus.OPEN
    await order_manager._orders[sample_order.id] = sample_order

    # 取消订单
    success = await order_manager.cancel_order(order_id=sample_order.id)
    assert success is True

    # 验证状态
    order = await order_manager.get_order(order_id=sample_order.id)
    assert order.status == OrderStatus.CANCELLED


@pytest.mark.asyncio
async def test_get_orders_by_symbol(order_manager):
    """测试按交易对获取订单"""
    # 创建多个订单
    symbols = ["BTC/USDT", "ETH/USDT", "BTC/USDT"]
    for i, symbol in enumerate(symbols):
        await order_manager.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.1,
            price=50000.0 + i * 1000,
            client_order_id=f"test_{i}"
        )

    # 获取BTC/USDT订单
    btc_orders = await order_manager.get_orders_by_symbol("BTC/USDT")
    assert len(btc_orders) == 2
    assert all(order.symbol == "BTC/USDT" for order in btc_orders)


@pytest.mark.asyncio
async def test_get_orders_by_status(order_manager):
    """测试按状态获取订单"""
    # 创建不同状态的订单
    orders_data = [
        ("BTC/USDT", OrderStatus.OPEN),
        ("ETH/USDT", OrderStatus.FILLED),
        ("BTC/USDT", OrderStatus.OPEN),
        ("BNB/USDT", OrderStatus.CANCELLED)
    ]

    for symbol, status in orders_data:
        order = await order_manager.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.1,
            price=50000.0
        )
        order.status = status

    # 获取OPEN状态的订单
    open_orders = await order_manager.get_orders_by_status(OrderStatus.OPEN)
    assert len(open_orders) == 2
    assert all(order.status == OrderStatus.OPEN for order in open_orders)


@pytest.mark.asyncio
async def test_get_active_orders(order_manager):
    """测试获取活跃订单"""
    # 创建订单
    statuses = [OrderStatus.OPEN, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.PENDING]
    for i, status in enumerate(statuses):
        order = await order_manager.create_order(
            symbol=f"TEST{i}/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.1,
            price=50000.0
        )
        order.status = status

    # 获取活跃订单（OPEN和PENDING）
    active_orders = await order_manager.get_active_orders()
    assert len(active_orders) == 2


@pytest.mark.asyncio
async def test_cleanup_completed_orders(order_manager):
    """测试清理已完成订单"""
    from datetime import timedelta

    # 创建订单
    now = datetime.utcnow()
    old_time = now - timedelta(days=35)  # 超过30天

    # 创建旧订单
    old_order = await order_manager.create_order(
        symbol="OLD/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=0.1,
        price=50000.0
    )
    old_order.status = OrderStatus.FILLED
    old_order.timestamp = old_time

    # 创建新订单
    new_order = await order_manager.create_order(
        symbol="NEW/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=0.1,
        price=50000.0
    )
    new_order.status = OrderStatus.FILLED

    # 清理30天前的订单
    cleaned_count = await order_manager.cleanup_completed_orders(max_age_days=30)
    assert cleaned_count == 1

    # 验证只剩新订单
    all_orders = await order_manager.get_all_orders()
    assert len(all_orders) == 1
    assert all_orders[0].symbol == "NEW/USDT"


def test_get_statistics(order_manager):
    """测试获取订单统计"""
    # 手动添加订单到统计
    order_manager._orders = {
        "order_1": Order(id="order_1", status=OrderStatus.OPEN),
        "order_2": Order(id="order_2", status=OrderStatus.FILLED),
        "order_3": Order(id="order_3", status=OrderStatus.FILLED),
        "order_4": Order(id="order_4", status=OrderStatus.CANCELLED),
        "order_5": Order(id="order_5", status=OrderStatus.REJECTED),
    }

    stats = order_manager.get_statistics()
    assert stats["total"] == 5
    assert stats["open"] == 1
    assert stats["filled"] == 2
    assert stats["cancelled"] == 1
    assert stats["rejected"] == 1
    assert stats["pending"] == 0
    assert stats["expired"] == 0


@pytest.mark.asyncio
async def test_estimate_fill_probability(order_manager):
    """测试估算成交概率"""
    # 市价单
    market_order = Order(
        id="test",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        amount=0.1
    )
    prob = await order_manager.estimate_fill_probability(market_order, 50000.0)
    assert prob == 0.9  # 市价单高概率

    # 限价单 - 价格低于当前价（买单）
    limit_order = Order(
        id="test",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        amount=0.1,
        price=49000.0  # 低于当前价50000
    )
    prob = await order_manager.estimate_fill_probability(limit_order, 50000.0)
    assert prob > 0.5  # 应该有较高概率


@pytest.mark.asyncio
async def test_get_order_impact(order_manager):
    """测试获取订单影响"""
    # 市价单
    market_order = Order(
        id="test",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        amount=0.1,
        price=0.0
    )
    impact = await order_manager.get_order_impact(market_order, 50000.0)
    assert impact["price_impact"] == 0.0
    assert impact["slippage_estimate"] == 0.01  # 估算1%滑点

    # 限价单
    limit_order = Order(
        id="test",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        amount=0.1,
        price=49000.0
    )
    impact = await order_manager.get_order_impact(limit_order, 50000.0)
    assert impact["price_impact"] < 0  # 买价低于市价
    assert impact["volume_impact"] == 0.1