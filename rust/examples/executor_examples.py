"""Trading Execution Engine Usage Examples"""

import asyncio
import time
import json
from deepalpha_rust import ExecutionEngine


def example_1_basic_usage():
    """Example 1: Basic order submission and management"""
    print("=" * 60)
    print("Example 1: Basic Order Submission")
    print("=" * 60)

    # Create execution engine
    engine = ExecutionEngine()

    # Check initial state
    print(f"Engine active: {engine.active}")
    stats = engine.get_stats()
    print(f"Initial stats: Total={stats.total_orders}, Filled={stats.filled_orders}")

    # Start the engine
    engine.start()
    time.sleep(0.1)  # Give it time to initialize

    print(f"\nEngine started: {engine.active}")

    # Submit a market order
    market_order = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "type": "market",
        "quantity": 1.5
    }

    order_id = engine.submit_order(market_order)
    print(f"\nSubmitted market order: {order_id}")

    # Submit a limit order
    limit_order = {
        "symbol": "ETH/USDT",
        "side": "sell",
        "type": "limit",
        "quantity": 2.0,
        "price": 3200.0
    }

    limit_order_id = engine.submit_order(limit_order)
    print(f"Submitted limit order: {limit_order_id}")

    # Wait for processing
    time.sleep(0.5)

    # Check order status
    order_info = engine.get_order(order_id)
    if order_info:
        print(f"\nMarket order status: {order_info['status']}")
        print(f"Filled quantity: {order_info['filled_quantity']}")
        print(f"Average fill price: {order_info['avg_fill_price']}")

    # Get updated statistics
    stats = engine.get_stats()
    print(f"\nUpdated stats:")
    print(f"  Total orders: {stats.total_orders}")
    print(f"  Filled orders: {stats.filled_orders}")
    print(f"  Cancelled orders: {stats.cancelled_orders}")
    print(f"  Rejected orders: {stats.rejected_orders}")
    print(f"  Average execution time: {stats.avg_execution_time_us:.2f} μs")

    # Stop the engine
    engine.stop()
    print(f"\nEngine stopped: {engine.active}")


def example_2_order_cancellation():
    """Example 2: Order cancellation lifecycle"""
    print("\n" + "=" * 60)
    print("Example 2: Order Cancellation")
    print("=" * 60)

    engine = ExecutionEngine()
    engine.start()
    time.sleep(0.1)

    # Submit multiple limit orders
    order_ids = []
    for i in range(5):
        order_data = {
            "symbol": "BTC/USDT",
            "side": "buy" if i % 2 == 0 else "sell",
            "type": "limit",
            "quantity": 1.0,
            "price": 49000.0 + i * 100,  # Different prices
        }

        order_id = engine.submit_order(order_data)
        order_ids.append(order_id)
        print(f"Submitted order {i+1}: {order_id}")

    time.sleep(0.2)

    # Check initial status
    for i, order_id in enumerate(order_ids):
        order_info = engine.get_order(order_id)
        if order_info:
            print(f"Order {i+1} status: {order_info['status']}")

    # Cancel some orders
    print("\nCancelling orders 2 and 4...")
    cancelled_2 = engine.cancel_order(order_ids[1])
    cancelled_4 = engine.cancel_order(order_ids[3])

    print(f"Order 2 cancelled: {cancelled_2}")
    print(f"Order 4 cancelled: {cancelled_4}")

    time.sleep(0.2)

    # Check final status
    print("\nFinal order statuses:")
    for i, order_id in enumerate(order_ids):
        order_info = engine.get_order(order_id)
        if order_info:
            print(f"Order {i+1}: {order_info['status']}")

    stats = engine.get_stats()
    print(f"\nFinal stats: Cancelled={stats.cancelled_orders}")

    engine.stop()


def example_3_position_tracking():
    """Example 3: Position and portfolio tracking"""
    print("\n" + "=" * 60)
    print("Example 3: Position Tracking")
    print("=" * 60)

    engine = ExecutionEngine()
    engine.start()
    time.sleep(0.1)

    # Build a position in BTC
    print("Building BTC position...")
    btc_orders = [
        {"symbol": "BTC/USDT", "side": "buy", "type": "market", "quantity": 2.0, "price": 50000.0},
        {"symbol": "BTC/USDT", "side": "buy", "type": "market", "quantity": 1.5, "price": 50100.0},
        {"symbol": "BTC/USDT", "side": "buy", "type": "market", "quantity": 1.0, "price": 50200.0},
    ]

    for order in btc_orders:
        order_id = engine.submit_order(order)
        print(f"Submitted BTC buy order: {order['quantity']} @ ${order['price']}")

    # Build a position in ETH
    print("\nBuilding ETH position...")
    eth_orders = [
        {"symbol": "ETH/USDT", "side": "buy", "type": "market", "quantity": 10.0, "price": 3100.0},
        {"symbol": "ETH/USDT", "side": "buy", "type": "market", "quantity": 5.0, "price": 3110.0},
    ]

    for order in eth_orders:
        order_id = engine.submit_order(order)
        print(f"Submitted ETH buy order: {order['quantity']} @ ${order['price']}")

    time.sleep(0.5)

    # Check positions
    positions = engine.get_positions()
    print(f"\nCurrent positions ({len(positions)}):")
    for pos in positions:
        print(f"  {pos['symbol']}: {pos['quantity']} @ ${pos['avg_price']:.2f}")
        print(f"    Unrealized PnL: ${pos['unrealized_pnl']:.2f}")

    # Partially close BTC position
    print("\nPartially closing BTC position...")
    close_order = {
        "symbol": "BTC/USDT",
        "side": "sell",
        "type": "market",
        "quantity": 2.5,
        "price": 50300.0,
    }

    close_id = engine.submit_order(close_order)
    print(f"Submitted BTC sell order: {close_order['quantity']} @ ${close_order['price']}")

    time.sleep(0.3)

    # Check updated positions
    positions = engine.get_positions()
    print(f"\nUpdated positions:")
    for pos in positions:
        print(f"  {pos['symbol']}: {pos['quantity']} @ ${pos['avg_price']:.2f}")
        print(f"    Unrealized PnL: ${pos['unrealized_pnl']:.2f}")

    stats = engine.get_stats()
    print(f"\nPortfolio PnL: ${stats.total_value:.2f}")

    engine.stop()


async def example_4_risk_management():
    """Example 4: Risk management integration"""
    print("\n" + "=" * 60)
    print("Example 4: Risk Management")
    print("=" * 60)

    engine = ExecutionEngine()
    engine.start()
    await asyncio.sleep(0.1)

    # Submit normal orders (should pass)
    print("Submitting normal orders...")
    for i in range(5):
        order_data = {
            "symbol": "RISK/USDT",
            "side": "buy",
            "type": "market",
            "quantity": 1.0,
            "price": 50000.0,
        }
        order_id = engine.submit_order(order_data)
        print(f"  Order {i+1}: {order_id}")

    await asyncio.sleep(0.2)

    # Submit large orders (might be rejected)
    print("\nSubmitting large orders (risk check)...")
    large_orders = [
        {"symbol": "RISK/USDT", "side": "buy", "type": "market", "quantity": 100.0, "price": 50000.0},
        {"symbol": "RISK/USDT", "side": "buy", "type": "market", "quantity": 500.0, "price": 50000.0},
        {"symbol": "RISK/USDT", "side": "buy", "type": "market", "quantity": 1000.0, "price": 50000.0},
    ]

    for i, order in enumerate(large_orders):
        try:
            order_id = engine.submit_order(order)
            print(f"  Large order {i+1} ({order['quantity']} units): {order_id}")
        except Exception as e:
            print(f"  Large order {i+1} rejected: {e}")

    await asyncio.sleep(0.5)

    # Check statistics
    stats = engine.get_stats()
    print(f"\nRisk Statistics:")
    print(f"  Total orders: {stats.total_orders}")
    print(f"  Filled orders: {stats.filled_orders}")
    print(f"  Rejected orders: {stats.rejected_orders}")
    print(f"  Risk violations: {stats.risk_violations}")

    engine.stop()


async def example_5_high_frequency_trading():
    """Example 5: High-frequency trading simulation"""
    print("\n" + "=" * 60)
    print("Example 5: High-Frequency Trading Simulation")
    print("=" * 60)

    engine = ExecutionEngine()
    engine.start()
    await asyncio.sleep(0.1)

    # Simulate HFT strategy
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    base_prices = {"BTC/USDT": 50000.0, "ETH/USDT": 3100.0, "BNB/USDT": 300.0}

    duration = 2.0  # seconds
    start_time = time.time()
    order_count = 0

    print(f"Running HFT simulation for {duration} seconds...")

    async def submit_hft_orders():
        nonlocal order_count
        while time.time() - start_time < duration:
            for symbol in symbols:
                # Simulate market making strategy
                base_price = base_prices[symbol]
                spread = base_price * 0.001  # 0.1% spread

                # Submit buy order (bid)
                buy_order = {
                    "symbol": symbol,
                    "side": "buy",
                    "type": "limit",
                    "quantity": 0.1,
                    "price": base_price - spread,
                }
                engine.submit_order(buy_order)

                # Submit sell order (ask)
                sell_order = {
                    "symbol": symbol,
                    "side": "sell",
                    "type": "limit",
                    "quantity": 0.1,
                    "price": base_price + spread,
                }
                engine.submit_order(sell_order)

                order_count += 2

            # Small delay
            await asyncio.sleep(0.001)

    # Run HFT simulation
    await submit_hft_orders()

    await asyncio.sleep(0.5)  # Wait for processing

    # Calculate statistics
    elapsed_time = time.time() - start_time
    stats = engine.get_stats()

    print(f"\n📊 HFT Results:")
    print(f"  Duration: {elapsed_time:.2f}s")
    print(f"  Orders submitted: {order_count}")
    print(f"  Submission rate: {order_count/elapsed_time:.0f} orders/sec")
    print(f"  Orders processed: {stats.total_orders}")
    print(f"  Processing rate: {stats.total_orders/elapsed_time:.0f} orders/sec")
    print(f"  Avg execution time: {stats.avg_execution_time_us:.2f} μs")
    print(f"  Pending orders: {stats.current_pending_orders}")

    engine.stop()


def example_6_error_handling():
    """Example 6: Error handling and edge cases"""
    print("\n" + "=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)

    engine = ExecutionEngine()
    engine.start()
    time.sleep(0.1)

    # Test invalid orders
    print("Testing invalid orders...")

    invalid_orders = [
        # Invalid side
        {"symbol": "BTC/USDT", "side": "invalid", "type": "market", "quantity": 1.0},
        # Invalid order type
        {"symbol": "BTC/USDT", "side": "buy", "type": "invalid", "quantity": 1.0},
        # Negative quantity
        {"symbol": "BTC/USDT", "side": "buy", "type": "market", "quantity": -1.0},
        # Zero quantity
        {"symbol": "BTC/USDT", "side": "buy", "type": "market", "quantity": 0.0},
        # Missing required fields
        {"side": "buy", "type": "market", "quantity": 1.0},
    ]

    for i, order in enumerate(invalid_orders):
        try:
            order_id = engine.submit_order(order)
            print(f"  Invalid order {i+1}: Unexpectedly accepted ({order_id})")
        except Exception as e:
            print(f"  Invalid order {i+1}: Correctly rejected - {e}")

    # Test order operations on non-existent orders
    print("\nTesting non-existent order operations...")
    fake_order_id = "non_existent_order_123"

    # Try to get status
    order_info = engine.get_order(fake_order_id)
    print(f"  Get non-existent order: {order_info}")

    # Try to cancel
    cancel_result = engine.cancel_order(fake_order_id)
    print(f"  Cancel non-existent order: {cancel_result}")

    # Test engine operations when not started
    print("\nTesting engine when not started...")
    engine.stop()

    try:
        order_data = {"symbol": "BTC/USDT", "side": "buy", "type": "market", "quantity": 1.0}
        order_id = engine.submit_order(order_data)
        print(f"  Submit when stopped: Unexpectedly accepted ({order_id})")
    except Exception as e:
        print(f"  Submit when stopped: Correctly rejected - {e}")

    print("\n✅ Error handling tests completed")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Trading Execution Engine Examples")
    print("=" * 60)

    # Run synchronous examples
    example_1_basic_usage()
    example_2_order_cancellation()
    example_3_position_tracking()
    example_6_error_handling()

    # Run asynchronous examples
    asyncio.run(example_4_risk_management())
    asyncio.run(example_5_high_frequency_trading())

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()