"""Test cases for Trading Execution Engine functionality"""

import pytest
import asyncio
import time
import json
from deepalpha_rust import ExecutionEngine


class TestExecutionEngine:
    """Test Execution Engine functionality"""

    @pytest.fixture
    def engine(self):
        """Create an ExecutionEngine instance"""
        return ExecutionEngine()

    def test_engine_creation(self, engine):
        """Test engine creation"""
        assert not engine.active
        stats = engine.get_stats()
        assert stats.total_orders == 0
        assert stats.filled_orders == 0
        assert stats.cancelled_orders == 0
        assert stats.rejected_orders == 0

    def test_submit_market_order(self, engine):
        """Test submitting a market order"""
        order_data = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "market",
            "quantity": 1.0
        }

        # Start engine first
        engine.start()
        time.sleep(0.1)  # Give time to start

        order_id = engine.submit_order(order_data)
        assert isinstance(order_id, str)
        assert len(order_id) > 0

        # Get order status
        time.sleep(0.1)  # Wait for processing
        order_info = engine.get_order(order_id)
        assert order_info is not None

        engine.stop()

    def test_submit_limit_order(self, engine):
        """Test submitting a limit order"""
        order_data = {
            "symbol": "ETH/USDT",
            "side": "sell",
            "type": "limit",
            "quantity": 2.0,
            "price": 3000.0
        }

        engine.start()
        time.sleep(0.1)

        order_id = engine.submit_order(order_data)
        assert isinstance(order_id, str)

        time.sleep(0.1)
        order_info = engine.get_order(order_id)
        assert order_info is not None

        engine.stop()

    def test_submit_invalid_order(self, engine):
        """Test submitting invalid orders"""
        engine.start()
        time.sleep(0.1)

        # Test invalid side
        invalid_order = {
            "symbol": "BTC/USDT",
            "side": "invalid",
            "type": "market",
            "quantity": 1.0
        }

        with pytest.raises(Exception):
            engine.submit_order(invalid_order)

        # Test invalid type
        invalid_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "invalid",
            "quantity": 1.0
        }

        with pytest.raises(Exception):
            engine.submit_order(invalid_order)

        engine.stop()

    @pytest.mark.asyncio
    async def test_order_lifecycle(self, engine):
        """Test complete order lifecycle"""
        engine.start()
        await asyncio.sleep(0.1)

        # Submit order
        order_data = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "quantity": 1.0,
            "price": 50000.0
        }

        order_id = engine.submit_order(order_data)

        # Check initial status
        await asyncio.sleep(0.05)
        order_info = engine.get_order(order_id)
        if order_info:
            print(f"Order status after submission: {order_info['status']}")

        # Try to cancel
        cancel_result = engine.cancel_order(order_id)
        print(f"Cancel result: {cancel_result}")

        # Check final status
        await asyncio.sleep(0.05)
        final_info = engine.get_order(order_id)
        if final_info:
            print(f"Order final status: {final_info['status']}")

        engine.stop()

    @pytest.mark.asyncio
    async def test_multiple_orders(self, engine):
        """Test handling multiple concurrent orders"""
        engine.start()
        await asyncio.sleep(0.1)

        # Submit multiple orders
        orders = []
        for i in range(10):
            order_data = {
                "symbol": f"SYM{i%5:02d}/USDT",
                "side": "buy" if i % 2 == 0 else "sell",
                "type": "market",
                "quantity": 1.0 + i * 0.1
            }
            order_id = engine.submit_order(order_data)
            orders.append(order_id)

        print(f"Submitted {len(orders)} orders")

        # Wait for processing
        await asyncio.sleep(1.0)

        # Check statistics
        stats = engine.get_stats()
        print(f"Total orders: {stats.total_orders}")
        print(f"Avg execution time: {stats.avg_execution_time_us:.2f} μs")
        print(f"Pending orders: {stats.current_pending_orders}")

        # Cancel remaining orders
        for order_id in orders:
            engine.cancel_order(order_id)

        await asyncio.sleep(0.5)
        final_stats = engine.get_stats()
        print(f"Final stats - Cancelled: {final_stats.cancelled_orders}")

        engine.stop()

    @pytest.mark.asyncio
    async def test_positions_tracking(self, engine):
        """Test position tracking"""
        engine.start()
        await asyncio.sleep(0.1)

        # Create a buy position
        buy_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "market",
            "quantity": 2.0,
            "price": 50000.0
        }

        buy_id = engine.submit_order(buy_order)
        print(f"Buy order ID: {buy_id}")

        # Create a sell order
        sell_order = {
            "symbol": "BTC/USDT",
            "side": "sell",
            "type": "limit",
            "quantity": 1.0,
            "price": 50100.0
        }

        sell_id = engine.submit_order(sell_order)
        print(f"Sell order ID: {sell_id}")

        # Wait for execution
        await asyncio.sleep(1.0)

        # Check positions
        positions = engine.get_positions()
        print(f"Number of positions: {len(positions)}")

        for pos in positions:
            print(f"  {pos['symbol']}: {pos['quantity']} @ ${pos['avg_price']:.2f}")
            print(f"    Unrealized PnL: ${pos['unrealized_pnl']:.2f}")

        engine.stop()

    @pytest.mark.asyncio
    async def test_performance_metrics(self, engine):
        """Test performance metrics under load"""
        engine.start()
        await asyncio.sleep(0.1)

        num_orders = 100
        start_time = time.time()

        # Submit many orders
        for i in range(num_orders):
            order_data = {
                "symbol": "BTC/USDT",
                "side": "buy" if i % 3 == 0 else "sell",
                "type": "market",
                "quantity": 1.0,
                "price": 50000.0 + (i % 1000) - 500
            }

            engine.submit_order(order_data)

        submission_time = time.time() - start_time

        # Wait for all processing
        await asyncio.sleep(2.0)

        # Get final statistics
        stats = engine.get_stats()
        total_time = time.time() - start_time

        print(f"\n📊 Performance Metrics:")
        print(f"  Orders submitted: {num_orders}")
        print(f"  Submission time: {submission_time:.3f}s")
        print(f"  Submission rate: {num_orders/submission_time:.0f} orders/sec")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Processed orders: {stats.total_orders}")
        print(f"  Filled orders: {stats.filled_orders}")
        print(f"  Cancelled orders: {stats.cancelled_orders}")
        print(f"  Rejected orders: {stats.rejected_orders}")
        print(f"  Avg execution time: {stats.avg_execution_time_us:.2f} μs")
        print(f"  Current pending: {stats.current_pending_orders}")
        print(f"  Risk violations: {stats.risk_violations}")

        # Performance assertions
        assert stats.total_orders >= num_orders * 0.9  # At least 90% processed
        if stats.avg_execution_time_us > 0:
            assert stats.avg_execution_time_us < 1000.0  # Less than 1ms

        engine.stop()

    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling"""
        engine.start()
        await asyncio.sleep(0.1)

        # Submit an order that might get rejected due to size
        large_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "market",
            "quantity": 10000.0  # Very large
        }

        order_id = engine.submit_order(large_order)
        print(f"Large order ID: {order_id}")

        await asyncio.sleep(0.5)

        # Check if it was rejected
        order_info = engine.get_order(order_id)
        if order_info:
            print(f"Large order status: {order_info['status']}")

        stats = engine.get_stats()
        if stats.rejected_orders > 0:
            print(f"Orders rejected: {stats.rejected_orders}")

        engine.stop()


@pytest.mark.integration
class TestExecutorIntegration:
    """Integration tests for execution engine"""

    @pytest.mark.asyncio
    async def test_risk_management(self, engine):
        """Test risk management integration"""
        engine.start()
        await asyncio.sleep(0.1)

        # This should be rejected by risk management (position size limit)
        large_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "quantity": 1000.0,  # Exceeds default limit
            "price": 50000.0
        }

        order_id = engine.submit_order(large_order)
        print(f"Large order ID: {order_id}")

        await asyncio.sleep(0.2)

        # Check if rejected
        order_info = engine.get_order(order_id)
        if order_info:
            print(f"Order status: {order_info['status']}")

        stats = engine.get_stats()
        if stats.risk_violations > 0:
            print(f"Risk violations: {stats.risk_violations}")

        engine.stop()

    @pytest.mark.asyncio
    async def test_concurrent_engines(self):
        """Test multiple engines running concurrently"""
        engines = []
        for i in range(3):
            e = ExecutionEngine()
            e.start()
            engines.append(e)
            await asyncio.sleep(0.1)

        # Submit orders to all engines
        order_ids = []
        for engine in engines:
            order_data = {
                "symbol": f"TEST{i}",
                "side": "buy",
                "type": "market",
                "quantity": 1.0
            }
            order_id = engine.submit_order(order_data)
            order_ids.append(order_id)

        print(f"Submitted {len(order_ids)} orders across {len(engines)} engines")

        # Wait for processing
        await asyncio.sleep(1.0)

        # Check all stats
        total_orders = 0
        for engine in engines:
            stats = engine.get_stats()
            total_orders += stats.total_orders
            engine.stop()
            print(f"Engine processed {stats.total_orders} orders")

        print(f"Total orders across all engines: {total_orders}")
        assert total_orders > 0

    @pytest.mark.asyncio
    async def test_continuous_trading(self, engine):
        """Test continuous trading simulation"""
        engine.start()
        await asyncio.sleep(0.1)

        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        base_prices = {"BTC/USDT": 50000.0, "ETH/USDT": 3000.0, "BNB/USDT": 300.0}

        # Simulate trading for 5 seconds
        duration = 5.0
        start_time = time.time()
        order_count = 0

        while time.time() - start_time < duration:
            for symbol in symbols:
                # Random walk price
                base_price = base_prices[symbol]
                price = base_price * (1.0 + (time.time() % 1000 - 500) / 10000.0)
                quantity = 1.0 + (time.time() % 10) * 0.1

                order_data = {
                    "symbol": symbol,
                    "side": "buy" if order_count % 2 == 0 else "sell",
                    "type": "market",
                    "quantity": quantity,
                    "price": price
                }

                try:
                    order_id = engine.submit_order(order_data)
                    order_count += 1

                    if order_count % 50 == 0:
                        stats = engine.get_stats()
                        elapsed = time.time() - start_time
                        rate = order_count / elapsed
                        print(f"  {elapsed:.1f}s: {order_count} orders, {rate:.1f} ops/s, {stats.pending_orders_count} pending")

                except Exception as e:
                    print(f"  Error submitting order: {e}")

                # Small delay to avoid overwhelming the system
                await asyncio.sleep(0.01)

        await asyncio.sleep(0.5)

        # Final statistics
        stats = engine.get_stats()
        total_time = time.time() - start_time

        print(f"\n📈 Trading Simulation Results:")
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Total orders: {stats.total_orders}")
        print(f"  Average rate: {order_count/total_time:.1f} orders/s")
        print(f"  PnL: ${stats.total_value:.2f}")
        print(f"  Final pending: {stats.current_pending_orders}")

        engine.stop()


if __name__ == "__main__":
    # Run standalone tests
    import sys

    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run pytest
    sys.exit(pytest.main([__file__, "-v"]))