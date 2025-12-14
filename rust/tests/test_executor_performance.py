"""Performance tests for Trading Execution Engine"""

import asyncio
import time
import json
import statistics
from deepalpha_rust import ExecutionEngine


class TestExecutorPerformance:
    """Test execution engine performance under various loads"""

    def test_single_order_latency(self):
        """Test single order submission latency"""
        engine = ExecutionEngine()
        engine.start()
        time.sleep(0.1)

        latencies = []

        for i in range(1000):
            start_time = time.perf_counter_ns()

            order_data = {
                "symbol": "BTC/USDT",
                "side": "buy" if i % 2 == 0 else "sell",
                "type": "market",
                "quantity": 1.0,
                "price": 50000.0 + (i % 1000)
            }

            order_id = engine.submit_order(order_data)

            end_time = time.perf_counter_ns()
            latency_ns = end_time - start_time
            latencies.append(latency_ns)

            if i % 100 == 0:
                print(f"  Order {i}: {latency_ns / 1000:.2f} μs")

        engine.stop()

        # Calculate statistics
        avg_latency_ns = statistics.mean(latencies)
        median_latency_ns = statistics.median(latencies)
        p95_latency_ns = latencies[int(0.95 * len(latencies))]
        p99_latency_ns = latencies[int(0.99 * len(latencies))]
        max_latency_ns = max(latencies)
        min_latency_ns = min(latencies)

        print(f"\n📊 Single Order Latency (1000 orders):")
        print(f"  Average: {avg_latency_ns / 1000:.2f} μs")
        print(f"  Median: {median_latency_ns / 1000:.2f} μs")
        print(f"  95th percentile: {p95_latency_ns / 1000:.2f} μs")
        print(f"  99th percentile: {p99_latency_ns / 1000:.2f} μs")
        print(f"  Min: {min_latency_ns / 1000:.2f} μs")
        print(f"  Max: {max_latency_ns / 1000:.2f} μs")

        # Performance assertions
        assert avg_latency_ns < 50000, f"Average latency too high: {avg_latency_ns / 1000:.2f} μs"
        assert p99_latency_ns < 100000, f"P99 latency too high: {p99_latency_ns / 1000:.2f} μs"

    def test_throughput_burst(self):
        """Test burst order throughput"""
        engine = ExecutionEngine()
        engine.start()
        time.sleep(0.1)

        # Test different batch sizes
        batch_sizes = [100, 500, 1000, 2000]

        for batch_size in batch_sizes:
            start_time = time.perf_counter()

            order_ids = []
            for i in range(batch_size):
                order_data = {
                    "symbol": f"SYM{i % 10:02d}/USDT",
                    "side": "buy" if i % 2 == 0 else "sell",
                    "type": "market",
                    "quantity": 1.0 + (i % 5) * 0.1,
                    "price": 50000.0 + (i % 1000)
                }

                order_id = engine.submit_order(order_data)
                order_ids.append(order_id)

            end_time = time.perf_counter()
            duration = end_time - start_time
            throughput = batch_size / duration

            print(f"  Batch {batch_size:4d}: {throughput:.0f} orders/sec ({duration:.3f}s)")

            # Verify all orders were accepted
            assert len(order_ids) == batch_size

            # Small delay between batches
            time.sleep(0.5)

        # Wait for processing
        time.sleep(2.0)

        final_stats = engine.get_stats()
        print(f"\n📈 Final Statistics:")
        print(f"  Total orders: {final_stats.total_orders}")
        print(f"  Filled orders: {final_stats.filled_orders}")
        print(f"  Cancelled orders: {final_stats.cancelled_orders}")
        print(f"  Rejected orders: {final_stats.rejected_orders}")
        print(f"  Avg execution time: {final_stats.avg_execution_time_us:.2f} μs")
        print(f"  Risk violations: {final_stats.risk_violations}")

        engine.stop()

        # Throughput assertions
        assert final_stats.total_orders >= 3000, f"Too few orders processed: {final_stats.total_orders}"

    def test_concurrent_submission(self):
        """Test concurrent order submission from multiple 'clients'"""
        async def submit_orders(client_id: int, num_orders: int, engine: ExecutionEngine):
            """Submit orders from a client"""
            order_ids = []
            for i in range(num_orders):
                order_data = {
                    "symbol": f"CLIENT{client_id:02d}_SYM{i % 5}/USDT",
                    "side": "buy" if (client_id + i) % 2 == 0 else "sell",
                    "type": "limit" if i % 3 == 0 else "market",
                    "quantity": 1.0 + (i % 3) * 0.5,
                    "price": 50000.0 + (i % 500) - 250,
                }

                order_id = engine.submit_order(order_data)
                order_ids.append(order_id)

                # Small delay to simulate realistic trading
                if i % 50 == 0:
                    await asyncio.sleep(0.001)

            return order_ids

        async def run_concurrent_test():
            engine = ExecutionEngine()
            engine.start()
            await asyncio.sleep(0.1)

            num_clients = 10
            orders_per_client = 200

            start_time = time.perf_counter()

            # Create tasks for each client
            tasks = []
            for client_id in range(num_clients):
                task = submit_orders(client_id, orders_per_client, engine)
                tasks.append(task)

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)

            end_time = time.perf_counter()
            duration = end_time - start_time
            total_orders = sum(len(client_orders) for client_orders in results)
            throughput = total_orders / duration

            print(f"\n⚡ Concurrent Submission Test:")
            print(f"  Clients: {num_clients}")
            print(f"  Orders per client: {orders_per_client}")
            print(f"  Total orders: {total_orders}")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Throughput: {throughput:.0f} orders/sec")

            # Wait for processing
            await asyncio.sleep(2.0)

            final_stats = engine.get_stats()
            print(f"\n📊 Final Stats:")
            print(f"  Processed orders: {final_stats.total_orders}")
            print(f"  Current pending: {final_stats.current_pending_orders}")
            print(f"  Total value: ${final_stats.total_value:.2f}")
            print(f"  Avg execution time: {final_stats.avg_execution_time_us:.2f} μs")

            engine.stop()

            # Performance assertions
            assert throughput > 500, f"Concurrent throughput too low: {throughput:.0f} orders/sec"
            assert final_stats.total_orders >= total_orders * 0.9, f"Too few orders processed: {final_stats.total_orders}/{total_orders}"

            return final_stats

        # Run the async test
        stats = asyncio.run(run_concurrent_test())

    def test_sustained_load(self):
        """Test sustained load over time"""
        engine = ExecutionEngine()
        engine.start()
        time.sleep(0.1)

        duration_seconds = 10
        target_rate = 1000  # orders per second
        batch_size = 100

        start_time = time.perf_counter()
        total_submitted = 0

        while time.perf_counter() - start_time < duration_seconds:
            batch_start = time.perf_counter()

            # Submit a batch of orders
            for i in range(batch_size):
                order_data = {
                    "symbol": f"LOAD{i % 20:02d}/USDT",
                    "side": "buy" if (total_submitted + i) % 2 == 0 else "sell",
                    "type": "market" if i % 4 != 0 else "limit",
                    "quantity": 1.0 + (i % 10) * 0.1,
                    "price": 50000.0 + (i % 2000) - 1000,
                }

                engine.submit_order(order_data)
                total_submitted += 1

            batch_time = time.perf_counter() - batch_start
            expected_batch_time = batch_size / target_rate

            # Sleep to maintain target rate
            if batch_time < expected_batch_time:
                time.sleep(expected_batch_time - batch_time)

            # Print progress
            elapsed = time.perf_counter() - start_time
            if int(elapsed) % 2 == 0 and batch_size == 100:  # Print every 2 seconds
                stats = engine.get_stats()
                current_rate = stats.total_orders / elapsed
                print(f"  {elapsed:.1f}s: {total_submitted} submitted, "
                      f"{stats.total_orders} processed, {current_rate:.0f} ops/s")

        total_time = time.perf_counter() - start_time

        # Wait for remaining orders to process
        time.sleep(3.0)

        final_stats = engine.get_stats()

        print(f"\n🚀 Sustained Load Test ({duration_seconds}s at {target_rate} ops/s target):")
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Orders submitted: {total_submitted}")
        print(f"  Orders processed: {final_stats.total_orders}")
        print(f"  Submission rate: {total_submitted/total_time:.0f} orders/sec")
        print(f"  Processing rate: {final_stats.total_orders/total_time:.0f} orders/sec")
        print(f"  Success rate: {final_stats.total_orders/total_submitted*100:.1f}%")
        print(f"  Avg execution time: {final_stats.avg_execution_time_us:.2f} μs")
        print(f"  Final pending: {final_stats.current_pending_orders}")
        print(f"  Risk violations: {final_stats.risk_violations}")

        engine.stop()

        # Performance assertions
        actual_rate = final_stats.total_orders / total_time
        assert actual_rate > target_rate * 0.8, f"Processing rate too low: {actual_rate:.0f} < {target_rate * 0.8:.0f}"
        assert final_stats.total_orders >= total_submitted * 0.9, f"Too many orders dropped: {final_stats.total_orders}/{total_submitted}"

    def test_memory_usage(self):
        """Test memory usage under load"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        engine = ExecutionEngine()
        engine.start()
        time.sleep(0.1)

        after_start_memory = process.memory_info().rss / 1024 / 1024
        engine_overhead = after_start_memory - initial_memory

        # Submit many orders
        num_orders = 10000
        for i in range(num_orders):
            order_data = {
                "symbol": f"MEM{i % 100}/USDT",
                "side": "buy" if i % 2 == 0 else "sell",
                "type": "market",
                "quantity": 1.0,
                "price": 50000.0 + (i % 1000),
            }
            engine.submit_order(order_data)

        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - after_start_memory

        # Wait for processing and potential cleanup
        time.sleep(2.0)

        final_memory = process.memory_info().rss / 1024 / 1024

        print(f"\n💾 Memory Usage Test ({num_orders} orders):")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Engine overhead: {engine_overhead:.1f} MB")
        print(f"  Peak memory: {peak_memory:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory per order: {memory_increase/num_orders*1024:.2f} KB")
        print(f"  Memory reclaimed: {peak_memory - final_memory:.1f} MB")

        engine.stop()

        # Memory assertions
        assert engine_overhead < 50, f"Engine overhead too high: {engine_overhead:.1f} MB"
        assert memory_increase < num_orders * 0.01, f"Memory per order too high: {memory_increase/num_orders:.3f} MB"

    def test_risk_check_performance(self):
        """Test risk management check performance"""
        engine = ExecutionEngine()
        engine.start()
        time.sleep(0.1)

        # Submit orders that might trigger risk checks
        risk_violations = 0

        # Normal orders
        for i in range(1000):
            order_data = {
                "symbol": "RISK/USDT",
                "side": "buy",
                "type": "market",
                "quantity": 1.0,
                "price": 50000.0,
            }
            engine.submit_order(order_data)

        # Large orders that might trigger violations
        for i in range(100):
            order_data = {
                "symbol": "RISK/USDT",
                "side": "buy",
                "type": "market",
                "quantity": 100.0 * (i + 1),  # Increasing size
                "price": 50000.0,
            }
            engine.submit_order(order_data)

        # Orders that might exceed position limits
        for i in range(50):
            order_data = {
                "symbol": f"RISK{i}/USDT",
                "side": "buy",
                "type": "limit",
                "quantity": 500.0,
                "price": 50000.0,
            }
            engine.submit_order(order_data)

        time.sleep(1.0)

        stats = engine.get_stats()
        risk_violations = stats.risk_violations

        print(f"\n⚠️  Risk Management Performance:")
        print(f"  Total orders: {stats.total_orders}")
        print(f"  Risk violations: {risk_violations}")
        print(f"  Violation rate: {risk_violations/stats.total_orders*100:.1f}%")
        print(f"  Avg execution time: {stats.avg_execution_time_us:.2f} μs")

        engine.stop()

        # Risk management should not significantly impact performance
        assert stats.avg_execution_time_us < 1000, f"Risk checks too slow: {stats.avg_execution_time_us:.2f} μs"


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Trading Execution Engine Performance Tests")
    print("=" * 60)

    test = TestExecutorPerformance()

    try:
        print("\n1. Single Order Latency Test")
        print("-" * 40)
        test.test_single_order_latency()

        print("\n2. Throughput Burst Test")
        print("-" * 40)
        test.test_throughput_burst()

        print("\n3. Concurrent Submission Test")
        print("-" * 40)
        test.test_concurrent_submission()

        print("\n4. Sustained Load Test")
        print("-" * 40)
        test.test_sustained_load()

        print("\n5. Memory Usage Test")
        print("-" * 40)
        test.test_memory_usage()

        print("\n6. Risk Check Performance Test")
        print("-" * 40)
        test.test_risk_check_performance()

        print("\n" + "=" * 60)
        print("✅ All performance tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)