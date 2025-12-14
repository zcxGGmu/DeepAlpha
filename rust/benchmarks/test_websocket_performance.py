"""WebSocket Performance Benchmarks

Test script to measure WebSocket server performance including:
- Connection handling capacity
- Message throughput
- Latency measurements
- Memory usage
"""

import asyncio
import websockets
import json
import time
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketBenchmark:
    """WebSocket performance benchmark suite"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        self.results = {}

    async def benchmark_connections(self, max_connections: int = 1000, step_size: int = 100):
        """Benchmark maximum concurrent connections"""
        logger.info(f"Testing concurrent connections up to {max_connections}...")

        results = {
            "max_connections": 0,
            "successful_connections": 0,
            "failed_connections": 0,
            "connection_times": [],
            "memory_usage": []
        }

        process = psutil.Process()

        for num_connections in range(step_size, max_connections + 1, step_size):
            logger.info(f"Testing {num_connections} connections...")

            # Record initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create connections in batches
            connection_tasks = []
            connection_times = []

            for i in range(num_connections):
                start_time = time.time()
                task = asyncio.create_task(self.create_connection(i))
                connection_tasks.append((task, start_time))

            # Wait for all connections
            successful = 0
            failed = 0

            for task, start_time in connection_tasks:
                try:
                    await task
                    connection_time = time.time() - start_time
                    connection_times.append(connection_time)
                    successful += 1
                except Exception as e:
                    logger.warning(f"Connection failed: {e}")
                    failed += 1

            # Record memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory

            logger.info(f"  Successful: {successful}, Failed: {failed}")
            logger.info(f"  Memory used: {memory_used:.2f} MB")

            results["successful_connections"] = successful
            results["failed_connections"] = failed
            results["connection_times"] = connection_times
            results["memory_usage"].append(memory_used)

            # Stop if too many failures
            if failed > successful * 0.5:  # More than 50% failures
                logger.warning("Too many connection failures, stopping test")
                break

            results["max_connections"] = successful

            # Close all connections
            await self.close_all_connections()

            # Brief pause between tests
            await asyncio.sleep(1)

        # Calculate statistics
        if results["connection_times"]:
            results["avg_connection_time"] = np.mean(results["connection_times"])
            results["median_connection_time"] = np.median(results["connection_times"])
            results["p95_connection_time"] = np.percentile(results["connection_times"], 95)
            results["p99_connection_time"] = np.percentile(results["connection_times"], 99)

        self.results["connections"] = results
        return results

    async def benchmark_throughput(self, num_clients: int = 100, duration: int = 10):
        """Benchmark message throughput"""
        logger.info(f"Testing throughput with {num_clients} clients for {duration} seconds...")

        results = {
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "messages_per_second": 0,
            "latencies": [],
            "client_results": []
        }

        # Create clients
        clients = []
        for i in range(num_clients):
            client = WebSocketClient(i, self.uri)
            clients.append(client)

        # Start all clients
        await asyncio.gather(*[client.start() for client in clients])

        # Give time for connections to establish
        await asyncio.sleep(1)

        # Start message sending
        start_time = time.time()
        message_task = asyncio.create_task(self.send_messages(duration))

        # Collect results
        receive_tasks = [client.collect_messages(duration) for client in clients]
        await asyncio.gather(*receive_tasks, return_exceptions=True)
        await message_task

        # Collect statistics
        for client in clients:
            client_result = {
                "client_id": client.client_id,
                "messages_sent": client.messages_sent,
                "messages_received": len(client.messages),
                "avg_latency": np.mean(client.latencies) if client.latencies else 0
            }
            results["client_results"].append(client_result)
            results["total_messages_sent"] += client.messages_sent
            results["total_messages_received"] += len(client.messages)
            results["latencies"].extend(client.latencies)

        # Calculate throughput
        actual_duration = time.time() - start_time
        results["messages_per_second"] = results["total_messages_received"] / actual_duration

        # Calculate latency statistics
        if results["latencies"]:
            results["avg_latency"] = np.mean(results["latencies"])
            results["median_latency"] = np.median(results["latencies"])
            results["p95_latency"] = np.percentile(results["latencies"], 95)
            results["p99_latency"] = np.percentile(results["latencies"], 99)

        # Clean up
        await asyncio.gather(*[client.stop() for client in clients], return_exceptions=True)

        self.results["throughput"] = results
        return results

    async def create_connection(self, client_id: int):
        """Create a single WebSocket connection"""
        uri = f"{self.uri}/client_{client_id}"
        async with websockets.connect(uri) as websocket:
            # Send identification
            await websocket.send(json.dumps({
                "type": "auth",
                "client_id": f"client_{client_id}"
            }))
            # Keep connection alive
            await asyncio.sleep(1)

    async def send_messages(self, duration: int):
        """Send broadcast messages continuously"""
        start_time = time.time()
        message_count = 0

        while time.time() - start_time < duration:
            message = {
                "type": "benchmark",
                "message_id": message_count,
                "timestamp": time.time(),
                "data": "x" * 100  # 100 bytes payload
            }

            # Send via manager (this would be connected to actual manager)
            # For now, just count
            message_count += 1
            await asyncio.sleep(0.001)  # 1000 messages per second max

    async def close_all_connections(self):
        """Close all active connections"""
        # This would be implemented to close connections via the manager
        pass

    def print_results(self):
        """Print benchmark results"""
        print("\n" + "=" * 60)
        print("WebSocket Performance Benchmark Results")
        print("=" * 60)

        if "connections" in self.results:
            print("\n📊 Connection Test Results:")
            results = self.results["connections"]
            print(f"  Max concurrent connections: {results['max_connections']}")
            if "avg_connection_time" in results:
                print(f"  Avg connection time: {results['avg_connection_time']*1000:.2f}ms")
                print(f"  P95 connection time: {results['p95_connection_time']*1000:.2f}ms")
            if results["memory_usage"]:
                print(f"  Memory per connection: {np.mean(results['memory_usage'])/results['max_connections']:.2f}MB")

        if "throughput" in self.results:
            print("\n🚀 Throughput Test Results:")
            results = self.results["throughput"]
            print(f"  Total messages sent: {results['total_messages_sent']:,}")
            print(f"  Total messages received: {results['total_messages_received']:,}")
            print(f"  Messages per second: {results['messages_per_second']:,.0f}")
            if "avg_latency" in results:
                print(f"  Avg latency: {results['avg_latency']*1000:.2f}ms")
                print(f"  P95 latency: {results['p95_latency']*1000:.2f}ms")
                print(f"  P99 latency: {results['p99_latency']*1000:.2f}ms")


class WebSocketClient:
    """Simple WebSocket client for testing"""

    def __init__(self, client_id: int, uri: str):
        self.client_id = client_id
        self.uri = uri
        self.websocket = None
        self.messages = []
        self.latencies = []
        self.messages_sent = 0

    async def start(self):
        """Start the client connection"""
        try:
            self.websocket = await websockets.connect(self.uri)
            # Send identification
            await self.websocket.send(json.dumps({
                "type": "auth",
                "client_id": f"client_{self.client_id}"
            }))
        except Exception as e:
            logger.error(f"Client {self.client_id} failed to connect: {e}")

    async def stop(self):
        """Stop the client connection"""
        if self.websocket:
            await self.websocket.close()

    async def collect_messages(self, duration: int):
        """Collect messages for the specified duration"""
        if not self.websocket:
            return

        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                received_time = time.time()

                # Parse and record
                try:
                    data = json.loads(message)
                    if "timestamp" in data:
                        latency = received_time - data["timestamp"]
                        self.latencies.append(latency)
                    self.messages.append(data)
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning(f"Client {self.client_id} error: {e}")
                break


async def main():
    """Run WebSocket benchmarks"""
    benchmark = WebSocketBenchmark()

    print("WebSocket Performance Benchmark Suite")
    print("=" * 60)

    # Test connections (with lower limits for safety)
    await benchmark.benchmark_connections(max_connections=500, step_size=50)

    # Test throughput
    await benchmark.benchmark_throughput(num_clients=50, duration=5)

    # Print results
    benchmark.print_results()


if __name__ == "__main__":
    # Install required packages if needed
    try:
        import psutil
        import numpy
        import websockets
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "psutil", "numpy", "websockets"])

    # Run benchmarks
    asyncio.run(main())