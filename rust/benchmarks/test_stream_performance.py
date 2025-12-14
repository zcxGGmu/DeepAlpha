"""Stream Processing Performance Benchmark

Test script to measure data stream processing performance including:
- Data ingestion rate
- Processing latency
- Memory usage
- Throughput with multiple processors
"""

import asyncio
import time
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from deepalpha_rust import MarketDataStream
except ImportError:
    print("DeepAlpha Rust module not found. Please build it first.")
    print("Run: cd rust && maturin develop")
    exit(1)


class StreamBenchmark:
    """Stream processing benchmark suite"""

    def __init__(self):
        self.results = {}

    async def benchmark_ingestion_rate(self, num_messages: int = 100000):
        """Benchmark data ingestion rate"""
        logger.info(f"Testing ingestion rate with {num_messages:,} messages...")

        stream = MarketDataStream(buffer_size=10000)
        stream.add_processor("validator", None)

        # Start stream
        stream.start()
        await asyncio.sleep(0.1)

        # Generate test data
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
        base_prices = {
            "BTC/USDT": 50000.0,
            "ETH/USDT": 3000.0,
            "BNB/USDT": 300.0,
            "SOL/USDT": 100.0,
            "ADA/USDT": 0.5
        }

        # Measure ingestion
        start_time = time.time()

        for i in range(num_messages):
            symbol = symbols[i % len(symbols)]
            base_price = base_prices[symbol]
            price = base_price + (i % 100 - 50) * 0.01
            volume = 1.0 + (i % 10) * 0.1

            # Mix of trades and quotes
            if i % 3 == 0:
                stream.push_trade(symbol, price, volume)
            else:
                stream.push_quote(symbol, price - 0.5, price + 0.5)

        ingestion_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(2.0)

        # Get statistics
        stats = stream.get_stats()
        final_time = time.time() - start_time

        # Calculate metrics
        ingestion_rate = num_messages / ingestion_time
        processing_rate = stats.processed_count / stats.avg_processing_time / 1000000  # per μs
        total_time = final_time

        results = {
            "num_messages": num_messages,
            "ingestion_time": ingestion_time,
            "ingestion_rate": ingestion_rate,
            "processed_count": stats.processed_count,
            "error_count": stats.error_count,
            "processing_rate": stats.processing_rate,
            "avg_processing_time_us": stats.avg_processing_time,
            "total_time": total_time,
            "success_rate": stats.processed_count / num_messages * 100
        }

        self.results["ingestion"] = results

        # Print results
        print(f"\n📊 Ingestion Benchmark Results:")
        print(f"  Messages: {num_messages:,}")
        print(f"  Ingestion rate: {ingestion_rate:,.0f} msg/s")
        print(f"  Processing rate: {stats.processing_rate:,.0f} msg/s")
        print(f"  Avg processing time: {stats.avg_processing_time:.2f} μs")
        print(f"  Success rate: {results['success_rate']:.2f}%")
        print(f"  Errors: {stats.error_count}")

        stream.stop()
        return results

    async def benchmark_multi_processor(self, num_messages: int = 50000):
        """Benchmark with multiple processors"""
        logger.info(f"Testing with multiple processors ({num_messages:,} messages)...")

        # Test with different processor combinations
        configs = [
            {"name": "Validator Only", "processors": ["validator"]},
            {"name": "Validator + Filter", "processors": ["validator", "filter"]},
            {"name": "Validator + Filter + Transform", "processors": ["validator", "filter", "transform"]},
            {"name": "All Processors", "processors": ["validator", "filter", "transform", "aggregator"]},
        ]

        results = {}

        for config in configs:
            logger.info(f"Testing: {config['name']}")
            stream = MarketDataStream(buffer_size=10000)

            # Add processors
            for proc_type in config["processors"]:
                stream.add_processor(proc_type, None)

            # Start stream
            stream.start()
            await asyncio.sleep(0.1)

            # Measure time
            start_time = time.time()

            # Push data
            for i in range(num_messages):
                stream.push_trade("BTC/USDT", 50000.0 + (i % 100), 1.0)

            push_time = time.time() - start_time

            # Wait for processing
            await asyncio.sleep(2.0)

            # Get stats
            stats = stream.get_stats()

            config_result = {
                "processors": config["processors"],
                "push_rate": num_messages / push_time,
                "processed_count": stats.processed_count,
                "processing_rate": stats.processing_rate,
                "avg_processing_time": stats.avg_processing_time,
                "error_count": stats.error_count
            }

            results[config["name"]] = config_result

            print(f"  {config['name']}: {stats.processing_rate:,.0f} msg/s, {stats.avg_processing_time:.2f} μs avg")

            stream.stop()

        self.results["multi_processor"] = results
        return results

    async def benchmark_memory_usage(self, num_messages: int = 100000, buffer_sizes: List[int] = None):
        """Benchmark memory usage with different buffer sizes"""
        if buffer_sizes is None:
            buffer_sizes = [100, 1000, 10000, 100000]

        logger.info(f"Testing memory usage with {num_messages:,} messages...")

        process = psutil.Process()
        results = {}

        for buffer_size in buffer_sizes:
            # Measure initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            stream = MarketDataStream(buffer_size=buffer_size)
            stream.add_processor("validator", None)
            stream.start()
            await asyncio.sleep(0.1)

            # Push data
            for i in range(num_messages):
                stream.push_trade(f"SYM{i%10}", 100.0 + i, 1.0)

            # Wait for processing
            await asyncio.sleep(2.0)

            # Measure final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory

            stats = stream.get_stats()

            buffer_result = {
                "buffer_size": buffer_size,
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_used_mb": memory_used,
                "memory_per_message_bytes": (memory_used * 1024 * 1024) / num_messages,
                "processed_count": stats.processed_count
            }

            results[f"buffer_{buffer_size}"] = buffer_result

            print(f"  Buffer {buffer_size:,}: {memory_used:.2f} MB used, {buffer_result['memory_per_message_bytes']:.2f} bytes/msg")

            stream.stop()

        self.results["memory"] = results
        return results

    async def benchmark_symbol_parallelism(self, num_symbols: int = 100, messages_per_symbol: int = 1000):
        """Benchmark processing multiple symbols in parallel"""
        logger.info(f"Testing {num_symbols} symbols with {messages_per_symbol} messages each...")

        stream = MarketDataStream(buffer_size=100000)
        stream.add_processor("validator", None)
        stream.add_processor("transform", None)

        stream.start()
        await asyncio.sleep(0.1)

        # Generate symbols
        symbols = [f"SYM{i:03d}" for i in range(num_symbols)]

        # Measure time
        start_time = time.time()

        # Push data for all symbols
        for i in range(messages_per_symbol):
            for symbol in symbols:
                price = 100.0 + (hash(f"{symbol}_{i}") % 1000) / 10.0
                volume = 1.0 + (i % 10) * 0.1
                stream.push_trade(symbol, price, volume)

        push_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(3.0)

        # Get stats
        stats = stream.get_stats()
        total_messages = num_symbols * messages_per_symbol
        total_time = time.time() - start_time

        results = {
            "num_symbols": num_symbols,
            "messages_per_symbol": messages_per_symbol,
            "total_messages": total_messages,
            "push_rate": total_messages / push_time,
            "processed_count": stats.processed_count,
            "processing_rate": stats.processing_rate,
            "avg_processing_time": stats.avg_processing_time,
            "error_count": stats.error_count
        }

        self.results["symbol_parallelism"] = results

        print(f"\n📈 Symbol Parallelism Results:")
        print(f"  Symbols: {num_symbols}")
        print(f"  Messages per symbol: {messages_per_symbol}")
        print(f"  Total messages: {total_messages:,}")
        print(f"  Processing rate: {stats.processing_rate:,.0f} msg/s")
        print(f"  Avg processing time: {stats.avg_processing_time:.2f} μs")

        stream.stop()
        return results

    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("Stream Processing Benchmark Summary")
        print("=" * 60)

        if "ingestion" in self.results:
            results = self.results["ingestion"]
            print(f"\n🚀 Ingestion Performance:")
            print(f"  Peak Rate: {results['ingestion_rate']:,.0f} msg/s")
            print(f"  Avg Latency: {results['avg_processing_time_us']:.2f} μs")
            print(f"  Success Rate: {results['success_rate']:.2f}%")

        if "multi_processor" in self.results:
            results = self.results["multi_processor"]
            print(f"\n⚙️  Processor Performance:")
            for name, config in results.items():
                print(f"  {name}:")
                print(f"    Rate: {config['processing_rate']:,.0f} msg/s")
                print(f"    Latency: {config['avg_processing_time']:.2f} μs")

        if "memory" in self.results:
            results = self.results["memory"]
            print(f"\n💾 Memory Usage:")
            for name, config in results.items():
                print(f"  {name}: {config['memory_used_mb']:.2f} MB")
                print(f"    Per message: {config['memory_per_message_bytes']:.2f} bytes")

        if "symbol_parallelism" in self.results:
            results = self.results["symbol_parallelism"]
            print(f"\n🔄 Symbol Parallelism:")
            print(f"  {results['num_symbols']} symbols processed")
            print(f"  {results['processed_count']:,} messages processed")
            print(f"  Rate: {results['processing_rate']:,.0f} msg/s")


async def main():
    """Run all stream benchmarks"""
    print("🔥 DeepAlpha Stream Processing Benchmark Suite")
    print("=" * 60)

    benchmark = StreamBenchmark()

    # Run benchmarks
    await benchmark.benchmark_ingestion_rate(100000)
    await benchmark.benchmark_multi_processor(50000)
    await benchmark.benchmark_memory_usage(100000)
    await benchmark.benchmark_symbol_parallelism(100, 1000)

    # Print summary
    benchmark.print_summary()


if __name__ == "__main__":
    # Check required packages
    try:
        import psutil
        import numpy
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "psutil", "numpy"])

    # Run benchmarks
    asyncio.run(main())