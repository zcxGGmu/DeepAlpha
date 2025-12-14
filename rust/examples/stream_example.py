"""Market Data Stream Example

Example of how to use the Rust market data stream for real-time processing.
"""

import asyncio
import time
import random
from deepalpha_rust import MarketDataStream


async def basic_stream_example():
    """Basic stream processing example"""
    print("\n📡 Basic Stream Processing Example")
    print("-" * 40)

    # Create stream with 1000-item buffer
    stream = MarketDataStream(buffer_size=1000)

    # Add processors
    stream.add_processor("validator", None)  # Validate data integrity
    stream.add_processor("filter", None)     # Filter out bad data
    stream.add_processor("transform", None) # Add calculated fields

    # Start processing
    stream.start()
    await asyncio.sleep(0.1)  # Give time to start

    print("Stream started! Pushing data...")

    # Simulate market data
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    base_prices = {"BTC/USDT": 50000.0, "ETH/USDT": 3000.0, "BNB/USDT": 300.0}

    # Generate and push data
    for i in range(100):
        symbol = symbols[i % len(symbols)]
        base_price = base_prices[symbol]

        # Add some randomness
        price_change = (random.random() - 0.5) * 100  # -50 to +50
        price = base_price + price_change
        volume = random.uniform(0.1, 10.0)

        # Mix of trades and quotes
        if i % 3 == 0:
            stream.push_trade(symbol, price, volume)
        else:
            stream.push_quote(symbol, price - 0.5, price + 0.5)

        if i % 20 == 0:
            print(f"  Pushed {i} data points...")

    await asyncio.sleep(1.0)  # Wait for processing

    # Get statistics
    stats = stream.get_stats()
    print(f"\n📊 Processing Statistics:")
    print(f"  Processed: {stats.processed_count:,}")
    print(f"  Errors: {stats.error_count}")
    print(f"  Processing rate: {stats.processing_rate:,.0f} msg/s")
    print(f"  Avg processing time: {stats.avg_processing_time:.2f} μs")
    print(f"  Buffer size: {stats.current_buffer_size}")

    # Get recent data
    recent = stream.get_recent_data(5)
    print(f"\n📈 Recent Data (last 5 items):")
    for i, data in enumerate(recent):
        print(f"  {i+1}. [{data['timestamp']}] {data['symbol']} - Type: {data['type']}")
        if 'price' in data:
            print(f"      Price: ${data['price']:.2f}")
        if 'volume' in data:
            print(f"      Volume: {data['volume']:.4f}")

    stream.stop()


async def high_frequency_example():
    """High-frequency data processing example"""
    print("\n⚡ High-Frequency Processing Example")
    print("-" * 40)

    # Create larger buffer for high frequency
    stream = MarketDataStream(buffer_size=10000)

    # Add all processors for full pipeline
    stream.add_processor("validator", None)
    stream.add_processor("filter", None)
    stream.add_processor("transform", None)
    stream.add_processor("aggregator", None)

    stream.start()
    await asyncio.sleep(0.1)

    print("Processing high-frequency data...")

    # Simulate high-frequency trading
    num_messages = 10000
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    start_time = time.time()

    for i in range(num_messages):
        symbol = symbols[i % len(symbols)]

        # Micro-level price movements
        price = 50000.0 + (i % 1000) * 0.01 + random.uniform(-0.001, 0.001)
        volume = random.uniform(0.001, 1.0)

        stream.push_trade(symbol, price, volume)

        if i % 2000 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            print(f"  Pushed {i:,} messages at {rate:,.0f} msg/s")

    # Wait for all processing
    await asyncio.sleep(2.0)

    # Final statistics
    stats = stream.get_stats()
    total_time = time.time() - start_time

    print(f"\n📊 High-Frequency Results:")
    print(f"  Total messages: {num_messages:,}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Push rate: {num_messages/total_time:,.0f} msg/s")
    print(f"  Processing rate: {stats.processing_rate:,.0f} msg/s")
    print(f"  Avg latency: {stats.avg_processing_time:.2f} μs")
    print(f"  Success rate: {(stats.processed_count/num_messages)*100:.2f}%")

    stream.stop()


async def multi_symbol_example():
    """Multiple symbols with different characteristics"""
    print("\n🔄 Multi-Symbol Processing Example")
    print("-" * 40)

    stream = MarketDataStream(buffer_size=5000)
    stream.add_processor("validator", None)
    stream.add_processor("transform", None)

    stream.start()
    await asyncio.sleep(0.1)

    # Different symbol categories
    symbols = {
        "BTC/USDT": {"base_price": 50000.0, "volatility": 100.0},
        "ETH/USDT": {"base_price": 3000.0, "volatility": 50.0},
        "BNB/USDT": {"base_price": 300.0, "volatility": 10.0},
        "SOL/USDT": {"base_price": 100.0, "volatility": 5.0},
        "ADA/USDT": {"base_price": 0.5, "volatility": 0.1},
    }

    print("Processing multiple symbol types...")

    # Generate data for each symbol with different characteristics
    for i in range(200):
        for symbol, config in symbols.items():
            # Different update frequencies for different symbols
            if symbol == "BTC/USDT" or i % 2 == 0:
                # Simulate realistic price movement
                change = random.gauss(0, config["volatility"] * 0.01)
                price = config["base_price"] + change
                volume = random.uniform(0.1, 5.0)

                if random.random() < 0.7:  # 70% trades
                    stream.push_trade(symbol, price, volume)
                else:  # 30% quotes
                    spread = price * 0.0001  # 0.01% spread
                    stream.push_quote(symbol, price - spread/2, price + spread/2)

        if i % 50 == 0:
            print(f"  Processed {i*len(symbols)} updates...")

    await asyncio.sleep(1.0)

    # Analyze results by symbol
    recent_data = stream.get_recent_data(100)
    symbol_counts = {}
    symbol_types = {}

    for data in recent_data:
        symbol = data['symbol']
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        data_type = data['type']
        if symbol not in symbol_types:
            symbol_types[symbol] = {"trade": 0, "quote": 0}
        symbol_types[symbol][data_type] += 1

    print(f"\n📊 Symbol Distribution:")
    for symbol, count in symbol_counts.items():
        trades = symbol_types[symbol]["trade"]
        quotes = symbol_types[symbol]["quote"]
        print(f"  {symbol}: {count} items ({trades} trades, {quotes} quotes)")

    # Overall stats
    stats = stream.get_stats()
    print(f"\n📈 Overall Statistics:")
    print(f"  Total processed: {stats.processed_count:,}")
    print(f"  Processing rate: {stats.processing_rate:,.0f} msg/s")
    print(f"  Buffer utilization: {stats.current_buffer_size}/5000")

    stream.stop()


async def error_handling_example():
    """Example with data validation and error handling"""
    print("\n⚠️  Error Handling Example")
    print("-" * 40)

    stream = MarketDataStream(buffer_size=1000)
    stream.add_processor("validator", None)  # Will catch invalid data
    stream.add_processor("filter", None)     # Will filter bad spreads

    stream.start()
    await asyncio.sleep(0.1)

    print("Pushing various data (valid and invalid)...")

    # Valid data
    stream.push_trade("BTC/USDT", 50000.0, 1.0)
    stream.push_quote("BTC/USDT", 49999.0, 50001.0)

    # Invalid data (should be filtered out)
    try:
        stream.push_trade("BTC/USDT", -100.0, 1.0)  # Negative price
    except:
        print("  Caught: Negative price (expected)")

    try:
        stream.push_quote("BTC/USDT", 50001.0, 49999.0)  # Bid > Ask
    except:
        print("  Caught: Invalid spread (expected)")

    # Edge cases
    stream.push_trade("BTC/USDT", 0.001, 0.0)  # Minimum valid price
    stream.push_trade("BTC/USDT", 999999.0, 999999.0)  # Large values

    # Wait for processing
    await asyncio.sleep(0.5)

    stats = stream.get_stats()
    print(f"\n📊 Error Handling Results:")
    print(f"  Processed: {stats.processed_count}")
    print(f"  Errors: {stats.error_count}")
    print(f"  Dropped: {stats.dropped_count}")
    print(f"  Success rate: {(stats.processed_count/(stats.processed_count+stats.error_count+1)*100):.1f}%")

    stream.stop()


async def custom_processor_example():
    """Example with custom processing logic"""
    print("\n🔧 Custom Processing Example")
    print("-" * 40)

    stream = MarketDataStream(buffer_size=1000)
    stream.add_processor("validator", None)
    stream.add_processor("transform", None)

    # Add custom processor if supported
    try:
        stream.add_processor("custom", None)
        print("Custom processor added")
    except:
        print("Custom processor not available (expected)")

    stream.start()
    await asyncio.sleep(0.1)

    # Push BTC data
    btc_prices = [50000, 50100, 50200, 50150, 50300, 50250, 50400]

    for i, price in enumerate(btc_prices):
        volume = 1.0 + i * 0.1
        stream.push_trade("BTC/USDT", float(price), volume)
        print(f"  BTC trade: ${price} -> {volume} BTC")
        await asyncio.sleep(0.1)

    await asyncio.sleep(0.5)

    # Get processed data with transformed fields
    recent = stream.get_recent_data(len(btc_prices))

    print(f"\n📊 Processed Data with Transformed Fields:")
    for i, data in enumerate(recent):
        print(f"  {i+1}. {data['symbol']} @ {data.get('timestamp', 'N/A')}")
        if 'price' in data:
            print(f"      Price: ${data['price']:.2f}")
        # Check for transformed fields (would be added by transform processor)

    stats = stream.get_stats()
    print(f"\nTransformation Statistics:")
    print(f"  Items processed: {stats.processed_count}")
    print(f"  Processing time: {stats.avg_processing_time:.2f} μs")

    stream.stop()


async def main():
    """Run all examples"""
    print("🌊 DeepAlpha Market Data Stream Examples")
    print("=" * 50)

    await basic_stream_example()
    await asyncio.sleep(0.5)

    await high_frequency_example()
    await asyncio.sleep(0.5)

    await multi_symbol_example()
    await asyncio.sleep(0.5)

    await error_handling_example()
    await asyncio.sleep(0.5)

    await custom_processor_example()

    print("\n✅ All examples completed!")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())