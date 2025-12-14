"""WebSocket Server Example

Example of how to use the Rust WebSocket manager for real-time data streaming.
"""

import asyncio
import json
import time
from deepalpha_rust import (
    WebSocketManager,
    TradeSignal,
    MarketData,
    CandlestickData,
    MessageType
)


async def main():
    """Main example function"""
    print("🚀 DeepAlpha WebSocket Server Example")
    print("=" * 50)

    # Create WebSocket manager
    manager = WebSocketManager("127.0.0.1", 8765)

    # Start the server
    print("\n📡 Starting WebSocket server on 127.0.0.1:8765...")
    manager.start()
    await asyncio.sleep(1)  # Give server time to start

    # Get initial stats
    stats = manager.get_stats()
    print(f"Initial stats: {stats.active_connections} active connections")

    # Simulate real-time market data
    print("\n📈 Simulating market data broadcast...")
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    base_prices = {
        "BTC/USDT": 50000.0,
        "ETH/USDT": 3000.0,
        "BNB/USDT": 300.0
    }

    # Broadcast market data for 30 seconds
    for i in range(30):
        # Update prices (random walk)
        for symbol in symbols:
            base_price = base_prices[symbol]
            change = (hash(f"{symbol}_{i}") % 100 - 50) / 10000  # -0.005 to 0.005
            new_price = base_price * (1 + change)
            base_prices[symbol] = new_price

            # Create market data
            market_data = MarketData(
                symbol=symbol,
                bid=new_price - 1,
                ask=new_price + 1,
                last=new_price,
                volume=float(hash(i) % 100)
            )

            # Broadcast
            manager.broadcast("market_data", {
                "symbol": market_data.symbol,
                "bid": market_data.bid,
                "ask": market_data.ask,
                "last": market_data.last,
                "volume": market_data.volume,
                "timestamp": market_data.timestamp
            })

        # Every 5 seconds, generate a trade signal
        if i % 5 == 0:
            signal = TradeSignal(
                symbol=symbols[i % len(symbols)],
                action="buy" if i % 10 < 5 else "sell",
                price=base_prices[symbols[i % len(symbols)]],
                quantity=0.1,
                strategy="MA_Cross",
                confidence=0.85,
                metadata={
                    "indicator": "RSI",
                    "value": 65.5,
                    "signal_strength": "strong"
                }
            )

            manager.broadcast("trade_signal", {
                "symbol": signal.symbol,
                "action": signal.action,
                "price": signal.price,
                "quantity": signal.quantity,
                "strategy": signal.strategy,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp,
                "metadata": signal.metadata
            })

            print(f"  Signal: {signal.action} {signal.quantity} {signal.symbol} @ {signal.price}")

        # Every 3 seconds, send candlestick data
        if i % 3 == 0:
            candle = CandlestickData(
                symbol="BTC/USDT",
                interval="1m",
                open=base_prices["BTC/USDT"] - 10,
                high=base_prices["BTC/USDT"] + 5,
                low=base_prices["BTC/USDT"] - 15,
                close=base_prices["BTC/USDT"],
                volume=float(hash(i) % 1000)
            )

            manager.broadcast("candlestick", {
                "symbol": candle.symbol,
                "interval": candle.interval,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "timestamp": candle.timestamp
            })

        # Print stats every 10 seconds
        if i % 10 == 0 and i > 0:
            stats = manager.get_stats()
            print(f"\n📊 Stats at {i}s:")
            print(f"  Active connections: {stats.active_connections}")
            print(f"  Messages sent: {stats.total_messages_sent:,}")
            print(f"  Bytes sent: {stats.bytes_sent:,}")
            print(f"  Errors: {stats.errors}")

        await asyncio.sleep(1)

    # Final stats
    print("\n✅ Broadcasting completed!")
    final_stats = manager.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Total messages sent: {final_stats.total_messages_sent:,}")
    print(f"  Total bytes sent: {final_stats.bytes_sent:,}")
    print(f"  Active connections: {final_stats.active_connections}")

    # Stop the server
    print("\n🛑 Stopping WebSocket server...")
    manager.stop()


async def client_example():
    """Example WebSocket client"""
    import websockets

    print("\n🔌 Connecting to WebSocket server...")
    uri = "ws://127.0.0.1:8765"

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")

            # Send authentication
            auth_msg = {
                "type": "auth",
                "client_id": "example_client",
                "token": "example_token"
            }
            await websocket.send(json.dumps(auth_msg))

            # Listen for messages
            message_count = 0
            start_time = time.time()

            while message_count < 50:  # Listen for 50 messages
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    message_count += 1

                    # Print based on message type
                    msg_type = data.get("message_type", "unknown")
                    if msg_type == "market_data":
                        symbol = data.get("symbol", "")
                        price = data.get("last", 0)
                        print(f"  📊 {symbol}: ${price:.2f}")
                    elif msg_type == "trade_signal":
                        symbol = data.get("symbol", "")
                        action = data.get("action", "")
                        print(f"  🚨 Signal: {action} {symbol}")
                    elif msg_type == "candlestick":
                        symbol = data.get("symbol", "")
                        close = data.get("close", 0)
                        print(f"  📈 Candle {symbol}: ${close:.2f}")

                except asyncio.TimeoutError:
                    print("  ⏰ No message received in 5 seconds")
                    break
                except Exception as e:
                    print(f"  ❌ Error receiving message: {e}")
                    break

            duration = time.time() - start_time
            print(f"\n📈 Received {message_count} messages in {duration:.2f} seconds")
            print(f"   Rate: {message_count/duration:.1f} messages/second")

    except Exception as e:
        print(f"❌ Failed to connect: {e}")


if __name__ == "__main__":
    # Run server example
    print("Choose an option:")
    print("1. Run WebSocket server")
    print("2. Run WebSocket client")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        asyncio.run(main())
    elif choice == "2":
        asyncio.run(client_example())
    else:
        print("Invalid choice")