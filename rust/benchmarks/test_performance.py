"""Performance benchmark tests for Rust indicators"""

import time
import numpy as np
from typing import List, Dict, Any

def generate_price_data(n: int = 50000) -> List[float]:
    """Generate realistic price data for testing"""
    np.random.seed(42)
    # Generate random walk with drift
    returns = np.random.normal(0.0001, 0.01, n)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices.tolist()

def benchmark_indicators():
    """Benchmark all technical indicators"""
    print("=" * 60)
    print("DeepAlpha Rust Indicators Performance Benchmark")
    print("=" * 60)

    # Generate test data
    print("\nGenerating test data...")
    data = generate_price_data(50000)
    print(f"Generated {len(data):,} price points")

    try:
        from deepalpha_rust import TechnicalIndicators
        indicators = TechnicalIndicators(data)

        results = {}

        # Test SMA
        print("\nTesting Simple Moving Average (SMA)...")
        start = time.time()
        sma_result = indicators.sma(period=20)
        end = time.time()
        sma_time = end - start
        results['SMA'] = sma_time
        print(f"  Time: {sma_time*1000:.2f}ms")
        print(f"  Throughput: {len(data)/sma_time:,.0f} points/sec")

        # Test EMA
        print("\nTesting Exponential Moving Average (EMA)...")
        start = time.time()
        ema_result = indicators.ema(period=20)
        end = time.time()
        ema_time = end - start
        results['EMA'] = ema_time
        print(f"  Time: {ema_time*1000:.2f}ms")
        print(f"  Throughput: {len(data)/ema_time:,.0f} points/sec")

        # Test RSI
        print("\nTesting Relative Strength Index (RSI)...")
        start = time.time()
        rsi_result = indicators.rsi(period=14)
        end = time.time()
        rsi_time = end - start
        results['RSI'] = rsi_time
        print(f"  Time: {rsi_time*1000:.2f}ms")
        print(f"  Throughput: {len(data)/rsi_time:,.0f} points/sec")

        # Test MACD
        print("\nTesting MACD...")
        start = time.time()
        macd_result = indicators.macd(fast=12, slow=26, signal=9)
        end = time.time()
        macd_time = end - start
        results['MACD'] = macd_time
        print(f"  Time: {macd_time*1000:.2f}ms")
        print(f"  Throughput: {len(data)/macd_time:,.0f} points/sec")

        # Test Bollinger Bands
        print("\nTesting Bollinger Bands...")
        start = time.time()
        bb_result = indicators.bollinger_bands(period=20, std_dev=2.0)
        end = time.time()
        bb_time = end - start
        results['Bollinger Bands'] = bb_time
        print(f"  Time: {bb_time*1000:.2f}ms")
        print(f"  Throughput: {len(data)/bb_time:,.0f} points/sec")

        # Test Stochastic
        print("\nTesting Stochastic Oscillator...")
        start = time.time()
        stoch_result = indicators.stochastic(k_period=14, d_period=3)
        end = time.time()
        stoch_time = end - start
        results['Stochastic'] = stoch_time
        print(f"  Time: {stoch_time*1000:.2f}ms")
        print(f"  Throughput: {len(data)/stoch_time:,.0f} points/sec")

        # Summary
        print("\n" + "=" * 60)
        print("Performance Summary")
        print("=" * 60)
        print(f"{'Indicator':<20} {'Time (ms)':<12} {'Throughput':<20}")
        print("-" * 60)
        for indicator, time_ms in results.items():
            throughput = len(data) / (time_ms / 1000)
            print(f"{indicator:<20} {time_ms*1000:<12.2f} {throughput:<20,.0f}")

        # Performance targets check
        print("\nPerformance Targets Check:")
        print("-" * 30)

        target = 50000  # Target: 50,000 points/second
        all_met = True

        for indicator, time_ms in results.items():
            throughput = len(data) / (time_ms / 1000)
            status = "✓ PASS" if throughput >= target else "✗ FAIL"
            print(f"{indicator}: {status} ({throughput:,.0f} points/sec)")
            if throughput < target:
                all_met = False

        if all_met:
            print(f"\n🎉 All performance targets met! (> {target:,} points/sec)")
        else:
            print(f"\n⚠️  Some indicators below target ({target:,} points/sec)")

        return results

    except ImportError:
        print("\n❌ Error: Rust module not built!")
        print("Please run 'python build.py' to build the Rust extensions.")
        return None

def compare_with_python():
    """Compare Rust performance with Python implementation"""
    print("\n" + "=" * 60)
    print("Rust vs Python Performance Comparison")
    print("=" * 60)

    # Generate smaller dataset for Python (for reasonable test time)
    data = generate_price_data(10000)

    try:
        # Test Rust implementation
        from deepalpha_rust import TechnicalIndicators
        rust_indicators = TechnicalIndicators(data)

        print("\nTesting Rust implementation...")
        start = time.time()
        rust_sma = rust_indicators.sma(period=20)
        rust_rsi = rust_indicators.rsi(period=14)
        rust_macd = rust_indicators.macd(fast=12, slow=26, signal=9)
        rust_time = time.time() - start

        print(f"  Rust time: {rust_time*1000:.2f}ms")

        # Test Python implementation (simple version)
        print("\nTesting Python implementation...")
        import pandas as pd

        df = pd.DataFrame({'close': data})
        start = time.time()

        # Simple SMA
        python_sma = df['close'].rolling(window=20).mean()

        # Simple RSI approximation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        python_rsi = 100 - (100 / (1 + rs))

        python_time = time.time() - start
        print(f"  Python time: {python_time*1000:.2f}ms")

        # Calculate speedup
        speedup = python_time / rust_time
        print(f"\n🚀 Rust speedup: {speedup:.1f}x faster")

    except ImportError:
        print("\n❌ Error: Could not run comparison (Rust module not built)")
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")

if __name__ == "__main__":
    # Run benchmarks
    results = benchmark_indicators()

    if results:
        # Run comparison if benchmarks succeeded
        compare_with_python()

        print("\n" + "=" * 60)
        print("Benchmark completed!")
        print("=" * 60)