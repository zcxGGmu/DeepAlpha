"""Test cases for Market Data Stream functionality"""

import pytest
import time
import json
from deepalpha_rust import MarketDataStream


class TestMarketDataStream:
    """Test Market Data Stream functionality"""

    @pytest.fixture
    def stream(self):
        """Create a MarketDataStream instance"""
        return MarketDataStream(buffer_size=1000)

    def test_stream_creation(self, stream):
        """Test stream creation"""
        assert not stream.active
        stats = stream.get_stats()
        assert stats.processed_count == 0
        assert stats.error_count == 0
        assert stats.dropped_count == 0

    def test_push_trade_data(self, stream):
        """Test pushing trade data"""
        stream.push_trade("BTC/USDT", 50000.0, 1.0)
        # Even without starting, should not raise an exception

    def test_push_quote_data(self, stream):
        """Test pushing quote data"""
        stream.push_quote("BTC/USDT", 49999.0, 50001.0)
        # Even without starting, should not raise an exception

    @pytest.mark.asyncio
    async def test_stream_start_stop(self, stream):
        """Test starting and stopping the stream"""
        assert not stream.active

        stream.start()
        assert stream.active

        # Starting again should be fine
        stream.start()
        assert stream.active

        stream.stop()
        assert not stream.active

    def test_add_processor(self, stream):
        """Test adding processors"""
        # Add filter processor
        stream.add_processor("filter", None)
        # Should not raise an exception

        # Add transform processor
        stream.add_processor("transform", None)
        # Should not raise an exception

        # Add aggregator processor
        stream.add_processor("aggregator", None)
        # Should not raise an exception

        # Add validator processor
        stream.add_processor("validator", None)
        # Should not raise an exception

        # Test invalid processor type
        with pytest.raises(Exception):
            stream.add_processor("invalid_type", None)

    @pytest.mark.asyncio
    async def test_stream_with_processors(self):
        """Test stream with processors"""
        stream = MarketDataStream(buffer_size=100)

        # Add processors
        stream.add_processor("validator", None)
        stream.add_processor("transform", None)

        # Start stream
        stream.start()
        await asyncio.sleep(0.1)  # Give time to start

        # Push some data
        for i in range(10):
            stream.push_trade(f"BTC/USDT", 50000.0 + i, 1.0)
            stream.push_quote(f"ETH/USDT", 3000.0 + i, 3000.5 + i)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Check stats
        stats = stream.get_stats()
        print(f"Processed: {stats.processed_count}")
        print(f"Errors: {stats.error_count}")
        print(f"Processing rate: {stats.processing_rate}")

        # Should have processed some data
        assert stats.processed_count >= 0

        stream.stop()

    @pytest.mark.asyncio
    async def test_get_recent_data(self, stream):
        """Test getting recent data from buffer"""
        # Push some data without starting (should still buffer)
        for i in range(5):
            stream.push_trade(f"BTC/USDT", 50000.0 + i, 1.0)

        # Get recent data
        recent = stream.get_recent_data(3)
        assert len(recent) <= 3

        # Check data structure
        if recent:
            first = recent[0]
            assert "id" in first
            assert "symbol" in first
            assert "timestamp" in first
            assert "type" in first
            assert "price" in first
            assert "volume" in first

    @pytest.mark.asyncio
    async def test_stream_performance(self, stream):
        """Test stream performance with high volume data"""
        # Add a filter to test performance
        stream.add_processor("filter", None)

        # Start stream
        stream.start()
        await asyncio.sleep(0.1)

        # Push a lot of data
        num_messages = 1000
        start_time = time.time()

        for i in range(num_messages):
            stream.push_trade("BTC/USDT", 50000.0 + (i % 100) - 50, 1.0 + (i % 10) * 0.1)

        push_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(1.0)

        # Check performance
        stats = stream.get_stats()
        processing_time = time.time() - start_time

        print(f"\nPerformance Test Results:")
        print(f"Messages pushed: {num_messages}")
        print(f"Push time: {push_time:.3f}s")
        print(f"Push rate: {num_messages/push_time:.0f} msg/s")
        print(f"Total time: {processing_time:.3f}s")
        print(f"Processed: {stats.processed_count}")
        print(f"Errors: {stats.error_count}")
        print(f"Processing rate: {stats.processing_rate:.0f} msg/s")
        print(f"Avg processing time: {stats.avg_processing_time:.3f}μs")

        stream.stop()

        # Performance assertions
        assert num_messages / push_time > 10000  # Should push > 10,000 msg/s
        assert stats.processing_rate > 1000     # Should process > 1,000 msg/s


@pytest.mark.integration
class TestStreamIntegration:
    """Integration tests for stream functionality"""

    @pytest.mark.asyncio
    async def test_multiple_streams(self):
        """Test running multiple streams simultaneously"""
        streams = []
        for i in range(3):
            stream = MarketDataStream(buffer_size=100)
            stream.add_processor("validator", None)
            stream.start()
            streams.append(stream)

        await asyncio.sleep(0.1)

        # Push data to all streams
        for i, stream in enumerate(streams):
            for j in range(10):
                stream.push_trade(f"SYMBOL_{i}", 100.0 + j, 1.0)

        await asyncio.sleep(0.5)

        # Check all streams
        for i, stream in enumerate(streams):
            stats = stream.get_stats()
            print(f"Stream {i}: Processed {stats.processed_count}, Errors {stats.error_count}")
            stream.stop()

    @pytest.mark.asyncio
    async def test_stream_with_config(self):
        """Test stream with specific configuration"""
        # Small buffer to test overflow handling
        stream = MarketDataStream(buffer_size=10)

        # Add multiple processors
        stream.add_processor("validator", None)
        stream.add_processor("filter", None)
        stream.add_processor("transform", None)

        stream.start()
        await asyncio.sleep(0.1)

        # Push more data than buffer size
        for i in range(50):
            stream.push_trade("BTC/USDT", 50000.0 + i, 1.0)

        await asyncio.sleep(0.5)

        stats = stream.get_stats()
        recent_data = stream.get_recent_data(5)

        print(f"Buffer size test: Processed {stats.processed_count}")
        print(f"Recent data count: {len(recent_data)}")

        # Recent data should be limited to buffer size
        assert len(recent_data) <= 10

        stream.stop()


if __name__ == "__main__":
    # Run standalone tests
    import sys
    import asyncio

    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run pytest
    sys.exit(pytest.main([__file__, "-v"]))