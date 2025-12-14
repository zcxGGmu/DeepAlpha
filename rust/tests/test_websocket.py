"""Test cases for WebSocket functionality"""

import pytest
import asyncio
import websockets
import json
import time
from concurrent.futures import ThreadPoolExecutor
from deepalpha_rust import WebSocketManager


class TestWebSocketManager:
    """Test WebSocket Manager functionality"""

    @pytest.fixture
    def websocket_manager(self):
        """Create a WebSocket manager instance"""
        return WebSocketManager("127.0.0.1", 8765)

    def test_manager_creation(self, websocket_manager):
        """Test WebSocket manager creation"""
        assert websocket_manager.host == "127.0.0.1"
        assert websocket_manager.port == 8765

    def test_get_initial_stats(self, websocket_manager):
        """Test getting initial statistics"""
        stats = websocket_manager.get_stats()
        assert stats.active_connections == 0
        assert stats.total_messages_sent == 0
        assert stats.total_messages_received == 0
        assert stats.bytes_sent == 0
        assert stats.bytes_received == 0
        assert stats.errors == 0

    @pytest.mark.asyncio
    async def test_broadcast_message(self, websocket_manager):
        """Test broadcasting a message"""
        test_data = {"symbol": "BTC/USDT", "price": 50000.0}

        # This should not raise an exception even without connections
        websocket_manager.broadcast("price_update", test_data)

        # Check stats
        stats = websocket_manager.get_stats()
        assert stats.total_messages_sent > 0

    def test_send_to_nonexistent_client(self, websocket_manager):
        """Test sending message to non-existent client"""
        test_data = {"test": "data"}
        result = websocket_manager.send_to_client("nonexistent", "test", test_data)
        assert result is False

    def test_disconnect_nonexistent_client(self, websocket_manager):
        """Test disconnecting non-existent client"""
        result = websocket_manager.disconnect_client("nonexistent")
        assert result is False

    def test_get_connected_clients(self, websocket_manager):
        """Test getting list of connected clients"""
        clients = websocket_manager.get_connected_clients()
        assert isinstance(clients, list)
        assert len(clients) == 0


@pytest.mark.slow
class TestWebSocketPerformance:
    """Performance tests for WebSocket"""

    @pytest.mark.asyncio
    async def test_concurrent_connections(self):
        """Test handling many concurrent connections"""
        manager = WebSocketManager("127.0.0.1", 8766)
        manager.start()

        # Give server time to start
        await asyncio.sleep(0.5)

        # Create multiple concurrent connections
        async def create_connection(client_id):
            try:
                uri = "ws://127.0.0.1:8766"
                async with websockets.connect(uri) as websocket:
                    # Send identification
                    await websocket.send(json.dumps({
                        "type": "auth",
                        "client_id": client_id
                    }))

                    # Wait for messages
                    try:
                        await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        return True
                    except asyncio.TimeoutError:
                        return True  # Success if connected
            except Exception:
                return False

        # Create 100 concurrent connections
        num_connections = 100
        tasks = [create_connection(f"client_{i}") for i in range(num_connections)]
        results = await asyncio.gather(*tasks)

        success_count = sum(results)
        print(f"\nSuccessfully connected {success_count}/{num_connections} clients")

        # Test broadcasting
        for i in range(10):
            test_data = {"broadcast": i, "timestamp": time.time()}
            manager.broadcast("test", test_data)
            await asyncio.sleep(0.1)

        # Check stats
        stats = manager.get_stats()
        print(f"Active connections: {stats.active_connections}")
        print(f"Messages sent: {stats.total_messages_sent}")

        # Clean up
        manager.stop()

    @pytest.mark.asyncio
    async def test_message_throughput(self):
        """Test message broadcasting throughput"""
        manager = WebSocketManager("127.0.0.1", 8767)
        manager.start()

        # Give server time to start
        await asyncio.sleep(0.5)

        # Create a single connection
        messages_received = []

        async def client_task():
            try:
                uri = "ws://127.0.0.1:8767"
                async with websockets.connect(uri) as websocket:
                    start_time = time.time()
                    timeout = 5.0  # 5 seconds test

                    while time.time() - start_time < timeout:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            messages_received.append(message)
                        except asyncio.TimeoutError:
                            continue

            except Exception as e:
                print(f"Client error: {e}")

        # Start client in background
        client_task = asyncio.create_task(client_task())

        # Wait for connection to establish
        await asyncio.sleep(0.5)

        # Broadcast messages as fast as possible
        num_messages = 1000
        start_time = time.time()

        for i in range(num_messages):
            test_data = {
                "message_id": i,
                "data": "x" * 100,  # 100 bytes payload
                "timestamp": time.time()
            }
            manager.broadcast("throughput_test", test_data)

        broadcast_time = time.time() - start_time

        # Wait for all messages to be received
        await asyncio.sleep(2.0)
        client_task.cancel()

        # Calculate metrics
        received_count = len(messages_received)
        broadcast_rate = num_messages / broadcast_time
        receive_rate = received_count / 5.0  # 5 seconds test window

        print(f"\nThroughput Test Results:")
        print(f"Messages broadcast: {num_messages}")
        print(f"Broadcast time: {broadcast_time:.2f}s")
        print(f"Broadcast rate: {broadcast_rate:.0f} msg/s")
        print(f"Messages received: {received_count}")
        print(f"Receive rate: {receive_rate:.0f} msg/s")

        # Check stats
        stats = manager.get_stats()
        print(f"Total messages sent (from stats): {stats.total_messages_sent}")
        print(f"Total bytes sent: {stats.bytes_sent}")

        # Clean up
        manager.stop()

        # Assertions
        assert broadcast_rate > 1000  # Should handle >1000 msg/s
        assert received_count > 0  # Should receive some messages


@pytest.mark.integration
class TestWebSocketIntegration:
    """Integration tests with actual WebSocket clients"""

    @pytest.mark.asyncio
    async def test_echo_server(self):
        """Test WebSocket server echo functionality"""
        manager = WebSocketManager("127.0.0.1", 8768)
        manager.start()

        # Give server time to start
        await asyncio.sleep(0.5)

        try:
            # Connect client
            uri = "ws://127.0.0.1:8768"
            async with websockets.connect(uri) as websocket:
                # Send a test message
                test_message = json.dumps({
                    "type": "test",
                    "data": "Hello WebSocket!"
                })
                await websocket.send(test_message)

                # The server should broadcast it back
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    response_data = json.loads(response)

                    # Verify it's a valid message structure
                    assert "id" in response_data
                    assert "message_type" in response_data
                    assert "timestamp" in response_data

                    print(f"\nReceived broadcast: {response_data}")

                except asyncio.TimeoutError:
                    pytest.fail("No response received from server")

        finally:
            manager.stop()


if __name__ == "__main__":
    # Run standalone tests
    import sys

    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run pytest
    sys.exit(pytest.main([__file__, "-v"]))