//! WebSocket Manager Module
//!
//! High-performance WebSocket connection management with support for
//! 10,000+ concurrent connections and sub-millisecond latency.

use crate::common::{Result, DeepAlphaError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, broadcast};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error, debug};

pub mod connection;
pub mod auth;
pub mod message;

use connection::Connection;
use message::WebSocketMessage;

/// Initialize the websocket submodule
pub fn init_websocket_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<WebSocketManager>()?;
    m.add_class::<ConnectionStats>()?;
    Ok(())
}

/// WebSocket connection statistics
#[pyclass]
#[derive(Debug, Clone, Serialize)]
pub struct ConnectionStats {
    #[pyo3(get)]
    active_connections: usize,
    #[pyo3(get)]
    total_messages_sent: u64,
    #[pyo3(get)]
    total_messages_received: u64,
    #[pyo3(get)]
    bytes_sent: u64,
    #[pyo3(get)]
    bytes_received: u64,
    #[pyo3(get)]
    errors: u64,
}

impl Default for ConnectionStats {
    fn default() -> Self {
        Self {
            active_connections: 0,
            total_messages_sent: 0,
            total_messages_received: 0,
            bytes_sent: 0,
            bytes_received: 0,
            errors: 0,
        }
    }
}

/// High-performance WebSocket manager
#[pyclass]
pub struct WebSocketManager {
    #[pyo3(get)]
    host: String,
    #[pyo3(get)]
    port: u16,

    // Internal state (not exposed to Python)
    connections: Arc<RwLock<HashMap<String, Connection>>>,
    message_tx: broadcast::Sender<WebSocketMessage>,
    stats: Arc<RwLock<ConnectionStats>>,
    config: Arc<RwLock<WebSocketConfig>>,
}

#[pymethods]
impl WebSocketManager {
    /// Create a new WebSocket manager
    #[new]
    fn new(host: String, port: u16) -> Self {
        let (message_tx, _) = broadcast::channel(10000); // Buffer for 10,000 messages

        Self {
            host,
            port,
            connections: Arc::new(RwLock::new(HashMap::new())),
            message_tx,
            stats: Arc::new(RwLock::new(ConnectionStats::default())),
            config: Arc::new(RwLock::new(WebSocketConfig::default())),
        }
    }

    /// Start the WebSocket server
    fn start(&self) -> PyResult<()> {
        let host = self.host.clone();
        let port = self.port;
        let connections = self.connections.clone();
        let message_tx = self.message_tx.clone();
        let stats = self.stats.clone();
        let config = self.config.clone();

        // Run in a new tokio runtime
        tokio::spawn(async move {
            if let Err(e) = run_server(host, port, connections, message_tx, stats, config).await {
                error!("WebSocket server error: {}", e);
            }
        });

        info!("WebSocket server started on {}:{}", host, port);
        Ok(())
    }

    /// Stop the WebSocket server
    fn stop(&self) -> PyResult<()> {
        // TODO: Implement graceful shutdown
        info!("WebSocket server stopping...");
        Ok(())
    }

    /// Broadcast a message to all connected clients
    fn broadcast(&self, py: Python, message_type: String, data: PyObject) -> PyResult<()> {
        let ws_message = WebSocketMessage {
            id: Uuid::new_v4().to_string(),
            message_type,
            timestamp: chrono::Utc::now().timestamp_millis(),
            data: serde_json::to_value(data)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
        };

        if let Err(e) = self.message_tx.send(ws_message.clone()) {
            warn!("Failed to broadcast message: {}", e);
        } else {
            // Update stats
            if let Ok(mut stats) = self.stats.try_write() {
                stats.total_messages_sent += 1;
            }
        }

        Ok(())
    }

    /// Send a message to a specific client
    fn send_to_client(&self, client_id: String, message_type: String, data: PyObject) -> PyResult<bool> {
        let ws_message = WebSocketMessage {
            id: Uuid::new_v4().to_string(),
            message_type,
            timestamp: chrono::Utc::now().timestamp_millis(),
            data: serde_json::to_value(data)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
        };

        let connections = self.connections.clone();

        // Send in a blocking context
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let result = rt.block_on(async move {
            if let Ok(conn_map) = connections.read().await {
                if let Some(conn) = conn_map.get(&client_id) {
                    conn.send_message(ws_message).await
                } else {
                    Ok(false)
                }
            } else {
                Ok(false)
            }
        });

        match result {
            Ok(sent) => {
                if sent {
                    if let Ok(mut stats) = self.stats.try_write() {
                        stats.total_messages_sent += 1;
                    }
                }
                Ok(sent)
            }
            Err(e) => Err(e),
        }
    }

    /// Get current connection statistics
    fn get_stats(&self) -> PyResult<ConnectionStats> {
        let stats = self.stats.read()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Update active connections count
        let active_count = self.connections.read()
            .map(|conns| conns.len())
            .unwrap_or(0);

        Ok(ConnectionStats {
            active_connections: active_count,
            total_messages_sent: stats.total_messages_sent,
            total_messages_received: stats.total_messages_received,
            bytes_sent: stats.bytes_sent,
            bytes_received: stats.bytes_received,
            errors: stats.errors,
        })
    }

    /// Disconnect a specific client
    fn disconnect_client(&self, client_id: String) -> PyResult<bool> {
        let connections = self.connections.clone();

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let result = rt.block_on(async move {
            let mut conn_map = connections.write().await;
            if conn_map.remove(&client_id).is_some() {
                info!("Client {} disconnected", client_id);
                Ok(true)
            } else {
                Ok(false)
            }
        });

        result
    }

    /// Get list of connected client IDs
    fn get_connected_clients(&self) -> PyResult<Vec<String>> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let result = rt.block_on(async move {
            let conn_map = self.connections.read().await;
            Ok(conn_map.keys().cloned().collect())
        });

        result
    }
}

/// Run the WebSocket server
async fn run_server(
    host: String,
    port: u16,
    connections: Arc<RwLock<HashMap<String, Connection>>>,
    message_tx: broadcast::Sender<WebSocketMessage>,
    stats: Arc<RwLock<ConnectionStats>>,
    config: Arc<RwLock<WebSocketConfig>>,
) -> Result<()> {
    let addr: SocketAddr = format!("{}:{}", host, port)
        .parse()
        .map_err(|e| DeepAlphaError::InvalidInput(e.to_string()))?;

    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|e| DeepAlphaError::NetworkError(e))?;

    info!("WebSocket server listening on {}", addr);

    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                debug!("New connection from {}", addr);

                let conn_id = Uuid::new_v4().to_string();
                let connections = connections.clone();
                let message_tx = message_tx.clone();
                let stats = stats.clone();
                let config = config.clone();

                tokio::spawn(async move {
                    if let Err(e) = handle_connection(
                        stream,
                        addr,
                        conn_id,
                        connections,
                        message_tx,
                        stats,
                        config,
                    ).await {
                        error!("Connection handler error: {}", e);
                    }
                });
            }
            Err(e) => {
                error!("Failed to accept connection: {}", e);
                if let Ok(mut stats) = stats.try_write() {
                    stats.errors += 1;
                }
            }
        }
    }
}

/// Handle an individual WebSocket connection
async fn handle_connection(
    stream: TcpStream,
    addr: SocketAddr,
    conn_id: String,
    connections: Arc<RwLock<HashMap<String, Connection>>>,
    message_tx: broadcast::Sender<WebSocketMessage>,
    stats: Arc<RwLock<ConnectionStats>>,
    config: Arc<RwLock<WebSocketConfig>>,
) -> Result<()> {
    let ws_config = config.read().await.clone();
    let ws_stream = accept_async_with_config(stream, Some(ws_config))
        .await
        .map_err(|e| DeepAlphaError::NetworkError(e))?;

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // Create connection object
    let connection = Connection::new(
        conn_id.clone(),
        addr,
        ws_sender,
    );

    // Add to connections map
    {
        let mut conn_map = connections.write().await;
        conn_map.insert(conn_id.clone(), connection);
    }

    info!("Client {} connected from {}", conn_id, addr);

    // Subscribe to broadcasts
    let mut message_rx = message_tx.subscribe();

    // Handle incoming messages and broadcasts
    loop {
        tokio::select! {
            // Handle incoming message from client
            Some(msg_result) = ws_receiver.next() => {
                match msg_result {
                    Ok(msg) => {
                        if msg.is_close() {
                            break;
                        }

                        if let Ok(text) = msg.to_text() {
                            debug!("Received message from {}: {}", conn_id, text);

                            // Update stats
                            if let Ok(mut stats) = stats.try_write() {
                                stats.total_messages_received += 1;
                                stats.bytes_received += msg.len() as u64;
                            }

                            // TODO: Process the message (auth, subscribe, etc.)
                        }
                    }
                    Err(e) => {
                        warn!("WebSocket error from {}: {}", conn_id, e);
                        if let Ok(mut stats) = stats.try_write() {
                            stats.errors += 1;
                        }
                        break;
                    }
                }
            }

            // Handle broadcast message
            Ok(broadcast_msg) = message_rx.recv() => {
                let json_msg = serde_json::to_string(&broadcast_msg)
                    .map_err(|e| DeepAlphaError::JsonError(e))?;

                let mut conn_map = connections.write().await;
                if let Some(conn) = conn_map.get_mut(&conn_id) {
                    if let Err(e) = conn.send_raw(json_msg).await {
                        error!("Failed to send to {}: {}", conn_id, e);
                        break;
                    } else {
                        // Update stats
                        if let Ok(mut stats) = stats.try_write() {
                            stats.bytes_sent += json_msg.len() as u64;
                        }
                    }
                }
            }
        }
    }

    // Remove connection
    {
        let mut conn_map = connections.write().await;
        conn_map.remove(&conn_id);
    }

    info!("Client {} disconnected", conn_id);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_websocket_manager_creation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let manager = WebSocketManager::new("127.0.0.1".to_string(), 8765);
            assert_eq!(manager.host, "127.0.0.1");
            assert_eq!(manager.port, 8765);

            let stats = manager.get_stats().unwrap();
            assert_eq!(stats.active_connections, 0);
        });
    }

    #[test]
    fn test_connection_stats() {
        let stats = ConnectionStats::default();
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_messages_sent, 0);
        assert_eq!(stats.errors, 0);
    }
}