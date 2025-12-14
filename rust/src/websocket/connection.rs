//! WebSocket connection management

use crate::common::Result;
use crate::websocket::message::WebSocketMessage;
use pyo3::prelude::*;
use std::net::SocketAddr;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_tungstenite::tungstenite::Message;
use tokio::sync::mpsc;
use futures_util::SinkExt;
use serde_json;

/// Represents a WebSocket connection
pub struct Connection {
    pub id: String,
    pub addr: SocketAddr,
    pub connected_at: u64,
    pub last_ping: u64,
    pub authenticated: bool,
    pub subscriptions: Vec<String>,
    sender: Option<mpsc::UnboundedSender<Message>>,
}

impl Connection {
    /// Create a new connection
    pub fn new<S>(
        id: S,
        addr: SocketAddr,
        sender: S,
    ) -> Self
    where
        S: Into<String>,
    {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: id.into(),
            addr,
            connected_at: now,
            last_ping: now,
            authenticated: false,
            subscriptions: Vec::new(),
            sender: Some(sender),
        }
    }

    /// Send a message to the connection
    pub async fn send_message(&mut self, message: WebSocketMessage) -> Result<bool> {
        let json = serde_json::to_string(&message)?;
        self.send_raw(json).await
    }

    /// Send a raw string message
    pub async fn send_raw(&mut self, message: String) -> Result<bool> {
        if let Some(sender) = &mut self.sender {
            let ws_message = Message::Text(message);

            match sender.send(ws_message).await {
                Ok(_) => Ok(true),
                Err(e) => {
                    // Connection is closed
                    warn!("Failed to send message: {}", e);
                    self.sender = None;
                    Ok(false)
                }
            }
        } else {
            Ok(false)
        }
    }

    /// Check if the connection is alive
    pub fn is_alive(&self) -> bool {
        self.sender.is_some()
    }

    /// Get connection uptime in seconds
    pub fn uptime(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - self.connected_at
    }

    /// Add a subscription
    pub fn subscribe<S>(&mut self, channel: S) -> bool
    where
        S: Into<String>,
    {
        let channel = channel.into();
        if !self.subscriptions.contains(&channel) {
            self.subscriptions.push(channel);
            true
        } else {
            false
        }
    }

    /// Remove a subscription
    pub fn unsubscribe(&mut self, channel: &str) -> bool {
        if let Some(pos) = self.subscriptions.iter().position(|s| s == channel) {
            self.subscriptions.remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if subscribed to a channel
    pub fn is_subscribed(&self, channel: &str) -> bool {
        self.subscriptions.contains(&channel.to_string())
    }

    /// Close the connection
    pub async fn close(&mut self) -> Result<()> {
        if let Some(sender) = &mut self.sender {
            let _ = sender.send(Message::Close(None)).await;
            self.sender = None;
        }
        Ok(())
    }
}

/// Connection information exposed to Python
#[pyclass]
#[derive(Clone, Debug)]
pub struct ConnectionInfo {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub address: String,
    #[pyo3(get)]
    pub connected_at: u64,
    #[pyo3(get)]
    pub uptime: u64,
    #[pyo3(get)]
    pub authenticated: bool,
    #[pyo3(get)]
    pub subscriptions: Vec<String>,
}

impl From<&Connection> for ConnectionInfo {
    fn from(conn: &Connection) -> Self {
        Self {
            id: conn.id.clone(),
            address: conn.addr.to_string(),
            connected_at: conn.connected_at,
            uptime: conn.uptime(),
            authenticated: conn.authenticated,
            subscriptions: conn.subscriptions.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;
    use tokio_tungstenite::tungstenite::Message;

    #[tokio::test]
    async fn test_connection_creation() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let addr = "127.0.0.1:8080".parse().unwrap();

        let conn = Connection::new("test_conn", addr, tx);

        assert_eq!(conn.id, "test_conn");
        assert!(!conn.authenticated);
        assert!(conn.is_alive());
        assert_eq!(conn.subscriptions.len(), 0);
    }

    #[tokio::test]
    async fn test_subscriptions() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let addr = "127.0.0.1:8080".parse().unwrap();

        let mut conn = Connection::new("test_conn", addr, tx);

        // Test subscription
        assert!(conn.subscribe("price:btcusdt"));
        assert!(conn.is_subscribed("price:btcusdt"));
        assert!(!conn.subscribe("price:btcusdt")); // Already subscribed

        // Test unsubscribe
        assert!(conn.unsubscribe("price:btcusdt"));
        assert!(!conn.is_subscribed("price:btcusdt"));
        assert!(!conn.unsubscribe("price:btcusdt")); // Not subscribed
    }
}