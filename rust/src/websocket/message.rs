//! WebSocket message types and handling

use crate::common::Result;
use pyo3::prelude::*;
use serde::{Serialize, Deserialize, Serializer};
use serde_json::Value;
use std::collections::HashMap;

/// WebSocket message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMessage {
    /// Unique message ID
    pub id: String,
    /// Message type (e.g., "price_update", "trade_signal", "auth")
    pub message_type: String,
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
    /// Message payload
    #[serde(flatten)]
    pub data: Value,
}

impl WebSocketMessage {
    /// Create a new message
    pub fn new<S, T>(message_type: S, data: T) -> Result<Self>
    where
        S: Into<String>,
        T: Serialize,
    {
        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            message_type: message_type.into(),
            timestamp: chrono::Utc::now().timestamp_millis(),
            data: serde_json::to_value(data)?,
        })
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Convert to JSON bytes
    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Create a price update message
    pub fn price_update(symbol: &str, price: f64, volume: Option<f64>) -> Result<Self> {
        let mut data = HashMap::new();
        data.insert("symbol".to_string(), Value::String(symbol.to_string()));
        data.insert("price".to_string(), Value::Number(serde_json::Number::from_f64(price).unwrap()));
        if let Some(vol) = volume {
            data.insert("volume".to_string(), Value::Number(serde_json::Number::from_f64(vol).unwrap()));
        }

        Self::new("price_update", data)
    }

    /// Create a trade signal message
    pub fn trade_signal(signal: &TradeSignal) -> Result<Self> {
        Self::new("trade_signal", signal)
    }

    /// Create an authentication response
    pub fn auth_response(success: bool, token: Option<&str>, error: Option<&str>) -> Result<Self> {
        let mut data = HashMap::new();
        data.insert("success".to_string(), Value::Bool(success));
        if let Some(tok) = token {
            data.insert("token".to_string(), Value::String(tok.to_string()));
        }
        if let Some(err) = error {
            data.insert("error".to_string(), Value::String(err.to_string()));
        }

        Self::new("auth_response", data)
    }

    /// Create a subscription response
    pub fn subscription_response(channel: &str, subscribed: bool, error: Option<&str>) -> Result<Self> {
        let mut data = HashMap::new();
        data.insert("channel".to_string(), Value::String(channel.to_string()));
        data.insert("subscribed".to_string(), Value::Bool(subscribed));
        if let Some(err) = error {
            data.insert("error".to_string(), Value::String(err.to_string()));
        }

        Self::new("subscription", data)
    }
}

/// Trade signal structure
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeSignal {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub action: String, // "buy" or "sell"
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub quantity: f64,
    #[pyo3(get)]
    pub strategy: String,
    #[pyo3(get)]
    pub confidence: f64, // 0.0 to 1.0
    #[pyo3(get)]
    pub timestamp: i64,
    #[pyo3(get)]
    pub metadata: Option<HashMap<String, Value>>,
}

#[pymethods]
impl TradeSignal {
    #[new]
    fn new(
        symbol: String,
        action: String,
        price: f64,
        quantity: f64,
        strategy: String,
        confidence: f64,
        metadata: Option<HashMap<String, Value>>,
    ) -> Self {
        Self {
            symbol,
            action,
            price,
            quantity,
            strategy,
            confidence,
            timestamp: chrono::Utc::now().timestamp_millis(),
            metadata,
        }
    }
}

/// Market data structure
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub bid: f64,
    #[pyo3(get)]
    pub ask: f64,
    #[pyo3(get)]
    pub last: f64,
    #[pyo3(get)]
    pub volume: f64,
    #[pyo3(get)]
    pub timestamp: i64,
}

#[pymethods]
impl MarketData {
    #[new]
    fn new(symbol: String, bid: f64, ask: f64, last: f64, volume: f64) -> Self {
        Self {
            symbol,
            bid,
            ask,
            last,
            volume,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }
}

/// OHLC candlestick data
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandlestickData {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub interval: String, // "1m", "5m", "1h", etc.
    #[pyo3(get)]
    pub open: f64,
    #[pyo3(get)]
    pub high: f64,
    #[pyo3(get)]
    pub low: f64,
    #[pyo3(get)]
    pub close: f64,
    #[pyo3(get)]
    pub volume: f64,
    #[pyo3(get)]
    pub timestamp: i64,
}

#[pymethods]
impl CandlestickData {
    #[new]
    fn new(
        symbol: String,
        interval: String,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            symbol,
            interval,
            open,
            high,
            low,
            close,
            volume,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }
}

/// Message types enumeration for Python
#[pyclass]
#[derive(Debug, Clone)]
pub enum MessageType {
    #[pyo3(name = "price")]
    Price,
    #[pyo3(name = "trade")]
    Trade,
    #[pyo3(name = "orderbook")]
    Orderbook,
    #[pyo3(name = "candlestick")]
    Candlestick,
    #[pyo3(name = "signal")]
    Signal,
    #[pyo3(name = "auth")]
    Auth,
    #[pyo3(name = "subscription")]
    Subscription,
    #[pyo3(name = "error")]
    Error,
}

#[pymethods]
impl MessageType {
    /// Convert to string
    fn to_string(&self) -> String {
        match self {
            MessageType::Price => "price_update",
            MessageType::Trade => "trade_update",
            MessageType::Orderbook => "orderbook_update",
            MessageType::Candlestick => "candlestick_update",
            MessageType::Signal => "trade_signal",
            MessageType::Auth => "auth_response",
            MessageType::Subscription => "subscription",
            MessageType::Error => "error",
        }.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_websocket_message_creation() {
        let msg = WebSocketMessage::new("test", {"key": "value"}).unwrap();
        assert_eq!(msg.message_type, "test");
        assert!(msg.timestamp > 0);
        assert!(!msg.id.is_empty());
    }

    #[test]
    fn test_price_update_message() {
        let msg = WebSocketMessage::price_update("BTC/USDT", 50000.0, Some(100.0)).unwrap();
        assert_eq!(msg.message_type, "price_update");
        assert_eq!(msg.data["symbol"], "BTC/USDT");
        assert_eq!(msg.data["price"], 50000.0);
        assert_eq!(msg.data["volume"], 100.0);
    }

    #[test]
    fn test_trade_signal() {
        let signal = TradeSignal::new(
            "BTC/USDT".to_string(),
            "buy".to_string(),
            50000.0,
            0.1,
            "MA_Cross".to_string(),
            0.85,
            None,
        );
        assert_eq!(signal.symbol, "BTC/USDT");
        assert_eq!(signal.action, "buy");
        assert_eq!(signal.confidence, 0.85);

        let msg = WebSocketMessage::trade_signal(&signal).unwrap();
        assert_eq!(msg.message_type, "trade_signal");
    }

    #[test]
    fn test_message_serialization() {
        let msg = WebSocketMessage::new("test", {"key": "value"}).unwrap();
        let json = msg.to_json().unwrap();
        assert!(json.contains("\"message_type\":\"test\""));
        assert!(json.contains("\"key\":\"value\""));
    }
}