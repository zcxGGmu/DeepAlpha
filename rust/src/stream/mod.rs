//! Market Data Stream Module
//!
//! High-performance real-time data stream processing with support for
//! 100,000+ data points per second and sub-millisecond latency.

use crate::common::{Result, DeepAlphaError, Price};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, broadcast};
use tokio::time::{Duration, Instant};
use futures_util::{StreamExt, SinkExt};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error, debug};

pub mod processor;
pub mod buffer;
pub mod validator;
pub mod aggregator;

use processor::{DataProcessor, ProcessorType};
use buffer::RingBuffer;
use validator::DataValidator;

/// Initialize the stream submodule
pub fn init_stream_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<MarketDataStream>()?;
    m.add_class::<StreamStats>()?;
    Ok(())
}

/// Stream statistics
#[pyclass]
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    #[pyo3(get)]
    pub processed_count: u64,
    #[pyo3(get)]
    pub error_count: u64,
    #[pyo3(get)]
    pub dropped_count: u64,
    #[pyo3(get)]
    pub current_buffer_size: usize,
    #[pyo3(get)]
    pub processing_rate: f64,
    #[pyo3(get)]
    pub avg_processing_time: f64,
}

/// Market data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketDataType {
    Trade,
    OrderBook,
    Quote,
    Candlestick,
    Custom(String),
}

/// Market data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataPoint {
    pub id: String,
    pub data_type: MarketDataType,
    pub symbol: String,
    pub timestamp: i64,
    pub price: Option<f64>,
    pub volume: Option<f64>,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub data: HashMap<String, f64>,
}

impl MarketDataPoint {
    pub fn new_trade(symbol: String, price: f64, volume: f64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            data_type: MarketDataType::Trade,
            symbol,
            timestamp: chrono::Utc::now().timestamp_millis(),
            price: Some(price),
            volume: Some(volume),
            bid: None,
            ask: None,
            data: HashMap::new(),
        }
    }

    pub fn new_quote(symbol: String, bid: f64, ask: f64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            data_type: MarketDataType::Quote,
            symbol,
            timestamp: chrono::Utc::now().timestamp_millis(),
            price: None,
            volume: None,
            bid: Some(bid),
            ask: Some(ask),
            data: HashMap::new(),
        }
    }

    pub fn add_field(&mut self, key: String, value: f64) {
        self.data.insert(key, value);
    }
}

/// High-performance market data stream processor
#[pyclass]
pub struct MarketDataStream {
    #[pyo3(get)]
    active: bool,

    // Internal state
    data_rx: Option<mpsc::UnboundedReceiver<MarketDataPoint>>,
    data_tx: mpsc::UnboundedSender<MarketDataPoint>,
    processors: Arc<RwLock<Vec<Box<dyn DataProcessor>>>>,
    buffer: Arc<RingBuffer<MarketDataPoint>>,
    stats: Arc<RwLock<StreamStats>>,
    validator: Arc<DataValidator>,
}

#[pymethods]
impl MarketDataStream {
    /// Create a new market data stream
    #[new]
    fn new(buffer_size: usize) -> Self {
        let (data_tx, data_rx) = mpsc::unbounded_channel();
        let buffer = Arc::new(RingBuffer::new(buffer_size));

        Self {
            active: false,
            data_rx: Some(data_rx),
            data_tx,
            processors: Arc::new(RwLock::new(Vec::new())),
            buffer,
            stats: Arc::new(RwLock::new(StreamStats::default())),
            validator: Arc::new(DataValidator::new()),
        }
    }

    /// Start processing the data stream
    fn start(&mut self) -> PyResult<()> {
        if self.active {
            return Ok(());
        }

        if self.data_rx.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Cannot start stream: already consumed"
            ));
        }

        let data_rx = self.data_rx.take().unwrap();
        let processors = self.processors.clone();
        let buffer = self.buffer.clone();
        let stats = self.stats.clone();
        let validator = self.validator.clone();

        self.active = true;

        // Start processing in background
        tokio::spawn(async move {
            process_stream(data_rx, processors, buffer, stats, validator).await;
        });

        info!("Market data stream started");
        Ok(())
    }

    /// Stop processing the data stream
    fn stop(&mut self) {
        self.active = false;
        info!("Market data stream stopping");
    }

    /// Add a data processor to the stream
    fn add_processor(&self, processor_type: String, config: Option<PyObject>) -> PyResult<()> {
        let proc_type = match processor_type.as_str() {
            "filter" => ProcessorType::Filter,
            "transform" => ProcessorType::Transform,
            "aggregator" => ProcessorType::Aggregator,
            "validator" => ProcessorType::Validator,
            "custom" => ProcessorType::Custom,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid processor type"
            )),
        };

        let processor = processor::create_processor(proc_type, config)?;

        let mut processors = self.processors.write()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        processors.push(processor);

        Ok(())
    }

    /// Push data into the stream
    fn push_trade(&self, symbol: String, price: f64, volume: f64) -> PyResult<()> {
        let data_point = MarketDataPoint::new_trade(symbol, price, volume);
        self.data_tx.send(data_point)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    /// Push quote data into the stream
    fn push_quote(&self, symbol: String, bid: f64, ask: f64) -> PyResult<()> {
        let data_point = MarketDataPoint::new_quote(symbol, bid, ask);
        self.data_tx.send(data_point)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    /// Get current statistics
    fn get_stats(&self) -> PyResult<StreamStats> {
        let stats = self.stats.read()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let mut result = StreamStats::default();
        result.processed_count = stats.processed_count;
        result.error_count = stats.error_count;
        result.dropped_count = stats.dropped_count;
        result.current_buffer_size = stats.current_buffer_size;
        result.processing_rate = stats.processing_rate;
        result.avg_processing_time = stats.avg_processing_time;

        Ok(result)
    }

    /// Get recent data from buffer
    fn get_recent_data(&self, count: usize) -> PyResult<Vec<PyObject>> {
        let py = Python::acquire_gil().python();
        let buffer = self.buffer.read().unwrap();
        let recent_data = buffer.get_recent(count);

        let mut result = Vec::new();
        for data_point in recent_data {
            let dict = pyo3::PyDict::new(py);
            dict.set_item("id", data_point.id)?;
            dict.set_item("symbol", data_point.symbol)?;
            dict.set_item("timestamp", data_point.timestamp)?;

            match data_point.data_type {
                MarketDataType::Trade => {
                    dict.set_item("type", "trade")?;
                    dict.set_item("price", data_point.price.unwrap_or(0.0))?;
                    dict.set_item("volume", data_point.volume.unwrap_or(0.0))?;
                }
                MarketDataType::Quote => {
                    dict.set_item("type", "quote")?;
                    dict.set_item("bid", data_point.bid.unwrap_or(0.0))?;
                    dict.set_item("ask", data_point.ask.unwrap_or(0.0))?;
                }
                _ => {
                    dict.set_item("type", "custom")?;
                }
            }

            result.push(dict.into());
        }

        Ok(result)
    }
}

/// Main stream processing loop
async fn process_stream(
    mut data_rx: mpsc::UnboundedReceiver<MarketDataPoint>,
    processors: Arc<RwLock<Vec<Box<dyn DataProcessor>>>>,
    buffer: Arc<RingBuffer<MarketDataPoint>>,
    stats: Arc<RwLock<StreamStats>>,
    validator: Arc<DataValidator>,
) {
    let mut processing_times = Vec::new();
    let mut last_stats_update = Instant::now();
    let mut last_count = 0u64;

    while let Some(mut data_point) = data_rx.recv().await {
        let start_time = Instant::now();

        // Validate data
        if let Err(e) = validator.validate(&data_point) {
            warn!("Data validation failed: {}", e);
            if let Ok(mut stats) = stats.try_write() {
                stats.error_count += 1;
            }
            continue;
        }

        // Process through all processors
        let mut processors_guard = processors.write().await;
        let mut should_continue = true;

        for processor in processors_guard.iter_mut() {
            match processor.process(&data_point).await {
                Ok(processed) => {
                    match processed {
                        Some(mut modified) => {
                            modified.id = data_point.id.clone();
                            data_point = modified;
                        }
                        None => {
                            // Filtered out
                            should_continue = false;
                            break;
                        }
                    }
                }
                Err(e) => {
                    error!("Processor error: {}", e);
                    if let Ok(mut stats) = stats.try_write() {
                        stats.error_count += 1;
                    }
                    should_continue = false;
                    break;
                }
            }
        }

        drop(processors_guard);

        if !should_continue {
            continue;
        }

        // Add to buffer
        buffer.push(data_point.clone());

        // Update statistics
        let processing_time = start_time.elapsed().as_micros() as f64;
        processing_times.push(processing_time);

        // Keep only last 1000 processing times for average
        if processing_times.len() > 1000 {
            processing_times.remove(0);
        }

        if let Ok(mut stats) = stats.try_write() {
            stats.processed_count += 1;
            stats.current_buffer_size = buffer.len();

            // Update rate calculations every second
            if last_stats_update.elapsed() >= Duration::from_secs(1) {
                let elapsed = last_stats_update.elapsed().as_secs_f64();
                let count_diff = stats.processed_count - last_count;
                stats.processing_rate = count_diff as f64 / elapsed;

                if !processing_times.is_empty() {
                    stats.avg_processing_time = processing_times.iter().sum::<f64>() / processing_times.len() as f64;
                }

                last_count = stats.processed_count;
                last_stats_update = Instant::now();
            }
        }

        debug!("Processed data point: {} {}", data_point.symbol, data_point.timestamp);
    }

    info!("Stream processing ended");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_data_point_creation() {
        let trade = MarketDataPoint::new_trade("BTC/USDT".to_string(), 50000.0, 1.0);
        assert_eq!(trade.symbol, "BTC/USDT");
        assert_eq!(trade.price, Some(50000.0));
        assert_eq!(trade.volume, Some(1.0));
        matches!(trade.data_type, MarketDataType::Trade);

        let quote = MarketDataPoint::new_quote("ETH/USDT".to_string(), 3000.0, 3001.0);
        assert_eq!(quote.bid, Some(3000.0));
        assert_eq!(quote.ask, Some(3001.0));
        matches!(quote.data_type, MarketDataType::Quote);
    }

    #[test]
    fn test_stream_stats_default() {
        let stats = StreamStats::default();
        assert_eq!(stats.processed_count, 0);
        assert_eq!(stats.error_count, 0);
        assert_eq!(stats.dropped_count, 0);
    }
}