//! Data processors for stream processing

use crate::common::Result;
use crate::stream::MarketDataPoint;
use pyo3::prelude::*;
use async_trait::async_trait;
use std::collections::HashMap;

/// Processor types
#[derive(Debug, Clone)]
pub enum ProcessorType {
    Filter,
    Transform,
    Aggregator,
    Validator,
    Custom,
}

/// Data processor trait
#[async_trait]
pub trait DataProcessor: Send + Sync {
    async fn process(&mut self, data: &MarketDataPoint) -> Result<Option<MarketDataPoint>>;
    fn name(&self) -> &str;
}

/// Filter processor - filters data based on criteria
pub struct FilterProcessor {
    symbol_filter: Option<String>,
    min_price: Option<f64>,
    max_price: Option<f64>,
}

impl FilterProcessor {
    pub fn new(config: HashMap<String, f64>) -> Self {
        Self {
            symbol_filter: config.get("symbol").map(|_| "BTC/USDT".to_string()), // Simplified
            min_price: config.get("min_price").copied(),
            max_price: config.get("max_price").copied(),
        }
    }
}

#[async_trait]
impl DataProcessor for FilterProcessor {
    async fn process(&mut self, data: &MarketDataPoint) -> Result<Option<MarketDataPoint>> {
        // Filter by price range
        if let Some(price) = data.price {
            if let Some(min) = self.min_price {
                if price < min {
                    return Ok(None);
                }
            }
            if let Some(max) = self.max_price {
                if price > max {
                    return Ok(None);
                }
            }
        }

        // Filter by bid/ask spread
        if let (Some(bid), Some(ask)) = (data.bid, data.ask) {
            let spread = ask - bid;
            let spread_pct = spread / ask;
            if spread_pct > 0.01 { // More than 1% spread
                return Ok(None);
            }
        }

        Ok(Some(data.clone()))
    }

    fn name(&self) -> &str {
        "FilterProcessor"
    }
}

/// Transform processor - transforms data
pub struct TransformProcessor {
    add_fields: HashMap<String, f64>,
    currency_conversion: Option<(String, String, f64)>, // (from, to, rate)
}

impl TransformProcessor {
    pub fn new() -> Self {
        Self {
            add_fields: HashMap::new(),
            currency_conversion: None,
        }
    }

    pub fn add_field(&mut self, name: String, value: f64) {
        self.add_fields.insert(name, value);
    }
}

#[async_trait]
impl DataProcessor for TransformProcessor {
    async fn process(&mut self, data: &MarketDataPoint) -> Result<Option<MarketDataPoint>> {
        let mut transformed = data.clone();

        // Add custom fields
        for (key, value) in &self.add_fields {
            transformed.data.insert(key.clone(), *value);
        }

        // Add calculated fields
        if let (Some(bid), Some(ask)) = (transformed.bid, transformed.ask) {
            let spread = ask - bid;
            transformed.data.insert("spread".to_string(), spread);
            transformed.data.insert("mid_price".to_string(), (bid + ask) / 2.0);
        }

        // Add timestamp fields
        transformed.data.insert("timestamp_ms".to_string(), transformed.timestamp as f64);

        Ok(Some(transformed))
    }

    fn name(&self) -> &str {
        "TransformProcessor"
    }
}

/// Aggregation processor - aggregates data over time windows
pub struct AggregatorProcessor {
    window_size_ms: u64,
    data_buffer: Vec<MarketDataPoint>,
    last_aggregate_time: u64,
}

impl AggregatorProcessor {
    pub fn new(window_size_ms: u64) -> Self {
        Self {
            window_size_ms,
            data_buffer: Vec::new(),
            last_aggregate_time: 0,
        }
    }
}

#[async_trait]
impl DataProcessor for AggregatorProcessor {
    async fn process(&mut self, data: &MarketDataPoint) -> Result<Option<MarketDataPoint>> {
        self.data_buffer.push(data.clone());

        let current_time = chrono::Utc::now().timestamp_millis() as u64;

        // Check if it's time to aggregate
        if current_time - self.last_aggregate_time < self.window_size_ms {
            return Ok(None); // Don't forward individual data points
        }

        if self.data_buffer.is_empty() {
            return Ok(None);
        }

        // Aggregate data
        let mut aggregated = MarketDataPoint {
            id: uuid::Uuid::new_v4().to_string(),
            data_type: data.data_type.clone(),
            symbol: data.symbol.clone(),
            timestamp: current_time as i64,
            price: None,
            volume: None,
            bid: None,
            ask: None,
            data: HashMap::new(),
        };

        // Calculate aggregates
        let mut total_volume = 0.0;
        let mut high_price = f64::MIN;
        let mut low_price = f64::MAX;
        let mut sum_price = 0.0;
        let mut count = 0u32;

        for point in &self.data_buffer {
            if let Some(price) = point.price {
                high_price = high_price.max(price);
                low_price = low_price.min(price);
                sum_price += price;
                count += 1;
            }
            if let Some(volume) = point.volume {
                total_volume += volume;
            }
        }

        if count > 0 {
            aggregated.price = Some(sum_price / count as f64);
            aggregated.data.insert("high".to_string(), high_price);
            aggregated.data.insert("low".to_string(), low_price);
            aggregated.data.insert("avg".to_string(), sum_price / count as f64);
        }

        aggregated.volume = Some(total_volume);
        aggregated.data.insert("count".to_string(), count as f64);

        // Clear buffer and update time
        self.data_buffer.clear();
        self.last_aggregate_time = current_time;

        Ok(Some(aggregated))
    }

    fn name(&self) -> &str {
        "AggregatorProcessor"
    }
}

/// Validation processor - validates data integrity
pub struct ValidationProcessor {
    max_price_change_pct: f64,
    max_timestamp_diff_ms: u64,
}

impl ValidationProcessor {
    pub fn new() -> Self {
        Self {
            max_price_change_pct: 10.0, // 10% max change
            max_timestamp_diff_ms: 60000, // 1 minute max timestamp diff
        }
    }
}

#[async_trait]
impl DataProcessor for ValidationProcessor {
    async fn process(&mut self, data: &MarketDataPoint) -> Result<Option<MarketDataPoint>> {
        // Validate timestamp
        let current_time = chrono::Utc::now().timestamp_millis();
        let time_diff = (current_time - data.timestamp).abs() as u64;

        if time_diff > self.max_timestamp_diff_ms {
            return Err(crate::common::DeepAlphaError::InvalidInput(
                format!("Timestamp too old: {}ms", time_diff)
            ));
        }

        // Validate price values
        if let Some(price) = data.price {
            if price <= 0.0 || !price.is_finite() {
                return Err(crate::common::DeepAlphaError::InvalidInput(
                    "Invalid price value".to_string()
                ));
            }
        }

        // Validate bid/ask
        if let (Some(bid), Some(ask)) = (data.bid, data.ask) {
            if bid >= ask {
                return Err(crate::common::DeepAlphaError::InvalidInput(
                    "Bid must be less than ask".to_string()
                ));
            }
        }

        // Validate volume
        if let Some(volume) = data.volume {
            if volume < 0.0 || !volume.is_finite() {
                return Err(crate::common::DeepAlphaError::InvalidInput(
                    "Invalid volume value".to_string()
                ));
            }
        }

        Ok(Some(data.clone()))
    }

    fn name(&self) -> &str {
        "ValidationProcessor"
    }
}

/// Custom processor for Python callbacks
pub struct CustomProcessor {
    name: String,
    callback: Option<PyObject>,
}

impl CustomProcessor {
    pub fn new(name: String) -> Self {
        Self {
            name,
            callback: None,
        }
    }
}

#[async_trait]
impl DataProcessor for CustomProcessor {
    async fn process(&mut self, data: &MarketDataPoint) -> Result<Option<MarketDataPoint>> {
        if let Some(callback) = &self.callback {
            let py = Python::acquire_gil().python();

            // Convert data point to Python dict
            let dict = pyo3::PyDict::new(py);
            dict.set_item("id", &data.id)?;
            dict.set_item("symbol", &data.symbol)?;
            dict.set_item("timestamp", data.timestamp)?;

            match data.data_type {
                crate::stream::MarketDataType::Trade => {
                    dict.set_item("type", "trade")?;
                    dict.set_item("price", data.price.unwrap_or(0.0))?;
                    dict.set_item("volume", data.volume.unwrap_or(0.0))?;
                }
                crate::stream::MarketDataType::Quote => {
                    dict.set_item("type", "quote")?;
                    dict.set_item("bid", data.bid.unwrap_or(0.0))?;
                    dict.set_item("ask", data.ask.unwrap_or(0.0))?;
                }
                _ => {
                    dict.set_item("type", "custom")?;
                }
            }

            // Call Python callback
            let result = callback.call1((dict,))?;

            // Return None if callback returned falsy
            if result.is_truthy().unwrap_or(false) {
                // TODO: Convert back to MarketDataPoint if needed
                Ok(Some(data.clone()))
            } else {
                Ok(None) // Filtered out
            }
        } else {
            Ok(Some(data.clone()))
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Create a processor based on type
pub fn create_processor(processor_type: ProcessorType, _config: Option<PyObject>) -> Result<Box<dyn DataProcessor>> {
    match processor_type {
        ProcessorType::Filter => Ok(Box::new(FilterProcessor::new(HashMap::new()))),
        ProcessorType::Transform => Ok(Box::new(TransformProcessor::new())),
        ProcessorType::Aggregator => Ok(Box::new(AggregatorProcessor::new(1000))), // 1 second window
        ProcessorType::Validator => Ok(Box::new(ValidationProcessor::new())),
        ProcessorType::Custom => Ok(Box::new(CustomProcessor::new("Custom".to_string()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::MarketDataType;

    #[tokio::test]
    async fn test_filter_processor() {
        let mut processor = FilterProcessor::new(HashMap::new());
        processor.min_price = Some(100.0);
        processor.max_price = Some(200.0);

        // Should pass
        let data = MarketDataPoint::new_trade("BTC/USDT".to_string(), 150.0, 1.0);
        let result = processor.process(&data).await.unwrap();
        assert!(result.is_some());

        // Should filter out (too low)
        let data = MarketDataPoint::new_trade("BTC/USDT".to_string(), 50.0, 1.0);
        let result = processor.process(&data).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_transform_processor() {
        let mut processor = TransformProcessor::new();
        processor.add_field("test_field".to_string(), 42.0);

        let data = MarketDataPoint::new_quote("BTC/USDT".to_string(), 100.0, 101.0);
        let result = processor.process(&data).await.unwrap();
        assert!(result.is_some());

        let transformed = result.unwrap();
        assert_eq!(transformed.data.get("test_field"), Some(&42.0));
        assert_eq!(transformed.data.get("spread"), Some(&1.0));
        assert_eq!(transformed.data.get("mid_price"), Some(&100.5));
    }
}