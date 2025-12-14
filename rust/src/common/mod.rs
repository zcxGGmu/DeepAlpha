//! Common utilities and types used across the crate

use thiserror::Error;

/// Common error types
#[derive(Error, Debug)]
pub enum DeepAlphaError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Calculation error: {0}")]
    CalculationError(String),

    #[error("Insufficient data: need at least {required} elements, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    #[error("Network error: {0}")]
    NetworkError(#[from] tokio_tungstenite::tungstenite::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, DeepAlphaError>;

/// Price data point
#[derive(Debug, Clone, Copy)]
pub struct Price {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: Option<f64>,
}

impl Price {
    pub fn new(open: f64, high: f64, low: f64, close: f64) -> Self {
        Self {
            open,
            high,
            low,
            close,
            volume: None,
        }
    }

    pub fn with_volume(mut self, volume: f64) -> Self {
        self.volume = Some(volume);
        self
    }
}

/// OHLC data trait
pub trait OHLC {
    fn open(&self) -> f64;
    fn high(&self) -> f64;
    fn low(&self) -> f64;
    fn close(&self) -> f64;
    fn volume(&self) -> Option<f64>;
}

impl OHLC for Price {
    fn open(&self) -> f64 {
        self.open
    }

    fn high(&self) -> f64 {
        self.high
    }

    fn low(&self) -> f64 {
        self.low
    }

    fn close(&self) -> f64 {
        self.close
    }

    fn volume(&self) -> Option<f64> {
        self.volume
    }
}

/// Moving average types
#[derive(Debug, Clone, Copy)]
pub enum MAType {
    SMA,
    EMA,
    WMA,
}

/// Validation utilities
pub mod validation {
    use super::Result;

    pub fn validate_period(period: usize) -> Result<()> {
        if period == 0 {
            Err(super::DeepAlphaError::InvalidInput(
                "Period must be greater than 0".to_string(),
            ))
        } else if period > 1000 {
            Err(super::DeepAlphaError::InvalidInput(
                "Period must be less than or equal to 1000".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    pub fn validate_data_length(data_len: usize, min_required: usize) -> Result<()> {
        if data_len < min_required {
            Err(super::DeepAlphaError::InsufficientData {
                required: min_required,
                actual: data_len,
            })
        } else {
            Ok(())
        }
    }
}