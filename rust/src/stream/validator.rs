//! Data validation for market data streams

use crate::common::Result;
use crate::stream::MarketDataPoint;
use std::collections::HashMap;

/// Data validator for market data
pub struct DataValidator {
    symbol_whitelist: Option<Vec<String>>,
    symbol_blacklist: Vec<String>,
    min_price: f64,
    max_price: f64,
    max_spread_pct: f64,
    max_timestamp_drift_ms: u64,
    required_fields: Vec<String>,
}

impl DataValidator {
    /// Create a new data validator with default settings
    pub fn new() -> Self {
        Self {
            symbol_whitelist: None,
            symbol_blacklist: Vec::new(),
            min_price: 0.001, // Minimum price
            max_price: 10_000_000.0, // Maximum price
            max_spread_pct: 0.5, // 50% max spread
            max_timestamp_drift_ms: 5000, // 5 seconds max timestamp drift
            required_fields: Vec::new(),
        }
    }

    /// Validate a market data point
    pub fn validate(&self, data: &MarketDataPoint) -> Result<()> {
        // Validate symbol
        self.validate_symbol(&data.symbol)?;

        // Validate timestamp
        self.validate_timestamp(data.timestamp)?;

        // Validate price data
        self.validate_prices(data)?;

        // Validate required fields
        self.validate_required_fields(data)?;

        Ok(())
    }

    /// Validate symbol against whitelist and blacklist
    fn validate_symbol(&self, symbol: &str) -> Result<()> {
        // Check blacklist first
        if self.symbol_blacklist.iter().any(|s| s == symbol) {
            return Err(crate::common::DeepAlphaError::InvalidInput(
                format!("Symbol {} is blacklisted", symbol)
            ));
        }

        // Check whitelist if configured
        if let Some(whitelist) = &self.symbol_whitelist {
            if !whitelist.iter().any(|s| s == symbol) {
                return Err(crate::common::DeepAlphaError::InvalidInput(
                    format!("Symbol {} is not in whitelist", symbol)
                ));
            }
        }

        // Check symbol format
        if !is_valid_symbol_format(symbol) {
            return Err(crate::common::DeepAlphaError::InvalidInput(
                format!("Invalid symbol format: {}", symbol)
            ));
        }

        Ok(())
    }

    /// Validate timestamp
    fn validate_timestamp(&self, timestamp: i64) -> Result<()> {
        let current_time = chrono::Utc::now().timestamp_millis();
        let drift = (current_time - timestamp).abs();

        if drift > self.max_timestamp_drift_ms as i64 {
            return Err(crate::common::DeepAlphaError::InvalidInput(
                format!("Timestamp drift too large: {}ms", drift)
            ));
        }

        // Check for future timestamps
        if timestamp > current_time {
            return Err(crate::common::DeepAlphaError::InvalidInput(
                "Timestamp is in the future".to_string()
            ));
        }

        // Check for very old timestamps
        if timestamp < 946684800000 { // 2000-01-01 in milliseconds
            return Err(crate::common::DeepAlphaError::InvalidInput(
                "Timestamp is too old (before year 2000)".to_string()
            ));
        }

        Ok(())
    }

    /// Validate price data
    fn validate_prices(&self, data: &MarketDataPoint) -> Result<()> {
        // Validate trade price
        if let Some(price) = data.price {
            self.validate_price_value(price)?;
        }

        // Validate bid/ask
        if let (Some(bid), Some(ask)) = (data.bid, data.ask) {
            self.validate_price_value(bid)?;
            self.validate_price_value(ask)?;

            // Bid must be less than ask
            if bid >= ask {
                return Err(crate::common::DeepAlphaError::InvalidInput(
                    "Bid must be less than ask".to_string()
                ));
            }

            // Check spread percentage
            let spread_pct = (ask - bid) / ask;
            if spread_pct > self.max_spread_pct {
                return Err(crate::common::DeepAlphaError::InvalidInput(
                    format!("Spread too large: {:.2}%", spread_pct * 100.0)
                ));
            }
        }

        // Validate volume
        if let Some(volume) = data.volume {
            self.validate_volume_value(volume)?;
        }

        Ok(())
    }

    /// Validate a single price value
    fn validate_price_value(&self, price: f64) -> Result<()> {
        // Check for finite numbers
        if !price.is_finite() {
            return Err(crate::common::DeepAlphaError::InvalidInput(
                "Price must be a finite number".to_string()
            ));
        }

        // Check for positive values
        if price <= 0.0 {
            return Err(crate::common::DeepAlphaError::InvalidInput(
                "Price must be positive".to_string()
            ));
        }

        // Check minimum price
        if price < self.min_price {
            return Err(crate::common::DeepAlphaError::InvalidInput(
                format!("Price {} below minimum {}", price, self.min_price)
            ));
        }

        // Check maximum price
        if price > self.max_price {
            return Err(crate::common::DeepAlphaError::InvalidInput(
                format!("Price {} above maximum {}", price, self.max_price)
            ));
        }

        Ok(())
    }

    /// Validate volume value
    fn validate_volume_value(&self, volume: f64) -> Result<()> {
        // Check for finite numbers
        if !volume.is_finite() {
            return Err(crate::common::DeepAlphaError::InvalidInput(
                "Volume must be a finite number".to_string()
            ));
        }

        // Check for non-negative values
        if volume < 0.0 {
            return Err(crate::common::DeepAlphaError::InvalidInput(
                "Volume cannot be negative".to_string()
            ));
        }

        // Reasonable maximum volume (10 billion)
        if volume > 10_000_000_000.0 {
            return Err(crate::common::DeepAlphaError::InvalidInput(
                "Volume too large".to_string()
            ));
        }

        Ok(())
    }

    /// Validate required fields
    fn validate_required_fields(&self, data: &MarketDataPoint) -> Result<()> {
        for field in &self.required_fields {
            match field.as_str() {
                "price" => {
                    if data.price.is_none() {
                        return Err(crate::common::DeepAlphaError::InvalidInput(
                            "Price field is required".to_string()
                        ));
                    }
                }
                "volume" => {
                    if data.volume.is_none() {
                        return Err(crate::common::DeepAlphaError::InvalidInput(
                            "Volume field is required".to_string()
                        ));
                    }
                }
                "bid" => {
                    if data.bid.is_none() {
                        return Err(crate::common::DeepAlphaError::InvalidInput(
                            "Bid field is required".to_string()
                        ));
                    }
                }
                "ask" => {
                    if data.ask.is_none() {
                        return Err(crate::common::DeepAlphaError::InvalidInput(
                            "Ask field is required".to_string()
                        ));
                    }
                }
                _ => {
                    if !data.data.contains_key(field) {
                        return Err(crate::common::DeepAlphaError::InvalidInput(
                            format!("Required field '{}' is missing", field)
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Add symbol to whitelist
    pub fn add_to_whitelist(&mut self, symbol: String) {
        if let Some(whitelist) = &mut self.symbol_whitelist {
            whitelist.push(symbol);
        } else {
            self.symbol_whitelist = Some(vec![symbol]);
        }
    }

    /// Add symbol to blacklist
    pub fn add_to_blacklist(&mut self, symbol: String) {
        self.symbol_blacklist.push(symbol);
    }

    /// Set required fields
    pub fn set_required_fields(&mut self, fields: Vec<String>) {
        self.required_fields = fields;
    }

    /// Set max spread percentage
    pub fn set_max_spread_pct(&mut self, pct: f64) {
        self.max_spread_pct = pct;
    }
}

/// Check if symbol format is valid
fn is_valid_symbol_format(symbol: &str) -> bool {
    // Basic symbol validation - adjust as needed
    let parts: Vec<&str> = symbol.split('/').collect();

    // Should have exactly 2 parts (e.g., BTC/USDT)
    if parts.len() != 2 {
        return false;
    }

    // Each part should be uppercase letters only
    for part in parts {
        if part.is_empty() || !part.chars().all(|c| c.is_ascii_uppercase()) {
            return false;
        }
    }

    true
}

/// Market data sanitizer
pub struct DataSanitizer {
    precision_digits: u32,
    normalize_symbols: bool,
}

impl DataSanitizer {
    pub fn new() -> Self {
        Self {
            precision_digits: 8,
            normalize_symbols: true,
        }
    }

    /// Sanitize a market data point
    pub fn sanitize(&self, data: &mut MarketDataPoint) {
        // Normalize symbol if enabled
        if self.normalize_symbols {
            data.symbol = data.symbol.to_uppercase();
        }

        // Round price values
        if let Some(ref mut price) = data.price {
            *price = round_to_precision(*price, self.precision_digits);
        }

        if let Some(ref mut bid) = data.bid {
            *bid = round_to_precision(*bid, self.precision_digits);
        }

        if let Some(ref mut ask) = data.ask {
            *ask = round_to_precision(*ask, self.precision_digits);
        }

        if let Some(ref mut volume) = data.volume {
            *volume = round_to_precision(*volume, self.precision_digits);
        }

        // Sanitize custom fields
        for (_, value) in data.data.iter_mut() {
            if value.is_finite() {
                *value = round_to_precision(*value, self.precision_digits);
            }
        }
    }

    /// Set precision for price rounding
    pub fn set_precision(&mut self, digits: u32) {
        self.precision_digits = digits;
    }
}

/// Round a float to specified precision
fn round_to_precision(value: f64, precision: u32) -> f64 {
    let factor = 10_f64.powi(precision as i32);
    (value * factor).round() / factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{MarketDataPoint, MarketDataType};

    #[test]
    fn test_symbol_validation() {
        assert!(is_valid_symbol_format("BTC/USDT"));
        assert!(is_valid_symbol_format("ETH/USD"));
        assert!(!is_valid_symbol_format("btc/usdt")); // Lowercase
        assert!(!is_valid_symbol_format("BTC/USDT/USDC")); // Too many parts
        assert!(!is_valid_symbol_format("BTCUSDT")); // No separator
    }

    #[test]
    fn test_data_validator() {
        let mut validator = DataValidator::new();

        // Valid data
        let data = MarketDataPoint::new_trade("BTC/USDT".to_string(), 50000.0, 1.0);
        assert!(validator.validate(&data).is_ok());

        // Invalid symbol
        let data = MarketDataPoint::new_trade("INVALID".to_string(), 50000.0, 1.0);
        assert!(validator.validate(&data).is_err());

        // Invalid price (negative)
        let data = MarketDataPoint::new_trade("BTC/USDT".to_string(), -100.0, 1.0);
        assert!(validator.validate(&data).is_err());

        // Invalid bid/ask
        let mut data = MarketDataPoint::new_quote("BTC/USDT".to_string(), 50001.0, 50000.0);
        assert!(validator.validate(&data).is_err()); // Bid > Ask
    }

    #[test]
    fn test_data_sanitizer() {
        let mut sanitizer = DataSanitizer::new();
        sanitizer.set_precision(2);

        let mut data = MarketDataPoint::new_trade("btc/usdt".to_string(), 50000.12345, 1.67890);
        sanitizer.sanitize(&mut data);

        assert_eq!(data.symbol, "BTC/USDT");
        assert_eq!(data.price, Some(50000.12));
        assert_eq!(data.volume, Some(1.68));
    }

    #[test]
    fn test_round_to_precision() {
        assert_eq!(round_to_precision(123.456, 2), 123.46);
        assert_eq!(round_to_precision(123.456, 0), 123.0);
        assert_eq!(round_to_precision(0.00123, 4), 0.0012);
        assert_eq!(round_to_precision(f64::INFINITY, 2), f64::INFINITY);
        assert_eq!(round_to_precision(f64::NAN, 2), f64::NAN);
    }
}