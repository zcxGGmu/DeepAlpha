//! Data aggregation utilities

use crate::stream::MarketDataPoint;
use std::collections::HashMap;

/// Time-based data aggregator
pub struct TimeAggregator {
    interval_ms: u64,
    data_by_symbol: HashMap<String, Vec<MarketDataPoint>>,
    last_aggregate_time: HashMap<String, u64>,
}

impl TimeAggregator {
    /// Create a new time-based aggregator
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval_ms,
            data_by_symbol: HashMap::new(),
            last_aggregate_time: HashMap::new(),
        }
    }

    /// Add data point to aggregator
    pub fn add_data(&mut self, data: MarketDataPoint) -> Option<AggregatedData> {
        let symbol = data.symbol.clone();
        let data_list = self.data_by_symbol.entry(symbol.clone()).or_insert_with(Vec::new);
        data_list.push(data.clone());

        // Check if it's time to aggregate
        let current_time = chrono::Utc::now().timestamp_millis() as u64;
        let last_time = self.last_aggregate_time.get(&symbol).copied().unwrap_or(0);

        if current_time - last_time >= self.interval_ms {
            if let Some(aggregated) = self.aggregate_symbol(&symbol) {
                self.last_aggregate_time.insert(symbol, current_time);
                return Some(aggregated);
            }
        }

        None
    }

    /// Aggregate data for a specific symbol
    fn aggregate_symbol(&mut self, symbol: &str) -> Option<AggregatedData> {
        let data_list = self.data_by_symbol.get(symbol)?;

        if data_list.is_empty() {
            return None;
        }

        let mut aggregated = AggregatedData {
            symbol: symbol.to_string(),
            interval_ms: self.interval_ms,
            timestamp: chrono::Utc::now().timestamp_millis(),
            open: None,
            high: f64::MIN,
            low: f64::MAX,
            close: None,
            volume: 0.0,
            trade_count: 0,
            vwap: 0.0,
        };

        let mut total_value = 0.0;

        // Process all data points
        for data in data_list {
            if let Some(price) = data.price {
                aggregated.high = aggregated.high.max(price);
                aggregated.low = aggregated.low.min(price);

                // Set open to first price
                if aggregated.open.is_none() {
                    aggregated.open = Some(price);
                }

                // Always update close to last price
                aggregated.close = Some(price);

                if let Some(volume) = data.volume {
                    total_value += price * volume;
                    aggregated.volume += volume;
                }

                aggregated.trade_count += 1;
            }
        }

        // Calculate VWAP
        if aggregated.volume > 0.0 {
            aggregated.vwap = total_value / aggregated.volume;
        }

        // Clear data for this symbol
        self.data_by_symbol.remove(symbol);

        Some(aggregated)
    }

    /// Force aggregation for all symbols
    pub fn aggregate_all(&mut self) -> Vec<AggregatedData> {
        let symbols: Vec<String> = self.data_by_symbol.keys().cloned().collect();
        let mut results = Vec::new();

        for symbol in symbols {
            if let Some(aggregated) = self.aggregate_symbol(&symbol) {
                results.push(aggregated);
            }
        }

        results
    }
}

/// Aggregated market data
#[derive(Debug, Clone)]
pub struct AggregatedData {
    pub symbol: String,
    pub interval_ms: u64,
    pub timestamp: i64,
    pub open: Option<f64>,
    pub high: f64,
    pub low: f64,
    pub close: Option<f64>,
    pub volume: f64,
    pub trade_count: u32,
    pub vwap: f64, // Volume Weighted Average Price
}

impl AggregatedData {
    /// Convert to OHLC format
    pub fn to_ohlc(&self) -> Option<(f64, f64, f64, f64)> {
        if let (Some(open), Some(close)) = (self.open, self.close) {
            Some((open, self.high, self.low, close))
        } else {
            None
        }
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Rolling statistics calculator
pub struct RollingStats {
    window_size: usize,
    values: Vec<f64>,
    sum: f64,
    sum_squares: f64,
    index: usize,
    count: usize,
}

impl RollingStats {
    /// Create a new rolling statistics calculator
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: vec![0.0; window_size],
            sum: 0.0,
            sum_squares: 0.0,
            index: 0,
            count: 0,
        }
    }

    /// Add a new value
    pub fn add(&mut self, value: f64) {
        // Subtract old value from sum if window is full
        if self.count >= self.window_size {
            let old_value = self.values[self.index];
            self.sum -= old_value;
            self.sum_squares -= old_value * old_value;
        }

        // Add new value
        self.values[self.index] = value;
        self.sum += value;
        self.sum_squares += value * value;

        self.index = (self.index + 1) % self.window_size;
        if self.count < self.window_size {
            self.count += 1;
        }
    }

    /// Get current mean
    pub fn mean(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.sum / self.count as f64)
        } else {
            None
        }
    }

    /// Get current variance
    pub fn variance(&self) -> Option<f64> {
        if self.count > 1 {
            let mean = self.mean()?;
            Some((self.sum_squares - 2.0 * mean * self.sum + self.count as f64 * mean * mean) / self.count as f64)
        } else {
            None
        }
    }

    /// Get current standard deviation
    pub fn std_dev(&self) -> Option<f64> {
        self.variance()?.sqrt()
    }

    /// Get minimum value in window
    pub fn min(&self) -> Option<f64> {
        if self.count > 0 {
            let end = if self.count < self.window_size {
                self.count
            } else {
                self.window_size
            };
            self.values[..end].iter().fold(f64::INFINITY, |a, &b| a.min(b))
        } else {
            None
        }
    }

    /// Get maximum value in window
    pub fn max(&self) -> Option<f64> {
        if self.count > 0 {
            let end = if self.count < self.window_size {
                self.count
            } else {
                self.window_size
            };
            self.values[..end].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        } else {
            None
        }
    }

    /// Reset the statistics
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.sum_squares = 0.0;
        self.index = 0;
        self.count = 0;
        for value in &mut self.values {
            *value = 0.0;
        }
    }
}

/// Volume profile calculator
pub struct VolumeProfile {
    price_levels: HashMap<u32, f64>, // price_hash -> volume
    precision: u32,
}

impl VolumeProfile {
    /// Create a new volume profile calculator
    pub fn new(precision: u32) -> Self {
        Self {
            price_levels: HashMap::new(),
            precision,
        }
    }

    /// Add trade data
    pub fn add_trade(&mut self, price: f64, volume: f64) {
        let price_hash = self.quantize_price(price);
        let entry = self.price_levels.entry(price_hash).or_insert(0.0);
        *entry += volume;
    }

    /// Quantize price to discrete level
    fn quantize_price(&self, price: f64) -> u32 {
        (price * 10_f64.powi(self.precision as i32)).round() as u32
    }

    /// Get volume profile as sorted vector
    pub fn get_profile(&self) -> Vec<(f64, f64)> {
        let mut profile: Vec<_> = self.price_levels
            .iter()
            .map(|(&hash, &volume)| {
                let price = hash as f64 / 10_f64.powi(self.precision as i32);
                (price, volume)
            })
            .collect();

        profile.sort_by_key(|(price, _)| *price);
        profile
    }

    /// Get volume at specific price level
    pub fn get_volume_at_price(&self, price: f64) -> f64 {
        let price_hash = self.quantize_price(price);
        self.price_levels.get(&price_hash).copied().unwrap_or(0.0)
    }

    /// Get total volume
    pub fn total_volume(&self) -> f64 {
        self.price_levels.values().sum()
    }

    /// Get high volume area (HVAs) - areas with volume above threshold
    pub fn get_high_volume_areas(&self, threshold_pct: f64) -> Vec<(f64, f64, f64)> {
        let total_volume = self.total_volume();
        if total_volume == 0.0 {
            return Vec::new();
        }

        let threshold = total_volume * threshold_pct;
        let profile = self.get_profile();

        let mut areas = Vec::new();
        let mut current_area_start = None;
        let mut current_area_volume = 0.0;

        for (price, volume) in profile {
            if volume >= threshold / 10.0 { // Individual threshold
                if current_area_start.is_none() {
                    current_area_start = Some(price);
                }
                current_area_volume += volume;
            } else {
                if let Some(start) = current_area_start {
                    areas.push((start, price, current_area_volume));
                    current_area_start = None;
                    current_area_volume = 0.0;
                }
            }
        }

        // Close final area if open
        if let Some(start) = current_area_start {
            let last_price = profile.last().unwrap().0;
            areas.push((start, last_price, current_area_volume));
        }

        areas
    }

    /// Reset the volume profile
    pub fn reset(&mut self) {
        self.price_levels.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_aggregator() {
        let mut aggregator = TimeAggregator::new(1000); // 1 second interval

        // Add some data
        let data1 = MarketDataPoint::new_trade("BTC/USDT".to_string(), 50000.0, 1.0);
        let data2 = MarketDataPoint::new_trade("BTC/USDT".to_string(), 50100.0, 2.0);

        assert!(aggregator.add_data(data1).is_none());
        assert!(aggregator.add_data(data2).is_none());
    }

    #[test]
    fn test_rolling_stats() {
        let mut stats = RollingStats::new(3);

        assert_eq!(stats.mean(), None);
        assert_eq!(stats.std_dev(), None);

        stats.add(1.0);
        stats.add(2.0);
        stats.add(3.0);

        assert_eq!(stats.mean(), Some(2.0));
        assert!((stats.std_dev().unwrap() - 0.8164965809).abs() < 0.0001);

        // Test rolling window
        stats.add(4.0);
        assert_eq!(stats.mean(), Some(3.0)); // (2+3+4)/3
    }

    #[test]
    fn test_volume_profile() {
        let mut profile = VolumeProfile::new(2);

        profile.add_trade(100.123, 1.0);
        profile.add_trade(100.124, 2.0);
        profile.add_trade(100.125, 1.0);

        assert_eq!(profile.total_volume(), 4.0);
        assert_eq!(profile.get_volume_at_price(100.12), 3.0);

        let profile_data = profile.get_profile();
        assert_eq!(profile_data.len(), 3);
    }

    #[test]
    fn test_round_to_precision() {
        use crate::stream::validator::round_to_precision;

        assert_eq!(round_to_precision(123.456, 2), 123.46);
        assert_eq!(round_to_precision(123.456, 0), 123.0);
    }
}