//! Volatility indicators

/// Bollinger Bands result
pub struct BollingerBandsResult {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
}

/// Bollinger Bands
pub fn bollinger_bands(data: &[f64], period: usize, std_dev: f64) -> BollingerBandsResult {
    if period == 0 || std_dev <= 0.0 || data.len() < period {
        let nan_len = data.len();
        return BollingerBandsResult {
            upper: vec![f64::NAN; nan_len],
            middle: vec![f64::NAN; nan_len],
            lower: vec![f64::NAN; nan_len],
        };
    }

    // Calculate middle band (SMA)
    let middle = super::basic::sma(data, period);

    // Calculate upper and lower bands
    let mut upper = Vec::with_capacity(data.len());
    let mut lower = Vec::with_capacity(data.len());

    // First period-1 values are NaN
    for _ in 0..period - 1 {
        upper.push(f64::NAN);
        lower.push(f64::NAN);
    }

    // Calculate bands for each window
    for i in period - 1..data.len() {
        let window = &data[i - period + 1..=i];
        let mean = window.iter().sum::<f64>() / period as f64;

        // Calculate standard deviation
        let variance = window
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / period as f64;
        let std = variance.sqrt();

        upper.push(mean + std_dev * std);
        lower.push(mean - std_dev * std);
    }

    BollingerBandsResult {
        upper,
        middle,
        lower,
    }
}

/// Average True Range (ATR)
pub fn atr(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Vec<f64> {
    if high.len() != low.len() || high.len() != close.len() || period == 0 {
        return vec![f64::NAN; high.len()];
    }

    if high.len() < period + 1 {
        return vec![f64::NAN; high.len()];
    }

    let mut true_ranges = Vec::with_capacity(high.len());

    // Calculate first TR
    let tr1 = high[0] - low[0];
    let tr2 = (high[0] - close[1]).abs();
    let tr3 = (low[0] - close[1]).abs();
    true_ranges.push(tr1.max(tr2).max(tr3));

    // Calculate subsequent TRs
    for i in 1..high.len() {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();
        true_ranges.push(tr1.max(tr2).max(tr3));
    }

    // Calculate ATR using EMA-like smoothing
    let mut atr_values = vec![f64::NAN; period];
    let initial_atr = true_ranges.iter().take(period).sum::<f64>() / period as f64;
    atr_values.push(initial_atr);

    let smoothing_factor = 1.0 / period as f64;

    for i in period + 1..true_ranges.len() {
        let prev_atr = atr_values[i - 1];
        let atr = (prev_atr * (period - 1) as f64 + true_ranges[i]) / period as f64;
        atr_values.push(atr);
    }

    atr_values
}

/// Keltner Channels
pub fn keltner_channels(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    multiplier: f64,
) -> KeltnerChannelsResult {
    if high.len() != low.len() || high.len() != close.len() || period == 0 {
        let nan_len = high.len();
        return KeltnerChannelsResult {
            upper: vec![f64::NAN; nan_len],
            middle: vec![f64::NAN; nan_len],
            lower: vec![f64::NAN; nan_len],
        };
    }

    // Calculate middle line (EMA)
    let middle = super::basic::ema(close, period);

    // Calculate ATR for band width
    let atr_values = atr(high, low, close, period);

    // Calculate upper and lower bands
    let upper: Vec<f64> = middle
        .iter()
        .zip(atr_values.iter())
        .map(|(m, a)| {
            if m.is_finite() && a.is_finite() {
                m + multiplier * a
            } else {
                f64::NAN
            }
        })
        .collect();

    let lower: Vec<f64> = middle
        .iter()
        .zip(atr_values.iter())
        .map(|(m, a)| {
            if m.is_finite() && a.is_finite() {
                m - multiplier * a
            } else {
                f64::NAN
            }
        })
        .collect();

    KeltnerChannelsResult {
        upper,
        middle,
        lower,
    }
}

pub struct KeltnerChannelsResult {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
}

/// Historical Volatility
pub fn historical_volatility(returns: &[f64], period: usize, annualization_factor: f64) -> Vec<f64> {
    if period == 0 || returns.len() < period {
        return vec![f64::NAN; returns.len()];
    }

    let mut result = vec![f64::NAN; period - 1];

    // Calculate rolling volatility
    for i in period - 1..returns.len() {
        let window = &returns[i - period + 1..=i];
        let mean = window.iter().sum::<f64>() / period as f64;

        let variance = window
            .iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / (period - 1) as f64;

        let volatility = variance.sqrt() * annualization_factor.sqrt();
        result.push(volatility);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bollinger_bands() {
        let data = vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0];
        let result = bollinger_bands(&data, 5, 2.0);

        assert_eq!(result.upper.len(), data.len());
        assert_eq!(result.middle.len(), data.len());
        assert_eq!(result.lower.len(), data.len());

        // Upper band should be > middle > lower band
        for i in 4..data.len() {
            assert!(result.upper[i] > result.middle[i]);
            assert!(result.middle[i] > result.lower[i]);
        }
    }

    #[test]
    fn test_atr() {
        let high = vec![10.0, 11.0, 12.0, 11.5, 12.5];
        let low = vec![9.0, 10.0, 11.0, 10.5, 11.5];
        let close = vec![9.5, 10.5, 11.5, 11.0, 12.0];

        let result = atr(&high, &low, &close, 2);
        assert_eq!(result.len(), high.len());
    }
}