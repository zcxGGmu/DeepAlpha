//! Basic technical indicators

use rayon::prelude::*;

/// Simple Moving Average (SMA)
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || data.len() < period {
        return vec![];
    }

    let mut result = Vec::with_capacity(data.len() - period + 1);

    // Initial sum
    let mut sum: f64 = data.iter().take(period).sum();
    result.push(sum / period as f64);

    // Sliding window
    for i in period..data.len() {
        sum += data[i] - data[i - period];
        result.push(sum / period as f64);
    }

    // Pad with NaN at the beginning to match input length
    let mut full_result = vec![f64::NAN; period - 1];
    full_result.extend(result);
    full_result
}

/// Exponential Moving Average (EMA)
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || data.is_empty() {
        return vec![f64::NAN; data.len()];
    }

    let mut result = Vec::with_capacity(data.len());
    let multiplier = 2.0 / (period as f64 + 1.0);

    // Start with SMA as the first EMA value
    let initial_sum: f64 = data.iter().take(period.min(data.len())).sum();
    let mut ema_val = initial_sum / period.min(data.len()) as f64;

    // First period-1 values are NaN
    for _ in 0..period - 1 {
        result.push(f64::NAN);
    }

    // First EMA value
    if !data.is_empty() {
        result.push(ema_val);

        // Calculate EMA for remaining values
        for &value in &data[period..] {
            ema_val = (value - ema_val) * multiplier + ema_val;
            result.push(ema_val);
        }
    }

    result
}

/// Weighted Moving Average (WMA)
pub fn wma(data: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || data.len() < period {
        return vec![f64::NAN; data.len()];
    }

    let mut result = Vec::with_capacity(data.len());

    // First period-1 values are NaN
    for _ in 0..period - 1 {
        result.push(f64::NAN);
    }

    // Calculate WMA for each window
    for i in period - 1..data.len() {
        let window = &data[i - period + 1..=i];
        let mut sum = 0.0;
        let mut weight_sum = 0.0;

        for (j, &value) in window.iter().enumerate() {
            let weight = (j + 1) as f64;
            sum += value * weight;
            weight_sum += weight;
        }

        result.push(sum / weight_sum);
    }

    result
}

/// Simple implementation for parallel processing
pub fn sma_parallel(data: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || data.len() < period {
        return vec![];
    }

    let len = data.len();
    let result: Vec<f64> = (period - 1..len)
        .into_par_iter()
        .map(|i| {
            let window = &data[i - period + 1..=i];
            window.iter().sum::<f64>() / period as f64
        })
        .collect();

    // Pad with NaN at the beginning
    let mut full_result = vec![f64::NAN; period - 1];
    full_result.extend(result);
    full_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);
        assert_eq!(result, vec![f64::NAN, f64::NAN, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&data, 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!result[2].is_finite();
    }

    #[test]
    fn test_wma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = wma(&data, 3);
        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!result[2].is_finite();
    }
}