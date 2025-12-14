//! Momentum indicators

/// Relative Strength Index (RSI)
pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || data.len() < period + 1 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = Vec::with_capacity(data.len());

    // First period values are NaN (need one extra for first difference)
    for _ in 0..period {
        result.push(f64::NAN);
    }

    // Calculate price changes
    let mut gains = Vec::new();
    let mut losses = Vec::new();

    for i in 1..data.len() {
        let change = data[i] - data[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    // Calculate initial average gain and loss
    let mut avg_gain = gains.iter().take(period).sum::<f64>() / period as f64;
    let mut avg_loss = losses.iter().take(period).sum::<f64>() / period as f64;

    // Calculate first RSI
    if avg_loss == 0.0 {
        result.push(100.0);
    } else {
        let rs = avg_gain / avg_loss;
        result.push(100.0 - (100.0 / (1.0 + rs)));
    }

    // Calculate RSI for remaining values using smoothing
    for i in period..gains.len() {
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

        if avg_loss == 0.0 {
            result.push(100.0);
        } else {
            let rs = avg_gain / avg_loss;
            result.push(100.0 - (100.0 / (1.0 + rs)));
        }
    }

    result
}

/// Stochastic Oscillator
pub fn stochastic(data: &[f64], k_period: usize, d_period: usize) -> StochasticResult {
    if k_period == 0 || d_period == 0 || data.len() < k_period {
        return StochasticResult {
            k: vec![f64::NAN; data.len()],
            d: vec![f64::NAN; data.len()],
        };
    }

    let mut k_values = Vec::with_capacity(data.len());

    // First k_period-1 values are NaN
    for _ in 0..k_period - 1 {
        k_values.push(f64::NAN);
    }

    // Calculate %K
    for i in k_period - 1..data.len() {
        let window = &data[i - k_period + 1..=i];
        let highest = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if highest == lowest {
            k_values.push(50.0); // Avoid division by zero
        } else {
            let k = ((data[i] - lowest) / (highest - lowest)) * 100.0;
            k_values.push(k);
        }
    }

    // Calculate %D as SMA of %K
    let d_values = super::basic::sma(&k_values, d_period);

    StochasticResult {
        k: k_values,
        d: d_values,
    }
}

pub struct StochasticResult {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

/// Rate of Change (ROC)
pub fn roc(data: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || data.len() <= period {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; period];

    for i in period..data.len() {
        if data[i - period] == 0.0 {
            result.push(f64::NAN);
        } else {
            let roc = ((data[i] - data[i - period]) / data[i - period]) * 100.0;
            result.push(roc);
        }
    }

    result
}

/// Momentum
pub fn momentum(data: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || data.len() <= period {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; period];

    for i in period..data.len() {
        result.push(data[i] - data[i - period]);
    }

    result
}

/// Williams %R
pub fn williams_r(data: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || data.len() < period {
        return vec![f64::NAN; data.len()];
    }

    let mut result = Vec::with_capacity(data.len());

    // First period-1 values are NaN
    for _ in 0..period - 1 {
        result.push(f64::NAN);
    }

    // Calculate Williams %R
    for i in period - 1..data.len() {
        let window = &data[i - period + 1..=i];
        let highest = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if highest == lowest {
            result.push(-50.0);
        } else {
            let wr = ((highest - data[i]) / (highest - lowest)) * -100.0;
            result.push(wr);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi() {
        let data = vec![44.0, 44.5, 45.0, 44.75, 45.25, 45.5, 45.75, 46.0, 45.5, 45.0];
        let result = rsi(&data, 5);
        assert!(result.len() == data.len());
        assert!(result[5].is_finite());
        assert!((0.0..=100.0).contains(&result[5]));
    }

    #[test]
    fn test_stochastic() {
        let data = vec![44.0, 44.5, 45.0, 44.75, 45.25, 45.5, 45.75, 46.0];
        let result = stochastic(&data, 3, 3);
        assert_eq!(result.k.len(), data.len());
        assert_eq!(result.d.len(), data.len());
    }
}