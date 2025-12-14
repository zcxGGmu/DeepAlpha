//! Trend indicators

/// MACD result
pub struct MACDResult {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub histogram: Vec<f64>,
}

/// Moving Average Convergence Divergence (MACD)
pub fn macd(data: &[f64], fast_period: usize, slow_period: usize, signal_period: usize) -> MACDResult {
    if fast_period >= slow_period || data.len() < slow_period {
        let nan_len = data.len();
        return MACDResult {
            macd: vec![f64::NAN; nan_len],
            signal: vec![f64::NAN; nan_len],
            histogram: vec![f64::NAN; nan_len],
        };
    }

    // Calculate fast and slow EMAs
    let fast_ema = super::basic::ema(data, fast_period);
    let slow_ema = super::basic::ema(data, slow_period);

    // Calculate MACD line
    let macd: Vec<f64> = fast_ema
        .iter()
        .zip(slow_ema.iter())
        .map(|(f, s)| {
            if f.is_finite() && s.is_finite() {
                f - s
            } else {
                f64::NAN
            }
        })
        .collect();

    // Calculate signal line (EMA of MACD)
    let signal = super::basic::ema(&macd, signal_period);

    // Calculate histogram
    let histogram: Vec<f64> = macd
        .iter()
        .zip(signal.iter())
        .map(|(m, s)| {
            if m.is_finite() && s.is_finite() {
                m - s
            } else {
                f64::NAN
            }
        })
        .collect();

    MACDResult {
        macd,
        signal,
        histogram,
    }
}

/// Directional Movement Index (DMI)
pub fn dmi(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> DMIResult {
    if high.len() != low.len() || high.len() != close.len() || period == 0 {
        let nan_len = high.len();
        return DMIResult {
            pdi: vec![f64::NAN; nan_len],
            ndi: vec![f64::NAN; nan_len],
            adx: vec![f64::NAN; nan_len],
        };
    }

    if high.len() < period + 1 {
        let nan_len = high.len();
        return DMIResult {
            pdi: vec![f64::NAN; nan_len],
            ndi: vec![f64::NAN; nan_len],
            adx: vec![f64::NAN; nan_len],
        };
    }

    let len = high.len();
    let mut up_moves = Vec::with_capacity(len - 1);
    let mut down_moves = Vec::with_capacity(len - 1);

    // Calculate up and down moves
    for i in 1..len {
        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];

        if up_move > down_move && up_move > 0.0 {
            up_moves.push(up_move);
            down_moves.push(0.0);
        } else if down_move > up_move && down_move > 0.0 {
            up_moves.push(0.0);
            down_moves.push(down_move);
        } else {
            up_moves.push(0.0);
            down_moves.push(0.0);
        }
    }

    // Calculate true ranges
    let mut true_ranges = Vec::with_capacity(len - 1);
    for i in 1..len {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();
        true_ranges.push(tr1.max(tr2).max(tr3));
    }

    // Smooth the values
    let mut smoothed_plus_dm = vec![0.0; len];
    let mut smoothed_minus_dm = vec![0.0; len];
    let mut smoothed_tr = vec![0.0; len];

    // Initialize with sum of first period
    smoothed_plus_dm[period] = up_moves.iter().take(period).sum();
    smoothed_minus_dm[period] = down_moves.iter().take(period).sum();
    smoothed_tr[period] = true_ranges.iter().take(period).sum();

    // Apply smoothing
    for i in period + 1..len {
        smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / period as f64)
            + up_moves[i - 1];
        smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - (smoothed_minus_dm[i - 1] / period as f64)
            + down_moves[i - 1];
        smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / period as f64) + true_ranges[i - 1];
    }

    // Calculate DI+ and DI-
    let mut pdi = vec![f64::NAN; period];
    let mut ndi = vec![f64::NAN; period];

    for i in period..len {
        if smoothed_tr[i] > 0.0 {
            pdi.push(100.0 * smoothed_plus_dm[i] / smoothed_tr[i]);
            ndi.push(100.0 * smoothed_minus_dm[i] / smoothed_tr[i]);
        } else {
            pdi.push(0.0);
            ndi.push(0.0);
        }
    }

    // Calculate ADX
    let mut dx = Vec::with_capacity(len);
    for i in period..len {
        let di_sum = pdi[i] + ndi[i];
        if di_sum > 0.0 {
            dx.push(100.0 * (pdi[i] - ndi[i]).abs() / di_sum);
        } else {
            dx.push(0.0);
        }
    }

    let mut adx = vec![f64::NAN; 2 * period - 1];
    if !dx.is_empty() {
        adx.push(dx.iter().take(period).sum::<f64>() / period as f64);

        for i in period + 1..dx.len() {
            let prev_adx = adx[i - 1];
            adx.push((prev_adx * (period - 1) as f64 + dx[i - 1]) / period as f64);
        }
    }

    // Ensure all vectors have the same length
    while pdi.len() < len {
        pdi.push(f64::NAN);
    }
    while ndi.len() < len {
        ndi.push(f64::NAN);
    }
    while adx.len() < len {
        adx.push(f64::NAN);
    }

    DMIResult { pdi, ndi, adx }
}

pub struct DMIResult {
    pub pdi: Vec<f64>,
    pub ndi: Vec<f64>,
    pub adx: Vec<f64>,
}

/// Parabolic SAR
pub fn parabolic_sar(
    high: &[f64],
    low: &[f64],
    initial_acceleration: f64,
    acceleration: f64,
    maximum_acceleration: f64,
) -> Vec<f64> {
    if high.len() != low.len() || high.len() < 2 {
        return vec![f64::NAN; high.len()];
    }

    let len = high.len();
    let mut psar = Vec::with_capacity(len);
    let mut is_uptrend = true;
    let mut ep = high[0]; // Extreme point
    let mut af = initial_acceleration; // Acceleration factor

    // Initialize PSAR
    psar.push(low[0]);

    for i in 1..len {
        let prev_psar = psar[i - 1];

        if is_uptrend {
            // Uptrend
            let mut new_psar = prev_psar + af * (ep - prev_psar);

            // Ensure PSAR is below current low
            if new_psar > low[i] {
                new_psar = low[i].min(high[i - 1]).min(low[i - 1]);
            }

            psar.push(new_psar);

            // Update extreme point and acceleration factor
            if high[i] > ep {
                ep = high[i];
                af = (af + acceleration).min(maximum_acceleration);
            }

            // Check for trend reversal
            if low[i] < new_psar {
                is_uptrend = false;
                psar[i] = ep;
                ep = low[i];
                af = initial_acceleration;
            }
        } else {
            // Downtrend
            let mut new_psar = prev_psar + af * (ep - prev_psar);

            // Ensure PSAR is above current high
            if new_psar < high[i] {
                new_psar = high[i].max(high[i - 1]).max(low[i - 1]);
            }

            psar.push(new_psar);

            // Update extreme point and acceleration factor
            if low[i] < ep {
                ep = low[i];
                af = (af + acceleration).min(maximum_acceleration);
            }

            // Check for trend reversal
            if high[i] > new_psar {
                is_uptrend = true;
                psar[i] = ep;
                ep = high[i];
                af = initial_acceleration;
            }
        }
    }

    psar
}

/// Ichimoku Cloud components
pub fn ichimoku(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    tenkan_period: usize,
    kijun_period: usize,
    senkou_b_period: usize,
) -> IchimokuResult {
    if high.len() != low.len() || high.len() != close.len() {
        let nan_len = high.len();
        return IchimokuResult {
            tenkan_sen: vec![f64::NAN; nan_len],
            kijun_sen: vec![f64::NAN; nan_len],
            senkou_span_a: vec![f64::NAN; nan_len],
            senkou_span_b: vec![f64::NAN; nan_len],
            chikou_span: vec![f64::NAN; nan_len],
        };
    }

    let len = high.len();
    let mut tenkan_sen = Vec::with_capacity(len);
    let mut kijun_sen = Vec::with_capacity(len);

    // Calculate Tenkan-sen (Conversion Line)
    for i in 0..len {
        if i < tenkan_period - 1 {
            tenkan_sen.push(f64::NAN);
        } else {
            let window_high = high[i - tenkan_period + 1..=i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let window_low = low[i - tenkan_period + 1..=i]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            tenkan_sen.push((window_high + window_low) / 2.0);
        }
    }

    // Calculate Kijun-sen (Base Line)
    for i in 0..len {
        if i < kijun_period - 1 {
            kijun_sen.push(f64::NAN);
        } else {
            let window_high = high[i - kijun_period + 1..=i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let window_low = low[i - kijun_period + 1..=i]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            kijun_sen.push((window_high + window_low) / 2.0);
        }
    }

    // Calculate Senkou Span A (Leading Span A)
    let mut senkou_span_a = Vec::with_capacity(len);
    for i in 0..len {
        if tenkan_sen[i].is_finite() && kijun_sen[i].is_finite() {
            senkou_span_a.push((tenkan_sen[i] + kijun_sen[i]) / 2.0);
        } else {
            senkou_span_a.push(f64::NAN);
        }
    }

    // Calculate Senkou Span B (Leading Span B)
    let mut senkou_span_b = Vec::with_capacity(len);
    for i in 0..len {
        if i < senkou_b_period - 1 {
            senkou_span_b.push(f64::NAN);
        } else {
            let window_high = high[i - senkou_b_period + 1..=i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let window_low = low[i - senkou_b_period + 1..=i]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            senkou_span_b.push((window_high + window_low) / 2.0);
        }
    }

    // Calculate Chikou Span (Lagging Span)
    let mut chikou_span = Vec::with_capacity(len);
    for i in 0..len {
        if i < kijun_period {
            chikou_span.push(f64::NAN);
        } else {
            chikou_span.push(close[i - kijun_period]);
        }
    }

    IchimokuResult {
        tenkan_sen,
        kijun_sen,
        senkou_span_a,
        senkou_span_b,
        chikou_span,
    }
}

pub struct IchimokuResult {
    pub tenkan_sen: Vec<f64>,
    pub kijun_sen: Vec<f64>,
    pub senkou_span_a: Vec<f64>,
    pub senkou_span_b: Vec<f64>,
    pub chikou_span: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macd() {
        let data = vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0];
        let result = macd(&data, 5, 8, 3);

        assert_eq!(result.macd.len(), data.len());
        assert_eq!(result.signal.len(), data.len());
        assert_eq!(result.histogram.len(), data.len());
    }

    #[test]
    fn test_parabolic_sar() {
        let high = vec![10.0, 11.0, 12.0, 11.5, 12.5];
        let low = vec![9.0, 10.0, 11.0, 10.5, 11.5];

        let result = parabolic_sar(&high, &low, 0.02, 0.02, 0.2);
        assert_eq!(result.len(), high.len());
    }
}