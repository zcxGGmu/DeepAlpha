//! Unit tests for indicators module

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_sma_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = basic::sma(&data, 3);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 3.0);
        assert_eq!(result[4], 4.0);
    }

    #[test]
    fn test_sma_empty_data() {
        let data = vec![];
        let result = basic::sma(&data, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_ema_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = basic::ema(&data, 3);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_finite());
        assert!(result[3].is_finite());
        assert!(result[4].is_finite());
    }

    #[test]
    fn test_rsi_basic() {
        let data = vec![44.0, 44.5, 45.0, 44.75, 45.25, 45.5, 45.75, 46.0, 45.5, 45.0];
        let result = momentum::rsi(&data, 5);

        assert_eq!(result.len(), data.len());

        // Check that RSI values are in valid range [0, 100]
        for i in 5..result.len() {
            assert!(result[i] >= 0.0);
            assert!(result[i] <= 100.0);
        }
    }

    #[test]
    fn test_bollinger_bands_basic() {
        let data = vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0];
        let result = volatility::bollinger_bands(&data, 5, 2.0);

        assert_eq!(result.upper.len(), data.len());
        assert_eq!(result.middle.len(), data.len());
        assert_eq!(result.lower.len(), data.len());

        // Check that upper > middle > lower for valid values
        for i in 4..data.len() {
            assert!(result.upper[i] > result.middle[i]);
            assert!(result.middle[i] > result.lower[i]);
        }
    }

    #[test]
    fn test_macd_basic() {
        let data = vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0];
        let result = trend::macd(&data, 5, 8, 3);

        assert_eq!(result.macd.len(), data.len());
        assert_eq!(result.signal.len(), data.len());
        assert_eq!(result.histogram.len(), data.len());

        // Check that histogram = macd - signal for valid values
        for i in 7..data.len() {
            if result.macd[i].is_finite() && result.signal[i].is_finite() {
                assert!((result.histogram[i] - (result.macd[i] - result.signal[i])).abs() < 0.0001);
            }
        }
    }

    #[test]
    fn test_parabolic_sar_basic() {
        let high = vec![10.0, 11.0, 12.0, 11.5, 12.5, 13.0];
        let low = vec![9.0, 10.0, 11.0, 10.5, 11.5, 12.0];
        let result = trend::parabolic_sar(&high, &low, 0.02, 0.02, 0.2);

        assert_eq!(result.len(), high.len());
        assert!(result[0].is_finite());
    }

    #[test]
    fn test_stochastic_basic() {
        let data = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0];
        let result = momentum::stochastic(&data, 5, 3);

        assert_eq!(result.k.len(), data.len());
        assert_eq!(result.d.len(), data.len());

        // Check that %K values are in [0, 100] range
        for i in 4..data.len() {
            assert!(result.k[i] >= 0.0);
            assert!(result.k[i] <= 100.0);
        }
    }

    #[test]
    fn test_atr_basic() {
        let high = vec![10.0, 11.0, 12.0, 11.5, 12.5, 13.0];
        let low = vec![9.0, 10.0, 11.0, 10.5, 11.5, 12.0];
        let close = vec![9.5, 10.5, 11.5, 11.0, 12.0, 12.5];
        let result = volatility::atr(&high, &low, &close, 2);

        assert_eq!(result.len(), high.len());
    }

    #[test]
    fn test_ichimoku_basic() {
        let high = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
        let low = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        let close = vec![9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5];
        let result = trend::ichimoku(&high, &low, &close, 9, 26, 52);

        assert_eq!(result.tenkan_sen.len(), high.len());
        assert_eq!(result.kijun_sen.len(), high.len());
        assert_eq!(result.senkou_span_a.len(), high.len());
        assert_eq!(result.senkou_span_b.len(), high.len());
        assert_eq!(result.chikou_span.len(), high.len());
    }
}