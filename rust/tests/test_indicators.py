"""Test cases for technical indicators"""

import pytest
import numpy as np
from deepalpha_rust.indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """Test technical indicators functionality"""

    @pytest.fixture
    def sample_data(self):
        """Sample price data for testing"""
        return [44.0, 44.5, 45.0, 44.75, 45.25, 45.5, 45.75, 46.0, 45.5, 45.0,
                44.75, 44.25, 44.0, 43.75, 43.5, 43.25, 43.0, 42.75, 42.5, 42.25]

    @pytest.fixture
    def indicators(self, sample_data):
        """Create TechnicalIndicators instance with sample data"""
        return TechnicalIndicators(sample_data)

    def test_init_empty_data(self):
        """Test initialization with empty data"""
        with pytest.raises(ValueError):
            TechnicalIndicators([])

    def test_init_valid_data(self, sample_data):
        """Test initialization with valid data"""
        indicators = TechnicalIndicators(sample_data)
        assert indicators.len() == len(sample_data)

    def test_sma(self, indicators):
        """Test Simple Moving Average"""
        period = 5
        sma_values = indicators.sma(period)

        assert len(sma_values) == indicators.len()
        assert np.isnan(sma_values[0])  # First period-1 values should be NaN
        assert np.isnan(sma_values[period-2])  # Last NaN before valid values
        assert not np.isnan(sma_values[period-1])  # First valid value

        # Test SMA calculation for first valid value
        expected = (44.0 + 44.5 + 45.0 + 44.75 + 45.25) / 5
        assert abs(sma_values[period-1] - expected) < 0.001

    def test_ema(self, indicators):
        """Test Exponential Moving Average"""
        period = 5
        ema_values = indicators.ema(period)

        assert len(ema_values) == indicators.len()
        assert np.isnan(ema_values[0])  # First period-1 values should be NaN
        assert not np.isnan(ema_values[period-1])  # First valid value

    def test_rsi(self, indicators):
        """Test Relative Strength Index"""
        period = 5
        rsi_values = indicators.rsi(period)

        assert len(rsi_values) == indicators.len()
        assert np.isnan(rsi_values[period-1])  # First period values should be NaN
        assert not np.isnan(rsi_values[period])  # First valid value

        # RSI should be between 0 and 100
        valid_values = rsi_values[~np.isnan(rsi_values)]
        assert np.all((valid_values >= 0) & (valid_values <= 100))

    def test_macd(self, indicators):
        """Test Moving Average Convergence Divergence"""
        macd_result = indicators.macd(fast=5, slow=8, signal=3)

        assert 'macd' in macd_result
        assert 'signal' in macd_result
        assert 'histogram' in macd_result

        macd = macd_result['macd']
        signal = macd_result['signal']
        histogram = macd_result['histogram']

        assert len(macd) == indicators.len()
        assert len(signal) == indicators.len()
        assert len(histogram) == indicators.len()

        # Histogram should be MACD - Signal
        valid_mask = ~(np.isnan(macd) | np.isnan(signal))
        np.testing.assert_array_almost_equal(
            histogram[valid_mask],
            macd[valid_mask] - signal[valid_mask],
            decimal=5
        )

    def test_bollinger_bands(self, indicators):
        """Test Bollinger Bands"""
        bb_result = indicators.bollinger_bands(period=5, std_dev=2.0)

        assert 'upper' in bb_result
        assert 'middle' in bb_result
        assert 'lower' in bb_result

        upper = bb_result['upper']
        middle = bb_result['middle']
        lower = bb_result['lower']

        assert len(upper) == indicators.len()
        assert len(middle) == indicators.len()
        assert len(lower) == indicators.len()

        # Upper band should be > middle > lower band for valid values
        valid_mask = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        assert np.all(upper[valid_mask] > middle[valid_mask])
        assert np.all(middle[valid_mask] > lower[valid_mask])

    def test_invalid_period(self, indicators):
        """Test invalid period values"""
        with pytest.raises(ValueError):
            indicators.sma(0)

        with pytest.raises(ValueError):
            indicators.ema(0)

        with pytest.raises(ValueError):
            indicators.rsi(0)

    def test_insufficient_data(self):
        """Test with insufficient data"""
        short_data = [1.0, 2.0, 3.0]
        indicators = TechnicalIndicators(short_data)

        with pytest.raises(ValueError):
            indicators.sma(5)

        with pytest.raises(ValueError):
            indicators.ema(5)

        with pytest.raises(ValueError):
            indicators.rsi(5)

    def test_macd_invalid_parameters(self, indicators):
        """Test MACD with invalid parameters"""
        # Fast period should be less than slow period
        with pytest.raises(ValueError):
            indicators.macd(fast=10, slow=5, signal=3)

    def test_bollinger_bands_invalid_std_dev(self, indicators):
        """Test Bollinger Bands with invalid standard deviation"""
        with pytest.raises(ValueError):
            indicators.bollinger_bands(period=5, std_dev=-1.0)

        with pytest.raises(ValueError):
            indicators.bollinger_bands(period=5, std_dev=0.0)