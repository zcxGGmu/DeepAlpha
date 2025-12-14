//! Technical Indicators Module
//!
//! High-performance implementations of common technical analysis indicators.

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1, IntoPyArray};
use std::f64;

pub mod basic;
pub mod momentum;
pub mod volatility;
pub mod trend;

#[cfg(test)]
mod tests;

use crate::common::{Result, DeepAlphaError, validation};

/// Initialize the indicators submodule
pub fn init_indicators_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<TechnicalIndicators>()?;
    Ok(())
}

/// Main technical indicators calculator
#[pyclass]
pub struct TechnicalIndicators {
    #[pyo3(get)]
    data: Vec<f64>,
}

#[pymethods]
impl TechnicalIndicators {
    /// Create a new TechnicalIndicators instance
    #[new]
    fn new(data: Vec<f64>) -> Result<Self> {
        if data.is_empty() {
            return Err(DeepAlphaError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }
        Ok(Self { data })
    }

    /// Get the length of the data
    #[getter]
    fn len(&self) -> usize {
        self.data.len()
    }

    /// Simple Moving Average (SMA)
    fn sma(&self, py: Python, period: usize) -> PyResult<PyArray1<f64>> {
        validation::validate_period(period)?;
        validation::validate_data_length(self.data.len(), period)?;

        let result = basic::sma(&self.data, period);
        Ok(result.into_pyarray(py))
    }

    /// Exponential Moving Average (EMA)
    fn ema(&self, py: Python, period: usize) -> PyResult<PyArray1<f64>> {
        validation::validate_period(period)?;
        validation::validate_data_length(self.data.len(), period)?;

        let result = basic::ema(&self.data, period);
        Ok(result.into_pyarray(py))
    }

    /// Weighted Moving Average (WMA)
    fn wma(&self, py: Python, period: usize) -> PyResult<PyArray1<f64>> {
        validation::validate_period(period)?;
        validation::validate_data_length(self.data.len(), period)?;

        let result = basic::wma(&self.data, period);
        Ok(result.into_pyarray(py))
    }

    /// Relative Strength Index (RSI)
    fn rsi(&self, py: Python, period: usize) -> PyResult<PyArray1<f64>> {
        validation::validate_period(period)?;
        validation::validate_data_length(self.data.len(), period + 1)?;

        let result = momentum::rsi(&self.data, period);
        Ok(result.into_pyarray(py))
    }

    /// Moving Average Convergence Divergence (MACD)
    fn macd(&self, py: Python, fast: usize, slow: usize, signal: usize) -> PyResult<PyObject> {
        validation::validate_period(fast)?;
        validation::validate_period(slow)?;
        validation::validate_period(signal)?;

        if fast >= slow {
            return Err(DeepAlphaError::InvalidInput(
                "Fast period must be less than slow period".to_string(),
            )
            .into());
        }

        validation::validate_data_length(self.data.len(), slow)?;

        let result = trend::macd(&self.data, fast, slow, signal);
        let dict = pyo3::PyDict::new(py);
        dict.set_item("macd", result.macd.into_pyarray(py))?;
        dict.set_item("signal", result.signal.into_pyarray(py))?;
        dict.set_item("histogram", result.histogram.into_pyarray(py))?;
        Ok(dict.into())
    }

    /// Bollinger Bands
    fn bollinger_bands(
        &self,
        py: Python,
        period: usize,
        std_dev: f64,
    ) -> PyResult<PyObject> {
        validation::validate_period(period)?;
        if std_dev <= 0.0 {
            return Err(DeepAlphaError::InvalidInput(
                "Standard deviation must be positive".to_string(),
            )
            .into());
        }

        validation::validate_data_length(self.data.len(), period)?;

        let result = volatility::bollinger_bands(&self.data, period, std_dev);
        let dict = pyo3::PyDict::new(py);
        dict.set_item("upper", result.upper.into_pyarray(py))?;
        dict.set_item("middle", result.middle.into_pyarray(py))?;
        dict.set_item("lower", result.lower.into_pyarray(py))?;
        Ok(dict.into())
    }

    /// Stochastic Oscillator
    fn stochastic(
        &self,
        py: Python,
        k_period: usize,
        d_period: usize,
    ) -> PyResult<PyObject> {
        validation::validate_period(k_period)?;
        validation::validate_period(d_period)?;
        validation::validate_data_length(self.data.len(), k_period)?;

        let result = momentum::stochastic(&self.data, k_period, d_period);
        let dict = pyo3::PyDict::new(py);
        dict.set_item("k", result.k.into_pyarray(py))?;
        dict.set_item("d", result.d.into_pyarray(py))?;
        Ok(dict.into())
    }
}