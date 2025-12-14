//! DeepAlpha Rust Performance Modules
//!
//! This crate provides high-performance Rust implementations for critical components
//! of the DeepAlpha quantitative trading system.

use pyo3::prelude::*;

pub mod common;
pub mod indicators;
pub mod websocket;
pub mod stream;
pub mod executor;

/// Python module definition
#[pymodule]
fn deepalpha_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Register submodules
    let indicators_module = PyModule::new(_py, "indicators")?;
    indicators::init_indicators_module(indicators_module)?;
    m.add_submodule(indicators_module)?;

    Ok(())
}