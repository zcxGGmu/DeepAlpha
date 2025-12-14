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

    let websocket_module = PyModule::new(_py, "websocket")?;
    websocket::init_websocket_module(websocket_module)?;
    m.add_submodule(websocket_module)?;

    let stream_module = PyModule::new(_py, "stream")?;
    stream::init_stream_module(stream_module)?;
    m.add_submodule(stream_module)?;

    let executor_module = PyModule::new(_py, "executor")?;
    executor::init_executor_module(executor_module)?;
    m.add_submodule(executor_module)?;

    // Add ExecutionEngine to main module for convenience
    m.add_class::<executor::ExecutionEngine>()?;

    Ok(())
}