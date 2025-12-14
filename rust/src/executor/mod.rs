//! Trading Execution Engine Module
//!
//! Ultra-low latency trading execution and risk management with support for
//! 1,000+ orders per second and sub-millisecond execution.

use crate::common::{Result, DeepAlphaError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, oneshot};
use tokio::time::{Duration, Instant};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error, debug};

pub mod order;
pub mod risk;
pub mod position;
pub mod portfolio;
pub mod gateway;

use order::{Order, OrderStatus, OrderType, OrderSide, OrderUpdate};
use risk::{RiskManager, RiskCheck};
use position::PositionManager;
use gateway::{ExchangeGateway, ExecutionResult};

/// Initialize the executor submodule
pub fn init_executor_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<ExecutionEngine>()?;
    m.add_class::<ExecutionStats>()?;
    Ok(())
}

/// Execution engine statistics
#[pyclass]
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    #[pyo3(get)]
    pub total_orders: u64,
    #[pyo3(get)]
    pub filled_orders: u64,
    #[pyo3(get)]
    pub cancelled_orders: u64,
    #[pyo3(get)]
    pub rejected_orders: u64,
    #[pyo3(get)]
    pub total_volume: f64,
    #[pyo3(get)]
    pub total_value: f64,
    #[pyo3(get)]
    pub avg_execution_time_us: f64,
    #[pyo3(get)]
    pub current_pending_orders: usize,
    #[pyo3(get)]
    pub risk_violations: u64,
}

/// High-performance trading execution engine
#[pyclass]
pub struct ExecutionEngine {
    #[pyo3(get)]
    active: bool,

    // Internal state
    orders: Arc<RwLock<HashMap<String, Order>>>,
    pending_orders: Arc<RwLock<Vec<String>>>,
    risk_manager: Arc<RwLock<RiskManager>>,
    position_manager: Arc<PositionManager>,
    gateway: Arc<dyn ExchangeGateway + Send + Sync>,
    order_tx: mpsc::UnboundedSender<OrderCommand>,
    stats: Arc<RwLock<ExecutionStats>>,
    execution_times: Arc<RwLock<Vec<f64>>>,
}

/// Internal command types
#[derive(Debug)]
enum OrderCommand {
    Submit {
        order: Order,
        response_tx: oneshot::Sender<Result<String>>,
    },
    Cancel {
        order_id: String,
        response_tx: oneshot::Sender<Result<bool>>,
    },
    Update {
        update: OrderUpdate,
    },
    GetOrder {
        order_id: String,
        response_tx: oneshot::Sender<Option<Order>>,
    },
    GetStats {
        response_tx: oneshot::Sender<ExecutionStats>,
    },
}

#[pymethods]
impl ExecutionEngine {
    /// Create a new execution engine
    #[new]
    fn new() -> Self {
        let (order_tx, order_rx) = mpsc::unbounded_channel();

        Self {
            active: false,
            orders: Arc::new(RwLock::new(HashMap::new())),
            pending_orders: Arc::new(RwLock::new(Vec::new())),
            risk_manager: Arc::new(RwLock::new(RiskManager::new())),
            position_manager: Arc::new(PositionManager::new()),
            gateway: Arc::new(gateway::SimulatedGateway::new()),
            order_tx,
            stats: Arc::new(RwLock::new(ExecutionStats::default())),
            execution_times: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start the execution engine
    fn start(&mut self) -> PyResult<()> {
        if self.active {
            return Ok(());
        }

        let order_rx = match std::mem::replace(&mut self.order_tx, mpsc::unbounded_channel().0) {
            Some(rx) => rx,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Cannot start engine: already consumed"
                ));
            }
        };

        let orders = self.orders.clone();
        let pending_orders = self.pending_orders.clone();
        let risk_manager = self.risk_manager.clone();
        let position_manager = self.position_manager.clone();
        let gateway = self.gateway.clone();
        let stats = self.stats.clone();
        let execution_times = self.execution_times.clone();

        self.active = true;

        // Start processing in background
        tokio::spawn(async move {
            process_orders(
                order_rx,
                orders,
                pending_orders,
                risk_manager,
                position_manager,
                gateway,
                stats,
                execution_times,
            ).await;
        });

        info!("Execution engine started");
        Ok(())
    }

    /// Stop the execution engine
    fn stop(&mut self) {
        self.active = false;
        info!("Execution engine stopping");
    }

    /// Submit a new order for execution
    fn submit_order(&self, py: Python, order_data: PyObject) -> PyResult<String> {
        // Extract order data from Python object
        let order_dict = order_data.downcast::<pyo3::PyDict>()?;

        let symbol = order_dict.get_item("symbol")?.extract::<String>()?;
        let side = match order_dict.get_item("side")?.extract::<String>()?.as_str() {
            "buy" => OrderSide::Buy,
            "sell" => OrderSide::Sell,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid side: must be 'buy' or 'sell'"
            )),
        };
        let order_type = match order_dict.get_item("type")?.extract::<String>()?.as_str() {
            "market" => OrderType::Market,
            "limit" => OrderType::Limit,
            "stop" => OrderType::Stop,
            "stop_limit" => OrderType::StopLimit,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid type: must be 'market', 'limit', 'stop', or 'stop_limit'"
            )),
        };
        let quantity = order_dict.get_item("quantity")?.extract::<f64>()?;
        let price = order_dict.get_item("price")?.extract::<Option<f64>>()?;
        let stop_price = order_dict.get_item("stop_price")?.extract::<Option<f64>>()?;
        let time_in_force = order_dict.get_item("time_in_force")
            .and_then(|v| v.extract::<Option<String>>().ok())
            .flatten()
            .unwrap_or_else(|| "GTC".to_string());

        // Create order
        let order = Order::new(
            symbol,
            side,
            order_type,
            quantity,
            price,
            stop_price,
            time_in_force,
        );

        // Send to processing loop
        let (response_tx, response_rx) = oneshot::channel();
        let command = OrderCommand::Submit {
            order,
            response_tx,
        };

        self.order_tx.send(command)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Wait for response (blocking in Python context)
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        match rt.block_on(response_rx) {
            Ok(result) => match result {
                Ok(order_id) => Ok(order_id),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())),
            },
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
        }
    }

    /// Cancel an existing order
    fn cancel_order(&self, order_id: String) -> PyResult<bool> {
        let (response_tx, response_rx) = oneshot::channel();
        let command = OrderCommand::Cancel {
            order_id,
            response_tx,
        };

        self.order_tx.send(command)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Wait for response
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        match rt.block_on(response_rx) {
            Ok(result) => match result {
                Ok(success) => Ok(success),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())),
            },
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
        }
    }

    /// Get order status
    fn get_order(&self, order_id: String) -> PyResult<Option<PyObject>> {
        let (response_tx, response_rx) = oneshot::channel();
        let command = OrderCommand::GetOrder {
            order_id,
            response_tx,
        };

        self.order_tx.send(command)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Wait for response
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        match rt.block_on(response_rx) {
            Ok(order_option) => {
                let py = Python::acquire_gil().python();
                match order_option {
                    Some(order) => {
                        let dict = pyo3::PyDict::new(py);
                        dict.set_item("id", order.id)?;
                        dict.set_item("symbol", order.symbol)?;
                        dict.set_item("side", match order.side {
                            OrderSide::Buy => "buy",
                            OrderSide::Sell => "sell",
                        })?;
                        dict.set_item("type", match order.order_type {
                            OrderType::Market => "market",
                            OrderType::Limit => "limit",
                            OrderType::Stop => "stop",
                            OrderType::StopLimit => "stop_limit",
                        })?;
                        dict.set_item("quantity", order.quantity)?;
                        if let Some(price) = order.price {
                            dict.set_item("price", price)?;
                        }
                        dict.set_item("status", match order.status {
                            OrderStatus::Pending => "pending",
                            OrderStatus::Submitted => "submitted",
                            OrderStatus::Partial => "partial",
                            OrderStatus::Filled => "filled",
                            OrderStatus::Cancelled => "cancelled",
                            OrderStatus::Rejected => "rejected",
                        })?;
                        dict.set_item("filled_quantity", order.filled_quantity)?;
                        dict.set_item("create_time", order.create_time)?;
                        if let Some(fill_time) = order.fill_time {
                            dict.set_item("fill_time", fill_time)?;
                        }
                        Ok(Some(dict.into()))
                    }
                    None => Ok(None),
                }
            },
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
        }
    }

    /// Get current execution statistics
    fn get_stats(&self) -> PyResult<ExecutionStats> {
        let (response_tx, response_rx) = oneshot::channel();
        let command = OrderCommand::GetStats {
            response_tx,
        };

        self.order_tx.send(command)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Wait for response
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        match rt.block_on(response_rx) {
            Ok(stats) => Ok(stats),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
        }
    }

    /// Get current positions
    fn get_positions(&self) -> PyResult<Vec<PyObject>> {
        let py = Python::acquire_gil().python();
        let positions = self.position_manager.get_all_positions();

        let mut result = Vec::new();
        for pos in positions {
            let dict = pyo3::PyDict::new(py);
            dict.set_item("symbol", pos.symbol)?;
            dict.set_item("quantity", pos.quantity)?;
            dict.set_item("avg_price", pos.avg_price)?;
            dict.set_item("unrealized_pnl", pos.unrealized_pnl)?;
            dict.set_item("realized_pnl", pos.realized_pnl)?;
            dict.set_item("total_pnl", pos.total_pnl)?;
            result.push(dict.into());
        }

        Ok(result)
    }
}

/// Main order processing loop
async fn process_orders(
    mut order_rx: mpsc::UnboundedReceiver<OrderCommand>,
    orders: Arc<RwLock<HashMap<String, Order>>>,
    pending_orders: Arc<RwLock<Vec<String>>>,
    risk_manager: Arc<RwLock<RiskManager>>,
    position_manager: Arc<PositionManager>,
    gateway: Arc<dyn ExchangeGateway + Send + Sync>,
    stats: Arc<RwLock<ExecutionStats>>,
    execution_times: Arc<RwLock<Vec<f64>>>,
) {
    while let Some(command) = order_rx.recv().await {
        match command {
            OrderCommand::Submit { order, response_tx } => {
                let start_time = Instant::now();
                let order_id = order.id.clone();

                // Risk check
                let risk_result = {
                    let risk_manager = risk_manager.read().await;
                    risk_manager.check_order(&order, &position_manager).await
                };

                match risk_result {
                    Ok(_) => {
                        // Store order
                        {
                            let mut orders_map = orders.write().await;
                            orders_map.insert(order_id.clone(), order.clone());
                        }

                        // Add to pending
                        {
                            let mut pending = pending_orders.write().await;
                            pending.push(order_id.clone());
                        }

                        // Submit to exchange
                        let result = gateway.submit_order(order.clone()).await;

                        match result {
                            Ok(_) => {
                                // Update order status
                                if let Some(mut stored_order) = orders.write().await.get_mut(&order_id) {
                                    stored_order.status = OrderStatus::Submitted;
                                    stored_order.submit_time = Some(chrono::Utc::now().timestamp_millis());
                                }

                                // Update stats
                                if let Ok(mut stats) = stats.try_write() {
                                    stats.total_orders += 1;
                                }

                                // Send response
                                let _ = response_tx.send(Ok(order_id));
                            }
                            Err(e) => {
                                // Update order status to rejected
                                if let Some(mut stored_order) = orders.write().await.get_mut(&order_id) {
                                    stored_order.status = OrderStatus::Rejected;
                                }

                                // Remove from pending
                                {
                                    let mut pending = pending_orders.write().await;
                                    pending.retain(|id| id != &order_id);
                                }

                                // Update stats
                                if let Ok(mut stats) = stats.try_write() {
                                    stats.rejected_orders += 1;
                                }

                                // Send error response
                                let _ = response_tx.send(Err(e));
                            }
                        }
                    }
                    Err(e) => {
                        // Risk violation
                        if let Ok(mut stats) = stats.try_write() {
                            stats.risk_violations += 1;
                        }

                        // Send error response
                        let _ = response_tx.send(Err(e));
                    }
                }

                // Record execution time
                let execution_time = start_time.elapsed().as_micros() as f64;
                if let Ok(mut times) = execution_times.try_write() {
                    times.push(execution_time);
                    // Keep only last 1000 times
                    if times.len() > 1000 {
                        times.remove(0);
                    }

                    // Update average
                    if !times.is_empty() {
                        let avg = times.iter().sum::<f64>() / times.len() as f64;
                        if let Ok(mut stats) = stats.try_write() {
                            stats.avg_execution_time_us = avg;
                        }
                    }
                }
            }

            OrderCommand::Cancel { order_id, response_tx } => {
                // Check if order exists and is cancellable
                let can_cancel = {
                    let orders_map = orders.read().await;
                    orders_map.get(&order_id)
                        .map(|order| order.status == OrderStatus::Submitted || order.status == OrderStatus::Partial)
                        .unwrap_or(false)
                };

                if can_cancel {
                    // Send cancel to exchange
                    let result = gateway.cancel_order(&order_id).await;

                    match result {
                        Ok(success) => {
                            if success {
                                // Update order status
                                if let Some(mut order) = orders.write().await.get_mut(&order_id) {
                                    order.status = OrderStatus::Cancelled;
                                    order.cancel_time = Some(chrono::Utc::now().timestamp_millis());
                                }

                                // Remove from pending
                                {
                                    let mut pending = pending_orders.write().await;
                                    pending.retain(|id| id != &order_id);
                                }

                                if let Ok(mut stats) = stats.try_write() {
                                    stats.cancelled_orders += 1;
                                }
                            }
                            let _ = response_tx.send(Ok(success));
                        }
                        Err(e) => {
                            let _ = response_tx.send(Err(e));
                        }
                    }
                } else {
                    // Order not found or not cancellable
                    let _ = response_tx.send(Ok(false));
                }
            }

            OrderCommand::Update { update } => {
                // Update order from exchange
                if let Some(mut order) = orders.write().await.get_mut(&update.order_id) {
                    // Apply update
                    match update.status {
                        OrderStatus::Partial => {
                            order.status = OrderStatus::Partial;
                            order.filled_quantity = update.filled_quantity.unwrap_or(0.0);
                            if let Some(price) = update.fill_price {
                                order.avg_fill_price = Some(price);
                            }
                        }
                        OrderStatus::Filled => {
                            order.status = OrderStatus::Filled;
                            order.filled_quantity = order.quantity;
                            order.fill_time = Some(update.timestamp);
                            if let Some(price) = update.fill_price {
                                order.avg_fill_price = Some(price);
                            }

                            // Update position
                            position_manager.update_position(&order);

                            // Update stats
                            if let Ok(mut stats) = stats.try_write() {
                                stats.filled_orders += 1;
                                stats.total_volume += order.filled_quantity;
                                if let Some(price) = order.avg_fill_price {
                                    stats.total_value += price * order.filled_quantity;
                                }
                            }
                        }
                        _ => {
                            order.status = update.status;
                        }
                    }
                }
            }

            OrderCommand::GetOrder { order_id, response_tx } => {
                let order = orders.read().await.get(&order_id).cloned();
                let _ = response_tx.send(order);
            }

            OrderCommand::GetStats { response_tx } => {
                let current_pending = pending_orders.read().await.len();
                let mut stats_copy = {
                    let stats = stats.read().await;
                    ExecutionStats {
                        total_orders: stats.total_orders,
                        filled_orders: stats.filled_orders,
                        cancelled_orders: stats.cancelled_orders,
                        rejected_orders: stats.rejected_orders,
                        total_volume: stats.total_volume,
                        total_value: stats.total_value,
                        avg_execution_time_us: stats.avg_execution_time_us,
                        current_pending_orders: current_pending,
                        risk_violations: stats.risk_violations,
                    }
                };
                let _ = response_tx.send(stats_copy);
            }
        }
    }

    info!("Order processing ended");
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_execution_stats_default() {
        let stats = ExecutionStats::default();
        assert_eq!(stats.total_orders, 0);
        assert_eq!(stats.filled_orders, 0);
        assert_eq!(stats.cancelled_orders, 0);
        assert_eq!(stats.rejected_orders, 0);
    }

    #[test]
    fn test_engine_creation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let engine = ExecutionEngine::new();
            assert!(!engine.active);

            let stats = engine.get_stats().unwrap();
            assert_eq!(stats.total_orders, 0);
        });
    }
}