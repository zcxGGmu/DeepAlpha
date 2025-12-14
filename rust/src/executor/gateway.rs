//! Exchange gateway interface and implementations

use crate::common::Result;
use crate::executor::{Order, ExecutionResult};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};

/// Exchange gateway trait
#[async_trait]
pub trait ExchangeGateway: Send + Sync {
    /// Submit an order to the exchange
    async fn submit_order(&self, order: Order) -> Result<ExecutionResult>;

    /// Cancel an existing order
    async fn cancel_order(&self, order_id: &str) -> Result<bool>;

    /// Get order status
    async fn get_order_status(&self, order_id: &str) -> Result<Option<OrderStatus>>;

    /// Get account balance
    async fn get_balance(&self) -> Result<f64>;

    /// Get open orders
    async fn get_open_orders(&self) -> Result<Vec<Order>>;

    /// Get position information
    async fn get_positions(&self) -> Result<Vec<PositionInfo>>;
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderStatus {
    Created,
    Pending,
    Submitted,
    Partial,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// Position information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionInfo {
    pub symbol: String,
    pub quantity: f64,
    pub avg_price: f64,
    pub unrealized_pnl: f64,
}

/// Exchange configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    pub name: String,
    pub api_key: String,
    pub secret_key: String,
    pub sandbox: bool,
    pub rate_limits: RateLimits,
    pub fees: FeeStructure,
}

/// Rate limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    pub orders_per_second: u32,
    pub orders_per_day: u32,
    pub api_requests_per_minute: u32,
}

/// Fee structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeStructure {
    pub maker_fee: f64,      // Percentage
    pub taker_fee: f64,      // Percentage
    pub min_fee: Option<f64>,
}

/// Simulated exchange for testing
pub struct SimulatedGateway {
    orders: std::collections::HashMap<String, Order>,
    balance: f64,
    config: ExchangeConfig,
}

impl SimulatedGateway {
    pub fn new() -> Self {
        Self {
            orders: std::collections::HashMap::new(),
            balance: 100000.0,
            config: ExchangeConfig {
                name: "Simulated".to_string(),
                api_key: "test_key".to_string(),
                secret_key: "test_secret".to_string(),
                sandbox: true,
                rate_limits: RateLimits {
                    orders_per_second: 100,
                    orders_per_day: 100000,
                    api_requests_per_minute: 1000,
                },
                fees: FeeStructure {
                    maker_fee: 0.001,
                    taker_fee: 0.002,
                    min_fee: Some(0.01),
                },
            },
        }
    }

    pub fn with_balance(balance: f64) -> Self {
        let mut gateway = Self::new();
        gateway.balance = balance;
        gateway
    }

    fn simulate_fill(&self, order: &mut Order) -> ExecutionResult {
        let price = order.price.unwrap_or_else(|| {
            // Generate a random market price
            50000.0 + (rand::random::<f64>() * 1000.0)
        });

        let fill_price = price + (rand::random::<f64>() - 0.5) * 0.1; // ±0.05 spread
        let filled_quantity = if rand::random::<f64>() > 0.05 {
            // 95% chance of full fill
            order.quantity
        } else {
            // 5% chance of partial fill
            order.quantity * (0.5 + rand::random::<f64>() * 0.5)
        };

        order.filled_quantity = filled_quantity;
        order.avg_fill_price = Some(fill_price);

        if order.filled_quantity >= order.quantity {
            order.status = OrderStatus::Filled;
            order.fill_time = Some(chrono::Utc::now().timestamp_millis());
        } else {
            order.status = OrderStatus::Partial;
        }

        ExecutionResult {
            order_id: order.id.clone(),
            symbol: order.symbol.clone(),
            side: order.side,
            quantity: filled_quantity,
            price: fill_price,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    fn calculate_fee(&self, result: &ExecutionResult) -> f64 {
        let base_fee = result.price * result.quantity;
        let fee_rate = result.side.order_type_is_market(); // Would check if taker or maker

        let fee = if fee_rate {
            self.config.fees.taker_fee
        } else {
            self.config.fees.maker_fee
        } * base_fee;

        fee.max(self.config.fees.min_fee.unwrap_or(0.0))
    }
}

trait OrderSideExt {
    fn order_type_is_market(&self) -> bool;
}

impl OrderSideExt for crate::executor::OrderSide {
    fn order_type_is_market(&self) -> bool {
        // Simulated: assume all orders are taker
        true
    }
}

#[async_trait]
impl ExchangeGateway for SimulatedGateway {
    async fn submit_order(&self, order: Order) -> Result<ExecutionResult> {
        // Simulate processing delay
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let mut order_copy = order.clone();

        // Check balance for buy orders
        if order_copy.side == crate::executor::OrderSide::Buy {
            let required = order_copy.price.unwrap_or(0.0) * order_copy.quantity;
            let fee = self.calculate_fee(&ExecutionResult {
                order_id: order_copy.id.clone(),
                symbol: order_copy.symbol.clone(),
                side: order_copy.side,
                quantity: order_copy.quantity,
                price: order_copy.price.unwrap_or(0.0),
                timestamp: 0,
            });

            if required + fee > self.balance {
                let mut rejected_order = order_copy.clone();
                rejected_order.status = OrderStatus::Rejected;
                self.orders.insert(order_copy.id.clone(), rejected_order);
                return Err(crate::common::DeepAlphaError::InvalidInput(
                    "Insufficient balance".to_string()
                ));
            }
        }

        // Simulate order execution
        let result = self.simulate_fill(&mut order_copy);

        // Update balance for filled orders
        if order_copy.status == OrderStatus::Filled || order_copy.status == OrderStatus::Partial {
            let fee = self.calculate_fee(&result);
            if order_copy.side == crate::executor::OrderSide::Sell {
                self.balance += (result.price * result.quantity) - fee;
            } else {
                self.balance -= (result.price * result.quantity) + fee;
            }
        }

        // Store the order
        self.orders.insert(order_copy.id.clone(), order_copy.clone());

        Ok(result)
    }

    async fn cancel_order(&self, order_id: &str) -> Result<bool> {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;

        if let Some(order) = self.orders.get(order_id) {
            if order.status == OrderStatus::Submitted || order.status == OrderStatus::Partial {
                // Can cancel
                let mut updated_order = order.clone();
                updated_order.status = OrderStatus::Cancelled;
                updated_order.cancel_time = Some(chrono::Utc::now().timestamp_millis());
                self.orders.insert(order_id.to_string(), updated_order);
                Ok(true)
            } else {
                Ok(false) // Cannot cancel
            }
        } else {
            Ok(false) // Order not found
        }
    }

    async fn get_order_status(&self, order_id: &str) -> Result<Option<OrderStatus>> {
        Ok(self.orders.get(order_id).map(|order| order.status))
    }

    async fn get_balance(&self) -> Result<f64> {
        Ok(self.balance)
    }

    async fn get_open_orders(&self) -> Result<Vec<Order>> {
        Ok(self
            .orders
            .values()
            .filter(|order| matches!(order.status, OrderStatus::Submitted | OrderStatus::Partial))
            .cloned()
            .collect())
    }

    async fn get_positions(&self) -> Result<Vec<PositionInfo>> {
        // Simulated implementation - would calculate from order history
        let mut positions = Vec::new();

        // Example position from filled orders
        for order in self.orders.values() {
            if order.status == OrderStatus::Filled || order.status == OrderStatus::Partial {
                if let Some(avg_price) = order.avg_fill_price {
                    let pnl = if order.side == crate::executor::OrderSide::Sell {
                        (avg_price - order.price.unwrap_or(0.0)) * order.filled_quantity
                    } else {
                        0.0
                    };

                    positions.push(PositionInfo {
                        symbol: order.symbol.clone(),
                        quantity: order.filled_quantity,
                        avg_price,
                        unrealized_pnl: pnl,
                    });
                }
            }
        }

        Ok(positions)
    }
}

/// Real exchange implementation (placeholder)
pub struct RealExchangeGateway {
    config: ExchangeConfig,
}

impl RealExchangeGateway {
    pub fn new(config: ExchangeConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl ExchangeGateway for RealExchangeGateway {
    async fn submit_order(&self, _order: Order) -> Result<ExecutionResult> {
        // TODO: Implement real exchange API calls
        Err(crate::common::DeepAlphaError::InvalidInput(
            "Real exchange gateway not implemented".to_string()
        ))
    }

    async fn cancel_order(&self, _order_id: &str) -> Result<bool> {
        Err(crate::common::DeepAlphaError::InvalidInput(
            "Real exchange gateway not implemented".to_string()
        ))
    }

    async fn get_order_status(&self, _order_id: &str) -> Result<Option<OrderStatus>> {
        Err(crate::common::DeepAlphaError::InvalidInput(
            "Real exchange gateway not implemented".to_string()
        ))
    }

    async fn get_balance(&self) -> Result<f64> {
        Err(crate::common::DeepAlphaError::InvalidInput(
            "Real exchange gateway not implemented".to_string()
        ))
    }

    async fn get_open_orders(&self) -> Result<Vec<Order>> {
        Err(crate::common::DeepAlphaError::InvalidInput(
            "Real exchange gateway not implemented".to_string()
        ))
    }

    async fn get_positions(&self) -> Result<Vec<PositionInfo>> {
        Err(crate::common::DeepAlphaError::InvalidInput(
            "Real exchange gateway not implemented".to_string()
        ))
    }
}

/// Gateway factory
pub struct GatewayFactory;

impl GatewayFactory {
    /// Create a gateway based on exchange name
    pub fn create_gateway(exchange_name: &str, config: ExchangeConfig) -> Box<dyn ExchangeGateway + Send + Sync> {
        match exchange_name.to_lowercase().as_str() {
            "simulated" => Box::new(SimulatedGateway::new()),
            "binance" => Box::new(RealExchangeGateway::new(config)),
            "coinbase" => Box::new(RealExchangeGateway::new(config)),
            "kraken" => Box::new(RealExchangeGateway::new(config)),
            _ => Box::new(SimulatedGateway::new()), // Default to simulated
        }
    }

    /// Create a test gateway
    pub fn create_test_gateway(balance: f64) -> Box<dyn ExchangeGateway + Send + Sync> {
        Box::new(SimulatedGateway::with_balance(balance))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{OrderSide, OrderType};

    #[tokio::test]
    async fn test_simulated_gateway() {
        let gateway = SimulatedGateway::new();

        let order = Order::new(
            "BTC/USDT".to_string(),
            OrderSide::Buy,
            OrderType::Market,
            1.0,
            Some(50000.0),
            None,
            "GTC".to_string(),
        );

        // Submit order
        let result = gateway.submit_order(order).await.unwrap();
        assert_eq!(result.symbol, "BTC/USDT");
        assert_eq!(result.quantity, 1.0);
        assert!(result.price > 0.0);
    }

    #[tokio::test]
    async fn test_gateway_factory() {
        let config = ExchangeConfig {
            name: "Simulated".to_string(),
            api_key: "test".to_string(),
            secret_key: "test".to_string(),
            sandbox: true,
            rate_limits: RateLimits {
                orders_per_second: 100,
                orders_per_day: 10000,
                api_requests_per_minute: 1000,
            },
            fees: FeeStructure {
                maker_fee: 0.001,
                taker_fee: 0.002,
                min_fee: Some(0.01),
            },
        };

        let gateway = GatewayFactory::create_gateway("Simulated", config);

        // Test basic functionality
        let balance = gateway.get_balance().await.unwrap();
        assert_eq!(balance, 100000.0);
    }

    #[test]
    fn test_fee_calculation() {
        let gateway = SimulatedGateway::new();

        let result = ExecutionResult {
            order_id: "test".to_string(),
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            quantity: 1.0,
            price: 50000.0,
            timestamp: 0,
        };

        let fee = gateway.calculate_fee(&result);
        // Should be taker fee * price * quantity
        assert_eq!(fee, 50000.0 * 0.002 * 1.0); // 100.0
    }

    #[test]
    fn test_fee_minimum() {
        let gateway = SimulatedGateway::new();

        let result = ExecutionResult {
            order_id: "test".to_string(),
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            quantity: 0.1, // Small quantity
            price: 100.0,   // Small price
            timestamp: 0,
        };

        let fee = gateway.calculate_fee(&result);
        // Should use minimum fee
        assert_eq!(fee, 0.01);
    }
}