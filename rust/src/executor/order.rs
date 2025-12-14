//! Order management for trading execution

use crate::common::Result;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    Submitted,
    Partial,
    Filled,
    Cancelled,
    Rejected,
}

/// Time in force
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Good Till Cancelled
    GTC,
    /// Immediate or Cancel
    IOC,
    /// Fill or Kill
    FOK,
    /// Day (valid until end of day)
    Day,
    /// Good Till Date
    GTD(DateTime<Utc>),
}

impl TimeInForce {
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "GTC" => TimeInForce::GTC,
            "IOC" => TimeInForce::IOC,
            "FOK" => TimeInForce::FOK,
            "DAY" => TimeInForce::Day,
            _ => TimeInForce::GTC, // Default
        }
    }
}

/// Trading order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub client_order_id: Option<String>,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub stop_price: Option<f64>,
    pub filled_quantity: f64,
    pub avg_fill_price: Option<f64>,
    pub status: OrderStatus,
    pub time_in_force: TimeInForce,
    pub create_time: i64,
    pub submit_time: Option<i64>,
    pub fill_time: Option<i64>,
    pub cancel_time: Option<i64>,
    pub expire_time: Option<i64>,
    pub metadata: HashMap<String, String>,
}

impl Order {
    /// Create a new order
    pub fn new(
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        quantity: f64,
        price: Option<f64>,
        stop_price: Option<f64>,
        time_in_force_str: String,
    ) -> Self {
        let time_in_force = TimeInForce::from_str(&time_in_force_str);
        let now = Utc::now().timestamp_millis();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            client_order_id: None,
            symbol,
            side,
            order_type,
            quantity,
            price,
            stop_price,
            filled_quantity: 0.0,
            avg_fill_price: None,
            status: OrderStatus::Pending,
            time_in_force,
            create_time: now,
            submit_time: None,
            fill_time: None,
            cancel_time: None,
            expire_time: None,
            metadata: HashMap::new(),
        }
    }

    /// Get remaining quantity
    pub fn remaining_quantity(&self) -> f64 {
        self.quantity - self.filled_quantity
    }

    /// Check if order is filled
    pub fn is_filled(&self) -> bool {
        self.filled_quantity >= self.quantity
    }

    /// Check if order is active
    pub fn is_active(&self) -> bool {
        matches!(self.status, OrderStatus::Submitted | OrderStatus::Partial)
    }

    /// Check if order can be cancelled
    pub fn is_cancellable(&self) -> bool {
        matches!(self.status, OrderStatus::Submitted | OrderStatus::Partial)
    }

    /// Get fill percentage
    pub fn fill_percentage(&self) -> f64 {
        if self.quantity > 0.0 {
            (self.filled_quantity / self.quantity) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate order value
    pub fn calculate_value(&self) -> Option<f64> {
        if self.filled_quantity > 0.0 {
            if let Some(avg_price) = self.avg_fill_price {
                Some(avg_price * self.filled_quantity)
            } else {
                None
            }
        } else {
            self.price.map(|p| p * self.quantity)
        }
    }

    /// Set client order ID
    pub fn with_client_order_id(mut self, client_id: String) -> Self {
        self.client_order_id = Some(client_id);
        self
    }

    /// Add metadata
    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Order update from exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderUpdate {
    pub order_id: String,
    pub status: OrderStatus,
    pub filled_quantity: Option<f64>,
    pub fill_price: Option<f64>,
    pub timestamp: i64,
    pub reason: Option<String>,
}

/// Order book entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookEntry {
    pub price: f64,
    pub quantity: f64,
    pub order_count: u32,
}

/// Order book
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderBookEntry>,
    pub asks: Vec<OrderBookEntry>,
    pub timestamp: i64,
}

impl OrderBook {
    /// Create a new order book
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            bids: Vec::new(),
            asks: Vec::new(),
            timestamp: Utc::now().timestamp_millis(),
        }
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|e| e.price)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|e| e.price)
    }

    /// Get spread
    pub fn spread(&self) -> Option<f64> {
        if let (Some(bid), Some(ask)) = (self.best_bid(), self.best_ask()) {
            Some(ask - bid)
        } else {
            None
        }
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        if let (Some(bid), Some(ask)) = (self.best_bid(), self.best_ask()) {
            Some((bid + ask) / 2.0)
        } else {
            None
        }
    }
}

/// Order matching engine
pub struct OrderMatcher {
    buy_orders: Vec<Order>,
    sell_orders: Vec<Order>,
}

impl OrderMatcher {
    /// Create a new order matcher
    pub fn new() -> Self {
        Self {
            buy_orders: Vec::new(),
            sell_orders: Vec::new(),
        }
    }

    /// Add order to matcher
    pub fn add_order(&mut self, order: Order) -> Vec<ExecutionResult> {
        let mut executions = Vec::new();

        match order.side {
            OrderSide::Buy => {
                self.match_order(&mut self.sell_orders, order, &mut executions);
            }
            OrderSide::Sell => {
                self.match_order(&mut self.buy_orders, order, &mut executions);
            }
        }

        executions
    }

    /// Match order against order book
    fn match_order(
        &mut self,
        book: &mut Vec<Order>,
        mut order: Order,
        executions: &mut Vec<ExecutionResult>,
    ) {
        while order.remaining_quantity() > 0.0 && !book.is_empty() {
            // Sort book by price (best price first)
            book.sort_by(|a, b| {
                if a.price.unwrap_or(0.0) < b.price.unwrap_or(0.0) {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            });

            let mut remove_index = None;

            for (i, book_order) in book.iter_mut().enumerate() {
                let can_match = match order.order_type {
                    OrderType::Market => true,
                    OrderType::Limit => {
                        if order.side == OrderSide::Buy {
                            book_order.price.unwrap_or(0.0) <= order.price.unwrap_or(f64::INFINITY)
                        } else {
                            book_order.price.unwrap_or(f64::INFINITY) >= order.price.unwrap_or(0.0)
                        }
                    }
                    _ => false,
                };

                if can_match {
                    let trade_quantity = order.remaining_quantity().min(book_order.remaining_quantity());
                    let trade_price = match order.order_type {
                        OrderType::Market => book_order.price.unwrap_or(0.0),
                        OrderType::Limit => order.price.unwrap_or(0.0),
                        _ => book_order.price.unwrap_or(0.0),
                    };

                    // Create execution
                    executions.push(ExecutionResult {
                        order_id: order.id.clone(),
                        symbol: order.symbol.clone(),
                        side: order.side,
                        quantity: trade_quantity,
                        price: trade_price,
                        timestamp: Utc::now().timestamp_millis(),
                    });

                    // Update order quantities
                    order.filled_quantity += trade_quantity;
                    book_order.filled_quantity += trade_quantity;

                    // Update average fill prices
                    if order.avg_fill_price.is_none() {
                        order.avg_fill_price = Some(trade_price);
                    } else {
                        let total_qty = order.filled_quantity;
                        let total_value = order.avg_fill_price.unwrap() * (total_qty - trade_quantity) + trade_price * trade_quantity;
                        order.avg_fill_price = Some(total_value / total_qty);
                    }

                    if book_order.avg_fill_price.is_none() {
                        book_order.avg_fill_price = Some(trade_price);
                    } else {
                        let total_qty = book_order.filled_quantity;
                        let total_value = book_order.avg_fill_price.unwrap() * (total_qty - trade_quantity) + trade_price * trade_quantity;
                        book_order.avg_fill_price = Some(total_value / total_qty);
                    }

                    // Remove filled orders
                    if book_order.is_filled() {
                        remove_index = Some(i);
                    }

                    if order.is_filled() {
                        order.status = OrderStatus::Filled;
                        order.fill_time = Some(Utc::now().timestamp_millis());
                        break;
                    } else {
                        order.status = OrderStatus::Partial;
                    }
                }
            }

            // Remove filled order from book
            if let Some(index) = remove_index {
                book.remove(index);
            }
        }

        // Add unfilled order back to appropriate book
        if !order.is_filled() {
            order.status = OrderStatus::Submitted;
            book.push(order);
        }
    }

    /// Get current order book
    pub fn get_order_book(&self, symbol: String) -> OrderBook {
        let mut bids = self.buy_orders.clone();
        let mut asks = self.sell_orders.clone();

        // Sort bids descending (highest price first)
        bids.sort_by(|a, b| b.price.unwrap_or(0.0).partial_cmp(&a.price.unwrap_or(0.0)).unwrap());

        // Sort asks ascending (lowest price first)
        asks.sort_by(|a, b| a.price.unwrap_or(f64::INFINITY).partial_cmp(&b.price.unwrap_or(0.0)).unwrap());

        // Convert to order book entries
        let bid_entries: Vec<OrderBookEntry> = bids.into_iter()
            .map(|order| OrderBookEntry {
                price: order.price.unwrap_or(0.0),
                quantity: order.remaining_quantity(),
                order_count: 1,
            })
            .collect();

        let ask_entries: Vec<OrderBookEntry> = asks.into_iter()
            .map(|order| OrderBookEntry {
                price: order.price.unwrap_or(0.0),
                quantity: order.remaining_quantity(),
                order_count: 1,
            })
            .collect();

        OrderBook {
            symbol,
            bids: bid_entries,
            asks: ask_entries,
            timestamp: Utc::now().timestamp_millis(),
        }
    }
}

/// Execution result from exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let order = Order::new(
            "BTC/USDT".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            1.0,
            Some(50000.0),
            None,
            "GTC".to_string(),
        );

        assert_eq!(order.symbol, "BTC/USDT");
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.order_type, OrderType::Limit);
        assert_eq!(order.quantity, 1.0);
        assert_eq!(order.price, Some(50000.0));
        assert_eq!(order.status, OrderStatus::Pending);
        assert_eq!(order.remaining_quantity(), 1.0);
        assert!(!order.is_filled());
    }

    #[test]
    fn test_order_filling() {
        let mut order = Order::new(
            "ETH/USDT".to_string(),
            OrderSide::Buy,
            OrderType::Market,
            10.0,
            None,
            None,
            "IOC".to_string(),
        );

        assert_eq!(order.fill_percentage(), 0.0);

        order.filled_quantity = 5.0;
        order.avg_fill_price = Some(3000.0);

        assert_eq!(order.remaining_quantity(), 5.0);
        assert_eq!(order.fill_percentage(), 50.0);
        assert!(!order.is_filled());
        assert_eq!(order.calculate_value(), Some(15000.0));
    }

    #[test]
    fn test_time_in_force() {
        assert!(matches!(TimeInForce::from_str("GTC"), TimeInForce::GTC));
        assert!(matches!(TimeInForce::from_str("ioc"), TimeInForce::IOC));
        assert!(matches!(TimeInForce::from_str("invalid"), TimeInForce::GTC));
    }

    #[test]
    fn test_order_book() {
        let mut book = OrderBook::new("BTC/USDT");

        assert_eq!(book.best_bid(), None);
        assert_eq!(book.best_ask(), None);
        assert_eq!(book.spread(), None);

        // Add some bids and asks
        book.bids.push(OrderBookEntry {
            price: 50000.0,
            quantity: 1.0,
            order_count: 1,
        });

        book.asks.push(OrderBookEntry {
            price: 50001.0,
            quantity: 1.0,
            order_count: 1,
        });

        assert_eq!(book.best_bid(), Some(50000.0));
        assert_eq!(book.best_ask(), Some(50001.0));
        assert_eq!(book.spread(), Some(1.0));
        assert_eq!(book.mid_price(), Some(50000.5));
    }

    #[test]
    fn test_order_matcher() {
        let mut matcher = OrderMatcher::new();

        // Create matching orders
        let mut buy_order = Order::new(
            "BTC/USDT".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            1.0,
            Some(50000.0),
            None,
            "GTC".to_string(),
        );

        let mut sell_order = Order::new(
            "BTC/USDT".to_string(),
            OrderSide::Sell,
            OrderType::Limit,
            1.0,
            Some(49999.0),
            None,
            "GTC".to_string(),
        );

        // Add sell order to matcher first
        let _ = matcher.add_order(sell_order);

        // Add buy order and get executions
        let executions = matcher.add_order(buy_order);

        assert_eq!(executions.len(), 1);
        assert_eq!(executions[0].quantity, 1.0);
        assert_eq!(executions[0].price, 49999.0);
    }
}