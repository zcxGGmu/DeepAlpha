//! Position management for trading

use crate::executor::Order;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Position data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64, // Positive for long, negative for short
    pub avg_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub total_pnl: f64,
    pub last_update: i64,
    pub trades_count: u32,
}

impl Position {
    /// Create a new position
    pub fn new(symbol: String, quantity: f64, price: f64) -> Self {
        Self {
            symbol,
            quantity,
            avg_price: price,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            total_pnl: 0.0,
            last_update: Utc::now().timestamp_millis(),
            trades_count: 1,
        }
    }

    /// Get position value
    pub fn get_value(&self) -> f64 {
        self.quantity.abs() * self.avg_price
    }

    /// Get position side
    pub fn get_side(&self) -> &str {
        if self.quantity > 0.0 {
            "long"
        } else if self.quantity < 0.0 {
            "short"
        } else {
            "flat"
        }
    }

    /// Check if position is flat
    pub fn is_flat(&self) -> bool {
        (self.quantity - 0.0).abs() < f64::EPSILON
    }

    /// Calculate unrealized PnL given current price
    pub fn calculate_unrealized_pnl(&mut self, current_price: f64) {
        if self.quantity != 0.0 {
            let current_value = self.quantity * current_price;
            let cost_basis = self.quantity * self.avg_price;
            self.unrealized_pnl = current_value - cost_basis;
            self.total_pnl = self.unrealized_pnl + self.realized_pnl;
        } else {
            self.unrealized_pnl = 0.0;
            self.total_pnl = self.realized_pnl;
        }
        self.last_update = Utc::now().timestamp_millis();
    }
}

/// Position manager
pub struct PositionManager {
    positions: HashMap<String, Position>,
    portfolio: Portfolio,
}

impl PositionManager {
    /// Create a new position manager
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
            portfolio: Portfolio::new(),
        }
    }

    /// Update position based on filled order
    pub fn update_position(&mut self, order: &Order) {
        if order.filled_quantity == 0.0 {
            return;
        }

        let symbol = &order.symbol;
        let fill_qty = order.filled_quantity;
        let fill_price = order.avg_fill_price.unwrap_or(0.0);

        if let Some(position) = self.positions.get_mut(symbol) {
            // Update existing position
            let old_qty = position.quantity;
            let old_value = old_qty * position.avg_price;

            // Add or subtract quantity based on side
            match order.side {
                crate::executor::OrderSide::Buy => {
                    position.quantity += fill_qty;
                }
                crate::executor::OrderSide::Sell => {
                    position.quantity -= fill_qty;
                }
            }

            // Calculate new average price
            if position.quantity != 0.0 {
                let new_value = old_value + (fill_qty * fill_price);
                position.avg_price = new_value / position.quantity;
            } else {
                // Position closed, reset avg_price
                position.avg_price = 0.0;
            }

            // Calculate realized PnL for position changes
            if old_qty != 0.0 && position.quantity == 0.0 {
                // Position fully closed
                let realized_pnl = old_qty * (fill_price - position.avg_price);
                position.realized_pnl += realized_pnl;
            } else if (old_qty > 0.0 && position.quantity < 0.0) ||
                      (old_qty < 0.0 && position.quantity > 0.0) {
                // Position flipped (short to long or vice versa)
                let realized_pnl = old_qty * fill_price.abs();
                position.realized_pnl += realized_pnl;
            }

            position.trades_count += 1;
            position.last_update = Utc::now().timestamp_millis();

            // Remove flat positions
            if position.is_flat() {
                self.positions.remove(symbol);
            }
        } else {
            // Create new position
            let quantity = match order.side {
                crate::executor::OrderSide::Buy => fill_qty,
                crate::executor::OrderSide::Sell => -fill_qty,
            };

            let new_position = Position::new(symbol.clone(), quantity, fill_price);
            self.positions.insert(symbol.clone(), new_position);
        }

        // Update portfolio
        self.update_portfolio();
    }

    /// Update unrealized PnL for all positions
    pub fn update_unrealized_pnl(&mut self, market_prices: &HashMap<String, f64>) {
        for (symbol, position) in self.positions.iter_mut() {
            if let Some(current_price) = market_prices.get(symbol) {
                position.calculate_unrealized_pnl(*current_price);
            }
        }
        self.update_portfolio();
    }

    /// Get position for symbol
    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Get position size for symbol
    pub fn get_position_size(&self, symbol: &str) -> f64 {
        self.positions
            .get(symbol)
            .map(|p| p.quantity)
            .unwrap_or(0.0)
    }

    /// Get all positions
    pub fn get_all_positions(&self) -> Vec<Position> {
        self.positions.values().cloned().collect()
    }

    /// Get total exposure
    pub fn get_total_exposure(&self) -> f64 {
        self.positions
            .values()
            .map(|p| p.quantity.abs() * p.avg_price)
            .sum()
    }

    /// Get total PnL
    pub fn get_total_pnl(&self) -> f64 {
        self.positions
            .values()
            .map(|p| p.total_pnl)
            .sum()
    }

    /// Get position value for symbol
    pub fn get_position_value(&self, symbol: &str) -> f64 {
        self.positions
            .get(symbol)
            .map(|p| p.get_value())
            .unwrap_or(0.0)
    }

    /// Update portfolio statistics
    fn update_portfolio(&mut self) {
        self.portfolio.update(&self.positions);
    }

    /// Get portfolio summary
    pub fn get_portfolio(&self) -> &Portfolio {
        &self.portfolio
    }
}

/// Portfolio summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub total_value: f64,
    pub total_pnl: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub positions_count: usize,
    pub last_update: i64,
}

impl Portfolio {
    pub fn new() -> Self {
        Self {
            total_value: 0.0,
            total_pnl: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            positions_count: 0,
            last_update: Utc::now().timestamp_millis(),
        }
    }

    /// Update portfolio based on current positions
    fn update(&mut self, positions: &HashMap<String, Position>) {
        self.total_value = positions
            .values()
            .map(|p| p.quantity.abs() * p.avg_price)
            .sum();

        self.total_pnl = positions
            .values()
            .map(|p| p.total_pnl)
            .sum();

        self.unrealized_pnl = positions
            .values()
            .map(|p| p.unrealized_pnl)
            .sum();

        self.realized_pnl = positions
            .values()
            .map(|p| p.realized_pnl)
            .sum();

        self.positions_count = positions.len();
        self.last_update = Utc::now().timestamp_millis();
    }

    /// Get PnL percentage
    pub fn pnl_percentage(&self) -> f64 {
        if self.total_value > 0.0 {
            (self.total_pnl / self.total_value) * 100.0
        } else {
            0.0
        }
    }
}

/// Position statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionStats {
    pub best_performer: Option<String>,
    pub worst_performer: Option<String>,
    pub biggest_position: Option<String>,
    pub most_traded: Option<String>,
    pub total_trades: u32,
    pub profitable_positions: usize,
    pub losing_positions: usize,
}

impl PositionManager {
    /// Get position statistics
    pub fn get_stats(&self) -> PositionStats {
        let positions: Vec<_> = self.positions.values().collect();

        if positions.is_empty() {
            return PositionStats {
                best_performer: None,
                worst_performer: None,
                biggest_position: None,
                most_traded: None,
                total_trades: 0,
                profitable_positions: 0,
                losing_positions: 0,
            };
        }

        let mut best_pnl = f64::MIN;
        let mut worst_pnl = f64::MAX;
        let mut best_symbol = None;
        let mut worst_symbol = None;
        let mut biggest_size = 0.0;
        let mut biggest_symbol = None;
        let mut most_trades = 0;
        let mut most_traded_symbol = None;
        let mut profitable_count = 0;
        let mut losing_count = 0;

        for position in &positions {
            // Best performer
            if position.total_pnl > best_pnl {
                best_pnl = position.total_pnl;
                best_symbol = Some(position.symbol.clone());
            }

            // Worst performer
            if position.total_pnl < worst_pnl {
                worst_pnl = position.total_pnl;
                worst_symbol = Some(position.symbol.clone());
            }

            // Biggest position
            let position_value = position.get_value();
            if position_value > biggest_size {
                biggest_size = position_value;
                biggest_symbol = Some(position.symbol.clone());
            }

            // Most traded
            if position.trades_count > most_trades {
                most_trades = position.trades_count;
                most_traded_symbol = Some(position.symbol.clone());
            }

            // PnL categories
            if position.total_pnl > 0.0 {
                profitable_count += 1;
            } else if position.total_pnl < 0.0 {
                losing_count += 1;
            }
        }

        let total_trades: u32 = positions.iter().map(|p| p.trades_count).sum();

        PositionStats {
            best_performer: best_symbol,
            worst_performer: worst_symbol,
            biggest_position: biggest_symbol,
            most_traded: most_traded_symbol,
            total_trades,
            profitable_positions: profitable_count,
            losing_positions: losing_count,
        }
    }
}

/// Position performance metrics
pub struct PositionPerformance {
    pub symbol: String,
    pub total_return: f64,
    pub win_rate: f64,
    pub avg_trade_pnl: f64,
    pub max_profit: f64,
    pub max_loss: f64,
    pub sharpe_ratio: Option<f64>,
    pub sortino_ratio: Option<f64>,
}

impl PositionManager {
    /// Calculate performance metrics for a position
    pub fn calculate_performance(&self, symbol: &str, trade_history: &[Trade]) -> Option<PositionPerformance> {
        let position = self.positions.get(symbol)?;

        if trade_history.is_empty() {
            return None;
        }

        let returns: Vec<f64> = trade_history.iter().map(|t| t.pnl).collect();
        let total_return = returns.iter().sum();
        let avg_trade_pnl = total_return / returns.len() as f64;

        let winning_trades = returns.iter().filter(|&r| *r > 0.0).count();
        let win_rate = winning_trades as f64 / returns.len() as f64;

        let max_profit = returns.iter().fold(f64::MIN, |a, &b| a.max(*b));
        let max_loss = returns.iter().fold(f64::MAX, |a, &b| a.min(*b));

        // Calculate Sharpe ratio
        let mean_return = avg_trade_pnl;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();

        let sharpe_ratio = if std_dev > 0.0 {
            Some(mean_return / std_dev * (365.0_f64).sqrt())
        } else {
            None
        };

        // Calculate Sortino ratio (downside deviation)
        let downside_variance = returns.iter()
            .filter(|&&r| *r < mean_return)
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let downside_dev = if downside_variance > 0.0 {
            downside_variance.sqrt()
        } else {
            0.0
        };

        let sortino_ratio = if downside_dev > 0.0 {
            Some(mean_return / downside_dev * (365.0_f64).sqrt())
        } else {
            None
        };

        Some(PositionPerformance {
            symbol: symbol.to_string(),
            total_return: position.total_pnl,
            win_rate,
            avg_trade_pnl,
            max_profit,
            max_loss,
            sharpe_ratio,
            sortino_ratio,
        })
    }
}

/// Trade record for performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub price: f64,
    pub pnl: f64,
    pub timestamp: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{OrderSide, OrderType};

    #[test]
    fn test_position_creation() {
        let position = Position::new("BTC/USDT".to_string(), 1.0, 50000.0);
        assert_eq!(position.symbol, "BTC/USDT");
        assert_eq!(position.quantity, 1.0);
        assert_eq!(position.avg_price, 50000.0);
        assert_eq!(position.get_side(), "long");
        assert!(!position.is_flat());
    }

    #[test]
    fn test_position_manager() {
        let mut manager = PositionManager::new();
        assert_eq!(manager.get_all_positions().len(), 0);

        // Add a buy order
        let order = Order::new(
            "BTC/USDT".to_string(),
            OrderSide::Buy,
            OrderType::Market,
            2.0,
            Some(50000.0),
            None,
            "GTC".to_string(),
        );

        // Simulate fill
        let mut filled_order = order.clone();
        filled_order.filled_quantity = 2.0;
        filled_order.avg_fill_price = Some(50000.0);

        manager.update_position(&filled_order);
        assert_eq!(manager.get_all_positions().len(), 1);
        assert_eq!(manager.get_position_size("BTC/USDT"), 2.0);
    }

    #[test]
    fn test_portfolio_update() {
        let mut portfolio = Portfolio::new();
        assert_eq!(portfolio.total_value, 0.0);

        let mut positions = HashMap::new();
        positions.insert("BTC/USDT".to_string(), Position::new("BTC/USDT".to_string(), 1.0, 50000.0));

        portfolio.update(&positions);
        assert_eq!(portfolio.total_value, 50000.0);
        assert_eq!(portfolio.positions_count, 1);
    }
}