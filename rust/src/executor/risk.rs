//! Risk management for trading execution

use crate::common::{Result, DeepAlphaError};
use crate::executor::{Order, OrderSide, OrderType};
use crate::executor::position::PositionManager;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Risk check types
#[derive(Debug, Clone, PartialEq)]
pub enum RiskCheck {
    PositionSize,
    Exposure,
    Leverage,
    DailyLoss,
    OrderRate,
    PriceDeviation,
    ShortSelling,
    InvalidSymbol,
    InsufficientBalance,
}

/// Risk limit types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLimitType {
    MaxPositionSize { symbol: String, max_size: f64 },
    MaxExposure { max_exposure: f64 },
    MaxLeverage { max_leverage: f64 },
    MaxDailyLoss { max_loss: f64 },
    MaxOrderRate { max_per_second: u32 },
    MaxPriceDeviation { max_deviation_pct: f64 },
    MinBalance { min_balance: f64 },
}

/// Risk limit configuration
#[derive(Debug, Clone)]
pub struct RiskLimit {
    pub limit_type: RiskLimitType,
    pub enabled: bool,
    pub strict: bool, // If true, violation prevents trading
    pub warning_threshold: Option<f64>, // Warning threshold as percentage of limit
}

impl RiskLimit {
    pub fn new(limit_type: RiskLimitType) -> Self {
        Self {
            limit_type,
            enabled: true,
            strict: true,
            warning_threshold: Some(0.8),
        }
    }

    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// Risk manager
pub struct RiskManager {
    limits: HashMap<RiskCheck, RiskLimit>,
    daily_stats: HashMap<String, DailyStats>,
    order_counts: Vec<(i64, u32)>, // (timestamp, count) for order rate limiting
    last_cleanup: i64,
    current_balance: f64,
    total_exposure: f64,
}

impl RiskManager {
    /// Create a new risk manager with default limits
    pub fn new() -> Self {
        let mut limits = HashMap::new();

        // Position size limits
        limits.insert(
            RiskCheck::PositionSize,
            RiskLimit::new(RiskLimitType::MaxPositionSize {
                symbol: "default".to_string(),
                max_size: 10.0,
            }).strict(true)
        );

        // Exposure limits
        limits.insert(
            RiskCheck::Exposure,
            RiskLimit::new(RiskLimitType::MaxExposure {
                max_exposure: 100000.0,
            }).strict(true)
        );

        // Leverage limits
        limits.insert(
            RiskCheck::Leverage,
            RiskLimit::new(RiskLimitType::MaxLeverage {
                max_leverage: 10.0,
            }).strict(true)
        );

        // Daily loss limits
        limits.insert(
            RiskCheck::DailyLoss,
            RiskLimit::new(RiskLimitType::MaxDailyLoss {
                max_loss: 10000.0,
            }).strict(false)
        );

        // Order rate limits
        limits.insert(
            RiskCheck::OrderRate,
            RiskLimit::new(RiskLimitType::MaxOrderRate {
                max_per_second: 100,
            }).strict(true)
        );

        // Price deviation limits
        limits.insert(
            RiskCheck::PriceDeviation,
            RiskLimit::new(RiskLimitType::MaxPriceDeviation {
                max_deviation_pct: 5.0, // 5%
            }).strict(false)
        );

        // Minimum balance
        limits.insert(
            RiskCheck::InsufficientBalance,
            RiskLimit::new(RiskLimitType::MinBalance {
                min_balance: 1000.0,
            }).strict(true)
        );

        Self {
            limits,
            daily_stats: HashMap::new(),
            order_counts: Vec::new(),
            last_cleanup: 0,
            current_balance: 100000.0, // Default starting balance
            total_exposure: 0.0,
        }
    }

    /// Add or update a risk limit
    pub fn set_limit(&mut self, risk_check: RiskCheck, limit: RiskLimit) {
        self.limits.insert(risk_check, limit);
    }

    /// Update account balance
    pub fn update_balance(&mut self, new_balance: f64) {
        self.current_balance = new_balance;
    }

    /// Check if an order passes all risk checks
    pub async fn check_order(
        &self,
        order: &Order,
        position_manager: &PositionManager,
    ) -> Result<()> {
        // Clean up old order counts periodically
        let now = chrono::Utc::now().timestamp();
        if now - self.last_cleanup > 3600 { // Clean every hour
            self.cleanup_old_data(now);
        }

        // Check position size
        self.check_position_size(order, position_manager)?;

        // Check exposure
        self.check_exposure(order, position_manager)?;

        // Check leverage
        self.check_leverage(order, position_manager)?;

        // Check daily loss
        self.check_daily_loss(order.symbol.as_str())?;

        // Check order rate
        self.check_order_rate(now)?;

        // Check price deviation (if limit order)
        if order.order_type == OrderType::Limit {
            self.check_price_deviation(order)?;
        }

        // Check insufficient balance
        self.check_insufficient_balance(order)?;

        Ok(())
    }

    /// Check position size limit
    fn check_position_size(
        &self,
        order: &Order,
        position_manager: &PositionManager,
    ) -> Result<()> {
        if let Some(limit) = self.limits.get(&RiskCheck::PositionSize) {
            if !limit.enabled {
                return Ok(());
            }

            if let RiskLimitType::MaxPositionSize { max_size, .. } = &limit.limit_type {
                let current_position = position_manager.get_position_size(&order.symbol);
                let new_position = if order.side == OrderSide::Buy {
                    current_position + order.quantity
                } else {
                    current_position - order.quantity
                };

                let abs_position = new_position.abs();

                if abs_position > *max_size {
                    let error_msg = format!(
                        "Position size limit exceeded for {}: {:.4} > {:.4}",
                        order.symbol, abs_position, max_size
                    );
                    return if limit.strict {
                        Err(DeepAlphaError::InvalidInput(error_msg))
                    } else {
                        warn!("{}", error_msg);
                        Ok(())
                    };
                }
            }
        }
        Ok(())
    }

    /// Check total exposure limit
    fn check_exposure(
        &self,
        order: &Order,
        position_manager: &PositionManager,
    ) -> Result<()> {
        if let Some(limit) = self.limits.get(&RiskCheck::Exposure) {
            if !limit.enabled {
                return Ok(());
            }

            if let RiskLimitType::MaxExposure { max_exposure } = &limit.limit_type {
                let current_exposure = position_manager.get_total_exposure();
                let order_value = order.price.unwrap_or(0.0) * order.quantity;
                let new_exposure = current_exposure + order_value;

                if new_exposure > *max_exposure {
                    let error_msg = format!(
                        "Exposure limit exceeded: {:.2} > {:.2}",
                        new_exposure, max_exposure
                    );
                    return if limit.strict {
                        Err(DeepAlphaError::InvalidInput(error_msg))
                    } else {
                        warn!("{}", error_msg);
                        Ok(())
                    };
                }
            }
        }
        Ok(())
    }

    /// Check leverage limit
    fn check_leverage(
        &self,
        order: &Order,
        position_manager: &PositionManager,
    ) -> Result<()> {
        if let Some(limit) = self.limits.get(&RiskCheck::Leverage) {
            if !limit.enabled {
                return Ok(());
            }

            if let RiskLimitType::MaxLeverage { max_leverage } = &limit.limit_type {
                let position_value = position_manager.get_position_value(&order.symbol);
                let new_position_value = if order.side == OrderSide::Buy {
                    position_value + (order.price.unwrap_or(0.0) * order.quantity)
                } else {
                    position_value - (order.price.unwrap_or(0.0) * order.quantity)
                };

                if new_position_value > 0.0 {
                    let leverage = new_position_value / self.current_balance;
                    if leverage > *max_leverage {
                        let error_msg = format!(
                            "Leverage limit exceeded: {:.2}x > {:.2}x",
                            leverage, max_leverage
                        );
                        return if limit.strict {
                            Err(DeepAlphaError::InvalidInput(error_msg))
                        } else {
                            warn!("{}", error_msg);
                            Ok(())
                        };
                    }
                }
            }
        }
        Ok(())
    }

    /// Check daily loss limit
    fn check_daily_loss(&self, symbol: &str) -> Result<()> {
        if let Some(limit) = self.limits.get(&RiskCheck::DailyLoss) {
            if !limit.enabled {
                return Ok(());
            }

            if let RiskLimitType::MaxDailyLoss { max_loss } = &limit.limit_type {
                let daily_pnl = self.daily_stats
                    .get(symbol)
                    .map(|stats| stats.realized_pnl + stats.unrealized_pnl)
                    .unwrap_or(0.0);

                if daily_pnl < -*max_loss {
                    let error_msg = format!(
                        "Daily loss limit exceeded for {}: {:.2} > {:.2}",
                        symbol, daily_pnl.abs(), max_loss
                    );
                    return if limit.strict {
                        Err(DeepAlphaError::InvalidInput(error_msg))
                    } else {
                        warn!("{}", error_msg);
                        Ok(())
                    };
                }
            }
        }
        Ok(())
    }

    /// Check order rate limit
    fn check_order_rate(&self, now: i64) -> Result<()> {
        if let Some(limit) = self.limits.get(&RiskCheck::OrderRate) {
            if !limit.enabled {
                return Ok(());
            }

            if let RiskLimitType::MaxOrderRate { max_per_second } = &limit.limit_type {
                // Count orders in the last second
                let recent_orders = self
                    .order_counts
                    .iter()
                    .filter(|(timestamp, _)| now - timestamp <= 1000)
                    .map(|(_, count)| count)
                    .sum();

                if recent_orders >= *max_per_second {
                    let error_msg = format!(
                        "Order rate limit exceeded: {}/s > {}/s",
                        recent_orders, max_per_second
                    );
                    return if limit.strict {
                        Err(DeepAlphaError::InvalidInput(error_msg))
                    } else {
                        warn!("{}", error_msg);
                        Ok(())
                    };
                }
            }
        }
        Ok(())
    }

    /// Check price deviation for limit orders
    fn check_price_deviation(&self, order: &Order) -> Result<()> {
        if let (Some(limit_price), Some(_)) = (order.price, order.stop_price) {
            // TODO: Check against market price
            // For now, just validate that price is reasonable
            if limit_price <= 0.0 || limit_price > 1000000.0 {
                return Err(DeepAlphaError::InvalidInput(
                    format!("Invalid limit price: {}", limit_price)
                ));
            }
        }
        Ok(())
    }

    /// Check if account has sufficient balance
    fn check_insufficient_balance(&self, order: &Order) -> Result<()> {
        if let Some(limit) = self.limits.get(&RiskCheck::InsufficientBalance) {
            if !limit.enabled {
                return Ok(());
            }

            if let RiskLimitType::MinBalance { min_balance } = &limit.limit_type {
                let required_margin = order.price.unwrap_or(0.0) * order.quantity;
                let available_balance = self.current_balance - self.total_exposure;

                if available_balance < required_margin + min_balance {
                    let error_msg = format!(
                        "Insufficient balance: required {:.2}, available {:.2}, min {:.2}",
                        required_margin, available_balance, min_balance
                    );
                    return if limit.strict {
                        Err(DeepAlphaError::InvalidInput(error_msg))
                    } else {
                        warn!("{}", error_msg);
                        Ok(())
                    };
                }
            }
        }
        Ok(())
    }

    /// Update daily statistics
    pub fn update_daily_stats(&mut self, symbol: &str, realized_pnl: f64, unrealized_pnl: f64) {
        let stats = self.daily_stats
            .entry(symbol.to_string())
            .or_insert_with(DailyStats::new);

        stats.realized_pnl = realized_pnl;
        stats.unrealized_pnl = unrealized_pnl;
        stats.last_update = chrono::Utc::now().timestamp();
    }

    /// Record an order
    pub fn record_order(&mut self, timestamp: i64) {
        self.order_counts.push((timestamp, 1));
    }

    /// Clean up old data
    fn cleanup_old_data(&mut self, now: i64) {
        // Clean old order counts (older than 1 hour)
        self.order_counts.retain(|(timestamp, _)| now - timestamp < 3600000);

        // Clean old daily stats (older than 1 day)
        let one_day_ago = now - 86400000;
        self.daily_stats.retain(|_, stats| stats.last_update > one_day_ago);

        self.last_cleanup = now;
    }

    /// Get all risk limits
    pub fn get_limits(&self) -> &HashMap<RiskCheck, RiskLimit> {
        &self.limits
    }

    /// Get daily statistics
    pub fn get_daily_stats(&self) -> &HashMap<String, DailyStats> {
        &self.daily_stats
    }
}

/// Daily trading statistics
#[derive(Debug, Clone)]
pub struct DailyStats {
    pub date: String,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub trades_count: u32,
    pub volume: f64,
    pub last_update: i64,
}

impl DailyStats {
    pub fn new() -> Self {
        let now = chrono::Utc::now();
        Self {
            date: now.format("%Y-%m-%d").to_string(),
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            trades_count: 0,
            volume: 0.0,
            last_update: now.timestamp_millis(),
        }
    }

    pub fn total_pnl(&self) -> f64 {
        self.realized_pnl + self.unrealized_pnl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_manager_creation() {
        let manager = RiskManager::new();
        assert_eq!(manager.limits.len(), 7);
        assert!(manager.limits.contains_key(&RiskCheck::PositionSize));
        assert!(manager.limits.contains_key(&RiskCheck::Exposure));
    }

    #[test]
    fn test_risk_limit() {
        let limit = RiskLimit::new(RiskLimitType::MaxPositionSize {
            symbol: "BTC/USDT".to_string(),
            max_size: 10.0,
        });

        assert!(limit.enabled);
        assert!(limit.strict);
    }

    #[test]
    fn test_daily_stats() {
        let stats = DailyStats::new();
        assert_eq!(stats.realized_pnl, 0.0);
        assert_eq!(stats.total_pnl(), 0.0);
    }

    #[test]
    fn test_position_size_check() {
        let manager = RiskManager::new();
        let position_manager = PositionManager::new();

        let order = Order::new(
            "BTC/USDT".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            15.0, // Exceeds default limit of 10
            Some(50000.0),
            None,
            "GTC".to_string(),
        );

        // This should fail due to position size limit
        let result = tokio::block_on(
            manager.check_order(&order, &position_manager)
        );

        assert!(result.is_err());
    }
}