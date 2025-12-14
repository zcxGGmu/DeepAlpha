//! Portfolio management module

use crate::executor::position::{Position, PositionManager};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Portfolio configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioConfig {
    pub max_positions: usize,
    pub max_position_size: f64,
    pub max_leverage: f64,
    pub rebalance_threshold: f64,
    pub sectors: HashMap<String, f64>, // sector name -> allocation percentage
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        let mut sectors = HashMap::new();
        sectors.insert("Technology".to_string(), 0.4);
        sectors.insert("Finance".to_string(), 0.3);
        sectors.insert("Healthcare".to_string(), 0.3);

        Self {
            max_positions: 20,
            max_position_size: 0.1, // 10% of portfolio
            max_leverage: 2.0,
            rebalance_threshold: 0.05, // 5%
            sectors,
        }
    }
}

/// Portfolio rebalancing recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceRecommendation {
    pub symbol: String,
    pub current_weight: f64,
    pub target_weight: f64,
    pub current_quantity: f64,
    pub target_quantity: f64,
    pub action: RebalanceAction,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RebalanceAction {
    Buy,
    Sell,
    Hold,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
}

/// Portfolio manager
pub struct PortfolioManager {
    config: PortfolioConfig,
    base_currency: String,
    start_date: DateTime<Utc>,
    benchmarks: HashMap<String, Vec<f64>>, // symbol -> price history
}

impl PortfolioManager {
    /// Create a new portfolio manager
    pub fn new(config: PortfolioConfig) -> Self {
        Self {
            config,
            base_currency: "USDT".to_string(),
            start_date: Utc::now(),
            benchmarks: HashMap::new(),
        }
    }

    /// Calculate current allocation weights
    pub fn calculate_allocations(&self, positions: &HashMap<String, Position>, prices: &HashMap<String, f64>) -> HashMap<String, f64> {
        let mut allocations = HashMap::new();
        let total_value: f64 = positions
            .values()
            .map(|p| p.quantity.abs() * p.avg_price)
            .sum();

        if total_value > 0.0 {
            for (symbol, position) in positions {
                let position_value = position.quantity.abs() * position.avg_price;
                let weight = position_value / total_value;
                allocations.insert(symbol.clone(), weight);
            }
        }

        allocations
    }

    /// Generate rebalancing recommendations
    pub fn generate_rebalance_recommendations(
        &self,
        positions: &HashMap<String, Position>,
        prices: &HashMap<String, f64>,
        total_value: f64,
    ) -> Vec<RebalanceRecommendation> {
        let allocations = self.calculate_allocations(positions, prices);
        let mut recommendations = Vec::new();

        for (symbol, position) in positions {
            let current_weight = allocations.get(symbol).unwrap_or(&0.0);
            let target_weight = self.get_target_weight(symbol);
            let deviation = (current_weight - target_weight).abs();

            if deviation > self.config.rebalance_threshold {
                let current_quantity = position.quantity;
                let target_value = total_value * target_weight;
                let target_quantity = if prices.get(symbol).unwrap_or(&0.0) > 0.0 {
                    target_value / prices.get(symbol).unwrap()
                } else {
                    0.0
                };

                let action = if target_quantity > current_quantity {
                    RebalanceAction::Buy
                } else if target_quantity < current_quantity {
                    RebalanceAction::Sell
                } else {
                    RebalanceAction::Hold
                };

                let priority = (deviation * 100.0) as u8; // Convert to priority score

                recommendations.push(RebalanceRecommendation {
                    symbol: symbol.clone(),
                    current_weight: *current_weight,
                    target_weight,
                    current_quantity,
                    target_quantity,
                    action,
                    priority,
                });
            }
        }

        // Sort by priority (highest deviation first)
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        recommendations
    }

    /// Get target weight for a symbol
    fn get_target_weight(&self, symbol: &str) -> f64 {
        // Default equal weight if not specified
        1.0 / self.config.max_positions as f64
    }

    /// Calculate portfolio metrics
    pub fn calculate_metrics(
        &self,
        positions: &HashMap<String, Position>,
        price_history: &HashMap<String, Vec<f64>>,
    ) -> PerformanceMetrics {
        let returns = self.calculate_returns(positions, price_history);

        if returns.is_empty() {
            return PerformanceMetrics {
                total_return: 0.0,
                annualized_return: 0.0,
                volatility: 0.0,
                sharpe_ratio: 0.0,
                max_drawdown: 0.0,
                calmar_ratio: 0.0,
                win_rate: 0.0,
                profit_factor: 1.0,
            };
        }

        let total_return = returns.iter().sum();
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        let volatility = variance.sqrt();

        // Annualize returns (assuming daily returns)
        let trading_days = 252;
        let annualized_return = total_return * (trading_days as f64 / returns.len() as f64);
        let annualized_volatility = volatility * (trading_days as f64).sqrt();

        // Sharpe ratio (assuming risk-free rate of 0%)
        let sharpe_ratio = if annualized_volatility > 0.0 {
            annualized_return / annualized_volatility
        } else {
            0.0
        };

        // Maximum drawdown
        let mut cumulative_value = 1.0;
        let mut peak = 1.0;
        let mut max_drawdown = 0.0;

        for ret in &returns {
            cumulative_value *= (1.0 + ret);
            if cumulative_value > peak {
                peak = cumulative_value;
            }
            let drawdown = (peak - cumulative_value) / peak;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Calmar ratio (annualized return / max drawdown)
        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        // Win rate
        let winning_returns = returns.iter().filter(|&&r| *r > 0.0).count();
        let win_rate = winning_returns as f64 / returns.len() as f64;

        // Profit factor (average win / average loss)
        let winning_trades: Vec<f64> = returns.iter().filter(|&&r| *r > 0.0).cloned().collect();
        let losing_trades: Vec<f64> = returns.iter().filter(|&&r| *r < 0.0).cloned().collect();

        let profit_factor = if losing_trades.is_empty() {
            f64::INFINITY
        } else {
            let avg_win = if winning_trades.is_empty() {
                0.0
            } else {
                winning_trades.iter().sum::<f64>() / winning_trades.len() as f64
            };
            let avg_loss = losing_trades.iter().sum::<f64>() / losing_trades.len() as f64;
            if avg_loss == 0.0 {
                1.0
            } else {
                avg_win / avg_loss.abs()
            }
        };

        PerformanceMetrics {
            total_return,
            annualized_return,
            volatility,
            sharpe_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            profit_factor,
        }
    }

    /// Calculate returns from price history
    fn calculate_returns(
        &self,
        positions: &HashMap<String, Position>,
        price_history: &HashMap<String, Vec<f64>>,
    ) -> Vec<f64> {
        // For simplicity, use BTC/USDT as benchmark if available
        if let Some(prices) = price_history.get("BTC/USDT") {
            let mut returns = Vec::new();
            for i in 1..prices.len() {
                let ret = (prices[i] - prices[i - 1]) / prices[i - 1];
                returns.push(ret);
            }
            returns
        } else {
            Vec::new()
        }
    }

    /// Check if portfolio needs rebalancing
    pub fn needs_rebalancing(&self, positions: &HashMap<String, Position>, prices: &HashMap<String, f64>) -> bool {
        let recommendations = self.generate_rebalance_recommendations(positions, prices, 100000.0);
        !recommendations.is_empty()
    }

    /// Get sector allocation
    pub fn get_sector_allocation(&self, positions: &HashMap<String, Position>) -> HashMap<String, f64> {
        let mut sector_allocation = HashMap::new();
        let total_value: f64 = positions
            .values()
            .map(|p| p.quantity.abs() * p.avg_price)
            .sum();

        if total_value > 0.0 {
            // Simple symbol to sector mapping (would normally come from metadata)
            for (symbol, position) in positions {
                let sector = self.get_symbol_sector(symbol);
                let entry = sector_allocation.entry(sector).or_insert(0.0);
                *entry += (position.quantity.abs() * position.avg_price) / total_value;
            }
        }

        sector_allocation
    }

    /// Get sector for a symbol
    fn get_symbol_sector(&self, symbol: &str) -> String {
        // Simple mapping - in practice, this would come from external data
        if symbol.starts_with("BTC") || symbol.starts_with("ETH") {
            "Cryptocurrency".to_string()
        } else if symbol.contains("AAPL") || symbol.contains("MSFT") {
            "Technology".to_string()
        } else if symbol.contains("JPM") || symbol.contains("BAC") {
            "Finance".to_string()
        } else if symbol.contains("JNJ") || symbol.contains("PFE") {
            "Healthcare".to_string()
        } else {
            "Other".to_string()
        }
    }
}

/// Portfolio optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub optimal_weights: HashMap<String, f64>,
    pub expected_return: f64,
    pub expected_risk: f64,
    pub sharpe_ratio: f64,
}

/// Portfolio optimizer (simplified mean-variance optimization)
pub struct PortfolioOptimizer {
    risk_free_rate: f64,
}

impl PortfolioOptimizer {
    pub fn new(risk_free_rate: f64) -> Self {
        Self { risk_free_rate }
    }

    /// Optimize portfolio weights (simplified implementation)
    pub fn optimize(
        &self,
        expected_returns: &HashMap<String, f64>,
        covariance_matrix: &HashMap<(String, String), f64>,
    ) -> OptimizationResult {
        let symbols: Vec<String> = expected_returns.keys().cloned().collect();
        let n = symbols.len();

        // For simplicity, use equal weights
        let weight = 1.0 / n as f64;
        let mut optimal_weights = HashMap::new();
        for symbol in &symbols {
            optimal_weights.insert(symbol.clone(), weight);
        }

        // Calculate expected portfolio return and risk
        let expected_return = expected_returns
            .values()
            .sum::<f64>() / n as f64;

        // Calculate portfolio variance (simplified)
        let portfolio_variance = symbols
            .iter()
            .map(|i| {
                symbols
                    .iter()
                    .map(|j| {
                        covariance_matrix
                            .get(&(i.clone(), j.clone()))
                            .unwrap_or(&0.0)
                    })
                    .sum::<f64>()
            })
            .sum::<f64>() / (n * n) as f64;

        let portfolio_risk = portfolio_variance.sqrt();
        let sharpe_ratio = if portfolio_risk > 0.0 {
            (expected_return - self.risk_free_rate) / portfolio_risk
        } else {
            0.0
        };

        OptimizationResult {
            optimal_weights,
            expected_return,
            expected_risk: portfolio_variance.sqrt(),
            sharpe_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_config_default() {
        let config = PortfolioConfig::default();
        assert_eq!(config.max_positions, 20);
        assert_eq!(config.max_position_size, 0.1);
        assert_eq!(config.max_leverage, 2.0);
    }

    #[test]
    fn test_portfolio_manager() {
        let config = PortfolioConfig::default();
        let manager = PortfolioManager::new(config);
        assert_eq!(manager.base_currency, "USDT");
    }

    #[test]
    fn test_rebalance_recommendation() {
        let rec = RebalanceRecommendation {
            symbol: "BTC/USDT".to_string(),
            current_weight: 0.15,
            target_weight: 0.10,
            current_quantity: 2.0,
            target_quantity: 1.0,
            action: RebalanceAction::Sell,
            priority: 50,
        };

        assert_eq!(rec.symbol, "BTC/USDT");
        assert_eq!(rec.action, RebalanceAction::Sell);
        assert_eq!(rec.priority, 50);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics {
            total_return: 0.15,
            annualized_return: 0.12,
            volatility: 0.20,
            sharpe_ratio: 0.6,
            max_drawdown: 0.05,
            calmar_ratio: 2.4,
            win_rate: 0.6,
            profit_factor: 1.5,
        };

        assert_eq!(metrics.total_return, 0.15);
        assert_eq!(metrics.sharpe_ratio, 0.6);
    }
}