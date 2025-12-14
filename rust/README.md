# DeepAlpha Rust 模块

这是 DeepAlpha 项目的 Rust 性能模块实现。

## 构建和安装

### 前置要求

- Rust 1.74+
- Python 3.8+
- maturin (Python 构建工具)

### 构建步骤

```bash
# 安装 maturin
pip install maturin

# 开发模式构建
maturin develop

# 或使用构建脚本
python build.py
```

## 使用示例

```python
from deepalpha_rust import TechnicalIndicators

# 创建指标计算器
prices = [44.0, 44.5, 45.0, 44.75, 45.25, 45.5, 45.75, 46.0]
indicators = TechnicalIndicators(prices)

# 计算各种指标
sma = indicators.sma(period=5)  # 简单移动平均
ema = indicators.ema(period=5)  # 指数移动平均
rsi = indicators.rsi(period=5)  # 相对强弱指数

# MACD 指标
macd_result = indicators.macd(fast=5, slow=8, signal=3)
print(f"MACD: {macd_result['macd']}")

# 布林带
bb_result = indicators.bollinger_bands(period=5, std_dev=2.0)
print(f"Upper band: {bb_result['upper']}")
```

## 测试

```bash
# Rust 单元测试
cargo test

# Python 集成测试
pytest tests/
```

## 性能基准

技术指标计算性能（50,000 K线数据）：
- SMA: < 10ms
- EMA: < 15ms
- RSI: < 20ms
- MACD: < 25ms
- 布林带: < 30ms

## 项目结构

```
rust/
├── src/
│   ├── indicators/      # 技术指标实现
│   │   ├── basic.rs     # 基础指标 (SMA, EMA, WMA)
│   │   ├── momentum.rs  # 动量指标 (RSI, Stochastic)
│   │   ├── volatility.rs# 波动率指标 (Bollinger Bands, ATR)
│   │   └── trend.rs     # 趋势指标 (MACD, Ichimoku)
│   ├── websocket/       # WebSocket 管理器
│   ├── stream/          # 数据流处理
│   ├── executor/        # 交易执行引擎
│   └── common/          # 通用工具
├── tests/               # Python 测试
└── python/              # Python 包装器
```