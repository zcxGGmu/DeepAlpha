# DeepAlpha - LLM-Driven Multi-Agent Quantitative Trading System

![DeepAlpha](https://img.shields.io/badge/DeepAlpha-v1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Rust](https://img.shields.io/badge/Rust-1.74+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

DeepAlpha is a Python/Rust hybrid architecture quantitative trading system driven by Large Language Models (LLMs), featuring a multi-agent architecture that makes trading decisions through AI models and integrates an ultra-high-performance Rust execution engine.

## 🌟 Core Features

### Multi-Agent System
- **Technical Agent**: Professional technical indicator analysis (EMA, RSI, MACD, ATR, etc.)
- **Pattern Agent**: Intelligent candlestick pattern recognition (head & shoulders, engulfing, evening star, etc.)
- **Trend Agent**: Multi-timeframe trend analysis and prediction
- **Risk Agent**: Intelligent risk assessment and position management
- **News Agent**: News sentiment analysis and market impact assessment

### AI Decision Engine
- Support for multiple LLM Providers (OpenAI, Anthropic Claude, DeepSeek, etc.)
- Intelligent voting mechanism and decision aggregation
- Decision caching and historical memory
- Dynamic weight adjustment

### 🚀 Rust Performance Optimization Modules
Complete Rust migration achieves ultra-high-performance core components:

1. **Technical Indicators Engine** (`rust/src/indicators`)
   - Supports 50+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
   - **Performance**: > 50,000 candles/second processing speed
   - **Optimization**: SIMD instruction set optimization, multi-core parallel computing
   - **Memory**: Zero-copy operations, 30-50% memory reduction

2. **WebSocket Manager** (`rust/src/websocket`)
   - Supports 10,000+ concurrent WebSocket connections
   - Automatic reconnection and heartbeat mechanism
   - Message broadcasting and subscription management
   - **Latency**: < 1ms message processing

3. **Data Stream Processor** (`rust/src/stream`)
   - Real-time data stream processing pipeline
   - Supports 100,000+ data points/second
   - Data validation and transformation
   - Asynchronous batch processing optimization

4. **Trading Execution Engine** (`rust/src/executor`)
   - **Ultra-low latency**: Average 50μs order submission, P99 < 100μs
   - **High throughput**: 1,000+ orders/second processing capability
   - Risk management and real-time control
   - Position and portfolio management

### Risk Management
- Real-time risk checking and position control
- Strict position management (1-2% net assets per trade)
- Dynamic stop-loss and take-profit mechanisms
- Maximum drawdown protection
- Portfolio risk diversification

### High-Performance Architecture
- Asynchronous concurrent processing (asyncio)
- Real-time WebSocket data streams
- **Rust Performance Modules**: 3-5x performance improvement, 30-50% memory optimization
- Distributed deployment support
- Comprehensive monitoring and logging system

## 🚀 Quick Start

### Requirements

- Python 3.10+
- Rust 1.74+ (for performance modules)
- Redis (for caching)
- PostgreSQL/SQLite (for data storage)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the project**
```bash
git clone https://github.com/zcxGGmu/DeepAlpha.git
cd DeepAlpha
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
```

3. **Install Python dependencies**
```bash
pip install -r requirements/prod.txt
```

4. **Install Rust Performance Modules**
```bash
# Quick installation script (recommended)
./scripts/install_rust.sh

# Or manual installation
cd rust
pip install maturin
maturin develop --release
```

5. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env file, fill in your API keys and configuration
```

6. **Initialize database**
```bash
python scripts/migrate.py
```

7. **Start the system**
```bash
python scripts/start.py
```

### Rust Performance Module Verification

After installation, run performance benchmarks to verify Rust module performance:

```bash
# Technical indicators performance test
cd rust/tests
python test_indicators_performance.py

# WebSocket performance test
python test_websocket_performance.py

# Data stream performance test
python test_stream_performance.py

# Execution engine performance test
python test_executor_performance.py
```

**Expected Performance Improvements**:
- Technical indicators: 50,000+ candles/second (5-10x improvement over Python)
- WebSocket connections: 10,000+ concurrent connections (< 1ms latency)
- Data stream processing: 100,000+ data points/second
- Order execution: 1,000+ orders/second (average latency < 50μs)
- Memory usage: 30-50% reduction

### Docker Deployment

```bash
# Build image with Rust performance modules
docker build -t deepalpha .

# Start services
docker-compose up -d
```

## 📖 System Architecture

### Overall Architecture Overview

```mermaid
graph TB
    %% External Data Sources
    subgraph "External Data Sources"
        B1[Binance API]
        B2[Other Exchanges]
        B3[News Feeds]
        B4[Sentiment APIs]
    end

    %% Data Collection Layer
    subgraph "Data Collection"
        C1[WebSocket Connectors]
        C2[REST API Pollers]
        C3[News Aggregator]
    end

    %% Rust High-Performance Layer
    subgraph "Rust Performance Layer"
        D1[WebSocket Manager<br/>10K+ Connections<br/><1ms Latency]
        D2[Stream Processor<br/>100K+ Points/sec]
        D3[Technical Indicators<br/>50K+ Candles/sec]
        D4[Execution Engine<br/>1K+ Orders/sec<br/><50μs Latency]
    end

    %% Storage Layer
    subgraph "Storage Layer"
        E1[(PostgreSQL<br/>Time Series Data)]
        E2[(Redis<br/>Cache & Session)]
        E3[(InfluxDB<br/>Market Data)]
    end

    %% Python Business Logic Layer
    subgraph "Python Business Logic"
        subgraph "Multi-Agent System"
            F1[Technical Agent<br/>TA Analysis]
            F2[Pattern Agent<br/>30+ Patterns]
            F3[Trend Agent<br/>Multi-Timeframe]
            F4[Risk Agent<br/>Risk Assessment]
            F5[News Agent<br/>Sentiment Analysis]
        end

        subgraph "Decision Engine"
            G1[Decision Aggregator<br/>Weighted Voting]
            G2[LLM Integration<br/>Multiple Providers]
            G3[Decision Cache<br/>Redis-backed]
        end
    end

    %% Execution Layer
    subgraph "Execution Layer"
        I1[Order Manager]
        I2[Position Manager]
        I3[Risk Manager]
    end

    %% Gateway Layer
    subgraph "Gateway Layer"
        J1[Exchange Gateway]
        J2[Notification Gateway]
        J3[API Gateway]
    end

    %% Interface Layer
    subgraph "Interface Layer"
        K1[REST API<br/>FastAPI]
        K2[WebSocket Server]
        K3[Web Dashboard]
    end

    %% Data Flow
    B1 --> C1
    B2 --> C2
    B3 --> C3
    C1 --> D1
    C2 --> D2
    C3 --> D2
    D1 --> E2
    D2 --> D3
    D3 --> E2
    E1 --> F1
    E2 --> F2
    E3 --> F3
    F1 --> G1
    F2 --> G1
    F3 --> G1
    F4 --> G1
    F5 --> G1
    G2 --> G1
    G1 --> I1
    I1 --> D4
    D4 --> J1
    I1 --> I2
    I1 --> I3
    K1 --> J3
    K2 --> K3

    classDef rust fill:#ff6b6b,stroke:#c92a2a,color:#fff
    classDef python fill:#4dabf7,stroke:#1864ab,color:#fff
    classDef storage fill:#69db7c,stroke:#2f9e44,color:#fff
    classDef external fill:#ffd43b,stroke:#fab005,color:#000

    class D1,D2,D3,D4 rust
    class F1,F2,F3,F4,F5,G1,G2,G3 python
    class E1,E2,E3 storage
    class B1,B2,B3,B4 external
```

### Core Components Details

#### 🚀 Rust Performance Optimization Layer

1. **WebSocket Manager** (`rust/src/websocket/`)
   - Supports 10,000+ concurrent connections
   - Message latency < 1ms
   - Automatic reconnection and heartbeat mechanism
   - JWT authentication and permission control

2. **Data Stream Processor** (`rust/src/stream/`)
   - 100,000+ data points/second processing capability
   - Zero-copy ring buffer
   - Real-time data validation and cleaning
   - Asynchronous batch processing optimization

3. **Technical Indicators Engine** (`rust/src/indicators/`)
   - 50+ technical indicators support
   - SIMD instruction set optimization
   - 50,000+ candles/second calculation speed
   - Batch calculation support

4. **Trading Execution Engine** (`rust/src/executor/`)
   - Ultra-low latency: average 50μs
   - 1,000+ orders/second throughput
   - Real-time risk control
   - Position and portfolio management

#### 🤖 Python Intelligent Agent System

1. **Technical Agent**
   - Professional technical indicator analysis
   - Multi-timeframe support
   - Indicator signal aggregation

2. **Pattern Agent**
   - 30+ candlestick pattern recognition
   - Support and resistance analysis
   - Pattern strength assessment

3. **Trend Agent**
   - Multi-timeframe trend analysis
   - Trend consistency check
   - Trend reversal prediction

4. **Risk Agent**
   - Real-time risk assessment
   - VaR calculation
   - Hedging recommendations

5. **News Agent**
   - Real-time news analysis
   - Sentiment scoring
   - Event impact assessment

### Detailed Architecture Documentation

For complete system architecture documentation, please refer to: [System Architecture](./docs/system-architecture.md)

This document includes:
- Detailed component descriptions
- Data flow analysis
- Performance optimization strategies
- Deployment architecture guide

## 📊 Sub-Function Module Architecture

### 1. Market Data Module Architecture

```mermaid
graph TB
    subgraph "Market Data Module Architecture"
        subgraph "Data Source Layer"
            DS1[Binance WebSocket]
            DS2[REST API Polling]
            DS3[News Data Source]
            DS4[Sentiment Data Source]
        end

        subgraph "Data Collection Layer"
            DC1[WebSocket Connector]
            DC2[REST Poller]
            DC3[News Aggregator]
            DC4[Data Validator]
        end

        subgraph "Data Stream Processing Layer"
            DP1[Real-time Data Stream<br/>100K+ pts/s]
            DP2[Historical Data Manager]
            DP3[Data Cleaner]
            DP4[Data Normalizer]
        end

        subgraph "Storage Layer"
            ST1[(Redis Cache<br/>ms-level Access)]
            ST2[(PostgreSQL<br/>Time Series Data)]
            ST3[(InfluxDB<br/>Market Data)]
        end

        subgraph "RUST Acceleration Layer"
            RU1[Batch Processor<br/>50K+ candles/s]
            RU2[Data Compressor<br/>30-50% Volume Optimization]
            RU3[Index Builder<br/>Real-time Index]
        end

        subgraph "Service Interface Layer"
            API1[Real-time Data API]
            API2[Historical Data API]
            API3[Subscription Manager]
        end
    end

    DS1 --> DC1
    DS2 --> DC2
    DS3 --> DC3
    DS4 --> DC4

    DC1 --> DP1
    DC2 --> DP2
    DC3 --> DP3
    DC4 --> DP4

    DP1 --> RU1
    DP2 --> RU2
    DP3 --> RU3
    DP4 --> ST1

    RU1 --> ST2
    RU2 --> ST3
    RU3 --> ST1

    ST1 --> API1
    ST2 --> API2
    ST3 --> API3

    classDef rust fill:#ff6b6b,stroke:#c92a2a,color:#fff
    classDef storage fill:#69db7c,stroke:#2f9e44,color:#fff
    classDef api fill:#4dabf7,stroke:#1864ab,color:#fff
    class RU1,RU2,RU3 rust
    class ST1,ST2,ST3 storage
    class API1,API2,API3 api
```

**Core Features**:
- **Multi-source Data Fusion**: Support for WebSocket, REST API, news, sentiment, and other multi-source data
- **Real-time Stream Processing**: 100,000+ data points/second processing capability
- **RUST Acceleration**: 50,000+ candles/second, 3-5x performance improvement
- **Smart Caching**: Redis millisecond-level access, InfluxDB efficient storage

### 2. Multi-Agent System Architecture

```mermaid
graph TB
    subgraph "Multi-Agent System Architecture"
        subgraph "Data Input Layer"
            DI1[Market Data Stream]
            DI2[Technical Indicator Data]
            DI3[News Sentiment Data]
            DI4[Historical Decision Records]
        end

        subgraph "AI Agent Layer"
            subgraph "Technical Agent"
                TA1[Technical Indicator Analyzer]
                TA2[Multi-timeframe Fusion]
                TA3[Signal Generator]
            end

            subgraph "Pattern Agent"
                PA1[Candlestick Pattern Recognizer<br/>30+ patterns]
                PA2[Support Resistance Analyzer]
                PA3[Pattern Strength Assessor]
            end

            subgraph "Trend Agent"
                TR1[Trend Analyzer]
                TR2[Multi-timeframe Validation]
                TR3[Reversal Predictor]
            end

            subgraph "Risk Agent"
                RA1[Risk Assessor]
                RA2[VaR Calculator]
                RA3[Hedging Recommender]
            end

            subgraph "News Agent"
                NA1[News Sentiment Analyzer]
                NA2[Event Impact Assessor]
                NA3[Market Correlation Analyzer]
            end
        end

        subgraph "LLM Management Layer"
            LM1[Load Balancer<br/>Dynamic Weight Allocation]
            LM2[LLM Provider Manager<br/>OpenAI/Claude/DeepSeek]
            LM3[Response Aggregator]
            LM4[Error Handler]
        end

        subgraph "Decision Fusion Layer"
            DF1[Signal Aggregator<br/>Agent Voting]
            DF2[Conflict Resolver<br/>Priority Mechanism]
            DF3[Decision Cache<br/>Redis-backed]
            DF4[Memory Manager<br/>100 Decision History]
        end

        subgraph "Output Layer"
            OU1[Trading Decision Signal]
            OU2[Confidence Score]
            OU3[Decision Explanation]
            OU4[Risk Warning]
        end
    end

    DI1 --> TA1
    DI2 --> TA2
    DI3 --> PA1
    DI4 --> DF3

    TA1 --> DF1
    PA2 --> DF1
    TR2 --> DF1
    RA3 --> DF2
    NA3 --> DF3

    DF1 --> LM1
    DF2 --> LM2
    DF3 --> LM3

    LM1 --> DF1
    LM2 --> DF2
    LM3 --> DF3
    LM4 --> OU4

    DF1 --> OU1
    DF2 --> OU2
    DF3 --> OU3
    DF4 --> OU4

    classDef ai fill:#f783ac,stroke:#a61e4d,color:#fff
    classDef llm fill:#845ef7,stroke:#5f3dc4,color:#fff
    classDef decision fill:#4dabf7,stroke:#1864ab,color:#fff
    class output fill:#69db7c,stroke:#2f9e44,color:#fff

    class TA1,TA2,TA3,PA1,PA2,PA3,TR1,TR2,TR3,RA1,RA2,RA3,NA1,NA2,NA3 ai
    class LM1,LM2,LM3,LM4 llm
    class DF1,DF2,DF3,DF4 decision
    class OU1,OU2,OU3,OU4 output
```

**Core Features**:
- **5 Specialized Agents**: Technical, Pattern, Trend, Risk, News agents
- **LLM Integration**: Support for OpenAI, Claude, DeepSeek and other providers
- **Intelligent Decision Fusion**: Voting mechanism, conflict resolution, dynamic weight adjustment
- **Decision Cache & Memory**: Redis cache, 100 decision history memory

### 3. Decision Engine Architecture

```mermaid
graph TB
    subgraph "Decision Engine Architecture"
        subgraph "Agent Input Layer"
            AI1[Technical Agent<br/>Technical Analysis Signal]
            AI2[Pattern Agent<br/>Pattern Recognition Signal]
            AI3[Trend Agent<br/>Trend Analysis Signal]
            AI4[Risk Agent<br/>Risk Assessment Signal]
            AI5[News Agent<br/>News Sentiment Signal]
        end

        subgraph "Input Preprocessing Layer"
            PR1[Signal Normalizer<br/>Unified Signal Format]
            PR2[Quality Assessor<br/>Confidence Scoring]
            PR3[Weight Calculator<br/>Dynamic Weight Allocation]
            PR4[Priority Sorter<br/>Signal Importance]
        end

        subgraph "Signal Synthesis Engine"
            SE1[Signal Aggregator<br/>1000+ signals/s]
            SE2[Conflict Detector<br/>Signal Conflict Detection]
            SE3[Weighted Voting Machine<br/>Weighted Decision]
            SE4[Confidence Calculator<br/>Decision Reliability]
        end

        subgraph "Conflict Resolution Mechanism"
            CR1[Risk Assessor<br/>Risk Priority Principle]
            CR2[Historical Validator<br/>Success Rate Statistics]
            CR3[Market Environment Adapter<br/>Trend Adaptation]
            CR4[Manual Rule Engine<br/>Trading Strategy]
        end

        subgraph "LLM Enhanced Decision"
            LL1[Context Builder<br/>Market State Snapshot]
            LL2[Prompt Generator<br/>Intelligent Prompt]
            LL3[LLM Call Manager<br/>Multi-provider Support]
            LL4[Response Parser<br/>Decision Extraction]
        end

        subgraph "Decision Optimizer"
            OP1[Real-time Optimizer<br/>Parameter Tuning]
            OP2[Performance Assessor<br/>Success Rate Statistics]
            OP3[Learning Engine<br/>Reinforcement Learning]
            OP4[Parameter Updater<br/>Dynamic Adjustment]
        end

        subgraph "Output Management Layer"
            OU1[Final Decision Maker<br/>Buy/Sell/Hold]
            OU2[Confidence Output<br/>0-100%]
            OU3[Decision Cache<br/>5min Cache]
            OU4[Explanation Generator<br/>Decision Rationale]
        end
    end

    AI1 --> PR1
    AI2 --> PR2
    AI3 --> PR3
    AI4 --> PR4
    AI5 --> PR1

    PR1 --> SE1
    PR2 --> SE2
    PR3 --> SE3
    PR4 --> SE4

    SE1 --> CR1
    SE2 --> CR2
    SE3 --> CR3
    SE4 --> CR4

    CR1 --> LL1
    CR2 --> LL2
    CR3 --> LL3
    CR4 --> LL4

    LL1 --> OP1
    LL2 --> OP2
    LL3 --> OP3
    LL4 --> OP4

    OP1 --> OU1
    OP2 --> OU2
    OP3 --> OU3
    OP4 --> OU4

    classDef input fill:#4dabf7,stroke:#1864ab,color:#fff
    classDef process fill:#ffd43b,stroke:#fab005,color:#000
    classDef llm fill:#845ef7,stroke:#5f3dc4,color:#fff
    classDef output fill:#69db7c,stroke:#2f9e44,color:#fff

    class AI1,AI2,AI3,AI4,AI5 input
    class PR1,PR2,PR3,PR4,SE1,SE2,SE3,SE4,CR1,CR2,CR3,CR4,OP1,OP2,OP3,OP4 process
    class LL1,LL2,LL3,LL4 llm
    class OU1,OU2,OU3,OU4 output
```

**Core Features**:
- **High-performance Processing**: 1,000+ signals/second, 10ms synthesis latency
- **Intelligent Conflict Resolution**: Risk assessment, historical validation, market environment adaptation
- **LLM Enhancement**: Multi-provider support, intelligent decision explanation
- **Real-time Optimization**: Reinforcement learning, dynamic parameter adjustment

### 4. Execution Engine Architecture

```mermaid
graph TB
    subgraph "Execution Engine Architecture"
        subgraph "Decision Input Layer"
            DI1[Decision Signal<br/>Buy/Sell/Hold]
            DI2[Target Symbol<br/>Symbol Info]
            DI3[Trading Parameters<br/>Quantity/Price]
            DI4[Risk Limits<br/>Max Position]
        end

        subgraph "Order Preprocessing Layer"
            OP1[Order Validator<br/>Parameter Compliance]
            OP2[Risk Pre-checker<br/>100μs Assessment]
            OP3[Position Calculator<br/>Dynamic Adjustment]
            OP4[Price Optimizer<br/>Optimal Price]
        end

        subgraph "RUST Execution Core"
            RU1[Order Router<br/>1,000+ orders/s]
            RU2[State Manager<br/>Real-time Sync]
            RU3[Queue Manager<br/>Priority Queue]
            RU4[Batch Processor<br/>Batch Optimization]
        end

        subgraph "Gateway Adapter Layer"
            GW1[Binance Gateway<br/>Primary Interface]
            GW2[Freqtrade Gateway<br/>Strategy Integration]
            GW3[Simulation Gateway<br/>Backtest Validation]
            GW4[Gateway Manager<br/>Failover]
        end

        subgraph "Risk Control Layer"
            RK1[Real-time Monitor<br/>100μs Check]
            RK2[Limit Executor<br/>1,000+ checks/s]
            RK3[Emergency Stopper<br/>Immediate Response]
            RK4[Recovery Manager<br/>Auto Recovery]
        end

        subgraph "State Sync Layer"
            SS1[Order Tracker<br/>Real-time Status]
            SS2[Position Synchronizer<br/>Account Sync]
            SS3[Trade Recorder<br/>Historical Records]
            SS4[Reconciliation Manager<br/>Consistency Check]
        end

        subgraph "Monitoring & Logging"
            ML1[Performance Monitor<br/><50μs Latency]
            ML2[Audit Logger<br/>Complete Records]
            ML3[Metrics Collector<br/>Real-time Metrics]
            ML4[Report Generator<br/>Periodic Reports]
        end
    end

    DI1 --> OP1
    DI2 --> OP2
    DI3 --> OP3
    DI4 --> OP4

    OP1 --> RU1
    OP2 --> RU2
    OP3 --> RU3
    OP4 --> RU4

    RU1 --> GW1
    RU2 --> GW2
    RU3 --> GW3
    RU4 --> GW4

    GW1 --> RK1
    GW2 --> RK2
    GW3 --> RK3
    GW4 --> RK4

    RK1 --> SS1
    RK2 --> SS2
    RK3 --> SS3
    RK4 --> SS4

    SS1 --> ML1
    SS2 --> ML2
    SS3 --> ML3
    SS4 --> ML4

    classDef rust fill:#ff6b6b,stroke:#c92a2a,color:#fff
    classDef gateway fill:#4dabf7,stroke:#1864ab,color:#fff
    classDef risk fill:#ffd43b,stroke:#fab005,color:#000
    classDef monitor fill:#69db7c,stroke:#2f9e44,color:#fff

    class RU1,RU2,RU3,RU4 rust
    class GW1,GW2,GW3,GW4 gateway
    class RK1,RK2,RK3,RK4 risk
    class ML1,ML2,ML3,ML4 monitor
```

**Core Features**:
- **Ultra-low Latency Execution**: Average 50μs, P99 < 100μs
- **High Throughput**: 1,000+ orders/second processing capability
- **Multi-gateway Support**: Binance, Freqtrade, simulation trading
- **Real-time Risk Control**: 100μs risk assessment, 1,000+ checks/second

### 5. Risk Management Architecture

```mermaid
graph TB
    subgraph "Risk Management Architecture"
        subgraph "Real-time Calculation Engine"
            RC1[VaR Calculator<br/>99% 1-day VaR]
            RC2[CVaR Calculator<br/>Conditional Value at Risk]
            RC3[Volatility Calculator<br/>Real-time Volatility]
            RC4[Correlation Calculator<br/>Asset Correlation]
        end

        subgraph "Panoramic Monitoring Dashboard"
            MD1[Real-time Risk Metrics<br/>Comprehensive Risk Score]
            MD2[Position Monitor<br/>Real-time Position Tracking]
            MD3[Drawdown Monitor<br/>Maximum Drawdown Protection]
            MD4[Concentration Monitor<br/>Risk Diversification]
        end

        subgraph "Multi-level Limit System"
            subgraph "Account Level Limits"
                AL1[Total Position Limit<br/>95% Net Assets]
                AL2[Total Leverage Limit<br/>3x Leverage]
                AL3[Daily Loss Limit<br/>5% Daily Loss]
                AL4[Monthly Drawdown Limit<br/>15% Monthly Drawdown]
            end

            subgraph "Strategy Level Limits"
                SL1[Single Strategy Position<br/>2% Net Assets]
                SL2[Single Strategy Leverage<br/>2x Leverage]
                SL3[Consecutive Loss Limit<br/>5 Consecutive]
                SL4[Sharpe Ratio Requirement<br/>>1.0]
            end

            subgraph "Instrument Level Limits"
                IL1[Single Instrument Position<br/>5% Net Assets]
                IL2[Single Instrument Stop Loss<br/>2% Fixed Stop Loss]
                IL3[Liquidity Limit<br/>10% Daily Volume]
                IL4[Volatility Limit<br/>50% Daily Volatility]
            end
        end

        subgraph "Dynamic Adjustment Mechanism"
            DA1[Volatility Adjuster<br/>Vol Scaling]
            DA2[Correlation Adjuster<br/>Correlation Scaling]
            DA3[Concentration Adjuster<br/>Concentration Scaling]
            DA4[Liquidity Adjuster<br/>Liquidity Scaling]
        end

        subgraph "Stress Testing Engine"
            ST1[Historical Scenario Test<br/>Historical Event Replay]
            ST2[Monte Carlo Simulation<br/>Random Scenarios]
            ST3[Extreme Scenario Test<br/>Black Swan]
            ST4[Portfolio Stress Test<br/>Portfolio Effects]
        end

        subgraph "Real-time Monitoring System"
            RM1[Risk Early Warning<br/>Multi-level Warning]
            RM2[Auto Position Reducer<br/>Intelligent Reduction]
            RM3[Emergency Stopper<br/>Immediate Stop]
            RM4[Risk Reporter<br/>Real-time Report]
        end
    end

    RC1 --> AL1
    RC2 --> SL1
    RC3 --> IL1
    RC4 --> DA1

    AL1 --> MD1
    AL2 --> MD2
    AL3 --> MD3
    AL4 --> MD4

    SL1 --> DA2
    SL2 --> DA3
    SL3 --> DA4
    SL4 --> ST1

    IL1 --> ST2
    IL2 --> ST3
    IL3 --> ST4
    IL4 --> RM1

    DA1 --> RM2
    DA2 --> RM3
    DA3 --> RM4
    DA4 --> RM1

    ST1 --> RM2
    ST2 --> RM3
    ST3 --> RM4
    ST4 --> RM1

    MD1 --> RM1
    MD2 --> RM2
    MD3 --> RM3
    MD4 --> RM4

    classDef calculate fill:#f783ac,stroke:#a61e4d,color:#fff
    classDef monitor fill:#4dabf7,stroke:#1864ab,color:#fff
    classDef limit fill:#ffd43b,stroke:#fab005,color:#000
    classDef test fill:#845ef7,stroke:#5f3dc4,color:#fff
    classDef system fill:#69db7c,stroke:#2f9e44,color:#fff

    class RC1,RC2,RC3,RC4 calculate
    class MD1,MD2,MD3,MD4 monitor
    class AL1,AL2,AL3,AL4,SL1,SL2,SL3,SL4,IL1,IL2,IL3,IL4 limit
    class ST1,ST2,ST3,ST4 test
    class RM1,RM2,RM3,RM4 system
```

**Core Features**:
- **Real-time Risk Calculation**: VaR/CVaR models, 100μs risk assessment
- **Multi-level Limits**: Account, strategy, instrument three-tier limit system
- **Dynamic Adjustment**: Volatility, correlation, liquidity-based dynamic adjustment
- **Stress Testing**: Historical scenarios, Monte Carlo, extreme scenario testing

### 6. WebSocket Management Architecture

```mermaid
graph TB
    subgraph "WebSocket Management Architecture"
        subgraph "Connection Management Layer"
            CM1[Connection Pool Manager<br/>10K+ connections]
            CM2[Load Balancer<br/>Intelligent Allocation]
            CM3[Health Checker<br/>Heartbeat Detection]
            CM4[Failover Controller<br/>Auto Switch]
        end

        subgraph "Message Routing System"
            MR1[Message Distributor<br/>10K+ msg/s]
            MR2[Subscription Manager<br/>topic-based routing]
            MR3[Filter Engine<br/>Conditional Filtering]
            MR4[Priority Queue<br/>QoS Guarantee]
        end

        subgraph "Data Processing Layer"
            DP1[Message Parser<br/>JSON/Binary]
            DP2[Data Validator<br/>Format Validation]
            DP3[Transformer<br/>Format Conversion]
            DP4[Compressor<br/>gzip/deflate]
        end

        subgraph "RUST High-Performance Core"
            RH1[Batch Processor<br/>Batch Message Processing]
            RH2[Zero-copy Buffer<br/>Ring Buffer]
            RH3[Parallel Processor<br/>Multi-core Parallel]
            RH4[Async I/O Processor<br/>Tokio Runtime]
        end

        subgraph "Caching Mechanism"
            CH1[Message Cache<br/>Recent Message Cache]
            CH2[Subscription Cache<br/>Subscription Cache]
            CH3[State Cache<br/>Connection State Cache]
            CH4[Data Cache<br/>Hot Data Cache]
        end

        subgraph "Monitoring & Logging"
            ML1[Performance Monitor<br/>Latency/Throughput]
            ML2[Connection Monitor<br/>Connection Status]
            ML3[Error Monitor<br/>Error Statistics]
            ML4[Audit Logger<br/>Complete Records]
        end

        subgraph "Security Layer"
            SC1[Authentication Manager<br/>JWT/OAuth]
            SC2[Permission Controller<br/>RBAC]
            SC3[Encryption Manager<br/>TLS/WSS]
            SC4[Rate Limiter<br/>Connection Rate Limiting]
        end
    end

    CM1 --> MR1
    CM2 --> MR2
    CM3 --> MR3
    CM4 --> MR4

    MR1 --> DP1
    MR2 --> DP2
    MR3 --> DP3
    MR4 --> DP4

    DP1 --> RH1
    DP2 --> RH2
    DP3 --> RH3
    DP4 --> RH4

    RH1 --> CH1
    RH2 --> CH2
    RH3 --> CH3
    RH4 --> CH4

    CH1 --> ML1
    CH2 --> ML2
    CH3 --> ML3
    CH4 --> ML4

    ML1 --> SC1
    ML2 --> SC2
    ML3 --> SC3
    ML4 --> SC4

    classDef manager fill:#4dabf7,stroke:#1864ab,color:#fff
    classDef rust fill:#ff6b6b,stroke:#c92a2a,color:#fff
    classDef cache fill:#69db7c,stroke:#2f9e44,color:#fff
    classDef security fill:#ffd43b,stroke:#fab005,color:#000

    class CM1,CM2,CM3,CM4,MR1,MR2,MR3,MR4,DP1,DP2,DP3,DP4 manager
    class RH1,RH2,RH3,RH4 rust
    class CH1,CH2,CH3,CH4 cache
    class ML1,ML2,ML3,ML4,SC1,SC2,SC3,SC4 security
```

**Core Features**:
- **High Concurrency Support**: 10,000+ concurrent connections, < 1ms message latency
- **Intelligent Routing**: topic-based routing, priority queue, QoS guarantee
- **RUST Acceleration**: Batch processing, zero-copy buffer, multi-core parallel
- **Security Guarantee**: JWT authentication, TLS encryption, RBAC permission control

### 7. Technical Indicators Architecture

```mermaid
graph TB
    subgraph "Technical Indicators Architecture"
        subgraph "Data Input Layer"
            DI1[Real-time K-line Data<br/>OHLCV]
            DI2[Historical K-line Data<br/>Batch Loading]
            DI3[Tick Data<br/>Tick-by-tick Trades]
            DI4[Auxiliary Data<br/>Volume/Open Interest]
        end

        subgraph "RUST Calculation Core"
            subgraph "SIMD Optimization Engine"
                SI1[Vectorized Calculator<br/>AVX2/AVX-512]
                SI2[Batch Processor<br/>10K+ candles/batch]
                SI3[Parallel Calculator<br/>Rayon Parallel]
                SI4[Memory Optimizer<br/>Zero-copy Operations]
            end

            subgraph "Indicator Calculators"
                IC1[Trend Indicators<br/>SMA/EMA/DEMA/TEMA]
                IC2[Momentum Indicators<br/>RSI/Stoch/CCI/Williams]
                IC3[Volatility Indicators<br/>BB/ATR/Keltner]
                IC4[Volume Indicators<br/>OBV/AD/MF/CMF]
            end

            subgraph "Advanced Indicators"
                AI1[Pattern Recognition<br/>30+ patterns]
                AI2[Support Resistance<br/>Auto Recognition]
                AI3[Harmonic Patterns<br/>Gartley/Butterfly]
                AI4[Multi-timeframe<br/>MTF Analysis]
            end
        end

        subgraph "Cache & Optimization"
            subgraph "Multi-layer Cache"
                CA1[L1 Cache<br/>Memory Cache]
                CA2[L2 Cache<br/>Redis Cache]
                CA3[Pre-calculated Cache<br/>Common Periods]
                CA4[Incremental Calculation<br/>Differential Update]
            end

            subgraph "Performance Optimization"
                PO1[Batch Optimization<br/>Batch Calculation]
                PO2[Parallel Optimization<br/>Multi-core Utilization]
                PO3[Memory Optimization<br/>Memory Pool]
                PO4[Algorithm Optimization<br/>Incremental Algorithm]
            end
        end

        subgraph "API Interface Layer"
            API1[Real-time Calculation API<br/>Single Calculation]
            API2[Batch Calculation API<br/>Batch Processing]
            API3[Streaming Calculation API<br/>Real-time Stream]
            API4[Historical Calculation API<br/>Historical Backtest]
        end

        subgraph "Monitoring & Diagnostics"
            MT1[Performance Monitor<br/>Calculation Time]
            MT2[Cache Monitor<br/>Hit Rate]
            MT3[Error Monitor<br/>Exception Handling]
            MT4[Indicator Analyzer<br/>Validity Analysis]
        end
    end

    DI1 --> SI1
    DI2 --> SI2
    DI3 --> SI3
    DI4 --> SI4

    SI1 --> IC1
    SI2 --> IC2
    SI3 --> IC3
    SI4 --> IC4

    IC1 --> AI1
    IC2 --> AI2
    IC3 --> AI3
    IC4 --> AI4

    AI1 --> CA1
    AI2 --> CA2
    AI3 --> CA3
    AI4 --> CA4

    CA1 --> PO1
    CA2 --> PO2
    CA3 --> PO3
    CA4 --> PO4

    PO1 --> API1
    PO2 --> API2
    PO3 --> API3
    PO4 --> API4

    API1 --> MT1
    API2 --> MT2
    API3 --> MT3
    API4 --> MT4

    classDef rust fill:#ff6b6b,stroke:#c92a2a,color:#fff
    classDef cache fill:#69db7c,stroke:#2f9e44,color:#fff
    classDef api fill:#4dabf7,stroke:#1864ab,color:#fff
    classDef monitor fill:#ffd43b,stroke:#fab005,color:#000

    class SI1,SI2,SI3,SI4,IC1,IC2,IC3,IC4,AI1,AI2,AI3,AI4 rust
    class CA1,CA2,CA3,CA4,PO1,PO2,PO3,PO4 cache
    class API1,API2,API3,API4 api
    class MT1,MT2,MT3,MT4 monitor
```

**Core Features**:
- **RUST High Performance**: 50,000+ candles/second, 3-5x performance improvement
- **SIMD Optimization**: AVX2/AVX-512 vectorized computing
- **50+ Technical Indicators**: Trend, momentum, volatility, volume indicators
- **Multi-layer Cache**: L1 memory cache, L2 Redis cache, incremental calculation

### 8. Configuration Management Architecture

```mermaid
graph TB
    subgraph "Configuration Management Architecture"
        subgraph "Configuration Hierarchy"
            subgraph "Environment Configuration"
                EV1[Development Environment<br/>development]
                EV2[Testing Environment<br/>staging]
                EV3[Production Environment<br/>production]
                EV4[Backtest Environment<br/>backtest]
            end

            subgraph "Application Configuration"
                AP1[Basic Configuration<br/>app.yaml]
                AP2[Trading Configuration<br/>trading.yaml]
                AP3[Risk Configuration<br/>risk.yaml]
                AP4[Agent Configuration<br/>agents.yaml]
            end

            subgraph "Module Configuration"
                MO1[Database Configuration<br/>database.yaml]
                MO2[Redis Configuration<br/>redis.yaml]
                MO3[LLM Configuration<br/>llm.yaml]
                MO4[Monitoring Configuration<br/>monitoring.yaml]
            end
        end

        subgraph "Configuration Providers"
            CP1[File Provider<br/>YAML/JSON]
            CP2[Environment Variables<br/>ENV Variables]
            CP3[Database Provider<br/>Dynamic Config]
            CP4[Remote Configuration Center<br/>Consul/Nacos]
        end

        subgraph "Configuration Management Core"
            CM1[Configuration Loader<br/>Multi-source Loading]
            CM2[Configuration Validator<br/>Schema Validation]
            CM3[Configuration Merger<br/>Hierarchical Merge]
            CM4[Configuration Parser<br/>Variable Replacement]
        end

        subgraph "Hot Update Mechanism"
            HU1[File Monitor<br/>fsnotify]
            HU2[Change Detector<br/>Difference Comparison]
            HU3[Hot Updater<br/>Seamless Update]
            HU4[Rollback Manager<br/>Version Rollback]
        end

        subgraph "Version Control"
            VC1[Version Manager<br/>Git-like]
            VC2[Change History<br/>Audit Log]
            VC3[Branch Manager<br/>Environment Isolation]
            VC4[Release Manager<br/>Gray Release]
        end

        subgraph "Configuration API"
            API1[Get Configuration<br/>Get Config]
            API2[Update Configuration<br/>Update Config]
            API3[Batch Operations<br/>Batch Ops]
            API4[Query Configuration<br/>Query Config]
        end

        subgraph "Monitoring & Alerting"
            MG1[Configuration Monitor<br/>Status Monitoring]
            MG2[Consistency Checker<br/>Multi-node Sync]
            MG3[Alert Manager<br/>Exception Alerts]
            MG4[Performance Monitor<br/>Loading Performance]
        end
    end

    EV1 --> CP1
    EV2 --> CP2
    EV3 --> CP3
    EV4 --> CP4

    CP1 --> CM1
    CP2 --> CM2
    CP3 --> CM3
    CP4 --> CM4

    CM1 --> HU1
    CM2 --> HU2
    CM3 --> HU3
    CM4 --> HU4

    HU1 --> VC1
    HU2 --> VC2
    HU3 --> VC3
    HU4 --> VC4

    VC1 --> API1
    VC2 --> API2
    VC3 --> API3
    VC4 --> API4

    API1 --> MG1
    API2 --> MG2
    API3 --> MG3
    API4 --> MG4

    classDef config fill:#4dabf7,stroke:#1864ab,color:#fff
    classDef core fill:#ff6b6b,stroke:#c92a2a,color:#fff
    classDef update fill:#ffd43b,stroke:#fab005,color:#000
    classDef api fill:#69db7c,stroke:#2f9e44,color:#fff

    class EV1,EV2,EV3,EV4,AP1,AP2,AP3,AP4,MO1,MO2,MO3,MO4,CP1,CP2,CP3,CP4 config
    class CM1,CM2,CM3,CM4 core
    class HU1,HU2,HU3,HU4,VC1,VC2,VC3,VC4 update
    class API1,API2,API3,API4,MG1,MG2,MG3,MG4 api
```

**Core Features**:
- **Multi-layer Configuration**: Environment, application, module three-tier configuration system
- **Multi-source Providers**: Files, environment variables, database, remote configuration center
- **Hot Update Mechanism**: File monitoring, seamless update, version rollback
- **5ms Fast Update**: Configuration changes effective within 5ms, 100% configuration validation

### 9. Database Layer Architecture

```mermaid
graph TB
    subgraph "Database Layer Architecture"
        subgraph "Database Cluster"
            subgraph "PostgreSQL Cluster"
                PG1[Master Node<br/>Read/Write]
                PG2[Slave Node 1<br/>Read Only]
                PG3[Slave Node 2<br/>Read Only]
                PG4[Backup Node<br/>WAL Archive]
            end

            subgraph "Redis Cluster"
                RD1[Master Node<br/>Cache]
                RD2[Slave Node 1<br/>Cache]
                RD3[Slave Node 2<br/>Cache]
                RD4[Sentinel Node<br/>High Availability]
            end

            subgraph "InfluxDB Cluster"
                ID1[Master Node<br/>Time Series Data]
                ID2[Slave Node<br/>Replica]
                ID3[Meta Node<br/>Metadata]
                ID4[Data Node<br/>Shard]
            end
        end

        subgraph "Connection Management Layer"
            CM1[Connection Pool Manager<br/>Max: 1000]
            CM2[Load Balancer<br/>Read/Write Separation]
            CM3[Failover Controller<br/>Auto Switch]
            CM4[Health Checker<br/>Heartbeat Detection]
        end

        subgraph "Smart Routing"
            IR1[Query Router<br/>Intelligent Distribution]
            IR2[Shard Manager<br/>Data Sharding]
            IR3[Index Optimizer<br/>Smart Index]
            IR4[Cache Router<br/>Cache Strategy]
        end

        subgraph "Transaction Management"
            TM1[Distributed Transaction<br/>XA Protocol]
            TM2[Transaction Coordinator<br/>Two-phase Commit]
            TM3[Lock Manager<br/>Optimistic Lock]
            TM4[Deadlock Detector<br/>Auto Resolution]
        end

        subgraph "Data Synchronization"
            DS1[Real-time Synchronizer<br/>CDC]
            DS2[Batch Synchronizer<br/>ETL]
            DS3[Incremental Synchronizer<br/>Change Log]
            DS4[Consistency Checker<br/>Data Validation]
        end

        subgraph "Monitoring & Operations"
            subgraph "Performance Monitoring"
                PM1[Query Monitor<br/>Slow Queries]
                PM2[Lock Monitor<br/>Lock Waiting]
                PM3[Cache Monitor<br/>Hit Rate]
                PM4[Connection Monitor<br/>Connection Pool]
            end

            subgraph "Alert System"
                AL1[Threshold Alert<br/>Performance Threshold]
                AL2[Exception Alert<br/>Error Rate]
                AL3[Capacity Alert<br/>Storage Capacity]
                AL4[Availability Alert<br/>Service Availability]
            end
        end

        subgraph "Backup & Recovery"
            subgraph "Backup Strategy"
                BK1[Full Backup<br/>Daily]
                BK2[Incremental Backup<br/>Hourly]
                BK3[WAL Backup<br/>Real-time]
                BK4[Cross-region Backup<br/>Disaster Recovery]
            end

            subgraph "Recovery Mechanism"
                RC1[Point-in-time Recovery<br/>PITR]
                RC2[Master-Slave Switch<br/>Failover]
                RC3[Data Reconstruction<br/>Data Repair]
                RC4[Disaster Recovery<br/>DRP]
            end
        end
    end

    PG1 --> CM1
    PG2 --> CM2
    PG3 --> CM3
    PG4 --> CM4

    RD1 --> IR1
    RD2 --> IR2
    RD3 --> IR3
    RD4 --> IR4

    ID1 --> TM1
    ID2 --> TM2
    ID3 --> TM3
    ID4 --> TM4

    CM1 --> DS1
    CM2 --> DS2
    CM3 --> DS3
    CM4 --> DS4

    DS1 --> PM1
    DS2 --> PM2
    DS3 --> PM3
    DS4 --> PM4

    PM1 --> BK1
    PM2 --> BK2
    PM3 --> BK3
    PM4 --> BK4

    BK1 --> RC1
    BK2 --> RC2
    BK3 --> RC3
    BK4 --> RC4

    classDef database fill:#69db7c,stroke:#2f9e44,color:#fff
    classDef connection fill:#4dabf7,stroke:#1864ab,color:#fff
    classDef router fill:#ff6b6b,stroke:#c92a2a,color:#fff
    classDef transaction fill:#ffd43b,stroke:#fab005,color:#000
    classDef monitor fill:#845ef7,stroke:#5f3dc4,color:#fff

    class PG1,PG2,PG3,PG4,RD1,RD2,RD3,RD4,ID1,ID2,ID3,ID4 database
    class CM1,CM2,CM3,CM4 connection
    class IR1,IR2,IR3,IR4 router
    class TM1,TM2,TM3,TM4,DS1,DS2,DS3,DS4 transaction
    class PM1,PM2,PM3,PM4,AL1,AL2,AL3,AL4,BK1,BK2,BK3,BK4,RC1,RC2,RC3,RC4 monitor
```

**Core Features**:
- **Multi-database Architecture**: PostgreSQL, Redis, InfluxDB clusters
- **Smart Sharding**: Data sharding, read-write separation, load balancing
- **Cross-database Transactions**: Distributed transactions, two-phase commit, consistency guarantee
- **High-performance Access**: 50,000+ QPS, < 10ms response latency

### 10. API Layer Architecture

```mermaid
graph TB
    subgraph "API Layer Architecture"
        subgraph "Access Layer"
            AL1[Load Balancer<br/>Nginx/HAProxy]
            AL2[API Gateway<br/>Kong/Istio]
            AL3[Rate Limiter<br/>Rate Limiter]
            AL4[Authentication Center<br/>Auth Service]
        end

        subgraph "Routing Layer"
            RT1[Path Router<br/>Path Router]
            RT2[Version Router<br/>Version Router]
            RT3[Method Router<br/>Method Router]
            RT4[Parameter Router<br/>Param Router]
        end

        subgraph "FastAPI Service Layer"
            subgraph "REST API Services"
                RA1[Market Data API<br/>Market Data]
                RA2[Trading API<br/>Trading]
                RA3[Account API<br/>Account]
                RA4[Analytics API<br/>Analytics]
            end

            subgraph "WebSocket Services"
                WS1[Real-time Quotes<br/>Real-time Quotes]
                WS2[Trade Updates<br/>Trade Updates]
                WS3[Event Stream<br/>Event Stream]
                WS4[Admin Panel<br/>Admin Panel]
            end

            subgraph "Middleware Layer"
                MW1[Authentication Middleware<br/>JWT/OAuth]
                MW2[Permission Middleware<br/>RBAC]
                MW3[Logging Middleware<br/>Logging]
                MW4[Monitoring Middleware<br/>Metrics]
            end
        end

        subgraph "Business Logic Layer"
            BL1[Market Data Service<br/>Data Service]
            BL2[Order Management Service<br/>Order Service]
            BL3[Risk Management Service<br/>Risk Service]
            BL4[Analysis Service<br/>Analysis Service]
        end

        subgraph "Data Access Layer"
            DA1[Cache Accessor<br/>Redis Cache]
            DA2[Database Accessor<br/>SQL Access]
            DA3[Time Series Data Accessor<br/>TSDB Access]
            DA4[File Storage Accessor<br/>File Storage]
        end

        subgraph "Monitoring & Operations"
            subgraph "Performance Monitoring"
                PF1[Response Time Monitor<br/>Response Time]
                PF2[Throughput Monitor<br/>Throughput]
                PF3[Error Rate Monitor<br/>Error Rate]
                PF4[Resource Usage Monitor<br/>Resource Usage]
            end

            subgraph "Distributed Tracing"
                TR1[Request Trace<br/>Request Trace]
                TR2[Dependency Trace<br/>Dependency Trace]
                TR3[Performance Profile<br/>Performance Profile]
                TR4[Bottleneck Analysis<br/>Bottleneck Analysis]
            end
        end
    end

    AL1 --> RT1
    AL2 --> RT2
    AL3 --> RT3
    AL4 --> RT4

    RT1 --> RA1
    RT2 --> WS1
    RT3 --> MW1
    RT4 --> WS2

    RA1 --> BL1
    RA2 --> BL2
    RA3 --> BL3
    RA4 --> BL4

    WS1 --> MW2
    WS2 --> MW3
    WS3 --> MW4
    WS4 --> RA1

    MW1 --> DA1
    MW2 --> DA2
    MW3 --> DA3
    MW4 --> DA4

    DA1 --> PF1
    DA2 --> PF2
    DA3 --> PF3
    DA4 --> PF4

    PF1 --> TR1
    PF2 --> TR2
    PF3 --> TR3
    PF4 --> TR4

    classDef gateway fill:#4dabf7,stroke:#1864ab,color:#fff
    classDef service fill:#ff6b6b,stroke:#c92a2a,color:#fff
    classDef business fill:#69db7c,stroke:#2f9e44,color:#fff
    classDef data fill:#ffd43b,stroke:#fab005,color:#000
    classDef monitor fill:#845ef7,stroke:#5f3dc4,color:#fff

    class AL1,AL2,AL3,AL4,RT1,RT2,RT3,RT4 gateway
    class RA1,RA2,RA3,RA4,WS1,WS2,WS3,WS4,MW1,MW2,MW3,MW4 service
    class BL1,BL2,BL3,BL4 business
    class DA1,DA2,DA3,DA4 data
    class PF1,PF2,PF3,PF4,TR1,TR2,TR3,TR4 monitor
```

**Core Features**:
- **High-performance API**: 10,000+ QPS, < 100ms response time
- **RESTful & WebSocket**: Complete REST API and real-time WebSocket services
- **Authentication & Authorization**: JWT/OAuth authentication, RBAC permission control
- **Full-chain Monitoring**: Request tracing, performance analysis, bottleneck identification

For detailed documentation of all architecture diagrams, please visit [docs/modules](./docs/modules) directory.

## 🚀 Deployment Architecture

### Production Environment Architecture

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Nginx/HAProxy]
    end

    subgraph "Kubernetes Cluster"
        subgraph "Application Pods"
            API1[DeepAlpha API Pod 1]
            API2[DeepAlpha API Pod 2]
            API3[DeepAlpha API Pod 3]
            EX1[Executor Pod 1<br/>Rust Engine]
            EX2[Executor Pod 2<br/>Rust Engine]
            WS1[WebSocket Pod<br/>10K Connections]
        end
    end

    subgraph "Data Layer"
        PG[(PostgreSQL Cluster)]
        RD[(Redis Cluster)]
        ID[(InfluxDB)]
        S3[(S3 Storage)]
    end

    subgraph "Monitoring"
        PROM[Prometheus]
        GRAF[Grafana]
        JAE[Jaeger]
    end

    LB --> API1
    LB --> API2
    LB --> API3
    API1 --> PG
    API2 --> PG
    API3 --> PG
    API1 --> RD
    API2 --> RD
    API3 --> RD
    EX1 --> PG
    EX2 --> PG
    WS1 --> RD
    PROM --> API1
    PROM --> API2
    PROM --> API3
    GRAF --> PROM
```

### Deployment Options

#### Docker Compose (Quick Start)

```bash
# Clone project
git clone https://github.com/zcxGGmu/DeepAlpha.git
cd DeepAlpha

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f deepalpha
```

#### Kubernetes (Production)

```bash
# Create namespace
kubectl create namespace deepalpha

# Deploy application
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n deepalpha
```

### Monitoring and Observability

- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboard
- **Jaeger**: Distributed tracing
- **ELK Stack**: Log aggregation and analysis

Access monitoring dashboards:
- Grafana: `http://your-domain:3000`
- Prometheus: `http://your-domain:9090`
- Jaeger: `http://your-domain:16686`

### Detailed Deployment Documentation

For complete deployment guide, please refer to: [Deployment Architecture](./docs/deployment-architecture.md)

Including:
- Production environment deployment architecture
- Kubernetes configuration manifests
- Docker Compose configuration
- Monitoring configuration
- Performance tuning recommendations
- Security configuration
- Disaster recovery plan

## 💡 Usage Examples

### Using Rust Performance Modules

```python
# Import Rust performance modules
from deepalpha_rust import TechnicalIndicators, WebSocketManager, ExecutionEngine

# 1. High-performance technical indicator calculation
indicators = TechnicalIndicators()

# Batch calculate indicators (ultra-high performance)
prices = [45000, 45100, 45200, 45300, 45400, 45300, 45200]
sma_20 = indicators.calculate_sma(prices, period=20)
ema_12 = indicators.calculate_ema(prices, period=12)
rsi = indicators.calculate_rsi(prices, period=14)
macd = indicators.calculate_macd(prices)

print(f"SMA20: {sma_20}, RSI14: {rsi}")

# 2. WebSocket connection management
ws_manager = WebSocketManager()
ws_manager.start()

# Batch add connections (supports 10,000+ concurrent)
for i in range(100):
    ws_manager.add_connection(f"conn_{i}", "wss://stream.binance.com/ws/btcusdt@trade")

# Broadcast message (< 1ms latency)
ws_manager.broadcast("market_update", {"symbol": "BTC/USDT", "price": 50000})

# 3. Ultra-low latency trading execution
engine = ExecutionEngine()
engine.start()

# Submit order (average latency < 50μs)
order_data = {
    "symbol": "BTC/USDT",
    "side": "buy",
    "type": "market",
    "quantity": 1.5
}
order_id = engine.submit_order(order_data)

# Get execution statistics
stats = engine.get_stats()
print(f"Total orders: {stats.total_orders}")
print(f"Average execution time: {stats.avg_execution_time_us} μs")
```

### Creating Trading Strategies

```python
from deepalpha import DeepAlphaEngine
from deepalpha.config import Settings

# Load configuration
settings = Settings.from_file("config/development.yaml")

# Create trading engine
engine = DeepAlphaEngine(settings)

# Add trading pairs
engine.add_symbol("BTC/USDT")
engine.add_symbol("ETH/USDT")

# Start trading
await engine.start()
```

### Custom Agents

```python
from deepalpha.agents import BaseAgent
from deepalpha.core import Signal
from deepalpha_rust import TechnicalIndicators  # Use Rust acceleration

class CustomAgent(BaseAgent):
    """Custom agent example (with Rust acceleration)"""

    def __init__(self):
        super().__init__()
        self.indicators = TechnicalIndicators()  # Rust-accelerated indicator calculation

    async def analyze(self, symbol: str, timeframe: str) -> Signal:
        # Get market data
        data = await self.get_market_data(symbol, timeframe)

        # Use Rust-accelerated indicator calculation
        rsi = self.indicators.calculate_rsi(data['close'], period=14)
        macd = self.indicators.calculate_macd(data['close'])

        # Custom analysis logic
        if rsi < 30 and macd['histogram'] > 0:
            return Signal.buy(confidence=0.8)
        elif rsi > 70 and macd['histogram'] < 0:
            return Signal.sell(confidence=0.7)

        return Signal.hold()
```

### Configuring LLM Providers

```yaml
# config/default.yaml
llm_providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    model: "gpt-4"
    weight: 0.4

  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: "claude-3-opus"
    weight: 0.3

  deepseek:
    api_key: ${DEEPSEEK_API_KEY}
    model: "deepseek-chat"
    weight: 0.3

# Rust performance module configuration
rust_modules:
  indicators:
    enabled: true
    batch_size: 10000
    parallel_workers: 8

  websocket:
    enabled: true
    max_connections: 10000
    heartbeat_interval: 30

  executor:
    enabled: true
    max_orders_per_second: 1000
    risk_check_interval: 1
```

## 📊 Monitoring Dashboard

The system provides a complete web monitoring interface:

- **Real-time Quotes Panel**: Display real-time prices and indicators for watched trading pairs
- **Trading History**: View all trading records and performance
- **Agent Status**: Monitor analysis results of each agent
- **Risk Monitoring**: Display real-time account risk metrics
- **Performance Monitoring**: Rust module performance metrics (latency, throughput, etc.)
- **Decision Logs**: View AI decision process and reasoning

Visit `http://localhost:8000` to access the monitoring interface.

## 🧪 Testing

### Run All Tests

```bash
# Python tests
pytest

# Rust module tests
cd rust && cargo test

# Performance benchmark tests
cd rust/tests
python test_indicators_performance.py
python test_websocket_performance.py
python test_stream_performance.py
python test_executor_performance.py
```

### Run Specific Tests

```bash
# Python unit tests
pytest tests/unit/test_agents.py
pytest tests/integration/test_engine.py

# Rust module examples
python rust/examples/executor_examples.py
python rust/examples/websocket_examples.py
python rust/examples/indicators_examples.py
python rust/examples/stream_examples.py
```

### Check Test Coverage

```bash
pytest --cov=deepalpha --cov-report=html
```

## 📚 API Documentation

After starting the service, visit the following URLs to view API documentation:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Main API Endpoints

- `GET /api/v1/health` - Health check
- `GET /api/v1/symbols` - Get supported trading pairs
- `POST /api/v1/trade` - Manual order placement
- `GET /api/v1/positions` - Get position information
- `GET /api/v1/decisions` - Get decision history
- `GET /api/v1/performance` - Get performance metrics

## 🔧 Configuration

### Main Configuration Items

```yaml
# Application configuration
app:
  name: "DeepAlpha"
  version: "1.0.0"
  debug: false

# Exchange configuration
exchange:
  name: "binance"
  api_key: ${BINANCE_API_KEY}
  api_secret: ${BINANCE_API_SECRET}
  sandbox: false

# Trading parameters
trading:
  max_position_size: 0.02  # Maximum position 2%
  stop_loss: 0.02  # 2% stop loss
  take_profit: 0.06  # 6% take profit
  max_drawdown: 0.10  # Maximum drawdown 10%

# Agent configuration
agents:
  technical:
    enabled: true
    timeframes: ["1m", "5m", "15m", "1h"]
  pattern:
    enabled: true
    patterns: ["hammer", "doji", "engulfing"]
  trend:
    enabled: true
    periods: [7, 25, 99]
  risk:
    enabled: true
    max_position_per_symbol: 0.05

# Decision engine
decision:
  min_confidence: 0.7
  cache_duration: 300  # 5 minutes
  memory_size: 100

# Rust performance module configuration
rust:
  indicators:
    enabled: true
    cache_size: 1000
    batch_size: 5000

  websocket:
    enabled: true
    max_connections: 10000
    reconnect_interval: 5

  executor:
    enabled: true
    max_orders_per_second: 1000
    risk_check_batch_size: 100
```

## 🚀 Performance Optimization

### System Tuning Recommendations

1. **Rust Performance Module Optimization**
   - Enable all Rust modules for maximum performance improvement
   - Adjust parallel worker count based on hardware
   - Use larger batch sizes for improved throughput

2. **Database Optimization**
   - Use connection pooling
   - Set up indexes appropriately
   - Clean up historical data regularly

3. **Caching Strategy**
   - Redis cache hot data
   - Local cache calculation results
   - Use CDN for static resource acceleration

4. **Concurrency Optimization**
   - Set reasonable coroutine count
   - Use async I/O
   - Batch process data

5. **Monitoring Metrics**
   - CPU usage < 80%
   - Memory usage < 70%
   - API response time < 100ms
   - Rust module latency monitoring

## 🛡️ Security

### Best Practices

1. **API Key Management**
   - Use environment variables to store keys
   - Rotate API keys regularly
   - Limit API permissions

2. **Network Security**
   - Use HTTPS/WSS
   - Set up firewall rules
   - Enable API access restrictions

3. **Fund Security**
   - Use API trading permissions, disable withdrawals
   - Set reasonable trading limits
   - Check trading records regularly

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

### Development Standards

- Follow PEP 8 coding standards
- Rust code follows rustfmt standards
- Write unit tests
- Update documentation
- Use conventional commit format for commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Binance](https://binance.com) - Excellent trading API
- [FastAPI](https://fastapi.tiangolo.com) - High-performance web framework
- [PyO3](https://pyo3.rs) - Bridge between Rust and Python
- [Tokio](https://tokio.rs) - Rust async runtime
- [OpenAI](https://openai.com) - Powerful AI capabilities

## 📞 Contact

- Project homepage: [https://github.com/zcxGGmu/DeepAlpha](https://github.com/zcxGGmu/DeepAlpha)
- Issue feedback: [Issues](https://github.com/zcxGGmu/DeepAlpha/issues)
- Email: deepalpha@example.com

## ⚠️ Disclaimer

This software is for learning and research purposes only. Quantitative trading involves risks, and using this software for actual trading may result in financial losses. Users need to bear all trading risks themselves, and developers are not responsible for any trading losses. Before using, please ensure you fully understand the associated risks and seek professional advice if necessary.

---

**Made with ❤️ by DeepAlpha Team**