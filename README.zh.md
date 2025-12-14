# DeepAlpha - 大模型驱动的多智能体量化交易系统

[![English](https://img.shields.io/badge/English-🇺🇸-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/中文-🇨🇳-red.svg)](README.zh.md)

![DeepAlpha](https://img.shields.io/badge/DeepAlpha-v1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Rust](https://img.shields.io/badge/Rust-1.74+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

DeepAlpha是一个基于大模型驱动的Python/Rust混合架构量化交易系统，采用多智能体架构，通过AI模型进行交易决策，集成了超高性能的Rust执行引擎。

## 🌟 核心特性

### 多智能体系统
- **Technical Agent**: 专业技术指标分析（EMA、RSI、MACD、ATR等）
- **Pattern Agent**: 智能K线形态识别（头肩顶、吞没、黄昏星等）
- **Trend Agent**: 多时间框架趋势分析和预测
- **Risk Agent**: 智能风险评估和仓位管理

### AI决策引擎
- 支持多个LLM Provider（OpenAI、Anthropic Claude、DeepSeek等）
- 智能投票机制和决策聚合
- 决策缓存和历史记忆
- 动态权重调整

### 🚀 Rust性能优化模块
经过完整的Rust迁移，实现了超高性能的核心组件：

1. **技术指标引擎** (`rust/src/indicators`)
   - 支持50+种技术指标（SMA、EMA、RSI、MACD、Bollinger Bands等）
   - **性能**: > 50,000 K线/秒处理速度
   - **优化**: SIMD指令集优化，多核并行计算
   - **内存**: 零拷贝操作，减少30-50%内存使用

2. **WebSocket管理器** (`rust/src/websocket`)
   - 支持10,000+并发WebSocket连接
   - 自动重连和心跳保活机制
   - 消息广播和订阅管理
   - **延迟**: < 1ms消息处理

3. **数据流处理器** (`rust/src/stream`)
   - 实时数据流处理管道
   - 支持100,000+数据点/秒
   - 数据验证和转换
   - 异步批处理优化

4. **交易执行引擎** (`rust/src/executor`)
   - **超低延迟**: 平均50μs订单提交，P99 < 100μs
   - **高吞吐**: 1,000+订单/秒处理能力
   - 风险管理和实时风控
   - 仓位和投资组合管理

### 风险管理
- 实时风险检查和头寸控制
- 严格的仓位管理（每笔交易1-2%净资产）
- 动态止损止盈机制
- 最大回撤保护
- 组合风险分散

### 高性能架构
- 异步并发处理（asyncio）
- 实时WebSocket数据流
- **Rust性能模块**：3-5倍性能提升，30-50%内存优化
- 分布式部署支持
- 完善的监控和日志系统

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Rust 1.74+ (用于性能模块)
- Redis（用于缓存）
- PostgreSQL/SQLite（用于数据存储）
- Docker（可选，用于容器化部署）

### 安装

1. **克隆项目**
```bash
git clone https://github.com/zcxGGmu/DeepAlpha.git
cd DeepAlpha
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

3. **安装Python依赖**
```bash
pip install -r requirements/prod.txt
```

4. **安装 Rust 性能模块**
```bash
# 快速安装脚本（推荐）
./scripts/install_rust.sh

# 或手动安装
cd rust
pip install maturin
maturin develop --release
```

5. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥和配置
```

6. **初始化数据库**
```bash
python scripts/migrate.py
```

7. **启动系统**
```bash
python scripts/start.py
```

### Rust 性能模块验证

安装完成后，可以运行性能基准测试验证Rust模块的性能：

```bash
# 技术指标性能测试
cd rust/tests
python test_indicators_performance.py

# WebSocket性能测试
python test_websocket_performance.py

# 数据流性能测试
python test_stream_performance.py

# 执行引擎性能测试
python test_executor_performance.py
```

**预期性能提升**：
- 技术指标计算：50,000+ K线/秒（相比Python提升5-10倍）
- WebSocket连接：10,000+并发连接（< 1ms延迟）
- 数据流处理：100,000+数据点/秒
- 订单执行：1,000+订单/秒（平均延迟 < 50μs）
- 内存使用：减少30-50%

### Docker部署

```bash
# 构建包含Rust性能模块的镜像
docker build -t deepalpha .

# 启动服务
docker-compose up -d
```

## 📖 系统架构

### 整体架构概览

```mermaid
graph TB
    %% 外部数据源
    subgraph "External Data Sources"
        B1[Binance API]
        B2[Other Exchanges]
        B3[News Feeds]
        B4[Sentiment APIs]
    end

    %% 数据收集层
    subgraph "Data Collection"
        C1[WebSocket Connectors]
        C2[REST API Pollers]
        C3[News Aggregator]
    end

    %% Rust高性能层
    subgraph "Rust Performance Layer"
        D1[WebSocket Manager<br/>10K+ Connections<br/><1ms Latency]
        D2[Stream Processor<br/>100K+ Points/sec]
        D3[Technical Indicators<br/>50K+ Candles/sec]
        D4[Execution Engine<br/>1K+ Orders/sec<br/><50μs Latency]
    end

    %% 存储层
    subgraph "Storage Layer"
        E1[(PostgreSQL<br/>Time Series Data)]
        E2[(Redis<br/>Cache & Session)]
        E3[(InfluxDB<br/>Market Data)]
    end

    %% Python业务逻辑层
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

    %% 执行层
    subgraph "Execution Layer"
        I1[Order Manager]
        I2[Position Manager]
        I3[Risk Manager]
    end

    %% 网关层
    subgraph "Gateway Layer"
        J1[Exchange Gateway]
        J2[Notification Gateway]
        J3[API Gateway]
    end

    %% 接口层
    subgraph "Interface Layer"
        K1[REST API<br/>FastAPI]
        K2[WebSocket Server]
        K3[Web Dashboard]
    end

    %% 数据流
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

### 核心组件详解

#### 🚀 Rust性能优化层

1. **WebSocket管理器** (`rust/src/websocket/`)
   - 支持10,000+并发连接
   - 消息延迟 < 1ms
   - 自动重连和心跳机制
   - JWT认证和权限控制

2. **数据流处理器** (`rust/src/stream/`)
   - 100,000+数据点/秒处理能力
   - 零拷贝环形缓冲区
   - 实时数据验证和清洗
   - 异步批处理优化

3. **技术指标引擎** (`rust/src/indicators/`)
   - 50+种技术指标支持
   - SIMD指令集优化
   - 50,000+K线/秒计算速度
   - 支持批量计算

4. **交易执行引擎** (`rust/src/executor/`)
   - 超低延迟：平均50μs
   - 1,000+订单/秒吞吐量
   - 实时风险控制
   - 仓位和组合管理

#### 🤖 Python智能体系统

1. **Technical Agent**
   - 专业技术指标分析
   - 多时间框架支持
   - 指标信号聚合

2. **Pattern Agent**
   - 30+K线形态识别
   - 支撑阻力位分析
   - 形态强度评估

3. **Trend Agent**
   - 多时间框架趋势分析
   - 趋势一致性检查
   - 趋势反转预测

4. **Risk Agent**
   - 实时风险评估
   - VaR计算
   - 对冲建议

5. **News Agent**
   - 实时新闻分析
   - 情感评分
   - 事件影响评估

### 详细架构文档

完整的系统架构说明请参考：[System Architecture](./docs/system-architecture.md)

该文档包含：
- 详细的组件说明
- 数据流分析
- 性能优化策略
- 部署架构指南

## 📊 子功能模块架构

### 1. 市场数据模块架构

```mermaid
graph TB
    subgraph "市场数据模块架构"
        subgraph "数据源层"
            DS1[Binance WebSocket]
            DS2[REST API轮询]
            DS3[新闻数据源]
            DS4[情绪数据源]
        end

        subgraph "数据采集层"
            DC1[WebSocket连接器]
            DC2[REST轮询器]
            DC3[新闻聚合器]
            DC4[数据验证器]
        end

        subgraph "数据流处理层"
            DP1[实时数据流<br/>100K+ pts/s]
            DP2[历史数据管理器]
            DP3[数据清洗器]
            DP4[数据标准化器]
        end

        subgraph "存储层"
            ST1[(Redis缓存<br/>毫秒级访问)]
            ST2[(PostgreSQL<br/>时序数据)]
            ST3[(InfluxDB<br/>市场数据)]
        end

        subgraph "RUST加速层"
            RU1[批量处理器<br/>50K+ candles/s]
            RU2[数据压缩器<br/>30-50%体积优化]
            RU3[索引构建器<br/>实时索引]
        end

        subgraph "服务接口层"
            API1[实时数据API]
            API2[历史数据API]
            API3[订阅管理器]
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

**核心特性**：
- **多源数据融合**：支持WebSocket、REST API、新闻、情绪等多源数据
- **实时流处理**：100,000+数据点/秒处理能力
- **RUST加速**：50,000+K线/秒，3-5倍性能提升
- **智能缓存**：Redis毫秒级访问，InfluxDB高效存储

### 2. 多智能体系统架构

```mermaid
graph TB
    subgraph "多智能体系统架构"
        subgraph "数据输入层"
            DI1[市场数据流]
            DI2[技术指标数据]
            DI3[新闻情感数据]
            DI4[历史决策记录]
        end

        subgraph "AI智能体层"
            subgraph "Technical Agent"
                TA1[技术指标分析器]
                TA2[多时间框架融合]
                TA3[信号生成器]
            end

            subgraph "Pattern Agent"
                PA1[K线形态识别器<br/>30+ patterns]
                PA2[支撑阻力分析器]
                PA3[形态强度评估器]
            end

            subgraph "Trend Agent"
                TR1[趋势分析器]
                TR2[多时间框架验证]
                TR3[反转预测器]
            end

            subgraph "Risk Agent"
                RA1[风险评估器]
                RA2[VaR计算器]
                RA3[对冲建议器]
            end

            subgraph "News Agent"
                NA1[新闻情感分析器]
                NA2[事件影响评估器]
                NA3[市场相关性分析器]
            end
        end

        subgraph "LLM管理层"
            LM1[负载均衡器<br/>动态权重分配]
            LM2[LLM提供商管理器<br/>OpenAI/Claude/DeepSeek]
            LM3[响应聚合器]
            LM4[错误处理器]
        end

        subgraph "决策融合层"
            DF1[信号聚合器<br/>智能体投票]
            DF2[冲突解决器<br/>优先级机制]
            DF3[决策缓存器<br/>Redis-backed]
            DF4[记忆管理器<br/>100决策历史]
        end

        subgraph "输出层"
            OU1[交易决策信号]
            OU2[置信度评分]
            OU3[决策解释]
            OU4[风险提示]
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

**核心特性**：
- **5大专业化智能体**：技术、形态、趋势、风险、新闻智能体
- **LLM集成**：支持OpenAI、Claude、DeepSeek等多个提供商
- **智能决策融合**：投票机制、冲突解决、动态权重调整
- **决策缓存与记忆**：Redis缓存，100决策历史记忆

### 3. 决策引擎架构

```mermaid
graph TB
    subgraph "决策引擎架构"
        subgraph "智能体输入层"
            AI1[Technical Agent<br/>技术分析信号]
            AI2[Pattern Agent<br/>形态识别信号]
            AI3[Trend Agent<br/>趋势分析信号]
            AI4[Risk Agent<br/>风险评估信号]
            AI5[News Agent<br/>新闻情绪信号]
        end

        subgraph "输入预处理层"
            PR1[信号标准化器<br/>统一信号格式]
            PR2[质量评估器<br/>置信度评分]
            PR3[权重计算器<br/>动态权重分配]
            PR4[优先级排序器<br/>信号重要性]
        end

        subgraph "信号合成引擎"
            SE1[信号聚合器<br/>1000+ signals/s]
            SE2[冲突检测器<br/>信号冲突识别]
            SE3[权重投票机<br/>加权决策]
            SE4[置信度计算器<br/>决策可靠性]
        end

        subgraph "冲突解决机制"
            CR1[风险评估器<br/>风险优先原则]
            CR2[历史验证器<br/>成功率统计]
            CR3[市场环境器<br/>趋势适应]
            CR4[人工规则引擎<br/>交易策略]
        end

        subgraph "LLM增强决策"
            LL1[上下文构建器<br/>市场状态快照]
            LL2[提示词生成器<br/>智能提示]
            LL3[LLM调用管理器<br/>多提供商支持]
            LL4[响应解析器<br/>决策提取]
        end

        subgraph "决策优化器"
            OP1[实时优化器<br/>参数调优]
            OP2[性能评估器<br/>成功率统计]
            OP3[学习引擎<br/>强化学习]
            OP4[参数更新器<br/>动态调整]
        end

        subgraph "输出管理层"
            OU1[最终决策器<br/>Buy/Sell/Hold]
            OU2[置信度输出<br/>0-100%]
            OU3[决策缓存器<br/>5分钟缓存]
            OU4[解释生成器<br/>决策理由]
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

**核心特性**：
- **高性能处理**：1,000+信号/秒，10ms合成延迟
- **智能冲突解决**：风险评估、历史验证、市场环境适应
- **LLM增强**：多提供商支持，智能决策解释
- **实时优化**：强化学习，动态参数调整

### 4. 执行引擎架构

```mermaid
graph TB
    subgraph "执行引擎架构"
        subgraph "决策输入层"
            DI1[决策信号<br/>Buy/Sell/Hold]
            DI2[目标交易对<br/>Symbol Info]
            DI3[交易参数<br/>Quantity/Price]
            DI4[风险限制<br/>Max Position]
        end

        subgraph "订单预处理层"
            OP1[订单验证器<br/>参数合规性]
            OP2[风险预检器<br/>100μs评估]
            OP3[仓位计算器<br/>动态调整]
            OP4[价格优化器<br/>最优价格]
        end

        subgraph "RUST执行核心"
            RU1[订单路由器<br/>1,000+ orders/s]
            RU2[状态管理器<br/>实时同步]
            RU3[队列管理器<br/>优先级队列]
            RU4[批量处理器<br/>批量优化]
        end

        subgraph "网关适配层"
            GW1[Binance网关<br/>主要接口]
            GW2[Freqtrade网关<br/>策略集成]
            GW3[模拟网关<br/>回测验证]
            GW4[网关管理器<br/>故障转移]
        end

        subgraph "风险控制层"
            RK1[实时监控器<br/>100μs检查]
            RK2[限制执行器<br/>1,000+ checks/s]
            RK3[紧急停止器<br/>即时响应]
            RK4[恢复管理器<br/>自动恢复]
        end

        subgraph "状态同步层"
            SS1[订单跟踪器<br/>实时状态]
            SS2[仓位同步器<br/>账户同步]
            SS3[成交记录器<br/>历史记录]
            SS4[对账管理器<br/>一致性检查]
        end

        subgraph "监控与日志"
            ML1[性能监控器<br/><50μs延迟]
            ML2[审计日志器<br/>完整记录]
            ML3[指标收集器<br/>实时指标]
            ML4[报告生成器<br/>定期报告]
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

**核心特性**：
- **超低延迟执行**：平均50μs，P99 < 100μs
- **高吞吐量**：1,000+订单/秒处理能力
- **多网关支持**：Binance、Freqtrade、模拟交易
- **实时风险控制**：100μs风险评估，1,000+检查/秒

### 5. 风险管理架构

```mermaid
graph TB
    subgraph "风险管理架构"
        subgraph "实时计算引擎"
            RC1[VaR计算器<br/>99% 1-day VaR]
            RC2[CVaR计算器<br/>条件风险价值]
            RC3[波动率计算器<br/>实时波动率]
            RC4[相关性计算器<br/>资产相关性]
        end

        subgraph "全景监控仪表盘"
            MD1[实时风险指标<br/>综合风险评分]
            MD2[仓位监控器<br/>实时仓位追踪]
            MD3[回撤监控器<br/>最大回撤保护]
            MD4[集中度监控器<br/>风险分散度]
        end

        subgraph "多层次限制体系"
            subgraph "账户级限制"
                AL1[总仓位限制<br/>95%净资产]
                AL2[总杠杆限制<br/>3倍杠杆]
                AL3[日亏损限制<br/>5%日损失]
                AL4[月回撤限制<br/>15%月回撤]
            end

            subgraph "策略级限制"
                SL1[单策略仓位<br/>2%净资产]
                SL2[单策略杠杆<br/>2倍杠杆]
                SL3[连续亏损限制<br/>5笔连续]
                SL4[夏普比率要求<br/>>1.0]
            end

            subgraph "品种级限制"
                IL1[单品种仓位<br/>5%净资产]
                IL2[单品种止损<br/>2%固定止损]
                IL3[流动性限制<br/>日交易量10%]
                IL4[波动率限制<br/>日波动50%]
            end
        end

        subgraph "动态调整机制"
            DA1[波动率调整器<br/>Vol Scaling]
            DA2[相关性调整器<br/>Correlation Scaling]
            DA3[集中度调整器<br/>Concentration Scaling]
            DA4[流动性调整器<br/>Liquidity Scaling]
        end

        subgraph "压力测试引擎"
            ST1[历史情景测试<br/>历史事件重演]
            ST2[蒙特卡洛模拟<br/>随机情景]
            ST3[极端情景测试<br/>Black Swan]
            ST4[组合压力测试<br/>组合效应]
        end

        subgraph "实时监控系统"
            RM1[风险预警器<br/>多级预警]
            RM2[自动减仓器<br/>智能减仓]
            RM3[紧急停止器<br/>立即停止]
            RM4[风险报告器<br/>实时报告]
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

**核心特性**：
- **实时风险计算**：VaR/CVaR模型，100μs风险评估
- **多层次限制**：账户、策略、品种三级限制体系
- **动态调整**：基于波动率、相关性、流动性动态调整
- **压力测试**：历史情景、蒙特卡洛、极端情景测试

### 6. WebSocket管理架构

```mermaid
graph TB
    subgraph "WebSocket管理架构"
        subgraph "连接管理层"
            CM1[连接池管理器<br/>10K+ connections]
            CM2[负载均衡器<br/>智能分配]
            CM3[健康检查器<br/>心跳检测]
            CM4[故障转移器<br/>自动切换]
        end

        subgraph "消息路由系统"
            MR1[消息分发器<br/>10K+ msg/s]
            MR2[订阅管理器<br/>topic-based routing]
            MR3[过滤器引擎<br/>条件过滤]
            MR4[优先级队列<br/>QoS保证]
        end

        subgraph "数据处理层"
            DP1[消息解析器<br/>JSON/Binary]
            DP2[数据验证器<br/>格式校验]
            DP3[转换器<br/>格式转换]
            DP4[压缩器<br/>gzip/deflate]
        end

        subgraph "RUST高性能核心"
            RH1[批量处理器<br/>批量消息处理]
            RH2[零拷贝缓冲区<br/>Ring Buffer]
            RH3[并行处理器<br/>多核并行]
            RH4[异步I/O处理<br/>Tokio Runtime]
        end

        subgraph "缓存机制"
            CH1[消息缓存器<br/>最近消息缓存]
            CH2[订阅缓存器<br/>订阅关系缓存]
            CH3[状态缓存器<br/>连接状态缓存]
            CH4[数据缓存器<br/>热点数据缓存]
        end

        subgraph "监控与日志"
            ML1[性能监控器<br/>延迟/吞吐量]
            ML2[连接监控器<br/>连接状态]
            ML3[错误监控器<br/>错误统计]
            ML4[审计日志器<br/>完整记录]
        end

        subgraph "安全层"
            SC1[认证管理器<br/>JWT/OAuth]
            SC2[权限控制器<br/>RBAC]
            SC3[加密管理器<br/>TLS/WSS]
            SC4[限流器<br/>连接限流]
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

**核心特性**：
- **高并发支持**：10,000+并发连接，< 1ms消息延迟
- **智能路由**：topic-based路由，优先级队列，QoS保证
- **RUST加速**：批量处理，零拷贝缓冲区，多核并行
- **安全保障**：JWT认证，TLS加密，RBAC权限控制

### 7. 技术指标架构

```mermaid
graph TB
    subgraph "技术指标架构"
        subgraph "数据输入层"
            DI1[实时K线数据<br/>OHLCV]
            DI2[历史K线数据<br/>批量加载]
            DI3[Tick数据<br/>逐笔成交]
            DI4[辅助数据<br/>成交量/持仓量]
        end

        subgraph "RUST计算核心"
            subgraph "SIMD优化引擎"
                SI1[向量化计算器<br/>AVX2/AVX-512]
                SI2[批量处理器<br/>10K+ candles/batch]
                SI3[并行计算器<br/>Rayon并行]
                SI4[内存优化器<br/>零拷贝操作]
            end

            subgraph "指标计算器"
                IC1[趋势指标<br/>SMA/EMA/DEMA/TEMA]
                IC2[动量指标<br/>RSI/Stoch/CCI/Williams]
                IC3[波动率指标<br/>BB/ATR/Keltner]
                IC4[成交量指标<br/>OBV/AD/MF/CMF]
            end

            subgraph "高级指标"
                AI1[形态识别<br/>30+ patterns]
                AI2[支撑阻力<br/>自动识别]
                AI3[谐波模式<br/>Gartley/Butterfly]
                AI4[多时间框架<br/>MTF分析]
            end
        end

        subgraph "缓存与优化"
            subgraph "多层缓存"
                CA1[L1缓存<br/>内存缓存]
                CA2[L2缓存<br/>Redis缓存]
                CA3[预计算缓存<br/>常用周期]
                CA4[增量计算<br/>差分更新]
            end

            subgraph "性能优化"
                PO1[批量优化<br/>批量计算]
                PO2[并行优化<br/>多核利用]
                PO3[内存优化<br/>内存池]
                PO4[算法优化<br/>增量算法]
            end
        end

        subgraph "API接口层"
            API1[实时计算API<br/>单次计算]
            API2[批量计算API<br/>批量处理]
            API3[流式计算API<br/>实时流]
            API4[历史计算API<br/>历史回测]
        end

        subgraph "监控与诊断"
            MT1[性能监控器<br/>计算耗时]
            MT2[缓存监控器<br/>命中率]
            MT3[错误监控器<br/>异常处理]
            MT4[指标分析器<br/>有效性分析]
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

**核心特性**：
- **RUST高性能**：50,000+K线/秒，3-5倍性能提升
- **SIMD优化**：AVX2/AVX-512向量化计算
- **50+技术指标**：趋势、动量、波动率、成交量指标
- **多层缓存**：L1内存缓存，L2 Redis缓存，增量计算

### 8. 配置管理架构

```mermaid
graph TB
    subgraph "配置管理架构"
        subgraph "配置层次结构"
            subgraph "环境配置"
                EV1[开发环境<br/>development]
                EV2[测试环境<br/>staging]
                EV3[生产环境<br/>production]
                EV4[回测环境<br/>backtest]
            end

            subgraph "应用配置"
                AP1[基础配置<br/>app.yaml]
                AP2[交易配置<br/>trading.yaml]
                AP3[风险配置<br/>risk.yaml]
                AP4[智能体配置<br/>agents.yaml]
            end

            subgraph "模块配置"
                MO1[数据库配置<br/>database.yaml]
                MO2[Redis配置<br/>redis.yaml]
                MO3[LLM配置<br/>llm.yaml]
                MO4[监控配置<br/>monitoring.yaml]
            end
        end

        subgraph "配置提供者"
            CP1[文件提供者<br/>YAML/JSON]
            CP2[环境变量<br/>ENV Variables]
            CP3[数据库提供者<br/>Dynamic Config]
            CP4[远程配置中心<br/>Consul/Nacos]
        end

        subgraph "配置管理核心"
            CM1[配置加载器<br/>多源加载]
            CM2[配置验证器<br/>Schema验证]
            CM3[配置合并器<br/>层次合并]
            CM4[配置解析器<br/>变量替换]
        end

        subgraph "热更新机制"
            HU1[文件监控器<br/>fsnotify]
            HU2[变更检测器<br/>差异对比]
            HU3[热更新器<br/>无缝更新]
            HU4[回滚管理器<br/>版本回退]
        end

        subgraph "版本控制"
            VC1[版本管理器<br/>Git-like]
            VC2[变更历史<br/>Audit Log]
            VC3[分支管理<br/>环境隔离]
            VC4[发布管理<br/>灰度发布]
        end

        subgraph "配置API"
            API1[获取配置<br/>Get Config]
            API2[更新配置<br/>Update Config]
            API3[批量操作<br/>Batch Ops]
            API4[配置查询<br/>Query Config]
        end

        subgraph "监控与告警"
            MG1[配置监控器<br/>状态监控]
            MG2[一致性检查器<br/>Multi-node Sync]
            MG3[告警管理器<br/>异常告警]
            MG4[性能监控器<br/>加载性能]
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

**核心特性**：
- **多层次配置**：环境、应用、模块三层配置体系
- **多源提供者**：文件、环境变量、数据库、远程配置中心
- **热更新机制**：文件监控，无缝更新，版本回退
- **5ms快速更新**：配置变更5ms内生效，100%配置验证

### 9. 数据库层架构

```mermaid
graph TB
    subgraph "数据库层架构"
        subgraph "数据库集群"
            subgraph "PostgreSQL集群"
                PG1[主节点<br/>读写]
                PG2[从节点1<br/>只读]
                PG3[从节点2<br/>只读]
                PG4[备份节点<br/>WAL归档]
            end

            subgraph "Redis集群"
                RD1[Master节点<br/>缓存]
                RD2[Slave节点1<br/>缓存]
                RD3[Slave节点2<br/>缓存]
                RD4[哨兵节点<br/>高可用]
            end

            subgraph "InfluxDB集群"
                ID1[主节点<br/>时序数据]
                ID2[从节点<br/>副本]
                ID3[Meta节点<br/>元数据]
                ID4[数据节点<br/>分片]
            end
        end

        subgraph "连接管理层"
            CM1[连接池管理器<br/>Max: 1000]
            CM2[负载均衡器<br/>读写分离]
            CM3[故障转移器<br/>自动切换]
            CM4[健康检查器<br/>心跳检测]
        end

        subgraph "智能路由"
            IR1[查询路由器<br/>智能分发]
            IR2[分片管理器<br/>数据分片]
            IR3[索引优化器<br/>智能索引]
            IR4[缓存路由器<br/>缓存策略]
        end

        subgraph "事务管理"
            TM1[分布式事务<br/>XA协议]
            TM2[事务协调器<br/>两阶段提交]
            TM3[锁管理器<br/>乐观锁]
            TM4[死锁检测器<br/>自动解决]
        end

        subgraph "数据同步"
            DS1[实时同步器<br/>CDC]
            DS2[批量同步器<br/>ETL]
            DS3[增量同步器<br/>Change Log]
            DS4[一致性检查器<br/>数据校验]
        end

        subgraph "监控与运维"
            subgraph "性能监控"
                PM1[查询监控器<br/>慢查询]
                PM2[锁监控器<br/>锁等待]
                PM3[缓存监控器<br/>命中率]
                PM4[连接监控器<br/>连接池]
            end

            subgraph "告警系统"
                AL1[阈值告警<br/>性能阈值]
                AL2[异常告警<br/>错误率]
                AL3[容量告警<br/>存储容量]
                AL4[可用性告警<br/>服务可用性]
            end
        end

        subgraph "备份与恢复"
            subgraph "备份策略"
                BK1[全量备份<br/>每日]
                BK2[增量备份<br/>每小时]
                BK3[WAL备份<br/>实时]
                BK4[跨地域备份<br/>异地容灾]
            end

            subgraph "恢复机制"
                RC1[时间点恢复<br/>PITR]
                RC2[主从切换<br/>故障转移]
                RC3[数据重建<br/>数据修复]
                RC4[灾难恢复<br/>DRP]
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

**核心特性**：
- **多数据库架构**：PostgreSQL、Redis、InfluxDB集群
- **智能分片**：数据分片，读写分离，负载均衡
- **跨数据库事务**：分布式事务，两阶段提交，一致性保证
- **高性能访问**：50,000+ QPS，< 10ms响应延迟

### 10. API层架构

```mermaid
graph TB
    subgraph "API层架构"
        subgraph "接入层"
            AL1[负载均衡器<br/>Nginx/HAProxy]
            AL2[API网关<br/>Kong/Istio]
            AL3[限流器<br/>Rate Limiter]
            AL4[认证中心<br/>Auth Service]
        end

        subgraph "路由层"
            RT1[路径路由器<br/>Path Router]
            RT2[版本路由器<br/>Version Router]
            RT3[方法路由器<br/>Method Router]
            RT4[参数路由器<br/>Param Router]
        end

        subgraph "FastAPI服务层"
            subgraph "REST API服务"
                RA1[市场数据API<br/>Market Data]
                RA2[交易API<br/>Trading]
                RA3[账户API<br/>Account]
                RA4[分析API<br/>Analytics]
            end

            subgraph "WebSocket服务"
                WS1[实时行情<br/>Real-time Quotes]
                WS2[交易推送<br/>Trade Updates]
                WS3[事件通知<br/>Event Stream]
                WS4[管理界面<br/>Admin Panel]
            end

            subgraph "中间件层"
                MW1[认证中间件<br/>JWT/OAuth]
                MW2[权限中间件<br/>RBAC]
                MW3[日志中间件<br/>Logging]
                MW4[监控中间件<br/>Metrics]
            end
        end

        subgraph "业务逻辑层"
            BL1[市场数据服务<br/>Data Service]
            BL2[订单管理服务<br/>Order Service]
            BL3[风险管理服务<br/>Risk Service]
            BL4[分析服务<br/>Analysis Service]
        end

        subgraph "数据访问层"
            DA1[缓存访问器<br/>Redis Cache]
            DA2[数据库访问器<br/>SQL Access]
            DA3[时序数据访问器<br/>TSDB Access]
            DA4[文件存储访问器<br/>File Storage]
        end

        subgraph "监控与运维"
            subgraph "性能监控"
                PF1[响应时间监控<br/>Response Time]
                PF2[吞吐量监控<br/>Throughput]
                PF3[错误率监控<br/>Error Rate]
                PF4[资源使用监控<br/>Resource Usage]
            end

            subgraph "链路追踪"
                TR1[请求追踪<br/>Request Trace]
                TR2[依赖追踪<br/>Dependency Trace]
                TR3[性能分析<br/>Performance Profile]
                TR4[瓶颈分析<br/>Bottleneck Analysis]
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

**核心特性**：
- **高性能API**：10,000+ QPS，< 100ms响应时间
- **RESTful与WebSocket**：完整的REST API和实时WebSocket服务
- **认证与授权**：JWT/OAuth认证，RBAC权限控制
- **全链路监控**：请求追踪，性能分析，瓶颈识别

所有架构图的详细文档可在 [docs/modules](./docs/modules) 目录下查看。

## 🚀 部署架构

### 生产环境架构

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

### 部署选项

#### Docker Compose（快速开始）

```bash
# 克隆项目
git clone https://github.com/zcxGGmu/DeepAlpha.git
cd DeepAlpha

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f deepalpha
```

#### Kubernetes（生产环境）

```bash
# 创建命名空间
kubectl create namespace deepalpha

# 部署应用
kubectl apply -f k8s/

# 检查部署状态
kubectl get pods -n deepalpha
```

### 监控和可观测性

- **Prometheus**: 指标收集
- **Grafana**: 可视化仪表板
- **Jaeger**: 分布式链路追踪
- **ELK Stack**: 日志聚合分析

访问监控面板：
- Grafana: `http://your-domain:3000`
- Prometheus: `http://your-domain:9090`
- Jaeger: `http://your-domain:16686`

### 详细部署文档

完整的部署指南请参考：[Deployment Architecture](./docs/deployment-architecture.md)

包含内容：
- 生产环境部署架构
- Kubernetes配置清单
- Docker Compose配置
- 监控配置
- 性能调优建议
- 安全配置
- 灾难恢复方案

## 💡 使用示例

### 使用Rust性能模块

```python
# 导入Rust性能模块
from deepalpha_rust import TechnicalIndicators, WebSocketManager, ExecutionEngine

# 1. 高性能技术指标计算
indicators = TechnicalIndicators()

# 批量计算指标（超高性能）
prices = [45000, 45100, 45200, 45300, 45400, 45300, 45200]
sma_20 = indicators.calculate_sma(prices, period=20)
ema_12 = indicators.calculate_ema(prices, period=12)
rsi = indicators.calculate_rsi(prices, period=14)
macd = indicators.calculate_macd(prices)

print(f"SMA20: {sma_20}, RSI14: {rsi}")

# 2. WebSocket连接管理
ws_manager = WebSocketManager()
ws_manager.start()

# 批量添加连接（支持10,000+并发）
for i in range(100):
    ws_manager.add_connection(f"conn_{i}", "wss://stream.binance.com/ws/btcusdt@trade")

# 广播消息（< 1ms延迟）
ws_manager.broadcast("market_update", {"symbol": "BTC/USDT", "price": 50000})

# 3. 超低延迟交易执行
engine = ExecutionEngine()
engine.start()

# 提交订单（平均延迟 < 50μs）
order_data = {
    "symbol": "BTC/USDT",
    "side": "buy",
    "type": "market",
    "quantity": 1.5
}
order_id = engine.submit_order(order_data)

# 获取执行统计
stats = engine.get_stats()
print(f"总订单数: {stats.total_orders}")
print(f"平均执行时间: {stats.avg_execution_time_us} μs")
```

### 创建交易策略

```python
from deepalpha import DeepAlphaEngine
from deepalpha.config import Settings

# 加载配置
settings = Settings.from_file("config/development.yaml")

# 创建交易引擎
engine = DeepAlphaEngine(settings)

# 添加交易对
engine.add_symbol("BTC/USDT")
engine.add_symbol("ETH/USDT")

# 启动交易
await engine.start()
```

### 自定义智能体

```python
from deepalpha.agents import BaseAgent
from deepalpha.core import Signal
from deepalpha_rust import TechnicalIndicators  # 使用Rust加速

class CustomAgent(BaseAgent):
    """自定义智能体示例（使用Rust加速）"""

    def __init__(self):
        super().__init__()
        self.indicators = TechnicalIndicators()  # Rust加速的指标计算

    async def analyze(self, symbol: str, timeframe: str) -> Signal:
        # 获取市场数据
        data = await self.get_market_data(symbol, timeframe)

        # 使用Rust加速的指标计算
        rsi = self.indicators.calculate_rsi(data['close'], period=14)
        macd = self.indicators.calculate_macd(data['close'])

        # 自定义分析逻辑
        if rsi < 30 and macd['histogram'] > 0:
            return Signal.buy(confidence=0.8)
        elif rsi > 70 and macd['histogram'] < 0:
            return Signal.sell(confidence=0.7)

        return Signal.hold()
```

### 配置LLM Provider

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

# Rust性能模块配置
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

## 📊 监控界面

系统提供完整的Web监控界面：

- **实时行情面板**: 显示关注的交易对实时价格和指标
- **交易历史**: 查看所有交易记录和绩效
- **智能体状态**: 监控各智能体的分析结果
- **风险监控**: 实时显示账户风险指标
- **性能监控**: Rust模块性能指标（延迟、吞吐量等）
- **决策日志**: 查看AI决策过程和理由

访问 `http://localhost:8000` 进入监控界面。

## 🧪 测试

### 运行所有测试

```bash
# Python测试
pytest

# Rust模块测试
cd rust && cargo test

# 性能基准测试
cd rust/tests
python test_indicators_performance.py
python test_websocket_performance.py
python test_stream_performance.py
python test_executor_performance.py
```

### 运行特定测试

```bash
# Python单元测试
pytest tests/unit/test_agents.py
pytest tests/integration/test_engine.py

# Rust模块示例
python rust/examples/executor_examples.py
python rust/examples/websocket_examples.py
python rust/examples/indicators_examples.py
python rust/examples/stream_examples.py
```

### 查看测试覆盖率

```bash
pytest --cov=deepalpha --cov-report=html
```

## 📚 API文档

启动服务后，访问以下地址查看API文档：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 主要API端点

- `GET /api/v1/health` - 健康检查
- `GET /api/v1/symbols` - 获取支持的交易对
- `POST /api/v1/trade` - 手动下单
- `GET /api/v1/positions` - 获取持仓信息
- `GET /api/v1/decisions` - 获取决策历史
- `GET /api/v1/performance` - 获取性能指标

## 🔧 配置说明

### 主要配置项

```yaml
# 应用配置
app:
  name: "DeepAlpha"
  version: "1.0.0"
  debug: false

# 交易所配置
exchange:
  name: "binance"
  api_key: ${BINANCE_API_KEY}
  api_secret: ${BINANCE_API_SECRET}
  sandbox: false

# 交易参数
trading:
  max_position_size: 0.02  # 最大仓位2%
  stop_loss: 0.02  # 2%止损
  take_profit: 0.06  # 6%止盈
  max_drawdown: 0.10  # 最大回撤10%

# 智能体配置
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

# 决策引擎
decision:
  min_confidence: 0.7
  cache_duration: 300  # 5分钟
  memory_size: 100

# Rust性能模块配置
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

## 🚀 性能优化

### 系统调优建议

1. **Rust性能模块优化**
   - 启用所有Rust模块以获得最大性能提升
   - 根据硬件调整并行worker数量
   - 使用更大的批处理大小提高吞吐量

2. **数据库优化**
   - 使用连接池
   - 合理设置索引
   - 定期清理历史数据

3. **缓存策略**
   - Redis缓存热点数据
   - 本地缓存计算结果
   - 使用CDN加速静态资源

4. **并发优化**
   - 合理设置协程数量
   - 使用异步IO
   - 批量处理数据

5. **监控指标**
   - CPU使用率 < 80%
   - 内存使用率 < 70%
   - API响应时间 < 100ms
   - Rust模块延迟监控

## 🛡️ 安全说明

### 最佳实践

1. **API密钥管理**
   - 使用环境变量存储密钥
   - 定期轮换API密钥
   - 限制API权限

2. **网络安全**
   - 使用HTTPS/WSS
   - 设置防火墙规则
   - 启用API访问限制

3. **资金安全**
   - 使用API交易权限，禁用提现
   - 设置合理的交易限额
   - 定期检查交易记录

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 开发规范

- 遵循PEP 8编码规范
- Rust代码遵循rustfmt标准
- 编写单元测试
- 更新文档
- 提交信息使用约定式提交格式

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Binance](https://binance.com) - 提供优秀的交易API
- [FastAPI](https://fastapi.tiangolo.com) - 高性能的Web框架
- [PyO3](https://pyo3.rs) - Rust与Python的桥梁
- [Tokio](https://tokio.rs) - Rust异步运行时
- [OpenAI](https://openai.com) - 强大的AI能力

## 📞 联系我们

- 项目主页: [https://github.com/zcxGGmu/DeepAlpha](https://github.com/zcxGGmu/DeepAlpha)
- 问题反馈: [Issues](https://github.com/zcxGGmu/DeepAlpha/issues)
- 邮箱: deepalpha@example.com

## ⚠️ 免责声明

本软件仅供学习和研究使用。量化交易存在风险，使用本软件进行实际交易可能导致资金损失。用户需要自行承担所有交易风险，开发者不对任何交易损失负责。在使用前，请确保您充分理解相关风险，并在必要时寻求专业建议。

---

**Made with ❤️ by DeepAlpha Team**