# NOFX Python 重构技术方案：A股港股智能交易系统（详细版）

## 项目概述

### 背景
本项目旨在将现有的 **NOFX**（基于 Go 语言开发的加密货币 AI 交易系统）重构为 **Python 版本**，并专注于 **A股** 和 **港股** 市场的智能交易。

### 目标
1. 保留 NOFX 的核心架构和设计理念
2. 使用 Python 生态系统的优势，简化开发和维护
3. 适配 A股和港股的交易规则和市场特性
4. 构建可扩展、高性能的智能交易系统

### 核心价值主张
- **统一架构**：一套代码支持 A股和港股两个市场
- **AI 驱动**：利用大语言模型进行交易决策
- **风控优先**：多层次风险控制保障资金安全
- **可观测性**：完整的决策链路追踪和性能分析

---

## 目录

- [一、NOFX 项目深度分析](#一nofx-项目深度分析)
- [二、Python 重构架构设计](#二python-重构架构设计)
- [三、核心模块详细设计](#三核心模块详细设计)
- [四、A股与港股深度适配](#四a股与港股深度适配)
- [五、数据库完整设计](#五数据库完整设计)
- [六、AI 决策引擎详细设计](#六ai-决策引擎详细设计)
- [七、交易执行层完整实现](#七交易执行层完整实现)
- [八、市场数据层深度实现](#八市场数据层深度实现)
- [九、API 服务层设计](#九api-服务层设计)
- [十、前端架构设计](#十前端架构设计)
- [十一、开发详细路线图](#十一开发详细路线图)
- [十二、测试策略](#十二测试策略)
- [十三、部署与运维](#十三部署与运维)
- [十四、监控与告警](#十四监控与告警)
- [十五、性能优化](#十五性能优化)
- [十六、安全设计](#十六安全设计)

---

## 一、NOFX 项目深度分析

### 1.1 Go 版本架构全景

```
┌─────────────────────────────────────────────────────────────────┐
│                        NOFX Go 架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Web UI    │    │  Bootstrap  │    │   Manager   │        │
│  │  (React)    │◄──►│  (启动流程)  │◄──►│ (Trader管理) │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             ▼                                   │
│                    ┌─────────────┐                              │
│                    │  API Server │                              │
│                    │ (HTTP/WebSocket)                          │
│                    └─────────────┘                              │
│                             │                                   │
│         ┌───────────────────┼───────────────────┐               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Auth      │    │   Crypto    │    │   Config    │        │
│  │  (JWT认证)  │    │  (加密服务)  │    │  (配置管理)  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     核心业务层                           │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                           │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐          │   │
│  │  │ Decision │    │  Trader  │    │  Market  │          │   │
│  │  │  Engine  │───►│  Interface│◄───│  Monitor │          │   │
│  │  │ (AI决策) │    │ (交易接口)│    │ (行情监控)│          │   │
│  │  └──────────┘    └──────────┘    └──────────┘          │   │
│  │        │                │                │              │   │
│  │        ▼                ▼                ▼              │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐          │   │
│  │  │    MCP   │    │Binance   │    │  Binance │          │   │
│  │  │  Clients │    │Futures   │    │  Streams │          │   │
│  │  │(LLM接口) │    │(交易执行) │    │ (行情源) │          │   │
│  │  └──────────┘    └──────────┘    └──────────┘          │   │
│  │                                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     辅助服务层                           │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Logger │ Pool │ Hook │ Migrations │ Prompts │ Scripts  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心模块代码分析

#### 1.2.1 决策引擎核心逻辑 (decision/engine.go)

Go 版本的决策引擎包含以下核心功能：

```go
// NOFX Go 版本决策流程
func GetFullDecision(ctx *Context, mcpClient mcp.AIClient) (*FullDecision, error) {
    // 1. 获取市场数据
    fetchMarketDataForContext(ctx)

    // 2. 构建 Prompt
    systemPrompt := buildSystemPrompt(...)
    userPrompt := buildUserPrompt(ctx)

    // 3. 调用 AI
    aiResponse := mcpClient.CallWithMessages(systemPrompt, userPrompt)

    // 4. 解析决策
    decisions := parseFullDecisionResponse(aiResponse, ...)

    // 5. 验证和执行
    validateAndExecuteDecisions(decisions, ctx)

    return decisions
}
```

**关键点**：
- 支持多 AI 模型（DeepSeek、Qwen）
- 动态 Prompt 构建
- 多轮决策验证
- 历史表现反馈

#### 1.2.2 交易接口设计 (trader/interface.go)

```go
// NOFX 交易接口
type Trader interface {
    GetBalance() (map[string]interface{}, error)
    GetPositions() ([]map[string]interface{}, error)
    OpenLong(symbol string, quantity float64, leverage int) (map[string]interface{}, error)
    OpenShort(symbol string, quantity float64, leverage int) (map[string]interface{}, error)
    CloseLong(symbol string, quantity float64) (map[string]interface{}, error)
    CloseShort(symbol string, quantity float64) (map[string]interface{}, error)
    SetLeverage(symbol string, leverage int) error
    SetStopLoss(symbol string, positionSide string, quantity, stopPrice float64) error
    SetTakeProfit(symbol string, positionSide string, quantity, takeProfitPrice float64) error
    CancelAllOrders(symbol string) error
    FormatQuantity(symbol string, quantity float64) (string, error)
}
```

#### 1.2.3 市场数据结构 (market/types.go)

```go
// NOFX 市场数据结构
type Data struct {
    Symbol            string
    CurrentPrice      float64
    PriceChange1h     float64
    PriceChange4h     float64
    CurrentEMA20      float64
    CurrentMACD       float64
    CurrentRSI7       float64
    OpenInterest      *OIData
    FundingRate       float64
    IntradaySeries    *IntradayData      // 3分钟数据
    LongerTermContext *LongerTermData    // 4小时数据
}

type IntradayData struct {
    MidPrices   []float64
    EMA20Values []float64
    MACDValues  []float64
    RSI7Values  []float64
    RSI14Values []float64
    Volume      []float64
    ATR14       float64
}
```

### 1.3 Go vs Python 技术栈对照表

| 层面 | Go 版本 | Python 版本 | 迁移要点 |
|------|---------|-------------|----------|
| **并发模型** | Goroutine + Channel | asyncio + aiohttp | 异步编程模式转换 |
| **类型系统** | 静态类型 | 动态类型 + Pydantic | 使用 Pydantic 增强类型安全 |
| **错误处理** | 显式 error 返回 | 异常处理 | 建立统一异常体系 |
| **依赖管理** | go.mod | poetry/pip | 虚拟环境管理 |
| **配置管理** | JSON + struct | YAML + Pydantic | 配置热加载 |
| **数据库** | SQLite (自研) | PostgreSQL + SQLAlchemy | ORM 映射 |
| **日志** | 自研 logger | structlog | 结构化日志 |
| **测试** | testing | pytest | 异步测试 |
| **部署** | 单一二进制 | Docker + uvicorn | 容器化部署 |

---

## 二、Python 重构架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DeepAlpha Python 架构                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │  Web UI      │    │   CLI        │    │   Admin API  │         │
│  │  (React/TS)  │    │  (Typer)     │    │  (FastAPI)   │         │
│  └──────────────┘    └──────────────┘    └──────────────┘         │
│         │                    │                    │                │
│         └────────────────────┼────────────────────┘                │
│                              ▼                                     │
│                     ┌──────────────┐                               │
│                     │  API Gateway │                               │
│                     │  (FastAPI)   │                               │
│                     └──────────────┘                               │
│                              │                                     │
│         ┌────────────────────┼────────────────────┐                │
│         ▼                    ▼                    ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │   Auth       │    │   Trader     │    │   Decision   │        │
│  │  Service     │    │   Service    │    │   Service    │        │
│  │  (JWT)       │    │  (生命周期)  │    │  (AI决策)    │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                        领域层                              │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │                                                              │   │
│  │  ┌────────────┐    ┌────────────┐    ┌────────────┐        │   │
│  │  │   Trader   │    │   Market   │    │   Risk     │        │   │
│  │  │   Domain   │    │   Domain   │    │   Domain   │        │   │
│  │  │  (交易领域) │    │  (行情领域) │    │  (风控领域) │        │   │
│  │  └────────────┘    └────────────┘    └────────────┘        │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                     │
│         ┌────────────────────┼────────────────────┐                │
│         ▼                    ▼                    ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │  Repository  │    │   Service    │    │   External   │        │
│  │   Layer      │    │   Layer      │    │    Layer     │        │
│  │ (数据访问)    │    │  (业务服务)  │    │  (外部集成)  │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│         │                    │                    │                │
│         ▼                    ▼                    ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ PostgreSQL   │    │    Redis     │    │  Data/LLM    │        │
│  │  (主数据库)  │    │   (缓存)     │    │  (外部服务)  │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 分层架构详解

#### 2.2.1 API 层 (backend/api/)

```python
backend/api/
├── __init__.py
├── app.py                      # FastAPI 应用工厂
├── dependencies.py             # 依赖注入
├── middleware.py               # 中间件
├── exceptions.py               # 异常处理
├── routes/
│   ├── __init__.py
│   ├── auth.py                 # 认证路由
│   ├── traders.py              # 交易员路由
│   ├── decisions.py            # 决策路由
│   ├── positions.py            # 持仓路由
│   ├── markets.py              # 行情路由
│   ├── analytics.py            # 分析路由
│   └── system.py               # 系统路由
└── websockets/
    ├── __init__.py
    ├── manager.py              # WebSocket 连接管理
    └── handlers.py             # 消息处理器
```

#### 2.2.2 服务层 (backend/services/)

```python
backend/services/
├── __init__.py
├── trader_service.py           # 交易员服务
├── decision_service.py         # 决策服务
├── market_service.py           # 市场数据服务
├── position_service.py         # 持仓服务
├── risk_service.py             # 风控服务
├── auth_service.py             # 认证服务
├── analysis_service.py         # 分析服务
└── notification_service.py     # 通知服务
```

#### 2.2.3 领域层 (backend/core/)

```python
backend/core/
├── __init__.py
├── trading_types.py            # 交易类型定义
├── trader.py                   # 交易员领域模型
├── position.py                 # 持仓领域模型
├── order.py                    # 订单领域模型
├── risk_manager.py             # 风险管理器
├── trading_calendar.py         # 交易日历
└── events.py                   # 领域事件
```

#### 2.2.4 数据访问层 (backend/repositories/)

```python
backend/repositories/
├── __init__.py
├── base.py                     # 基础 Repository
├── trader_repository.py        # 交易员数据访问
├── decision_repository.py      # 决策数据访问
├── position_repository.py      # 持仓数据访问
├── market_repository.py        # 市场数据访问
└── analytics_repository.py     # 分析数据访问
```

### 2.3 项目完整目录结构

```
deepalpha/
├── backend/
│   ├── __init__.py
│   ├── main.py                 # 应用入口
│   ├── config.py               # 配置管理
│   ├── constants.py            # 常量定义
│   ├── exceptions.py           # 自定义异常
│   │
│   ├── api/                    # API 层
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── dependencies.py
│   │   ├── middleware.py
│   │   ├── exceptions.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── traders.py
│   │   │   ├── decisions.py
│   │   │   ├── positions.py
│   │   │   ├── markets.py
│   │   │   ├── analytics.py
│   │   │   └── system.py
│   │   └── websockets/
│   │       ├── __init__.py
│   │       ├── manager.py
│   │       └── handlers.py
│   │
│   ├── core/                   # 领域层
│   │   ├── __init__.py
│   │   ├── trading_types.py
│   │   ├── trader.py
│   │   ├── position.py
│   │   ├── order.py
│   │   ├── risk_manager.py
│   │   ├── trading_calendar.py
│   │   └── events.py
│   │
│   ├── models/                 # 数据模型
│   │   ├── __init__.py
│   │   ├── database.py         # SQLAlchemy 模型
│   │   ├── schemas.py          # Pydantic 模型
│   │   └── enums.py
│   │
│   ├── services/               # 业务服务
│   │   ├── __init__.py
│   │   ├── trader_service.py
│   │   ├── decision_service.py
│   │   ├── market_service.py
│   │   ├── position_service.py
│   │   ├── risk_service.py
│   │   ├── auth_service.py
│   │   ├── analysis_service.py
│   │   └── notification_service.py
│   │
│   ├── repositories/           # 数据访问
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── trader_repository.py
│   │   ├── decision_repository.py
│   │   ├── position_repository.py
│   │   ├── market_repository.py
│   │   └── analytics_repository.py
│   │
│   ├── traders/                # 交易执行层
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── exceptions.py
│   │   ├── atrader/
│   │   │   ├── __init__.py
│   │   │   ├── simulator.py
│   │   │   ├── broker.py
│   │   │   └── types.py
│   │   └── hkstock/
│   │       ├── __init__.py
│   │       ├── simulator.py
│   │       ├── broker.py
│   │       └── types.py
│   │
│   ├── markets/                # 市场数据层
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── data_types.py
│   │   ├── indicators.py
│   │   ├── calendar.py
│   │   ├── atrader/
│   │   │   ├── __init__.py
│   │   │   ├── tushare.py
│   │   │   ├── akshare.py
│   │   │   ├── eastmoney.py
│   │   │   └── websocket.py
│   │   └── hkstock/
│   │       ├── __init__.py
│   │       ├── yahoo.py
│   │       ├── eastmoney.py
│   │       └── websocket.py
│   │
│   ├── decisions/              # AI 决策引擎
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── prompt_manager.py
│   │   ├── context_builder.py
│   │   ├── parser.py
│   │   ├── validator.py
│   │   └── feedback.py
│   │
│   ├── llm/                    # LLM 客户端
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── deepseek.py
│   │   ├── qwen.py
│   │   ├── kimi.py
│   │   ├── provider.py
│   │   └── rate_limiter.py
│   │
│   ├── utils/                  # 工具函数
│   │   ├── __init__.py
│   │   ├── crypto.py
│   │   ├── logger.py
│   │   ├── datetime.py
│   │   ├── helpers.py
│   │   └── validators.py
│   │
│   └── tasks/                  # 后台任务
│       ├── __init__.py
│       ├── scheduler.py
│       ├── decision_task.py
│       ├── market_data_task.py
│       └── snapshot_task.py
│
├── frontend/                   # 前端
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── routes/
│   │   ├── stores/
│   │   ├── hooks/
│   │   ├── lib/
│   │   └── types.ts
│   ├── public/
│   └── package.json
│
├── config/                     # 配置文件
│   ├── settings.yaml
│   ├── logging.yaml
│   ├── prompts/
│   │   ├── a_stock_default.txt
│   │   ├── a_stock_aggressive.txt
│   │   ├── hk_stock_default.txt
│   │   └── competition.txt
│   └── stock_pools/
│       ├── a_stock_blue_chips.yaml
│       ├── a_stock_growth.yaml
│       └── hk_stock_tech.yaml
│
├── tests/                      # 测试
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── conftest.py
│
├── scripts/                    # 脚本
│   ├── init_db.py
│   ├── migrate.py
│   ├── seed.py
│   └── benchmark.py
│
├── migrations/                 # 数据库迁移
│   └── versions/
│
├── docs/                       # 文档
│   ├── api/
│   ├── architecture/
│   └── user-guide/
│
├── deployment/                 # 部署配置
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── Dockerfile.dev
│   │   └── docker-compose.yml
│   ├── kubernetes/
│   └── nginx/
│
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

### 2.4 技术栈详解

#### 2.4.1 后端技术栈

| 技术 | 版本 | 用途 | 说明 |
|------|------|------|------|
| **Python** | 3.11+ | 运行时 | 类型提示增强 |
| **FastAPI** | 0.115+ | Web 框架 | 高性能异步 API |
| **SQLAlchemy** | 2.0+ | ORM | 异步 ORM 支持 |
| **Pydantic** | 2.9+ | 数据验证 | V2 版本性能提升 |
| **asyncio** | - | 异步框架 | 核心异步支持 |
| **aiohttp** | 3.10+ | HTTP 客户端 | 异步 HTTP 请求 |
| **Redis** | 5.2+ | 缓存 | 数据缓存和消息队列 |
| **PostgreSQL** | 16+ | 数据库 | 主数据存储 |
| **Alembic** | 1.13+ | 数据库迁移 | 版本化管理 |
| **structlog** | 24.4+ | 日志 | 结构化日志 |
| **cryptography** | 43+ | 加密 | 敏感数据加密 |

#### 2.4.2 数据分析技术栈

| 技术 | 版本 | 用途 | 说明 |
|------|------|------|------|
| **pandas** | 2.2+ | 数据处理 | DataFrame 操作 |
| **polars** | 1.9+ | 高性能数据 | 替代 pandas 部分场景 |
| **numpy** | 2.1+ | 数值计算 | 基础计算库 |
| **pandas-ta** | 0.3.14b | 技术指标 | 技术分析指标库 |
| **TA-Lib** | 0.4.28 | 技术指标 | C 扩展，性能更好 |
| **plotly** | - | 可视化 | 交互式图表 |

#### 2.4.3 LLM 集成技术栈

| 技术 | 版本 | 用途 | 说明 |
|------|------|------|------|
| **openai** | 1.54+ | LLM SDK | 通用 LLM 接口 |
| **httpx** | 0.27+ | HTTP 客户端 | 异步 HTTP |
| **tenacity** | - | 重试机制 | API 调用重试 |
| **tiktoken** | - | Token 计数 | Token 使用统计 |

---

## 三、核心模块详细设计

### 3.1 配置管理系统

#### 3.1.1 配置类设计

```python
# backend/config.py
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """数据库配置"""
    url: str = Field(
        default="postgresql+asyncpg://user:password@localhost/deepalpha",
        description="数据库连接 URL"
    )
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_pre_ping: bool = True

    model_config = SettingsConfigDict(env_prefix="DB_")


class RedisConfig(BaseSettings):
    """Redis 配置"""
    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis 连接 URL"
    )
    decode_responses: bool = True
    socket_connect_timeout: int = 5
    socket_timeout: int = 5
    max_connections: int = 50

    model_config = SettingsConfigDict(env_prefix="REDIS_")


class SecurityConfig(BaseSettings):
    """安全配置"""
    jwt_secret: str = Field(
        default="change-this-secret-in-production",
        description="JWT 签名密钥"
    )
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 86400  # 24 小时
    password_min_length: int = 8

    # 加密配置
    encryption_key: str | None = None

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters")
        return v

    model_config = SettingsConfigDict(env_prefix="SECURITY_")


class LoggingConfig(BaseSettings):
    """日志配置"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "text"] = "json"
    file_path: str | None = None
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5

    model_config = SettingsConfigDict(env_prefix="LOG_")


class MarketConfig(BaseSettings):
    """市场配置"""
    # A股配置
    a_stock_enabled: bool = True
    a_stock_min_quantity: int = 100
    a_stock_t_plus_one: bool = True
    a_stock_commission_rate: float = 0.0003  # 万分之三
    a_stock_stamp_duty_rate: float = 0.001    # 千分之一（卖出）
    a_stock_transfer_fee_rate: float = 0.00001  # 万分之0.1

    # 港股配置
    hk_stock_enabled: bool = True
    hk_stock_t_plus_one: bool = False
    hk_stock_commission_rate: float = 0.001   # 千分之一
    hk_stock_stamp_duty_rate: float = 0.001   # 千分之一（双向）

    # 数据源配置
    tushare_token: str = ""
    tushare_priority: int = 1

    model_config = SettingsConfigDict(env_prefix="MARKET_")


class LLMConfig(BaseSettings):
    """LLM 配置"""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    deepseek_max_tokens: int = 4096
    deepseek_temperature: float = 0.7

    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"
    qwen_max_tokens: int = 4096
    qwen_temperature: float = 0.7

    kimi_api_key: str = ""
    kimi_base_url: str = "https://api.moonshot.cn/v1"
    kimi_model: str = "moonshot-v1-8k"
    kimi_max_tokens: int = 4096
    kimi_temperature: float = 0.7

    # 限流配置
    max_concurrent_requests: int = 5
    request_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0

    model_config = SettingsConfigDict(env_prefix="LLM_")


class RiskConfig(BaseSettings):
    """风险配置"""
    max_position_ratio: float = 0.2        # 单只股票最大持仓比例
    max_total_position_ratio: float = 0.8  # 总持仓最大比例
    max_drawdown: float = 0.15             # 最大回撤
    stop_loss_pct: float = 0.05            # 默认止损 5%
    take_profit_ratio: float = 2.0         # 止盈/止损比
    max_daily_loss: float = 0.05           # 最大日亏损 5%

    model_config = SettingsConfigDict(env_prefix="RISK_")


class TraderConfig(BaseSettings):
    """交易员配置"""
    decision_interval: int = 300           # 决策间隔（秒）
    max_concurrent_decisions: int = 5      # 最大并发决策数
    enable_auto_trading: bool = False      # 是否启用自动交易

    model_config = SettingsConfigDict(env_prefix="TRADER_")


class AppConfig(BaseSettings):
    """应用主配置"""
    app_name: str = "DeepAlpha Trading System"
    version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # 目录配置
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    config_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "config")
    log_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")

    # CORS 配置
    cors_origins: list[str] = ["*"]
    cors_credentials: bool = True
    cors_methods: list[str] = ["*"]
    cors_headers: list[str] = ["*"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    model_config = SettingsConfigDict(env_prefix="APP_")

    # 子配置
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    trader: TraderConfig = Field(default_factory=TraderConfig)

    def create_directories(self):
        """创建必要的目录"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_config() -> AppConfig:
    """获取配置单例"""
    return AppConfig()


# 全局配置实例
config = get_config()
```

#### 3.1.2 配置文件示例

```yaml
# config/settings.yaml
app:
  name: "DeepAlpha Trading System"
  version: "1.0.0"
  environment: "development"
  debug: true
  api_host: "0.0.0.0"
  api_port: 8000
  api_workers: 1
  cors_origins: "http://localhost:3000,http://localhost:5173"

database:
  url: "postgresql+asyncpg://postgres:postgres@localhost:5432/deepalpha"
  echo: false
  pool_size: 10
  max_overflow: 20

redis:
  url: "redis://localhost:6379/0"
  decode_responses: true

security:
  jwt_secret: "${JWT_SECRET:your-secret-key-min-32-characters-long}"
  jwt_algorithm: "HS256"
  jwt_expiration: 86400

logging:
  level: "INFO"
  format: "json"
  file_path: "logs/deepalpha.log"

market:
  a_stock_enabled: true
  a_stock_min_quantity: 100
  a_stock_t_plus_one: true
  a_stock_commission_rate: 0.0003
  a_stock_stamp_duty_rate: 0.001
  hk_stock_enabled: true
  hk_stock_t_plus_one: false
  tushare_token: "${TUSHARE_TOKEN:}"

llm:
  deepseek_api_key: "${DEEPSEEK_API_KEY:}"
  deepseek_model: "deepseek-chat"
  qwen_api_key: "${QWEN_API_KEY:}"
  qwen_model: "qwen-plus"
  max_concurrent_requests: 5
  request_timeout: 60
  max_retries: 3

risk:
  max_position_ratio: 0.2
  max_total_position_ratio: 0.8
  max_drawdown: 0.15
  stop_loss_pct: 0.05
  take_profit_ratio: 2.0
  max_daily_loss: 0.05

trader:
  decision_interval: 300
  max_concurrent_decisions: 5
  enable_auto_trading: false
```

### 3.2 异常处理体系

```python
# backend/exceptions.py
from typing import Any

from fastapi import HTTPException, status


class DeepAlphaException(Exception):
    """基础异常类"""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(DeepAlphaException):
    """数据验证错误"""

    def __init__(self, message: str, field: str | None = None, **details):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details={"field": field, **details} if field else details,
        )


class TradingError(DeepAlphaException):
    """交易错误"""

    def __init__(self, message: str, symbol: str | None = None, **details):
        super().__init__(
            message=message,
            code="TRADING_ERROR",
            details={"symbol": symbol, **details} if symbol else details,
        )


class MarketDataError(DeepAlphaException):
    """市场数据错误"""

    def __init__(self, message: str, symbol: str | None = None, **details):
        super().__init__(
            message=message,
            code="MARKET_DATA_ERROR",
            details={"symbol": symbol, **details} if symbol else details,
        )


class LLMError(DeepAlphaException):
    """LLM 调用错误"""

    def __init__(self, message: str, provider: str | None = None, **details):
        super().__init__(
            message=message,
            code="LLM_ERROR",
            details={"provider": provider, **details} if provider else details,
        )


class RiskLimitError(DeepAlphaException):
    """风险限制错误"""

    def __init__(
        self,
        message: str,
        limit_type: str | None = None,
        current_value: float | None = None,
        limit_value: float | None = None,
        **details,
    ):
        super().__init__(
            message=message,
            code="RISK_LIMIT_ERROR",
            details={
                "limit_type": limit_type,
                "current_value": current_value,
                "limit_value": limit_value,
                **details,
            },
        )


class AuthenticationError(DeepAlphaException):
    """认证错误"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message=message, code="AUTHENTICATION_ERROR")


class AuthorizationError(DeepAlphaException):
    """授权错误"""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message=message, code="AUTHORIZATION_ERROR")


class RateLimitError(DeepAlphaException):
    """限流错误"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        **details,
    ):
        super().__init__(
            message=message,
            code="RATE_LIMIT_ERROR",
            details={"retry_after": retry_after, **details},
        )


# FastAPI 异常处理器
def setup_exception_handlers(app):
    """设置异常处理器"""

    @app.exception_handler(DeepAlphaException)
    async def deepalpha_exception_handler(request, exc: DeepAlphaException):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RateLimitError)
    async def rate_limit_exception_handler(request, exc: RateLimitError):
        from fastapi.responses import JSONResponse

        retry_after = exc.details.get("retry_after", 60)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(retry_after)},
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(AuthenticationError)
    async def auth_exception_handler(request, exc: AuthenticationError):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(AuthorizationError)
    async def authorization_exception_handler(request, exc: AuthorizationError):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )
```

### 3.3 日志系统

```python
# backend/utils/logger.py
import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import Processor

from backend.config import get_config

config = get_config()


def add_app_context(logger, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """添加应用上下文到日志"""
    event_dict.update(
        {
            "app": config.app_name,
            "version": config.version,
            "environment": config.environment,
        }
    )
    return event_dict


def filter_exc_info(logger, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """过滤异常信息"""
    if event_dict.get("exc_info"):
        # 在生产环境中不输出详细异常栈
        if config.environment == "production":
            event_dict.pop("exc_info", None)
    return event_dict


def setup_logging():
    """配置结构化日志"""

    # 确保日志目录存在
    if config.logging.file_path:
        log_file = Path(config.logging.file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # 配置标准库日志
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.logging.level),
    )

    # 配置处理器链
    processors: list[Processor] = [
        # 添加日志级别
        structlog.stdlib.add_log_level,
        # 添加日志名称
        structlog.stdlib.add_logger_name,
        # 添加时间戳
        structlog.processors.TimeStamper(fmt="iso"),
        # 添加调用位置
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        # 添加应用上下文
        add_app_context,
    ]

    # 根据格式选择输出处理器
    if config.logging.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend(
            [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ]
        )

    # 开发环境添加异常信息
    if config.environment == "development":
        processors.append(filter_exc_info)

    # 配置 structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 配置文件处理器（如果指定）
    if config.logging.file_path:
        file_handler = logging.handlers.RotatingFileHandler(
            config.logging.file_path,
            maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count,
        )
        file_handler.setLevel(getattr(logging, config.logging.level))

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


# 获取 logger
def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """获取结构化 logger"""
    return structlog.get_logger(name)


# 模块级 logger
logger = get_logger(__name__)
```

### 3.4 工具函数

```python
# backend/utils/helpers.py
from datetime import date, datetime, time, timedelta
from typing import Any

from .datetime import ensure_timezone, now_ms


def format_number(value: float, decimals: int = 2) -> str:
    """格式化数字"""
    return f"{value:,.{decimals}f}"


def format_percent(value: float, decimals: int = 2) -> str:
    """格式化百分比"""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, currency: str = "CNY") -> str:
    """格式化货币"""
    if currency == "CNY":
        return f"¥{format_number(value)}"
    elif currency == "HKD":
        return f"HK${format_number(value)}"
    elif currency == "USD":
        return f"${format_number(value)}"
    return format_number(value)


def truncate_float(value: float, decimals: int) -> float:
    """截断浮点数"""
    multiplier = 10 ** decimals
    return int(value * multiplier) / multiplier


def round_to_lot_size(quantity: int, lot_size: int) -> int:
    """按手数调整数量"""
    return (quantity // lot_size) * lot_size


def calculate_return(entry_price: float, current_price: float) -> float:
    """计算收益率"""
    if entry_price == 0:
        return 0.0
    return (current_price - entry_price) / entry_price


def calculate_pnl(
    entry_price: float,
    current_price: float,
    quantity: int,
    side: str,
) -> float:
    """计算盈亏"""
    if side == "long":
        return (current_price - entry_price) * quantity
    else:
        return (entry_price - current_price) * quantity


def calculate_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.03,
) -> float:
    """计算夏普比率"""
    if not returns:
        return 0.0

    import numpy as np

    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / 252  # 日化

    if np.std(excess_returns) == 0:
        return 0.0

    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    """计算最大回撤"""
    if not equity_curve:
        return 0.0

    import numpy as np

    equity_array = np.array(equity_curve)
    cummax = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - cummax) / cummax

    return abs(drawdown.min())


def calculate_win_rate(trades: list[dict[str, Any]]) -> float:
    """计算胜率"""
    if not trades:
        return 0.0

    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    return len(winning_trades) / len(trades)


def calculate_profit_factor(trades: list[dict[str, Any]]) -> float:
    """计算盈亏比"""
    if not trades:
        return 0.0

    gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def is_trading_time(market: str = "a_stock") -> bool:
    """判断是否在交易时间"""
    now = ensure_timezone(datetime.now())
    current_time = now.time()
    current_date = now.date()

    # 检查是否是交易日（需要交易日历）
    from backend.core.trading_calendar import get_trading_calendar

    calendar = get_trading_calendar(market)
    if not calendar.is_trading_day(current_date):
        return False

    if market == "a_stock":
        # A股交易时间: 9:30-11:30, 13:00-15:00
        morning_start = time(9, 30)
        morning_end = time(11, 30)
        afternoon_start = time(13, 0)
        afternoon_end = time(15, 0)

        return (morning_start <= current_time <= morning_end) or (
            afternoon_start <= current_time <= afternoon_end
        )

    elif market == "hk_stock":
        # 港股交易时间: 9:30-12:00, 13:00-16:00
        morning_start = time(9, 30)
        morning_end = time(12, 0)
        afternoon_start = time(13, 0)
        afternoon_end = time(16, 0)

        return (morning_start <= current_time <= morning_end) or (
            afternoon_start <= current_time <= afternoon_end
        )

    return False


def parse_stock_code(code: str, market: str) -> tuple[str, str]:
    """解析股票代码"""
    code = code.strip().upper()

    if market == "a_stock":
        # A股代码格式: 000001.SZ 或 600000.SH
        if "." not in code:
            # 自动判断交易所
            if code.startswith("6"):
                return f"{code}.SH", "SH"
            elif code.startswith(("0", "3")):
                return f"{code}.SZ", "SZ"
            elif code.startswith("8") or code.startswith("4"):
                return f"{code}.BJ", "BJ"
        return code, code.split(".")[1]

    elif market == "hk_stock":
        # 港股代码格式: 00700 或 0700.HK
        if "." not in code:
            # 补全到5位
            code = code.zfill(5)
            return f"{code}.HK", "HK"
        return code, code.split(".")[1]

    return code, ""


def mask_sensitive_data(value: str, visible_chars: int = 4) -> str:
    """脱敏敏感数据"""
    if len(value) <= visible_chars:
        return "*" * len(value)
    return value[:visible_chars] + "*" * (len(value) - visible_chars)


def debounce(func):
    """防抖装饰器"""
    import asyncio
    from functools import wraps

    @wraps(func)
    async def wrapper(*args, **kwargs):
        if hasattr(wrapper, "_task"):
            wrapper._task.cancel()
        wrapper._task = asyncio.create_task(func(*args, **kwargs))
        return await wrapper._task

    return wrapper


def throttle(func, delay: float = 1.0):
    """节流装饰器"""
    import asyncio
    from functools import wraps
    from typing import Any, Coroutine

    last_called = [0.0]
    waiting: list[Coroutine[Any, Any, Any]] = []

    @wraps(func)
    async def wrapper(*args, **kwargs):
        last_called[0] = now_ms()

        async def run():
            result = await func(*args, **kwargs)
            waiting.clear()
            last_called[0] = 0
            return result

        if last_called[0] == 0:
            return await run()

        waiting.append(run())
        await asyncio.sleep(delay - (now_ms() - last_called[0]) / 1000)
        return await waiting[-1]

    return wrapper
```

---

## 四、A股与港股深度适配

### 4.1 交易日历系统

```python
# backend/core/trading_calendar.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import ClassVar

import requests

from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradingSession:
    """交易时段"""
    morning_start: str  # "09:30"
    morning_end: str    # "11:30"
    afternoon_start: str  # "13:00"
    afternoon_end: str   # "15:00" 或 "16:00"


class TradingCalendar(ABC):
    """交易日历抽象基类"""

    MARKET: ClassVar[str]
    SESSION: ClassVar[TradingSession]

    def __init__(self):
        self._holidays: set[date] = set()
        self._half_days: set[date] = set()

    @abstractmethod
    async def load_holidays(self, year: int):
        """加载节假日"""
        pass

    def is_trading_day(self, dt: date | datetime) -> bool:
        """判断是否是交易日"""
        if isinstance(dt, datetime):
            dt = dt.date()

        # 周末
        if dt.weekday() >= 5:
            return False

        # 节假日
        if dt in self._holidays:
            return False

        return True

    def is_half_trading_day(self, dt: date | datetime) -> bool:
        """判断是否是半交易日"""
        if isinstance(dt, datetime):
            dt = dt.date()
        return dt in self._half_days

    def is_trading_time(self, dt: datetime) -> bool:
        """判断是否在交易时间内"""
        if not self.is_trading_day(dt):
            return False

        current_time = dt.time()
        from datetime import time

        morning_start = time(*map(int, self.SESSION.morning_start.split(":")))
        morning_end = time(*map(int, self.SESSION.morning_end.split(":")))
        afternoon_start = time(*map(int, self.SESSION.afternoon_start.split(":")))
        afternoon_end = time(*map(int, self.SESSION.afternoon_end.split(":")))

        return (
            morning_start <= current_time <= morning_end
            or afternoon_start <= current_time <= afternoon_end
        )

    def next_trading_day(self, dt: date | datetime) -> date:
        """获取下一个交易日"""
        if isinstance(dt, datetime):
            dt = dt.date()

        next_day = dt + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day

    def prev_trading_day(self, dt: date | datetime) -> date:
        """获取上一个交易日"""
        if isinstance(dt, datetime):
            dt = dt.date()

        prev_day = dt - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day

    def get_trading_days(self, start: date, end: date) -> list[date]:
        """获取日期范围内的交易日"""
        days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days


class AStockTradingCalendar(TradingCalendar):
    """A股交易日历"""

    MARKET = "a_stock"
    SESSION = TradingSession(
        morning_start="09:30",
        morning_end="11:30",
        afternoon_start="13:00",
        afternoon_end="15:00",
    )

    async def load_holidays(self, year: int):
        """从 Tushare 加载 A股节假日"""
        try:
            import tushare as ts

            # 从配置获取 token
            from backend.config import get_config

            config = get_config()
            if not config.market.tushare_token:
                logger.warning("Tushare token not configured, using default holidays")
                self._load_default_holidays(year)
                return

            ts.set_token(config.market.tushare_token)
            pro = ts.pro_api()

            # 获取节假日
            df = pro.trade_cal_exchange(
                exchange="SSE",  # 上交所
                start_date=f"{year}0101",
                end_date=f"{year}1231",
            )

            # 筛选非交易日
            holidays = df[df["is_open"] == 0]["cal_date"].tolist()
            self._holidays = {
                datetime.strptime(d, "%Y%m%d").date() for d in holidays
            }

            logger.info(f"Loaded {len(self._holidays)} holidays for A-Stock {year}")

        except Exception as e:
            logger.error(f"Failed to load holidays from Tushare: {e}")
            self._load_default_holidays(year)

    def _load_default_holidays(self, year: int):
        """加载默认节假日"""
        # A股主要节假日（简化版）
        default_holidays = {
            # 元旦
            f"{year}-01-01",
            # 春节（需要根据农历调整）
            f"{year}-02-10",
            f"{year}-02-11",
            f"{year}-02-12",
            f"{year}-02-13",
            f"{year}-02-14",
            f"{year}-02-15",
            f"{year}-02-16",
            f"{year}-02-17",
            # 清明节
            f"{year}-04-04",
            f"{year}-04-05",
            f"{year}-04-06",
            # 劳动节
            f"{year}-05-01",
            f"{year}-05-02",
            f"{year}-05-03",
            f"{year}-05-04",
            f"{year}-05-05",
            # 端午节
            f"{year}-06-10",
            f"{year}-06-11",
            f"{year}-06-12",
            # 中秋节
            f"{year}-09-15",
            f"{year}-09-16",
            f"{year}-09-17",
            # 国庆节
            f"{year}-10-01",
            f"{year}-10-02",
            f"{year}-10-03",
            f"{year}-10-04",
            f"{year}-10-05",
            f"{year}-10-06",
            f"{year}-10-07",
        }

        self._holidays = {
            datetime.strptime(d, "%Y-%m-%d").date() for d in default_holidays
        }


class HKStockTradingCalendar(TradingCalendar):
    """港股交易日历"""

    MARKET = "hk_stock"
    SESSION = TradingSession(
        morning_start="09:30",
        morning_end="12:00",
        afternoon_start="13:00",
        afternoon_end="16:00",
    )

    async def load_holidays(self, year: int):
        """从公开数据源加载港股节假日"""
        try:
            # 港股节假日可以从港交所网站获取
            url = f"https://www.hkex.com.hk/eng/market/sec_tradinfo/tradcal/{year}.htm"

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # 解析 HTML 提取节假日（这里需要具体实现）
            # 简化版：使用默认节假日
            self._load_default_holidays(year)

            logger.info(f"Loaded holidays for HK-Stock {year}")

        except Exception as e:
            logger.error(f"Failed to load HK holidays: {e}")
            self._load_default_holidays(year)

    def _load_default_holidays(self, year: int):
        """加载默认港股节假日"""
        # 港股主要节假日
        default_holidays = {
            # 元旦
            f"{year}-01-01",
            # 农历新年（需要根据农历调整）
            f"{year}-02-10",
            f"{year}-02-11",
            f"{year}-02-12",
            # 耶稣受难日
            f"{year}-03-29",
            # 复活节星期一
            f"{year}-04-01",
            # 劳动节
            f"{year}-05-01",
            # 端午节
            f"{year}-06-10",
            # 香港特区成立纪念日
            f"{year}-07-01",
            # 中秋节翌日
            f"{year}-09-18",
            # 国庆日
            f"{year}-10-01",
            # 重阳节
            f"{year}-10-23",
            # 圣诞节
            f"{year}-12-25",
            f"{year}-12-26",
        }

        self._holidays = {
            datetime.strptime(d, "%Y-%m-%d").date() for d in default_holidays
        }


# 全局交易日历缓存
_calendars: dict[str, TradingCalendar] = {}


def get_trading_calendar(market: str) -> TradingCalendar:
    """获取交易日历"""
    if market not in _calendars:
        if market == "a_stock":
            _calendars[market] = AStockTradingCalendar()
        elif market == "hk_stock":
            _calendars[market] = HKStockTradingCalendar()
        else:
            raise ValueError(f"Unknown market: {market}")
    return _calendars[market]


async def ensure_holidays_loaded(market: str, year: int):
    """确保节假日已加载"""
    calendar = get_trading_calendar(market)
    if not calendar._holidays:
        await calendar.load_holidays(year)
```

### 4.2 费用计算引擎

```python
# backend/core/fee_calculator.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class Side(Enum):
    """交易方向"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class FeeResult:
    """费用计算结果"""
    commission: Decimal  # 佣金
    stamp_duty: Decimal  # 印花税
    transfer_fee: Decimal  # 过户费
    trading_fee: Decimal  # 交易规费
    total: Decimal  # 总费用

    def __add__(self, other: "FeeResult") -> "FeeResult":
        return FeeResult(
            commission=self.commission + other.commission,
            stamp_duty=self.stamp_duty + other.stamp_duty,
            transfer_fee=self.transfer_fee + other.transfer_fee,
            trading_fee=self.trading_fee + other.trading_fee,
            total=self.total + other.total,
        )


class FeeCalculator(ABC):
    """费用计算器抽象基类"""

    @abstractmethod
    def calculate(
        self,
        price: float,
        quantity: int,
        side: Side,
    ) -> FeeResult:
        """计算交易费用"""
        pass


class AStockFeeCalculator(FeeCalculator):
    """A 股费用计算器"""

    # A股费用配置（可从配置文件读取）
    COMMISSION_RATE = Decimal("0.0003")  # 万分之三
    COMMISSION_MIN = Decimal("5")  # 最低5元

    STAMP_DUTY_RATE = Decimal("0.001")  # 千分之一（仅卖出）
    TRANSFER_FEE_RATE = Decimal("0.00001")  # 万分之0.1

    def calculate(
        self,
        price: float,
        quantity: int,
        side: Side,
    ) -> FeeResult:
        """
        计算 A股交易费用

        Args:
            price: 成交价格
            quantity: 成交数量（股）
            side: 交易方向

        Returns:
            FeeResult: 费用明细
        """
        amount = Decimal(str(price)) * Decimal(str(quantity))

        # 1. 佣金（双向收取，有最低收费）
        commission = max(
            amount * self.COMMISSION_RATE,
            self.COMMISSION_MIN,
        )

        # 2. 印花税（仅卖出收取）
        stamp_duty = (
            amount * self.STAMP_DUTY_RATE
            if side == Side.SELL
            else Decimal("0")
        )

        # 3. 过户费（双向收取）
        transfer_fee = amount * self.TRANSFER_FEE_RATE

        # 4. 交易规费（经手费+证管费，约万分之0.687）
        trading_fee = amount * Decimal("0.0000687")

        total = commission + stamp_duty + transfer_fee + trading_fee

        return FeeResult(
            commission=commission.quantize(Decimal("0.01")),
            stamp_duty=stamp_duty.quantize(Decimal("0.01")),
            transfer_fee=transfer_fee.quantize(Decimal("0.01")),
            trading_fee=trading_fee.quantize(Decimal("0.01")),
            total=total.quantize(Decimal("0.01")),
        )


class HKStockFeeCalculator(FeeCalculator):
    """港股费用计算器"""

    # 港股费用配置
    COMMISSION_RATE = Decimal("0.001")  # 千分之一
    COMMISSION_MIN = Decimal("3")  # 最低3港币

    STAMP_DUTY_RATE = Decimal("0.001")  # 千分之一（双向）
    TRADING_FEE_RATE = Decimal("0.00005")  # 万分之0.5
    CCASS_FEE_RATE = Decimal("0.00002")  # 万分之0.2

    def calculate(
        self,
        price: float,
        quantity: int,
        side: Side,
    ) -> FeeResult:
        """
        计算港股交易费用

        Args:
            price: 成交价格（港币）
            quantity: 成交数量（股）
            side: 交易方向

        Returns:
            FeeResult: 费用明细
        """
        amount = Decimal(str(price)) * Decimal(str(quantity))

        # 1. 佣金（双向收取，有最低收费）
        commission = max(
            amount * self.COMMISSION_RATE,
            self.COMMISSION_MIN,
        )

        # 2. 印花税（双向收取）
        stamp_duty = amount * self.STAMP_DUTY_RATE

        # 3. 交易费（双向收取）
        trading_fee = amount * self.TRADING_FEE_RATE

        # 4. 结算费（双向收取）
        transfer_fee = amount * self.CCASS_FEE_RATE

        total = commission + stamp_duty + trading_fee + transfer_fee

        return FeeResult(
            commission=commission.quantize(Decimal("0.01")),
            stamp_duty=stamp_duty.quantize(Decimal("0.01")),
            transfer_fee=transfer_fee.quantize(Decimal("0.01")),
            trading_fee=trading_fee.quantize(Decimal("0.01")),
            total=total.quantize(Decimal("0.01")),
        )


def get_fee_calculator(market: str) -> FeeCalculator:
    """获取费用计算器"""
    if market == "a_stock":
        return AStockFeeCalculator()
    elif market == "hk_stock":
        return HKStockFeeCalculator()
    else:
        raise ValueError(f"Unknown market: {market}")


# 示例使用
async def example_fee_calculation():
    """费用计算示例"""

    # A股买入 1000 股，价格 10 元
    a_calculator = get_fee_calculator("a_stock")
    a_buy_fees = a_calculator.calculate(
        price=10.0,
        quantity=1000,
        side=Side.BUY,
    )
    logger.info(f"A股买入费用: {a_buy_fees}")

    # A股卖出 1000 股，价格 11 元
    a_sell_fees = a_calculator.calculate(
        price=11.0,
        quantity=1000,
        side=Side.SELL,
    )
    logger.info(f"A股卖出费用: {a_sell_fees}")

    # 港股买入 1000 股，价格 300 港币
    hk_calculator = get_fee_calculator("hk_stock")
    hk_buy_fees = hk_calculator.calculate(
        price=300.0,
        quantity=1000,
        side=Side.BUY,
    )
    logger.info(f"港股买入费用: {hk_buy_fees}")

    # 计算总费用
    total_fees = a_buy_fees + a_sell_fees
    logger.info(f"A股完整交易费用: {total_fees}")
```

---

## 五、数据库完整设计

### 5.1 完整的数据库 Schema

```sql
-- DeepAlpha 数据库 Schema
-- PostgreSQL 16+

-- 启用 UUID 扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 启用 pg_trgm 用于模糊搜索
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- =============================================
-- 用户和认证相关表
-- =============================================

-- 用户表
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    last_login_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 用户会话表
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP
);

-- =============================================
-- AI 模型配置表
-- =============================================

CREATE TABLE ai_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(50) NOT NULL,
    provider VARCHAR(50) NOT NULL, -- 'deepseek', 'qwen', 'kimi'
    model_name VARCHAR(100) NOT NULL,
    api_key_encrypted TEXT NOT NULL,
    base_url VARCHAR(255),
    max_tokens INT DEFAULT 4096,
    temperature DECIMAL(3, 2) DEFAULT 0.7,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, name)
);

-- =============================================
-- 交易员相关表
-- =============================================

CREATE TABLE traders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,

    -- 配置引用
    ai_model_id UUID REFERENCES ai_models(id) ON DELETE SET NULL,
    market_type VARCHAR(20) NOT NULL, -- 'a_stock' or 'hk_stock'
    stock_pool_id UUID, -- 关联股票池

    -- 账户配置
    initial_balance DECIMAL(20, 2) NOT NULL,
    current_balance DECIMAL(20, 2) DEFAULT 0, -- 可用余额
    total_equity DECIMAL(20, 2), -- 总权益（含持仓）

    -- 运行状态
    is_running BOOLEAN DEFAULT FALSE,
    is_paused BOOLEAN DEFAULT FALSE,
    last_decision_at TIMESTAMP,

    -- 性能指标
    total_pnl DECIMAL(20, 2) DEFAULT 0,
    total_pnl_pct DECIMAL(10, 4) DEFAULT 0,
    win_rate DECIMAL(5, 4) DEFAULT 0,
    total_trades INT DEFAULT 0,
    winning_trades INT DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_traders_user_id ON traders(user_id);
CREATE INDEX idx_traders_market_type ON traders(market_type);
CREATE INDEX idx_traders_is_running ON traders(is_running);

-- =============================================
-- 股票池表
-- =============================================

CREATE TABLE stock_pools (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    market_type VARCHAR(20) NOT NULL,
    symbols JSONB NOT NULL, -- 存储股票代码列表及元数据
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- 决策记录表
-- =============================================

CREATE TABLE decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trader_id UUID REFERENCES traders(id) ON DELETE CASCADE,

    -- 决策输入
    system_prompt TEXT,
    user_prompt TEXT,
    market_context JSONB,

    -- 决策输出
    cot_trace TEXT, -- Chain of Thought 推理过程
    decisions JSONB NOT NULL, -- 决策列表

    -- 执行状态
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'executing', 'completed', 'failed'
    execution_result JSONB,
    error_message TEXT,

    -- 性能统计
    ai_request_duration_ms INT,
    execution_duration_ms INT,

    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP
);

CREATE INDEX idx_decisions_trader_id ON decisions(trader_id);
CREATE INDEX idx_decisions_status ON decisions(status);
CREATE INDEX idx_decisions_created_at ON decisions(created_at DESC);

-- =============================================
-- 持仓记录表
-- =============================================

CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trader_id UUID REFERENCES traders(id) ON DELETE CASCADE,

    -- 持仓信息
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'long' or 'short'
    quantity INT NOT NULL,

    -- 价格信息
    entry_price DECIMAL(20, 4) NOT NULL,
    current_price DECIMAL(20, 4),
    stop_loss DECIMAL(20, 4),
    take_profit DECIMAL(20, 4),

    -- 盈亏信息
    unrealized_pnl DECIMAL(20, 2) DEFAULT 0,
    unrealized_pnl_pct DECIMAL(10, 4) DEFAULT 0,
    realized_pnl DECIMAL(20, 2) DEFAULT 0,

    -- 风险指标
    peak_pnl_pct DECIMAL(10, 4), -- 历史最高收益率
    drawdown_pct DECIMAL(10, 4), -- 回撤幅度

    -- T+1 信息（仅A股）
    can_sell_date DATE, -- 可卖出日期

    -- 状态
    is_open BOOLEAN DEFAULT TRUE,
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,

    UNIQUE(trader_id, symbol, side)
);

CREATE INDEX idx_positions_trader_id ON positions(trader_id);
CREATE INDEX idx_positions_is_open ON positions(is_open);
CREATE INDEX idx_positions_symbol ON positions(symbol);

-- =============================================
-- 交易记录表
-- =============================================

CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trader_id UUID REFERENCES traders(id) ON DELETE CASCADE,
    position_id UUID REFERENCES positions(id) ON DELETE SET NULL,

    -- 交易信息
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
    quantity INT NOT NULL,
    price DECIMAL(20, 4) NOT NULL,

    -- 费用
    commission DECIMAL(10, 2),
    stamp_duty DECIMAL(10, 2),
    other_fees DECIMAL(10, 2),
    total_fees DECIMAL(10, 2),

    -- 盈亏（仅平仓时有值）
    pnl DECIMAL(20, 2),
    pnl_pct DECIMAL(10, 4),

    -- 关联决策
    decision_id UUID REFERENCES decisions(id),

    -- 状态
    status VARCHAR(20) DEFAULT 'filled', -- 'pending', 'filled', 'cancelled', 'failed'

    -- 时间戳
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trades_trader_id ON trades(trader_id);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_executed_at ON trades(executed_at DESC);

-- =============================================
-- 账户快照表（用于权益曲线）
-- =============================================

CREATE TABLE account_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trader_id UUID REFERENCES traders(id) ON DELETE CASCADE,

    -- 账户快照
    total_equity DECIMAL(20, 2) NOT NULL,
    available_balance DECIMAL(20, 2) NOT NULL,
    positions_value DECIMAL(20, 2) DEFAULT 0,
    total_pnl DECIMAL(20, 2),
    total_pnl_pct DECIMAL(10, 4),

    -- 持仓统计
    position_count INT DEFAULT 0,
    long_positions INT DEFAULT 0,
    short_positions INT DEFAULT 0,

    -- 风险指标
    margin_used DECIMAL(20, 2) DEFAULT 0,
    margin_ratio DECIMAL(5, 4) DEFAULT 0,
    max_drawdown DECIMAL(10, 4),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_account_snapshots_trader_id ON traders(trader_id);
CREATE INDEX idx_account_snapshots_created_at ON account_snapshots(created_at DESC);

-- =============================================
-- 市场数据表（可选，用于缓存）
-- =============================================

CREATE TABLE market_data_cache (
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, interval, timestamp)
);

-- =============================================
-- 系统配置表
-- =============================================

CREATE TABLE system_configs (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    value_type VARCHAR(20) DEFAULT 'string', -- 'string', 'int', 'float', 'bool', 'json'
    description TEXT,
    is_sensitive BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by UUID REFERENCES users(id)
);

-- =============================================
-- 审计日志表
-- =============================================

CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);

-- =============================================
-- 通知表
-- =============================================

CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    trader_id UUID REFERENCES traders(id) ON DELETE CASCADE,

    -- 通知内容
    type VARCHAR(50) NOT NULL, -- 'info', 'warning', 'error', 'success'
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    data JSONB,

    -- 状态
    is_read BOOLEAN DEFAULT FALSE,
    read_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_notifications_user_id ON notifications(user_id);
CREATE INDEX idx_notifications_is_read ON notifications(is_read);

-- =============================================
-- 函数和触发器
-- =============================================

-- 更新 updated_at 字段的函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为需要的表添加触发器
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_traders_updated_at
    BEFORE UPDATE ON traders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

### 5.2 SQLAlchemy 完整模型定义

```python
# backend/models/database.py
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Decimal as SQLDecimal,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class TimestampMixin:
    """时间戳混入类"""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


# =============================================
# 用户模型
# =============================================

class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 关系
    sessions: Mapped[list["UserSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    ai_models: Mapped[list["AIModel"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    traders: Mapped[list["Trader"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    stock_pools: Mapped[list["StockPool"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class UserSession(Base, TimestampMixin):
    __tablename__ = "user_sessions"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(String(255), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 关系
    user: Mapped["User"] = relationship(back_populates="sessions")


# =============================================
# AI 模型配置
# =============================================

class AIModel(Base, TimestampMixin):
    __tablename__ = "ai_models"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    api_key_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    base_url: Mapped[str | None] = mapped_column(String(255))
    max_tokens: Mapped[int] = mapped_column(Integer, default=4096)
    temperature: Mapped[Decimal] = mapped_column(SQLDecimal(3, 2), default=Decimal("0.7"))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # 关系
    user: Mapped["User"] = relationship(back_populates="ai_models")
    traders: Mapped[list["Trader"]] = relationship(
        back_populates="ai_model", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_ai_models_user_name"),
    )


# =============================================
# 股票池
# =============================================

class StockPool(Base, TimestampMixin):
    __tablename__ = "stock_pools"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    market_type: Mapped[str] = mapped_column(String(20), nullable=False)
    symbols: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)

    # 关系
    user: Mapped["User"] = relationship(back_populates="stock_pools")
    traders: Mapped[list["Trader"]] = relationship(
        back_populates="stock_pool", cascade="all, delete-orphan"
    )


# =============================================
# 交易员
# =============================================

class Trader(Base, TimestampMixin):
    __tablename__ = "traders"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    # 配置
    ai_model_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_models.id", ondelete="SET NULL"),
    )
    market_type: Mapped[str] = mapped_column(String(20), nullable=False)
    stock_pool_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("stock_pools.id", ondelete="SET NULL"),
    )

    # 账户
    initial_balance: Mapped[Decimal] = mapped_column(
        SQLDecimal(20, 2), nullable=False
    )
    current_balance: Mapped[Decimal] = mapped_column(
        SQLDecimal(20, 2), default=Decimal("0")
    )
    total_equity: Mapped[Decimal | None] = mapped_column(SQLDecimal(20, 2))

    # 运行状态
    is_running: Mapped[bool] = mapped_column(Boolean, default=False)
    is_paused: Mapped[bool] = mapped_column(Boolean, default=False)
    last_decision_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 性能指标
    total_pnl: Mapped[Decimal] = mapped_column(SQLDecimal(20, 2), default=Decimal("0"))
    total_pnl_pct: Mapped[Decimal] = mapped_column(
        SQLDecimal(10, 4), default=Decimal("0")
    )
    win_rate: Mapped[Decimal] = mapped_column(SQLDecimal(5, 4), default=Decimal("0"))
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)

    # 关系
    user: Mapped["User"] = relationship(back_populates="traders")
    ai_model: Mapped["AIModel"] = relationship(back_populates="traders")
    stock_pool: Mapped["StockPool"] = relationship(back_populates="traders")
    decisions: Mapped[list["Decision"]] = relationship(
        back_populates="trader", cascade="all, delete-orphan"
    )
    positions: Mapped[list["Position"]] = relationship(
        back_populates="trader", cascade="all, delete-orphan"
    )
    trades: Mapped[list["Trade"]] = relationship(
        back_populates="trader", cascade="all, delete-orphan"
    )
    snapshots: Mapped[list["AccountSnapshot"]] = relationship(
        back_populates="trader", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_traders_user_id", "user_id"),
        Index("ix_traders_market_type", "market_type"),
        Index("ix_traders_is_running", "is_running"),
    )


# =============================================
# 决策记录
# =============================================

class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    trader_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("traders.id", ondelete="CASCADE"),
        nullable=False,
    )

    # 输入
    system_prompt: Mapped[str | None] = mapped_column(Text)
    user_prompt: Mapped[str | None] = mapped_column(Text)
    market_context: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # 输出
    cot_trace: Mapped[str | None] = mapped_column(Text)
    decisions: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # 执行
    status: Mapped[str] = mapped_column(String(20), default="pending")
    execution_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error_message: Mapped[str | None] = mapped_column(Text)

    # 性能
    ai_request_duration_ms: Mapped[int | None] = mapped_column(Integer)
    execution_duration_ms: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 关系
    trader: Mapped["Trader"] = relationship(back_populates="decisions")
    trades: Mapped[list["Trade"]] = relationship(
        back_populates="decision", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_decisions_trader_id", "trader_id"),
        Index("ix_decisions_status", "status"),
        Index("ix_decisions_created_at", "created_at"),
    )


# =============================================
# 持仓
# =============================================

class Position(Base, TimestampMixin):
    __tablename__ = "positions"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    trader_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("traders.id", ondelete="CASCADE"),
        nullable=False,
    )

    # 持仓信息
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)

    # 价格
    entry_price: Mapped[Decimal] = mapped_column(SQLDecimal(20, 4), nullable=False)
    current_price: Mapped[Decimal | None] = mapped_column(SQLDecimal(20, 4))
    stop_loss: Mapped[Decimal | None] = mapped_column(SQLDecimal(20, 4))
    take_profit: Mapped[Decimal | None] = mapped_column(SQLDecimal(20, 4))

    # 盈亏
    unrealized_pnl: Mapped[Decimal] = mapped_column(SQLDecimal(20, 2), default=Decimal("0"))
    unrealized_pnl_pct: Mapped[Decimal] = mapped_column(
        SQLDecimal(10, 4), default=Decimal("0")
    )
    realized_pnl: Mapped[Decimal] = mapped_column(SQLDecimal(20, 2), default=Decimal("0"))

    # 风险
    peak_pnl_pct: Mapped[Decimal | None] = mapped_column(SQLDecimal(10, 4))
    drawdown_pct: Mapped[Decimal | None] = mapped_column(SQLDecimal(10, 4))

    # T+1
    can_sell_date: Mapped[date | None] = mapped_column()

    # 状态
    is_open: Mapped[bool] = mapped_column(Boolean, default=True)
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 关系
    trader: Mapped["Trader"] = relationship(back_populates="positions")
    trades: Mapped[list["Trade"]] = relationship(
        back_populates="position", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("trader_id", "symbol", "side", name="uq_positions_trader_symbol_side"),
        Index("ix_positions_trader_id", "trader_id"),
        Index("ix_positions_is_open", "is_open"),
        Index("ix_positions_symbol", "symbol"),
    )


# =============================================
# 交易记录
# =============================================

class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    trader_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("traders.id", ondelete="CASCADE"),
        nullable=False,
    )
    position_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("positions.id", ondelete="SET NULL"),
    )

    # 交易信息
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[Decimal] = mapped_column(SQLDecimal(20, 4), nullable=False)

    # 费用
    commission: Mapped[Decimal | None] = mapped_column(SQLDecimal(10, 2))
    stamp_duty: Mapped[Decimal | None] = mapped_column(SQLDecimal(10, 2))
    other_fees: Mapped[Decimal | None] = mapped_column(SQLDecimal(10, 2))
    total_fees: Mapped[Decimal | None] = mapped_column(SQLDecimal(10, 2))

    # 盈亏
    pnl: Mapped[Decimal | None] = mapped_column(SQLDecimal(20, 2))
    pnl_pct: Mapped[Decimal | None] = mapped_column(SQLDecimal(10, 4))

    # 关联
    decision_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("decisions.id"),
    )

    # 状态
    status: Mapped[str] = mapped_column(String(20), default="filled")

    # 时间
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # 关系
    trader: Mapped["Trader"] = relationship(back_populates="trades")
    position: Mapped["Position"] = relationship(back_populates="trades")
    decision: Mapped["Decision"] = relationship(back_populates="trades")

    __table_args__ = (
        Index("ix_trades_trader_id", "trader_id"),
        Index("ix_trades_symbol", "symbol"),
        Index("ix_trades_executed_at", "executed_at"),
    )


# =============================================
# 账户快照
# =============================================

class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    trader_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("traders.id", ondelete="CASCADE"),
        nullable=False,
    )

    # 账户快照
    total_equity: Mapped[Decimal] = mapped_column(SQLDecimal(20, 2), nullable=False)
    available_balance: Mapped[Decimal] = mapped_column(SQLDecimal(20, 2), nullable=False)
    positions_value: Mapped[Decimal] = mapped_column(SQLDecimal(20, 2), default=Decimal("0"))
    total_pnl: Mapped[Decimal | None] = mapped_column(SQLDecimal(20, 2))
    total_pnl_pct: Mapped[Decimal | None] = mapped_column(SQLDecimal(10, 4))

    # 持仓统计
    position_count: Mapped[int] = mapped_column(Integer, default=0)
    long_positions: Mapped[int] = mapped_column(Integer, default=0)
    short_positions: Mapped[int] = mapped_column(Integer, default=0)

    # 风险
    margin_used: Mapped[Decimal] = mapped_column(SQLDecimal(20, 2), default=Decimal("0"))
    margin_ratio: Mapped[Decimal] = mapped_column(SQLDecimal(5, 4), default=Decimal("0"))
    max_drawdown: Mapped[Decimal | None] = mapped_column(SQLDecimal(10, 4))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # 关系
    trader: Mapped["Trader"] = relationship(back_populates="snapshots")

    __table_args__ = (
        Index("ix_account_snapshots_trader_id_created_at", "trader_id", "created_at"),
    )
```

---

由于篇幅限制，我将创建一个补充文档来包含剩余的详细章节（六到十六章）。让我继续完成这个详细的技术方案：