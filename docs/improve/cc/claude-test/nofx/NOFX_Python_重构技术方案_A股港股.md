# NOFX Python 重构技术方案：A股港股智能交易系统

## 项目概述

### 背景
本项目旨在将现有的 **NOFX**（基于 Go 语言开发的加密货币 AI 交易系统）重构为 **Python 版本**，并专注于 **A股** 和 **港股** 市场的智能交易。

### 目标
1. 保留 NOFX 的核心架构和设计理念
2. 使用 Python 生态系统的优势，简化开发和维护
3. 适配 A股和港股的交易规则和市场特性
4. 构建可扩展、高性能的智能交易系统

---

## 一、NOFX 项目架构分析

### 1.1 技术栈分析

| 模块 | 原有技术 | Python 替代方案 |
|------|----------|-----------------|
| 后端语言 | Go 1.21+ | Python 3.11+ |
| 前端框架 | React 18 + TypeScript | 保留（或可选 FastAPI + Jinja2） |
| 数据库 | SQLite | PostgreSQL / SQLite |
| Web框架 | 自研 HTTP Server | FastAPI |
| WebSocket | 自研实现 | FastAPI WebSocket |
| 配置管理 | JSON + 加密 | YAML/TOML + cryptography |
| 日志系统 | 自研 | structlog |
| 加密服务 | RSA | cryptography |

### 1.2 核心模块分析

#### 1.2.1 后端核心模块

```
nofx/
├── main.go                 # 程序入口
├── api/                    # API 服务器
│   ├── server.go          # HTTP 服务器
│   └── utils.go           # API 工具函数
├── auth/                   # 认证授权
│   └── auth.go            # JWT 认证
├── config/                 # 配置管理
│   ├── config.go          # 配置结构
│   └── database.go        # 数据库操作
├── crypto/                 # 加密服务
│   ├── crypto.go          # 加密工具
│   └── secure_storage.go  # 安全存储
├── decision/               # AI 决策引擎
│   ├── engine.go          # 决策引擎
│   └── prompt_manager.go  # Prompt 管理
├── trader/                 # 交易执行层
│   ├── interface.go       # 交易接口定义
│   ├── binance_futures.go # Binance 实现
│   ├── hyperliquid_trader.go
│   └── lighter_trader.go
├── market/                 # 市场数据层
│   ├── data.go            # 市场数据结构
│   ├── monitor.go         # WebSocket 监控
│   └── websocket_client.go
├── mcp/                    # AI 模型客户端
│   ├── deepseek_client.go
│   └── qwen_client.go
├── manager/                # 交易员管理
│   └── trader_manager.go  # 交易员生命周期管理
├── pool/                   # 币种池管理
│   └── coin_pool.go
└── logger/                 # 日志服务
    ├── logger.go
    └── decision_logger.go
```

#### 1.2.2 前端架构 (React + TypeScript)

```
web/
├── src/
│   ├── components/        # UI 组件
│   ├── pages/            # 页面组件
│   ├── routes/           # 路由配置
│   ├── stores/           # 状态管理 (Zustand)
│   ├── hooks/            # 自定义 Hooks
│   ├── lib/              # 工具库
│   └── types.ts          # TypeScript 类型定义
```

---

## 二、Python 重构技术方案

### 2.1 整体架构设计

采用 **分层架构** 和 **插件化设计**：

```
deepalpha/                     # 项目根目录
├── backend/                   # 后端服务
│   ├── main.py               # 程序入口
│   ├── api/                  # FastAPI 接口层
│   ├── core/                 # 核心业务逻辑
│   ├── models/               # 数据模型
│   ├── services/             # 业务服务层
│   ├── traders/              # 交易执行层（插件化）
│   ├── markets/              # 市场数据层
│   ├── decisions/            # AI 决策引擎
│   ├── llm/                  # LLM 客户端
│   └── utils/                # 工具函数
├── frontend/                  # 前端（保留 React 或重构）
├── config/                    # 配置文件
├── tests/                     # 测试代码
├── scripts/                   # 脚本工具
└── docs/                      # 文档
```

### 2.2 技术选型

#### 2.2.1 后端技术栈

| 类别 | 技术选择 | 说明 |
|------|----------|------|
| **Web框架** | FastAPI | 高性能、异步支持、自动文档 |
| **数据库** | PostgreSQL | 生产环境；SQLite 用于开发 |
| **ORM** | SQLAlchemy 2.0 | 异步支持、类型提示 |
| **数据验证** | Pydantic V2 | 数据模型和验证 |
| **异步框架** | asyncio + aiohttp | 异步 I/O 和 HTTP 请求 |
| **WebSocket** | FastAPI WebSocket | 实时数据推送 |
| **任务调度** | APScheduler | 定时任务管理 |
| **日志** | structlog | 结构化日志 |
| **配置管理** | pydantic-settings | 类型安全的配置管理 |
| **加密** | cryptography | 加密服务 |
| **认证** | FastAPI Security + JWT | 认证授权 |
| **测试** | pytest + pytest-asyncio | 异步测试支持 |

#### 2.2.2 数据分析技术栈

| 类别 | 技术选择 | 说明 |
|------|----------|------|
| **数据处理** | pandas | 数据分析 |
| **数值计算** | numpy | 数值计算 |
| **技术指标** | ta-lib / pandas-ta | 技术分析指标 |
| **可视化** | plotly | 交互式图表 |
| **时间序列** | polars | 高性能时间序列处理 |

#### 2.2.3 LLM 集成

| 类别 | 技术选择 | 说明 |
|------|----------|------|
| **LLM SDK** | LangChain / LiteLLM | 统一 LLM 接口 |
| **向量数据库** | ChromaDB / Qdrant | 存储 AI 决策历史 |
| **提示工程** | PromptLayer | Prompt 版本管理 |

### 2.3 核心模块设计

#### 2.3.1 项目结构

```python
deepalpha/
├── backend/
│   ├── __init__.py
│   ├── main.py                 # 应用入口
│   ├── config.py               # 配置管理
│   │
│   ├── api/                    # API 层
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI 应用
│   │   ├── routes/             # 路由定义
│   │   │   ├── traders.py
│   │   │   ├── decisions.py
│   │   │   ├── positions.py
│   │   │   └── auth.py
│   │   ├── dependencies.py     # 依赖注入
│   │   └── middleware.py       # 中间件
│   │
│   ├── core/                   # 核心业务
│   │   ├── __init__.py
│   │   ├── trading_types.py    # 交易类型定义
│   │   ├── risk_manager.py     # 风险管理
│   │   └── position_manager.py # 持仓管理
│   │
│   ├── models/                 # 数据模型
│   │   ├── __init__.py
│   │   ├── database.py         # 数据库模型 (SQLAlchemy)
│   │   ├── schemas.py          # Pydantic 模型
│   │   └── enums.py            # 枚举类型
│   │
│   ├── services/               # 业务服务
│   │   ├── __init__.py
│   │   ├── trader_service.py   # 交易员服务
│   │   ├── decision_service.py # 决策服务
│   │   ├── market_service.py   # 市场数据服务
│   │   └── auth_service.py     # 认证服务
│   │
│   ├── traders/                # 交易执行层（插件化）
│   │   ├── __init__.py
│   │   ├── base.py             # 抽象基类
│   │   ├── atrader/            # A股交易接口
│   │   │   ├── simulator.py    # 模拟交易
│   │   │   └── broker.py       # 券商接口
│   │   └── hkstock/            # 港股交易接口
│   │       ├── simulator.py
│   │       └── broker.py
│   │
│   ├── markets/                # 市场数据层
│   │   ├── __init__.py
│   │   ├── base.py             # 抽象基类
│   │   ├── data_types.py       # 数据结构
│   │   ├── atrader/            # A股数据源
│   │   │   ├── tushare.py      # Tushare
│   │   │   ├── akshare.py      # AKShare
│   │   │   └── eastmoney.py    # 东方财富
│   │   └── hkstock/            # 港股数据源
│   │       ├── yahoo.py        # Yahoo Finance
│   │       └── eastmoney.py    # 东方财富
│   │
│   ├── decisions/              # AI 决策引擎
│   │   ├── __init__.py
│   │   ├── engine.py           # 决策引擎
│   │   ├── prompt_manager.py   # Prompt 管理
│   │   ├── context_builder.py  # 上下文构建
│   │   └── parser.py           # 决策解析
│   │
│   ├── llm/                    # LLM 客户端
│   │   ├── __init__.py
│   │   ├── base.py             # 抽象基类
│   │   ├── deepseek.py
│   │   ├── qwen.py
│   │   ├── kimi.py
│   │   └── provider.py         # LLM 提供者工厂
│   │
│   ├── utils/                  # 工具函数
│   │   ├── __init__.py
│   │   ├── crypto.py           # 加密工具
│   │   ├── logger.py           # 日志配置
│   │   └── helpers.py          # 辅助函数
│   │
│   └── websockets/             # WebSocket 处理
│       ├── __init__.py
│       └── manager.py          # 连接管理
│
├── frontend/                   # 前端（可选保留 React）
│   └── ...                     # 与 NOFX 类似
│
├── config/                     # 配置文件
│   ├── settings.yaml           # 主配置
│   ├── prompts/                # Prompt 模板
│   └── stocks/                 # 股票池配置
│
├── tests/                      # 测试
│   ├── unit/
│   ├── integration/
│   └── conftest.py
│
├── scripts/                    # 脚本
│   ├── init_db.py
│   └── migrate.py
│
├── pyproject.toml              # 项目配置
├── requirements.txt            # 依赖列表
└── README.md
```

#### 2.3.2 核心类设计

##### (1) 交易接口抽象类

```python
# backend/traders/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class Side(Enum):
    LONG = "long"
    SHORT = "short"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

@dataclass
class Position:
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    leverage: int = 1

@dataclass
class AccountInfo:
    total_equity: float
    available_balance: float
    positions: list[Position]
    margin_used: float
    margin_ratio: float

class BaseTrader(ABC):
    """交易接口抽象基类"""

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """获取账户信息"""
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """获取所有持仓"""
        pass

    @abstractmethod
    async def open_position(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        """开仓"""
        pass

    @abstractmethod
    async def close_position(
        self,
        symbol: str,
        side: Side,
        quantity: Optional[float] = None,
    ) -> dict:
        """平仓"""
        pass

    @abstractmethod
    async def set_stop_loss(
        self,
        symbol: str,
        side: Side,
        stop_price: float,
    ) -> dict:
        """设置止损"""
        pass

    @abstractmethod
    async def set_take_profit(
        self,
        symbol: str,
        side: Side,
        take_profit_price: float,
    ) -> dict:
        """设置止盈"""
        pass

    @abstractmethod
    async def cancel_orders(self, symbol: str) -> dict:
        """取消所有挂单"""
        pass
```

##### (2) A股交易实现

```python
# backend/traders/atrader/simulator.py
from ..base import BaseTrader, Side, OrderType, Position, AccountInfo

class ATraderSimulator(BaseTrader):
    """A股模拟交易实现"""

    def __init__(self, initial_balance: float = 1_000_000):
        self.balance = initial_balance
        self.positions: dict[str, Position] = {}
        self.equity = initial_balance

    async def get_account(self) -> AccountInfo:
        total_unrealized_pnl = sum(
            p.unrealized_pnl for p in self.positions.values()
        )
        self.equity = self.balance + total_unrealized_pnl

        return AccountInfo(
            total_equity=self.equity,
            available_balance=self.balance,
            positions=list(self.positions.values()),
            margin_used=total_unrealized_pnl,
            margin_ratio=total_unrealized_pnl / self.equity if self.equity > 0 else 0,
        )

    async def get_positions(self) -> list[Position]:
        return list(self.positions.values())

    async def open_position(
        self,
        symbol: str,
        side: Side,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        # 实现开仓逻辑
        pass

    # ... 其他方法实现
```

##### (3) 市场数据接口

```python
# backend/markets/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None

@dataclass
class Kline:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class TechnicalIndicators:
    symbol: str
    timestamp: datetime
    ema20: float
    ema50: float
    macd: float
    rsi6: float
    rsi12: float
    rsi24: float
    atr: float

class BaseMarketDataProvider(ABC):
    """市场数据提供者抽象基类"""

    @abstractmethod
    async def get_realtime_price(self, symbol: str) -> MarketData:
        """获取实时价格"""
        pass

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> list[Kline]:
        """获取K线数据"""
        pass

    @abstractmethod
    async def get_technical_indicators(
        self,
        symbol: str,
    ) -> TechnicalIndicators:
        """获取技术指标"""
        pass

    @abstractmethod
    async def subscribe_realtime(
        self,
        symbols: list[str],
        callback,
    ):
        """订阅实时行情"""
        pass
```

##### (4) AI 决策引擎

```python
# backend/decisions/engine.py
from dataclasses import dataclass
from typing import Optional
from ..llm.base import BaseLLMClient
from ..markets.base import MarketData, TechnicalIndicators
from ..traders.base import AccountInfo, Position
from .context_builder import ContextBuilder
from .parser import DecisionParser

@dataclass
class Decision:
    symbol: str
    action: str  # buy, sell, hold, close
    quantity: Optional[int] = None
    confidence: Optional[int] = None
    reasoning: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class DecisionContext:
    account: AccountInfo
    positions: list[Position]
    market_data: dict[str, MarketData]
    indicators: dict[str, TechnicalIndicators]
    candidate_stocks: list[str]
    performance_history: Optional[dict] = None

class DecisionEngine:
    """AI 决策引擎"""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        context_builder: ContextBuilder,
        parser: DecisionParser,
    ):
        self.llm_client = llm_client
        self.context_builder = context_builder
        self.parser = parser

    async def make_decision(
        self,
        context: DecisionContext,
        custom_prompt: Optional[str] = None,
    ) -> list[Decision]:
        """
        执行 AI 决策流程

        Args:
            context: 决策上下文
            custom_prompt: 自定义提示词（可选）

        Returns:
            决策列表
        """
        # 1. 构建提示词
        system_prompt = self._build_system_prompt()
        user_prompt = await self.context_builder.build(context)

        if custom_prompt:
            user_prompt = custom_prompt + "\n\n" + user_prompt

        # 2. 调用 LLM
        response = await self.llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # 3. 解析决策
        decisions = self.parser.parse(response, context)

        return decisions

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专业的股票交易 AI 助手，专注于 A股和港股市场。

## 交易规则
- A股 T+1 交易制度
- 港股 T+0 交易制度
- A股最小交易单位：100股（1手）
- 港股最小交易单位：视股票而定

## 风险管理
- 单只股票持仓不超过总资金的 20%
- 止损幅度不超过 5%
- 止盈目标至少 1:2 风险收益比

## 输出格式
请以 JSON 格式输出决策，包含以下字段：
- symbol: 股票代码
- action: 操作类型 (buy/sell/hold/close)
- quantity: 交易数量（股数）
- confidence: 信心度 (0-100)
- reasoning: 决策理由
- stop_loss: 止损价格（可选）
- take_profit: 止盈价格（可选）

请基于提供的市场数据和账户状态进行分析，给出合理的交易决策。"""
```

#### 2.3.3 FastAPI 接口设计

```python
# backend/api/app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .routes import traders, decisions, positions, auth
from ..websockets.manager import ConnectionManager

def create_app() -> FastAPI:
    app = FastAPI(
        title="DeepAlpha Trading API",
        description="A股港股智能交易系统 API",
        version="1.0.0",
    )

    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(auth.router, prefix="/api/auth", tags=["认证"])
    app.include_router(traders.router, prefix="/api/traders", tags=["交易员"])
    app.include_router(decisions.router, prefix="/api/decisions", tags=["决策"])
    app.include_router(positions.router, prefix="/api/positions", tags=["持仓"])

    # WebSocket 端点
    websocket_manager = ConnectionManager()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await websocket_manager.broadcast(data)
        except WebSocketDisconnect:
            websocket_manager.disconnect(websocket)

    return app
```

---

## 三、A股与港股适配方案

### 3.1 A股市场特性

#### 3.1.1 交易规则

| 特性 | 说明 | 实现要点 |
|------|------|----------|
| **交易时间** | 周一至周五 9:30-11:30, 13:00-15:00 | 时间校验，禁止非交易时间下单 |
| **T+1 制度** | 当日买入次日才能卖出 | 持仓管理中增加 T+1 检查 |
| **涨跌停限制** | 主板 ±10%, 创业板/科创板 ±20% | 价格限制校验 |
| **最小单位** | 100股（1手） | 订单数量必须是 100 的整数倍 |
| **印花税** | 卖出时 0.1% | 费用计算 |
| **过户费** | 0.001% | 费用计算 |

#### 3.1.2 数据接口

```python
# backend/markets/atrader/tushare.py
import tushare as ts
from ..base import BaseMarketDataProvider, MarketData, Kline

class TushareDataProvider(BaseMarketDataProvider):
    """Tushare 数据源"""

    def __init__(self, token: str):
        ts.set_token(token)
        self.pro = ts.pro_api()

    async def get_realtime_price(self, symbol: str) -> MarketData:
        df = self.pro.daily(
            ts_code=symbol,
            trade_date=datetime.now().strftime("%Y%m%d"),
        )
        # 处理数据...
        pass

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> list[Kline]:
        # interval 映射: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
        df = self.pro.daily(
            ts_code=symbol,
            start_date=(datetime.now() - timedelta(days=limit)).strftime("%Y%m%d"),
            end_date=datetime.now().strftime("%Y%m%d"),
        )
        # 转换为 Kline 对象列表
        pass
```

### 3.2 港股市场特性

#### 3.2.1 交易规则

| 特性 | 说明 | 实现要点 |
|------|------|----------|
| **交易时间** | 周一至周五 9:30-12:00, 13:00-16:00 | 时间校验 |
| **T+0 制度** | 当日买入可当日卖出 | 无 T+1 限制 |
| **无涨跌停** | 无涨跌停限制 | 需要额外风险控制 |
| **最小单位** | 每只股票不同 | 需查询每只股票的 trading_unit |
| **印花税** | 买卖双向 0.1% | 费用计算 |
| **交易货币** | 港币 (HKD) | 汇率转换 |

#### 3.2.2 数据接口

```python
# backend/markets/hkstock/yahoo.py
import yfinance as yf
from ..base import BaseMarketDataProvider, MarketData, Kline

class YahooFinanceHK(BaseMarketDataProvider):
    """Yahoo Finance 港股数据源"""

    async def get_realtime_price(self, symbol: str) -> MarketData:
        # 港股代码格式: 0700.HK (腾讯)
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            price=info.get("currentPrice") or info.get("regularMarketPrice"),
            volume=info.get("volume", 0),
        )

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> list[Kline]:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            period=f"{limit}d",
            interval=interval,
        )
        # 转换为 Kline 对象列表
        pass
```

### 3.3 技术指标适配

A股和港股技术指标计算方法相同，但需要考虑：

1. **交易时间差异**：港股有午间休市，需要处理数据连续性
2. **节假日差异**：两地节假日不同，需要交易日历校验
3. **除权除息**：需要复权处理

```python
# backend/markets/indicators.py
import pandas as pd
import pandas_ta as ta

class TechnicalIndicatorCalculator:
    """技术指标计算器"""

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
        """计算 EMA"""
        return df["close"].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_macd(df: pd.DataFrame) -> dict:
        """计算 MACD"""
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        return {
            "macd": macd_df["MACD_12_26_9"],
            "signal": macd_df["MACDs_12_26_9"],
            "histogram": macd_df["MACDh_12_26_9"],
        }

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算 RSI"""
        return ta.rsi(df["close"], length=period)

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算 ATR"""
        return ta.atr(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            length=period,
        )
```

---

## 四、数据库设计

### 4.1 核心数据表

```sql
-- 交易员表
CREATE TABLE traders (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    name VARCHAR(100) NOT NULL,
    ai_model_id VARCHAR(50) NOT NULL,
    market_type VARCHAR(20) NOT NULL, -- 'a_stock' or 'hk_stock'
    is_running BOOLEAN DEFAULT FALSE,
    initial_balance DECIMAL(20, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AI 模型配置表
CREATE TABLE ai_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    provider VARCHAR(50) NOT NULL, -- 'deepseek', 'qwen', 'kimi'
    api_key_encrypted TEXT NOT NULL,
    base_url VARCHAR(255),
    max_tokens INT DEFAULT 4096,
    temperature DECIMAL(3, 2) DEFAULT 0.7,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 决策记录表
CREATE TABLE decisions (
    id SERIAL PRIMARY KEY,
    trader_id INTEGER REFERENCES traders(id),
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(20) NOT NULL,
    quantity INT,
    price DECIMAL(20, 4),
    stop_loss DECIMAL(20, 4),
    take_profit DECIMAL(20, 4),
    confidence INT,
    reasoning TEXT,
    system_prompt TEXT,
    user_prompt TEXT,
    cot_trace TEXT,
    executed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 持仓记录表
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    trader_id INTEGER REFERENCES traders(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'long' or 'short'
    quantity INT NOT NULL,
    entry_price DECIMAL(20, 4) NOT NULL,
    current_price DECIMAL(20, 4),
    unrealized_pnl DECIMAL(20, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trader_id, symbol, side)
);

-- 交易记录表
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    trader_id INTEGER REFERENCES traders(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INT NOT NULL,
    price DECIMAL(20, 4) NOT NULL,
    commission DECIMAL(20, 2),
    pnl DECIMAL(20, 2),
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 账户快照表（用于权益曲线）
CREATE TABLE account_snapshots (
    id SERIAL PRIMARY KEY,
    trader_id INTEGER REFERENCES traders(id),
    total_equity DECIMAL(20, 2) NOT NULL,
    available_balance DECIMAL(20, 2) NOT NULL,
    total_pnl DECIMAL(20, 2),
    position_count INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 股票池表
CREATE TABLE stock_pools (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    name VARCHAR(100) NOT NULL,
    market_type VARCHAR(20) NOT NULL,
    symbols JSONB NOT NULL, -- 存储股票代码列表
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 系统配置表
CREATE TABLE system_configs (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4.2 SQLAlchemy 模型

```python
# backend/models/database.py
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    Numeric,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

class Trader(Base):
    __tablename__ = "traders"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), nullable=False)
    name = Column(String(100), nullable=False)
    ai_model_id = Column(String(50), nullable=False)
    market_type = Column(String(20), nullable=False)
    is_running = Column(Boolean, default=False)
    initial_balance = Column(Numeric(20, 2), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    decisions = relationship("Decision", back_populates="trader")
    positions = relationship("Position", back_populates="trader")
    trades = relationship("Trade", back_populates="trader")
    snapshots = relationship("AccountSnapshot", back_populates="trader")

class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True)
    trader_id = Column(Integer, ForeignKey("traders.id"))
    symbol = Column(String(20), nullable=False)
    action = Column(String(20), nullable=False)
    quantity = Column(Integer)
    price = Column(Numeric(20, 4))
    stop_loss = Column(Numeric(20, 4))
    take_profit = Column(Numeric(20, 4))
    confidence = Column(Integer)
    reasoning = Column(Text)
    system_prompt = Column(Text)
    user_prompt = Column(Text)
    cot_trace = Column(Text)
    executed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    trader = relationship("Trader", back_populates="decisions")

class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    trader_id = Column(Integer, ForeignKey("traders.id"))
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Numeric(20, 4), nullable=False)
    current_price = Column(Numeric(20, 4))
    unrealized_pnl = Column(Numeric(20, 2))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    trader = relationship("Trader", back_populates="positions")
    __table_args__ = (
        UniqueConstraint("trader_id", "symbol", "side", name="unique_position"),
    )

class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"

    id = Column(Integer, primary_key=True)
    trader_id = Column(Integer, ForeignKey("traders.id"))
    total_equity = Column(Numeric(20, 2), nullable=False)
    available_balance = Column(Numeric(20, 2), nullable=False)
    total_pnl = Column(Numeric(20, 2))
    position_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    trader = relationship("Trader", back_populates="snapshots")
```

---

## 五、开发路线规划

### 5.1 阶段划分

#### 第一阶段：基础框架搭建（2-3周）

**目标**：建立项目基础架构和核心接口

| 任务 | 说明 | 优先级 |
|------|------|--------|
| 项目初始化 | 目录结构、依赖配置、基础工具类 | P0 |
| 数据库设计 | SQLAlchemy 模型定义、迁移脚本 | P0 |
| 配置管理 | Pydantic Settings、加密配置 | P0 |
| 日志系统 | structlog 配置、日志格式 | P1 |
| API 框架 | FastAPI 初始化、中间件、依赖注入 | P0 |

**交付物**：
- 可运行的项目骨架
- 数据库初始化脚本
- 基础配置和日志系统

#### 第二阶段：市场数据层（2-3周）

**目标**：实现 A股和港股数据获取

| 任务 | 说明 | 优先级 |
|------|------|--------|
| 数据接口抽象 | BaseMarketDataProvider 定义 | P0 |
| A股数据源 | Tushare / AKShare 实现 | P0 |
| 港股数据源 | Yahoo Finance 实现 | P0 |
| 技术指标计算 | EMA, MACD, RSI, ATR 等 | P0 |
| 数据缓存 | Redis 缓存实时数据 | P1 |
| WebSocket 推送 | 实时行情推送 | P1 |

**交付物**：
- 完整的市场数据获取模块
- 技术指标计算工具
- WebSocket 实时推送接口

#### 第三阶段：交易执行层（2-3周）

**目标**：实现模拟交易功能

| 任务 | 说明 | 优先级 |
|------|------|--------|
| 交易接口抽象 | BaseTrader 定义 | P0 |
| A股模拟交易 | Simulator 实现（含 T+1） | P0 |
| 港股模拟交易 | Simulator 实现（T+0） | P0 |
| 订单管理 | 订单状态跟踪 | P0 |
| 风险控制 | 仓位限制、止损检查 | P0 |
| 费用计算 | 印花税、佣金、过户费 | P1 |

**交付物**：
- 完整的模拟交易系统
- 风险控制和费用计算模块

#### 第四阶段：AI 决策引擎（3-4周）

**目标**：实现 AI 决策流程

| 任务 | 说明 | 优先级 |
|------|------|--------|
| LLM 客户端 | DeepSeek, Qwen, Kimi 接口 | P0 |
| Prompt 管理 | Prompt 模板加载、动态替换 | P0 |
| 上下文构建 | 市场数据、账户状态整合 | P0 |
| 决策解析 | LLM 输出解析、校验 | P0 |
| 决策记录 | 决策历史存储、分析 | P0 |
| 自学习机制 | 历史表现分析 | P1 |

**交付物**：
- 完整的 AI 决策引擎
- Prompt 管理系统
- 决策历史分析工具

#### 第五阶段：API 和前端（2-3周）

**目标**：实现用户交互界面

| 任务 | 说明 | 优先级 |
|------|------|--------|
| REST API | 交易员、决策、持仓接口 | P0 |
| 认证授权 | JWT 认证 | P0 |
| WebSocket | 实时数据推送 | P0 |
| 前端开发 | React 组件开发 | P1 |
| 图表可视化 | Recharts 权益曲线 | P1 |

**交付物**：
- 完整的 REST API
- 前端管理界面
- 实时数据展示

#### 第六阶段：测试和优化（2-3周）

**目标**：确保系统稳定性

| 任务 | 说明 | 优先级 |
|------|------|--------|
| 单元测试 | pytest 覆盖核心模块 | P0 |
| 集成测试 | 端到端测试 | P0 |
| 性能优化 | 数据库查询、异步优化 | P1 |
| 安全加固 | API 加密、输入验证 | P0 |
| 文档完善 | API 文档、使用手册 | P1 |

**交付物**：
- 完整的测试用例
- 性能优化报告
- 用户文档

### 5.2 里程碑

| 里程碑 | 时间 | 标志 |
|--------|------|------|
| M1: 框架完成 | 第 3 周 | 项目可启动，数据库可连接 |
| M2: 数据可用 | 第 6 周 | 可获取实时行情和技术指标 |
| M3: 交易可用 | 第 9 周 | 可进行模拟交易 |
| M4: AI 可用 | 第 13 周 | AI 可自主决策 |
| M5: 系统可用 | 第 16 周 | 完整功能可演示 |
| M6: 生产就绪 | 第 19 周 | 通过测试，可部署 |

---

## 六、关键技术难点与解决方案

### 6.1 A股 T+1 限制

**问题**：当日买入的股票当日不能卖出

**解决方案**：
```python
class ATraderSimulator(BaseTrader):
    def __init__(self):
        self.positions: dict[str, Position] = {}
        self.pending_sell: dict[str, datetime] = {}  # 记录可卖出日期

    async def open_position(self, symbol: str, side: Side, quantity: int):
        if side == Side.LONG:
            # 买入后，次日才能卖出
            tomorrow = datetime.now().date() + timedelta(days=1)
            # 跳过非交易日
            while tomorrow in self.holidays or tomorrow.weekday() >= 5:
                tomorrow += timedelta(days=1)
            self.pending_sell[symbol] = tomorrow
        # ...

    async def close_position(self, symbol: str, side: Side, quantity: int):
        if side == Side.LONG:
            sell_date = self.pending_sell.get(symbol)
            if sell_date and datetime.now().date() < sell_date:
                raise ValueError("T+1 限制：当日买入次日才能卖出")
        # ...
```

### 6.2 A股最小交易单位

**问题**：A 股必须以 100 股的整数倍交易

**解决方案**：
```python
def validate_order_quantity(quantity: int, market: str) -> bool:
    if market == "a_stock":
        return quantity % 100 == 0 and quantity > 0
    elif market == "hk_stock":
        return quantity > 0  # 港股每只股票交易单位不同
    return True
```

### 6.3 涨跌停限制

**问题**：A股有涨跌停限制，订单价格不能超出

**解决方案**：
```python
def check_price_limit(symbol: str, price: float, side: str, prev_close: float):
    # 根据股票类型判断涨跌停幅度
    # 主板 ±10%, 创业板/科创板 ±20%, ST ±5%
    limit_type = get_stock_limit_type(symbol)

    if limit_type == "main_board":
        limit_pct = 0.10
    elif limit_type in ("chi_next", "star_market"):
        limit_pct = 0.20
    elif limit_type == "st":
        limit_pct = 0.05

    upper_limit = prev_close * (1 + limit_pct)
    lower_limit = prev_close * (1 - limit_pct)

    if side == "buy" and price > upper_limit:
        raise ValueError(f"买入价格超过涨停价 {upper_limit}")
    if side == "sell" and price < lower_limit:
        raise ValueError(f"卖出价格低于跌停价 {lower_limit}")
```

### 6.4 港股交易单位

**问题**：港股每只股票的最小交易单位不同

**解决方案**：
```python
# 预先获取每只港股的交易单位
HK_STOCK_UNITS = {
    "0700.HK": 100,    # 腾讯
    "9988.HK": 100,    # 阿里巴巴
    "0941.HK": 500,    # 中国移动
    # ... 更多股票
}

def get_hk_trading_unit(symbol: str) -> int:
    if symbol not in HK_STOCK_UNITS:
        # 从数据源获取
        unit = fetch_trading_unit(symbol)
        HK_STOCK_UNITS[symbol] = unit
    return HK_STOCK_UNITS[symbol]

def validate_hk_order(quantity: int, symbol: str) -> bool:
    unit = get_hk_trading_unit(symbol)
    return quantity % unit == 0 and quantity > 0
```

### 6.5 实时数据延迟

**问题**：免费数据源通常有延迟

**解决方案**：
1. 使用多个数据源交叉验证
2. 实现本地数据缓存和更新机制
3. 对于关键数据，考虑付费实时数据源

```python
class MultiSourceMarketData(BaseMarketDataProvider):
    def __init__(self):
        self.sources = [
            TushareDataProvider(),
            AkShareDataProvider(),
            EastmoneyDataProvider(),
        ]

    async def get_realtime_price(self, symbol: str) -> MarketData:
        # 尝试多个数据源
        for source in self.sources:
            try:
                return await source.get_realtime_price(symbol)
            except Exception as e:
                logger.warning(f"{source.__class__.__name__} 获取数据失败: {e}")

        # 所有源都失败，返回缓存数据
        return self._get_cached_data(symbol)
```

---

## 七、依赖清单

### 7.1 核心依赖

```txt
# requirements.txt

# Web 框架
fastapi==0.115.0
uvicorn[standard]==0.32.0
websockets==13.1

# 数据库
sqlalchemy==2.0.35
asyncpg==0.29.0
alembic==1.13.3
redis==5.2.0

# 数据验证
pydantic==2.9.2
pydantic-settings==2.6.0
email-validator==2.2.0

# 认证
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.12

# 数据分析
pandas==2.2.3
numpy==2.1.3
polars==1.9.0

# 技术分析
pandas-ta==0.3.14b0
TA-Lib==0.4.28

# A股数据源
tushare==1.5.4
akshare==1.14.88

# 港股数据源
yfinance==0.2.50

# HTTP 客户端
httpx==0.27.2
aiohttp==3.10.10

# LLM SDK
langchain==0.3.7
langchain-openai==0.2.8
openai==1.54.4

# 加密
cryptography==43.0.3

# 日志
structlog==24.4.0

# 任务调度
apscheduler==3.10.4

# 工具
python-dotenv==1.0.1
typer==0.13.1
rich==13.9.4

# 测试
pytest==8.3.3
pytest-asyncio==0.24.0
pytest-cov==6.0.0
httpx==0.27.2  # 用于测试 API
```

### 7.2 开发依赖

```txt
# requirements-dev.txt

# 代码质量
ruff==0.8.0
mypy==1.13.0
black==24.10.0

# 类型存根
types-redis==4.6.0.20240903

# 文档
mkdocs==1.6.1
mkdocs-material==9.5.44

# 性能分析
py-spy==0.3.14
memory-profiler==0.61.0
```

---

## 八、配置文件示例

### 8.1 主配置文件

```yaml
# config/settings.yaml

app:
  name: "DeepAlpha Trading System"
  version: "1.0.0"
  environment: "development"
  debug: true

server:
  host: "0.0.0.0"
  port: 8000
  reload: true

database:
  url: "postgresql+asyncpg://user:password@localhost/deepalpha"
  echo: false
  pool_size: 10
  max_overflow: 20

redis:
  url: "redis://localhost:6379/0"
  decode_responses: true

security:
  jwt_secret: "${JWT_SECRET:your-secret-key-change-in-production}"
  jwt_algorithm: "HS256"
  jwt_expiration: 86400  # 24 hours

logging:
  level: "INFO"
  format: "json"
  handlers:
    - type: "console"
      level: "INFO"
    - type: "file"
      level: "DEBUG"
      filename: "logs/deepalpha.log"

# A股配置
a_stock:
  enabled: true
  trading_hours:
    morning:
      start: "09:30"
      end: "11:30"
    afternoon:
      start: "13:00"
      end: "15:00"
  min_quantity: 100  # 最小100股
  t_plus_one: true
  fees:
    commission: 0.0003  # 万分之三
    stamp_duty: 0.001   # 千分之一（仅卖出）
    transfer_fee: 0.00001  # 万分之0.1

# 港股配置
hk_stock:
  enabled: true
  trading_hours:
    morning:
      start: "09:30"
      end: "12:00"
    afternoon:
      start: "13:00"
      end: "16:00"
  t_plus_one: false
  fees:
    commission: 0.001  # 千分之一
    stamp_duty: 0.001  # 千分之一（双向）
    trading_fee: 0.00005  # 万分之0.5

# 数据源配置
data_sources:
  tushare:
    token: "${TUSHARE_TOKEN:}"
    priority: 1

  akshare:
    priority: 2

  yahoo_finance:
    priority: 3

# AI 模型配置
ai_models:
  deepseek:
    provider: "deepseek"
    base_url: "https://api.deepseek.com/v1"
    api_key: "${DEEPSEEK_API_KEY:}"
    model: "deepseek-chat"
    max_tokens: 4096
    temperature: 0.7

  qwen:
    provider: "qwen"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: "${QWEN_API_KEY:}"
    model: "qwen-plus"
    max_tokens: 4096
    temperature: 0.7

# 风险管理
risk_management:
  max_position_ratio: 0.2  # 单只股票最大持仓比例
  max_total_position_ratio: 0.8  # 总持仓最大比例
  max_drawdown: 0.15  # 最大回撤
  stop_loss_pct: 0.05  # 默认止损 5%
  take_profit_ratio: 2.0  # 止盈/止损比

# 交易员管理
trader:
  decision_interval: 300  # 决策间隔（秒）
  max_concurrent_decisions: 5  # 最大并发决策数

# Prompt 配置
prompts:
  directory: "config/prompts"
  default: "default.txt"
```

### 8.2 Prompt 模板示例

```txt
# config/prompts/a_stock_default.txt

你是一个专业的A股交易 AI 助手。

## 市场数据
{{market_data}}

## 技术指标
{{indicators}}

## 账户状态
- 总资产: {{total_equity}}
- 可用资金: {{available_balance}}
- 持仓数量: {{position_count}}

## 当前持仓
{% for position in positions %}
- {{position.symbol}}: {{position.side}} {{position.quantity}}股, 成本{{position.entry_price}}, 浮盈{{position.unrealized_pnl}}
{% endfor %}

## 历史表现（最近20次交易）
{{performance_summary}}

## 候选股票池
{{candidate_stocks}}

## 任务
请基于以上信息，分析市场形势，给出交易决策。

## 输出格式
```json
[
  {
    "symbol": "股票代码",
    "action": "buy/sell/hold/close",
    "quantity": 股数,
    "confidence": 0-100,
    "reasoning": "决策理由",
    "stop_loss": 止损价格（可选）,
    "take_profit": 止盈价格（可选）
  }
]
```

## 注意事项
1. 严格遵守风险管理规则
2. A股T+1，当日买入次日才能卖出
3. 最小交易单位100股
4. 考虑涨跌停限制
5. 避免过度交易
```

---

## 九、部署方案

### 9.1 开发环境

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 API Keys

# 4. 初始化数据库
alembic upgrade head

# 5. 启动服务
python -m backend.main
```

### 9.2 生产环境（Docker）

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装 TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/deepalpha
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: deepalpha
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 9.3 Nginx 配置

```nginx
# nginx.conf

server {
    listen 80;
    server_name your-domain.com;

    # 前端
    location / {
        root /var/www/frontend;
        try_files $uri $uri/ /index.html;
    }

    # 后端 API
    location /api {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket
    location /ws {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 十、风险与注意事项

### 10.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 数据延迟 | 交易决策滞后 | 多数据源验证、本地缓存 |
| LLM 输出不稳定 | 决策错误 | 多轮校验、规则兜底 |
| 系统稳定性 | 资金损失 | 充分测试、模拟运行 |
| API 限流 | 数据获取失败 | 请求队列、降级策略 |

### 10.2 业务风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 市场波动 | 亏损 | 严格止损、仓位控制 |
| 政策变化 | 交易受限 | 关注公告、灵活调整 |
| 技术故障 | 无法交易 | 高可用架构、监控告警 |

### 10.3 法律合规

1. **投资咨询资质**：AI 决策仅供参考，不构成投资建议
2. **数据使用合规**：遵守数据源使用条款
3. **风险提示**：充分披露交易风险
4. **用户协议**：明确责任边界

---

## 十一、总结

### 11.1 项目优势

1. **成熟架构**：基于 NOFX 验证过的架构设计
2. **Python 生态**：丰富的数据分析和 AI 库支持
3. **市场专注**：专注 A股和港股，深度适配
4. **可扩展性**：插件化设计，易于扩展

### 11.2 关键成功因素

1. **数据质量**：稳定、准确的市场数据
2. **AI 效果**：高质量的 Prompt 和 LLM 选择
3. **风险控制**：严格的风险管理规则
4. **用户体验**：友好的操作界面

### 11.3 后续展望

1. **实盘对接**：接入券商实盘接口
2. **策略优化**：基于历史数据的策略优化
3. **多市场支持**：扩展到美股、期货等
4. **社区建设**：构建策略分享和竞争社区

---

**文档版本**: v1.0
**创建日期**: 2026-01-05
**作者**: DeepAlpha Team
