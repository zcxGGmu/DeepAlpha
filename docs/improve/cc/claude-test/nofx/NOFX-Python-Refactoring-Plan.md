# NOFX Python重构技术方案：A股港股智能交易系统

## 项目概述

本方案旨在将基于 Go 语言的 NOFX 加密货币交易系统重构为 Python 版本，专注于**A股**和**港股**市场的智能交易。保留 NOFX 的核心架构优势（多智能体决策、统一风控、模块化设计），同时针对中国股市特性进行适配。

---

## 一、技术栈选型

### 1.1 后端技术栈

| 类别 | NOFX (Go) | Python重构方案 | 理由 |
|------|-----------|----------------|------|
| Web框架 | Gin | FastAPI | 异步高性能，自动API文档，类型提示 |
| 数据库ORM | 原生SQLite | SQLAlchemy | 成熟ORM，支持异步，多数据库兼容 |
| 任务调度 | time.Ticker | APScheduler | 功能丰富的Python任务调度库 |
| 并发模型 | Goroutine | asyncio + asyncio.run | 异步IO，高并发处理 |
| 数据处理 | 自定义结构 | pandas + numpy | 金融数据处理标准库 |
| 技术指标 | 自计算 | ta-lib | C语言加速，专业金融指标库 |
| 缓存 | 内存 | Redis | 分布式缓存，支持消息队列 |
| 认证 | JWT + bcrypt | FastAPI Security + passlib | Python生态标准方案 |
| 加密 | AES-GCM | cryptography | Python标准加密库 |

### 1.2 前端技术栈（复用NOFX）

- **框架**: React 18 + TypeScript 5.0
- **构建工具**: Vite 6.0
- **状态管理**: Zustand
- **UI组件**: Radix UI + Tailwind CSS
- **图表**: Recharts / ECharts
- **HTTP客户端**: Axios + SWR

### 1.3 券商/数据源接口

| 市场类型 | 券商接口 | 数据源 | 备注 |
|----------|----------|--------|------|
| A股 | 富途OpenD / 同花顺 / 东方财富 | Tushare / AKShare | 富途支持实盘交易 |
| 港股 | 富途OpenD / 老虎证券 | Yahoo Finance / 富途 | 富途同时支持A股和港股 |

---

## 二、系统架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         前端 (React + TS)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ 登录注册 │ │ 交易员   │ │ 交易     │ │ AI决策分析       │   │
│  │ 页面     │ │ 管理页面 │ │ 仪表盘   │ │ 页面             │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▲ HTTP/WebSocket
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI 网关层                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ 认证中间件│ │ CORS中间件│ │限流中间件│ │ WebSocket 路由  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        核心业务层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  交易员管理   │  │  账户管理    │  │   决策引擎          │  │
│  │ TraderManager │  │ AccountMgr   │  │   DecisionEngine    │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  订单管理    │  │  持仓管理    │  │   AI客户端           │  │
│  │ OrderManager │  │ PositionMgr  │  │   AIClient           │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        服务支撑层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  券商接口    │  │  行情服务    │  │   风控服务          │  │
│  │ BrokerAdapter│  │ MarketService│  │   RiskService       │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  技术指标    │  │  日志服务    │  │   通知服务          │  │
│  │ Indicator    │  │ Logger       │  │   Notification      │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        数据持久层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ PostgreSQL/  │  │    Redis     │  │   文件存储          │  │
│  │   MySQL      │  │   缓存/队列   │  │   决策日志          │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        外部接口层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  富途OpenD   │  │  Tushare     │  │   AI模型API         │  │
│  │  (实盘交易)  │  │  (行情数据)  │  │   (DeepSeek/Qwen)   │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 项目目录结构

```
deepalpha/
├── api/                        # FastAPI路由
│   ├── __init__.py
│   ├── auth.py                 # 认证相关接口
│   ├── traders.py              # 交易员管理接口
│   ├── accounts.py             # 账户查询接口
│   ├── orders.py               # 订单管理接口
│   ├── positions.py            # 持仓查询接口
│   ├── market.py               # 行情数据接口
│   └── websocket.py            # WebSocket路由
│
├── core/                       # 核心配置
│   ├── __init__.py
│   ├── config.py               # 配置管理
│   ├── security.py             # 安全相关(JWT, 加密)
│   ├── deps.py                 # 依赖注入
│   └── logger.py               # 日志配置
│
├── models/                     # SQLAlchemy模型
│   ├── __init__.py
│   ├── user.py                 # 用户模型
│   ├── ai_model.py             # AI模型配置
│   ├── broker.py               # 券商配置
│   ├── trader.py               # 交易员配置
│   ├── signal_source.py        # 信号源配置
│   ├── order.py                # 订单记录
│   ├── position.py             # 持仓记录
│   └── decision_log.py         # 决策日志
│
├── schemas/                    # Pydantic模式
│   ├── __init__.py
│   ├── user.py                 # 用户相关Schema
│   ├── trader.py               # 交易员相关Schema
│   ├── account.py              # 账户相关Schema
│   ├── market.py               # 行情相关Schema
│   └── decision.py             # 决策相关Schema
│
├── brokers/                    # 券商接口
│   ├── __init__.py
│   ├── base.py                 # 抽象券商接口
│   ├── futu.py                 # 富途OpenD实现
│   └── mock.py                 # 模拟券商(测试用)
│
├── market/                     # 行情与指标
│   ├── __init__.py
│   ├── data_provider.py        # 数据提供者抽象
│   ├── tushare_provider.py     # Tushare数据源
│   ├── akshare_provider.py     # AKShare数据源
│   ├── indicators.py           # 技术指标计算
│   └── cache.py                # 行情缓存管理
│
├── trading/                    # 交易核心
│   ├── __init__.py
│   ├── engine.py               # 交易引擎
│   ├── order_manager.py        # 订单管理器
│   ├── position_manager.py     # 持仓管理器
│   └── execution.py            # 订单执行
│
├── decision/                   # AI决策引擎
│   ├── __init__.py
│   ├── engine.py               # 决策引擎
│   ├── context.py              # 决策上下文构建
│   ├── parser.py               # 决策解析器
│   └── prompts/                # 提示词模板
│       ├── base.txt
│       ├── a_share.txt         # A股策略
│       └── hk_stock.txt        # 港股策略
│
├── ai/                         # AI客户端
│   ├── __init__.py
│   ├── base.py                 # AI客户端抽象
│   ├── deepseek.py             # DeepSeek客户端
│   ├── qwen.py                 # 通义千问客户端
│   └── custom.py               # 自定义API客户端
│
├── risk/                       # 风控管理
│   ├── __init__.py
│   ├── validator.py            # 交易验证器
│   ├── monitor.py              # 实时监控
│   └── limits.py               # 限额管理
│
├── scheduler/                  # 任务调度
│   ├── __init__.py
│   ├── manager.py              # 调度管理器
│   └── jobs.py                 # 定时任务
│
├── logger/                     # 决策日志
│   ├── __init__.py
│   ├── decision_logger.py      # 决策日志记录
│   └── performance.py          # 性能分析
│
├── web/                        # 前端(复用NOFX)
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── lib/
│   └── package.json
│
├── migrations/                 # 数据库迁移(Alembic)
│
├── tests/                      # 测试
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_brokers/
│   ├── test_decision/
│   └── test_risk/
│
├── scripts/                    # 工具脚本
│   ├── init_db.py
│   └── backup.py
│
├── docker/                     # Docker配置
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .env.example                # 环境变量示例
├── pyproject.toml              # 项目配置
├── requirements.txt            # 依赖清单
└── main.py                     # 应用入口
```

---

## 三、核心模块详细设计

### 3.1 券商接口层 (brokers/)

#### 3.1.1 抽象接口定义

```python
# brokers/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime
from decimal import Decimal

class BrokerInterface(ABC):
    """券商抽象接口，所有券商实现必须遵循"""

    @abstractmethod
    async def connect(self) -> bool:
        """建立连接"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """断开连接"""
        pass

    @abstractmethod
    async def get_account_info(self) -> Dict:
        """获取账户信息"""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """获取当前持仓"""
        pass

    @abstractmethod
    async def get_orders(self, status: Optional[str] = None) -> List[Dict]:
        """获取订单列表"""
        pass

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        order_type: str,  # 'limit', 'market'
        quantity: int,
        price: Optional[Decimal] = None,
    ) -> Dict:
        """下单"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        pass

    @abstractmethod
    async def get_market_data(
        self,
        symbols: List[str],
        period: str = "1d",
        count: int = 100,
    ) -> Dict[str, Dict]:
        """获取行情数据"""
        pass

    @abstractmethod
    def is_trading_time(self) -> bool:
        """判断是否在交易时间内"""
        pass
```

#### 3.1.2 富途OpenD实现

```python
# brokers/futu.py
from typing import Dict, List
from futu import (
    OpenSecTradeContext,
    OpenQuoteContext,
    TrdEnv,
    Market,
    SecurityFirm,
)
from .base import BrokerInterface

class FutuBroker(BrokerInterface):
    """富途OpenD券商实现"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        trd_env: str = "SIMULATE",  # SIMULATE or REAL
        markets: List[str] = ["CN", "HK"],  # CN=A股, HK=港股
    ):
        self.host = host
        self.port = port
        self.trd_env = TrdEnv.SIMULATE if trd_env == "SIMULATE" else TrdEnv.REAL
        self.markets = [Market.CN if m == "CN" else Market.HK for m in markets]

        self.quote_ctx = None
        self.trd_ctx = None

    async def connect(self) -> bool:
        """连接富途OpenD"""
        try:
            self.quote_ctx = OpenQuoteContext(
                host=self.host, port=self.port
            )
            self.trd_ctx = OpenSecTradeContext(
                host=self.host,
                port=self.port,
                security_firm=SecurityFirm.FUTUSECURITIES,
                env=self.trd_env,
            )
            return True
        except Exception as e:
            print(f"富途连接失败: {e}")
            return False

    async def get_account_info(self) -> Dict:
        """获取账户信息"""
        ret, data = self.trd_ctx.acctradinginfo_query()
        if ret == 0:
            return {
                "total_assets": data["cash"][0],
                "available_cash": data["cash"][1],
                "market_value": data["market_val"][0],
                "currency": "CNY" if Market.CN in self.markets else "HKD",
            }
        return {}

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: int,
        price: Decimal = None,
    ) -> Dict:
        """下单"""
        # 富途下单逻辑
        ret, data = self.trd_ctx.place_order(
            price=float(price) if price else None,
            qty=quantity,
            code=symbol,
            trd_side="BUY" if side == "buy" else "SELL",
            order_type=order_type.upper(),
        )
        return {"success": ret == 0, "order_id": data.get("order_id")}

    def is_trading_time(self) -> bool:
        """判断是否在交易时间内"""
        from datetime import datetime, time

        now = datetime.now().time()

        # A股交易时间: 9:30-11:30, 13:00-15:00
        a_stock_periods = [
            (time(9, 30), time(11, 30)),
            (time(13, 0), time(15, 0)),
        ]

        # 港股交易时间: 9:30-12:00, 13:00-16:00
        hk_stock_periods = [
            (time(9, 30), time(12, 0)),
            (time(13, 0), time(16, 0)),
        ]

        periods = []
        if Market.CN in self.markets:
            periods.extend(a_stock_periods)
        if Market.HK in self.markets:
            periods.extend(hk_stock_periods)

        return any(start <= now <= end for start, end in periods)
```

### 3.2 决策引擎 (decision/)

#### 3.2.1 决策上下文构建

```python
# decision/context.py
from typing import Dict, List
from datetime import datetime
from decimal import Decimal

class DecisionContextBuilder:
    """构建AI决策上下文"""

    def __init__(
        self,
        trader_id: str,
        broker,
        market_service,
        position_manager,
    ):
        self.trader_id = trader_id
        self.broker = broker
        self.market_service = market_service
        self.position_manager = position_manager

    async def build(
        self,
        candidate_symbols: List[str],
        include_indicators: bool = True,
    ) -> Dict:
        """构建完整决策上下文"""

        # 1. 账户信息
        account = await self.broker.get_account_info()

        # 2. 当前持仓
        positions = await self.broker.get_positions()

        # 3. 候选股票行情
        market_data = {}
        for symbol in candidate_symbols:
            data = await self.market_service.get_quote(symbol)
            if include_indicators:
                data["indicators"] = await self.market_service.get_indicators(
                    symbol, periods=["1d", "1w", "1m"]
                )
            market_data[symbol] = data

        # 4. 构建上下文
        context = {
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trader_id": self.trader_id,
            "account": {
                "total_assets": account.get("total_assets", 0),
                "available_cash": account.get("available_cash", 0),
                "position_market_value": account.get("market_value", 0),
                "position_ratio": self._calc_position_ratio(account),
            },
            "positions": [
                {
                    "symbol": p["symbol"],
                    "quantity": p["quantity"],
                    "cost_price": p["cost_price"],
                    "current_price": p["current_price"],
                    "market_value": p["market_value"],
                    "pnl": p["pnl"],
                    "pnl_ratio": p["pnl_ratio"],
                }
                for p in positions
            ],
            "candidates": [
                {
                    "symbol": symbol,
                    **market_data.get(symbol, {}),
                }
                for symbol in candidate_symbols
            ],
            "market_overview": await self._get_market_overview(),
        }

        return context

    def _calc_position_ratio(self, account: Dict) -> float:
        """计算持仓比例"""
        total = account.get("total_assets", 0)
        if total == 0:
            return 0
        return account.get("market_value", 0) / total

    async def _get_market_overview(self) -> Dict:
        """获取市场概况"""
        # 获取主要指数表现
        indices = {
            "A股": ["000001.SH", "399001.SZ"],  # 上证、深证
            "港股": ["HSI.HK"],  # 恒生指数
        }
        overview = {}
        for market, codes in indices.items():
            data = []
            for code in codes:
                quote = await self.market_service.get_quote(code)
                data.append({
                    "name": code,
                    "change": quote.get("change_percent", 0),
                })
            overview[market] = data
        return overview
```

#### 3.2.2 决策引擎

```python
# decision/engine.py
from typing import Dict, List, Optional
from .context import DecisionContextBuilder
from .parser import DecisionParser
from ai.base import AIClient

class DecisionEngine:
    """AI决策引擎"""

    def __init__(
        self,
        ai_client: AIClient,
        prompt_template: str,
        context_builder: DecisionContextBuilder,
        parser: DecisionParser,
    ):
        self.ai_client = ai_client
        self.prompt_template = prompt_template
        self.context_builder = context_builder
        self.parser = parser

    async def make_decision(
        self,
        candidate_symbols: List[str],
        additional_context: Optional[Dict] = None,
    ) -> List[Dict]:
        """执行决策循环"""

        # 1. 构建上下文
        context = await self.context_builder.build(
            candidate_symbols=candidate_symbols,
            include_indicators=True,
        )

        # 2. 合并额外上下文
        if additional_context:
            context.update(additional_context)

        # 3. 构建提示词
        system_prompt = self._load_system_prompt()
        user_prompt = self._format_user_prompt(context)

        # 4. 调用AI
        response = await self.ai_client.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # 5. 解析决策
        decisions = self.parser.parse(response)

        # 6. 验证决策
        validated_decisions = await self._validate_decisions(
            decisions, context
        )

        return validated_decisions

    def _load_system_prompt(self) -> str:
        """加载系统提示词"""
        with open(f"decision/prompts/{self.prompt_template}.txt") as f:
            return f.read()

    def _format_user_prompt(self, context: Dict) -> str:
        """格式化用户提示词"""
        # 使用 Jinja2 或 f-string 格式化
        return f"""
当前时间: {context['current_time']}

账户状态:
- 总资产: {context['account']['total_assets']:.2f}
- 可用资金: {context['account']['available_cash']:.2f}
- 持仓市值: {context['account']['position_market_value']:.2f}
- 持仓比例: {context['account']['position_ratio']:.2%}

当前持仓:
{self._format_positions(context['positions'])}

候选股票:
{self._format_candidates(context['candidates'])}

市场概况:
{self._format_market_overview(context['market_overview'])}

请基于以上信息，输出交易决策。决策格式要求:
{{"decisions": [{{"action": "buy/sell/hold", "symbol": "代码", "quantity": 数量, "reason": "理由"}}]}}
"""

    async def _validate_decisions(
        self, decisions: List[Dict], context: Dict
    ) -> List[Dict]:
        """验证决策的合法性"""
        # 1. 检查资金是否充足
        # 2. 检查持仓是否存在
        # 3. 检查交易时间
        # 4. 检查涨跌停限制
        valid_decisions = []
        for decision in decisions:
            if await self._is_valid_decision(decision, context):
                valid_decisions.append(decision)
        return valid_decisions
```

### 3.3 风控管理 (risk/)

#### 3.3.1 交易验证器

```python
# risk/validator.py
from typing import Dict, List
from decimal import Decimal

class TradingValidator:
    """交易验证器"""

    def __init__(
        self,
        max_position_ratio: float = 0.3,  # 单只股票最大持仓比例
        max_total_position_ratio: float = 0.8,  # 总持仓比例
        max_daily_trades: int = 10,  # 每日最大交易次数
        min_trade_interval: int = 60,  # 最小交易间隔(秒)
    ):
        self.max_position_ratio = max_position_ratio
        self.max_total_position_ratio = max_total_position_ratio
        self.max_daily_trades = max_daily_trades
        self.min_trade_interval = min_trade_interval
        self._trade_history = {}

    async def validate_buy(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        account: Dict,
        positions: List[Dict],
    ) -> tuple[bool, str]:
        """验证买入交易"""

        # 1. 检查资金是否充足
        required_amount = Decimal(quantity) * price
        available_cash = account.get("available_cash", 0)
        if required_amount > available_cash:
            return False, f"资金不足，需要 {required_amount}，可用 {available_cash}"

        # 2. 检查单只股票持仓比例
        total_assets = account.get("total_assets", 1)
        position_value = required_amount
        existing_position = next(
            (p for p in positions if p["symbol"] == symbol), None
        )
        if existing_position:
            position_value += existing_position["market_value"]

        if position_value / total_assets > self.max_position_ratio:
            return False, f"超过单只股票最大持仓限制 {self.max_position_ratio:.0%}"

        # 3. 检查总持仓比例
        current_total_position = sum(p["market_value"] for p in positions)
        new_total_position = current_total_position + position_value
        if new_total_position / total_assets > self.max_total_position_ratio:
            return False, f"超过总持仓限制 {self.max_total_position_ratio:.0%}"

        # 4. 检查每日交易次数
        if not self._check_daily_trade_limit(symbol):
            return False, f"超过每日最大交易次数 {self.max_daily_trades}"

        # 5. 检查交易间隔
        if not self._check_trade_interval(symbol):
            return False, f"交易间隔过短，最小间隔 {self.min_trade_interval} 秒"

        # 6. 检查涨跌停(仅A股)
        if await self._is_limit_up(symbol, price):
            return False, "目标股涨停，无法买入"

        return True, "验证通过"

    async def validate_sell(
        self,
        symbol: str,
        quantity: int,
        positions: List[Dict],
    ) -> tuple[bool, str]:
        """验证卖出交易"""

        # 1. 检查持仓是否足够
        position = next((p for p in positions if p["symbol"] == symbol), None)
        if not position:
            return False, "未持有该股票"

        if position["quantity"] < quantity:
            return False, f"持仓不足，持有 {position['quantity']}，欲卖出 {quantity}"

        # 2. 检查跌停(仅A股)
        current_price = position.get("current_price", 0)
        if await self._is_limit_down(symbol, current_price):
            return False, "目标股跌停，可能无法卖出"

        return True, "验证通过"

    async def _is_limit_up(self, symbol: str, price: Decimal) -> bool:
        """判断是否涨停"""
        # A股主板10%, 创业板/科创板20%, 北交所30%
        # 需要获取昨日收盘价计算
        # 这里简化处理
        return False

    async def _is_limit_down(self, symbol: str, price: Decimal) -> bool:
        """判断是否跌停"""
        return False
```

### 3.4 技术指标 (market/indicators.py)

```python
# market/indicators.py
import pandas as pd
import numpy as np
from typing import Dict, List

class TechnicalIndicators:
    """技术指标计算"""

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
        """计算指数移动平均线"""
        return df["close"].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame, period: int = 20, std_dev: int = 2
    ) -> Dict[str, pd.Series]:
        """计算布林带"""
        sma = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return {
            "middle": sma,
            "upper": upper_band,
            "lower": lower_band,
        }

    @staticmethod
    def calculate_kdj(df: pd.DataFrame, period: int = 9) -> Dict[str, pd.Series]:
        """计算KDJ指标"""
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        rsv = (df["close"] - low_min) / (high_max - low_min) * 100

        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d

        return {"k": k, "d": d, "j": j}
```

---

## 四、A股港股特性适配

### 4.1 交易时间管理

```python
# core/trading_hours.py
from datetime import datetime, time, timedelta
from typing import Optional

class TradingHoursManager:
    """交易时间管理"""

    # A股交易时间
    A_STOCK_MORNING = (time(9, 30), time(11, 30))
    A_STOCK_AFTERNOON = (time(13, 0), time(15, 0))

    # 港股交易时间
    HK_STOCK_MORNING = (time(9, 30), time(12, 0))
    HK_STOCK_AFTERNOON = (time(13, 0), time(16, 0))

    @classmethod
    def is_trading_time(cls, market: str = "CN") -> bool:
        """判断是否在交易时间"""
        now = datetime.now().time()

        if market == "CN":  # A股
            periods = [cls.A_STOCK_MORNING, cls.A_STOCK_AFTERNOON]
        elif market == "HK":  # 港股
            periods = [cls.HK_STOCK_MORNING, cls.HK_STOCK_AFTERNOON]
        else:
            raise ValueError(f"不支持的市场: {market}")

        return any(start <= now <= end for start, end in periods)

    @classmethod
    def get_next_trading_time(cls, market: str = "CN") -> Optional[datetime]:
        """获取下一个交易时间"""
        now = datetime.now()

        if market == "CN":
            periods = [
                (datetime.combine(now.date(), time(9, 30)), datetime.combine(now.date(), time(11, 30))),
                (datetime.combine(now.date(), time(13, 0)), datetime.combine(now.date(), time(15, 0))),
            ]
        else:
            periods = [
                (datetime.combine(now.date(), time(9, 30)), datetime.combine(now.date(), time(12, 0))),
                (datetime.combine(now.date(), time(13, 0)), datetime.combine(now.date(), time(16, 0))),
            ]

        for start, end in periods:
            if now < start:
                return start
            if start <= now <= end:
                return now

        # 今天的交易时段已过，返回明天早上
        if market == "CN":
            return datetime.combine(now.date() + timedelta(days=1), time(9, 30))
        else:
            return datetime.combine(now.date() + timedelta(days=1), time(9, 30))

    @classmethod
    def is_auction_period(cls, market: str = "CN") -> bool:
        """判断是否在集合竞价时间"""
        now = datetime.now().time()
        current_date = datetime.now().date()

        # A股集合竞价时间: 9:15-9:25
        if market == "CN":
            auction_start = time(9, 15)
            auction_end = time(9, 25)
            return auction_start <= now <= auction_end

        # 港股集合竞价时间: 9:00-9:30 (上午)
        if market == "HK":
            auction_start = time(9, 0)
            auction_end = time(9, 30)
            return auction_start <= now <= auction_end

        return False
```

### 4.2 涨跌停限制

```python
# core/limit_rules.py
from decimal import Decimal
from typing import Dict
from enum import Enum

class BoardType(Enum):
    """板块类型"""
    MAIN = "main"  # 主板
    SME = "sme"  # 中小板
    STARTUP = "startup"  # 创业板
    STAR = "star"  # 科创板
    BSE = "bse"  # 北交所

class LimitRules:
    """涨跌停规则"""

    # 涨跌停幅度
    LIMIT_RATES = {
        BoardType.MAIN: 0.10,  # 主板10%
        BoardType.SME: 0.10,  # 中小板10%
        BoardType.STARTUP: 0.20,  # 创业板20%
        BoardType.STAR: 0.20,  # 科创板20%
        BoardType.BSE: 0.30,  # 北交所30%
    }

    # ST股票涨跌幅限制为5%
    ST_LIMIT_RATE = 0.05

    # 港股无涨跌停限制
    HK_NO_LIMIT = None

    @classmethod
    def get_board_type(cls, symbol: str) -> BoardType:
        """判断股票所属板块"""
        # 600xxx, 000xxx, 001xxx: 主板
        # 002xxx: 中小板
        # 300xxx: 创业板
        # 688xxx: 科创板
        # 8xxxxx, 4xxxxx: 北交所
        code = symbol.split(".")[0]

        if code.startswith("688"):
            return BoardType.STAR
        elif code.startswith("300"):
            return BoardType.STARTUP
        elif code.startswith("002"):
            return BoardType.SME
        elif code.startswith("8") or code.startswith("4"):
            return BoardType.BSE
        else:
            return BoardType.MAIN

    @classmethod
    def is_st_stock(cls, symbol: str) -> bool:
        """判断是否为ST股票"""
        # ST股票名称包含ST、*ST等标识
        # 这里简化处理，实际需要查询股票名称
        return False

    @classmethod
    def get_limit_price(
        cls, symbol: str, prev_close: Decimal, is_upper: bool = True
    ) -> Decimal:
        """计算涨跌停价格"""
        # 港股无涨跌停
        if ".HK" in symbol:
            return None

        board_type = cls.get_board_type(symbol)
        limit_rate = cls.LIMIT_RATES.get(board_type, 0.10)

        # ST股票特殊处理
        if cls.is_st_stock(symbol):
            limit_rate = cls.ST_LIMIT_RATE

        if is_upper:
            return prev_close * (1 + limit_rate)
        else:
            return prev_close * (1 - limit_rate)

    @classmethod
    def is_limit_up(cls, symbol: str, current_price: Decimal, prev_close: Decimal) -> bool:
        """判断是否涨停"""
        limit_price = cls.get_limit_price(symbol, prev_close, is_upper=True)
        return limit_price and current_price >= limit_price

    @classmethod
    def is_limit_down(cls, symbol: str, current_price: Decimal, prev_close: Decimal) -> bool:
        """判断是否跌停"""
        limit_price = cls.get_limit_price(symbol, prev_close, is_upper=False)
        return limit_price and current_price <= limit_price
```

### 4.3 T+1 交易限制

```python
# trading/settlement.py
from datetime import datetime, date
from typing import Dict, List

class SettlementManager:
    """T+1交收管理"""

    def __init__(self):
        self._buy_records: Dict[str, List[datetime]] = {}

    def record_buy(self, symbol: str, quantity: int, buy_time: datetime) -> None:
        """记录买入信息"""
        if symbol not in self._buy_records:
            self._buy_records[symbol] = []
        for _ in range(quantity):
            self._buy_records[symbol].append(buy_time)

    def get_sellable_quantity(self, symbol: str, current_time: datetime) -> int:
        """获取可卖出数量（T+1）"""
        if symbol not in self._buy_records:
            return 0

        # A股T+1，港股T+0
        if ".HK" in symbol:
            return len(self._buy_records[symbol])

        # A股只能卖出昨天之前买入的
        yesterday = date.fromordinal(current_time.date().toordinal() - 1)
        sellable = sum(
            1
            for buy_time in self._buy_records[symbol]
            if buy_time.date() <= yesterday
        )
        return sellable

    def can_sell_today(self, symbol: str, quantity: int, current_time: datetime) -> bool:
        """判断今日是否可卖出指定数量"""
        sellable = self.get_sellable_quantity(symbol, current_time)
        return sellable >= quantity
```

### 4.4 手数和最小单位

```python
# core/lot_rules.py
from typing import Dict
from enum import Enum

class MarketType(Enum):
    """市场类型"""
    CN = "CN"  # A股
    HK = "HK"  # 港股

class LotRules:
    """手数规则"""

    # 每手股数
    LOT_SIZE = {
        MarketType.CN: 100,  # A股100股为1手
        MarketType.HK: 1000,  # 港股通常1000股为1手，部分股票不同
    }

    # 港股特殊手数
    HK_SPECIAL_LOT = {
        "00700.HK": 100,  # 腾讯
        "00388.HK": 500,  # 港交所
        # ... 其他特殊手数
    }

    @classmethod
    def get_lot_size(cls, symbol: str) -> int:
        """获取股票每手股数"""
        if ".HK" in symbol:
            return cls.HK_SPECIAL_LOT.get(symbol, 1000)
        else:
            return 100

    @classmethod
    def round_to_lot(cls, symbol: str, quantity: int) -> int:
        """向上取整到手数"""
        lot_size = cls.get_lot_size(symbol)
        return ((quantity + lot_size - 1) // lot_size) * lot_size

    @classmethod
    def is_valid_lot(cls, symbol: str, quantity: int) -> bool:
        """判断数量是否为整手"""
        lot_size = cls.get_lot_size(symbol)
        return quantity % lot_size == 0
```

---

## 五、开发路线规划

### 5.1 第一阶段：基础框架搭建（2-3周）

**目标**: 建立项目骨架，实现核心基础设施

| 序号 | 任务 | 优先级 | 产出 |
|------|------|--------|------|
| 1.1 | 项目初始化，搭建目录结构 | P0 | 项目骨架 |
| 1.2 | 配置 FastAPI + SQLAlchemy + Alembic | P0 | 数据库迁移 |
| 1.3 | 实现用户认证(JWT + OTP) | P0 | 登录注册功能 |
| 1.4 | 定义数据模型(用户、券商、交易员) | P0 | 数据库表结构 |
| 1.5 | 实现配置管理(环境变量、系统配置) | P0 | 配置系统 |
| 1.6 | 搭建日志系统 | P1 | 日志模块 |
| 1.7 | Docker 部署配置 | P1 | Dockerfile |

### 5.2 第二阶段：券商接口实现（3-4周）

**目标**: 实现富途OpenD集成，支持实盘交易

| 序号 | 任务 | 优先级 | 产出 |
|------|------|--------|------|
| 2.1 | 设计抽象券商接口 | P0 | BrokerInterface |
| 2.2 | 实现富途OpenD连接 | P0 | FutuBroker |
| 2.3 | 实现账户查询功能 | P0 | get_account_info |
| 2.4 | 实现持仓查询功能 | P0 | get_positions |
| 2.5 | 实现下单功能 | P0 | place_order |
| 2.6 | 实现撤单功能 | P0 | cancel_order |
| 2.7 | 实现模拟券商(测试用) | P1 | MockBroker |
| 2.8 | 券商接口单元测试 | P1 | 测试用例 |

### 5.3 第三阶段：行情与指标（2-3周）

**目标**: 实现行情数据获取和技术指标计算

| 序号 | 任务 | 优先级 | 产出 |
|------|------|--------|------|
| 3.1 | 集成 Tushare 数据源 | P0 | TushareProvider |
| 3.2 | 实现实时行情获取 | P0 | get_quote |
| 3.3 | 实现历史K线获取 | P0 | get_kline |
| 3.4 | 实现技术指标计算(MA/EMA/MACD/RSI) | P0 | Indicator |
| 3.5 | 实现指标缓存 | P1 | Redis缓存 |
| 3.6 | 集成 AKShare 备用数据源 | P2 | AKShareProvider |

### 5.4 第四阶段：AI决策引擎（3-4周）

**目标**: 实现AI驱动的交易决策

| 序号 | 任务 | 优先级 | 产出 |
|------|------|--------|------|
| 4.1 | 设计决策上下文结构 | P0 | DecisionContext |
| 4.2 | 实现上下文构建器 | P0 | ContextBuilder |
| 4.3 | 实现 DeepSeek AI 客户端 | P0 | DeepSeekClient |
| 4.4 | 实现通义千问 AI 客户端 | P0 | QwenClient |
| 4.5 | 设计A股决策提示词模板 | P0 | a_share.txt |
| 4.6 | 设计港股决策提示词模板 | P0 | hk_stock.txt |
| 4.7 | 实现决策解析器 | P0 | DecisionParser |
| 4.8 | 实现决策执行器 | P0 | DecisionExecutor |
| 4.9 | 实现决策日志记录 | P1 | DecisionLogger |

### 5.5 第五阶段：风控系统（2-3周）

**目标**: 实现全面的风控管理

| 序号 | 任务 | 优先级 | 产出 |
|------|------|--------|------|
| 5.1 | 实现交易验证器 | P0 | TradingValidator |
| 5.2 | 实现持仓限额管理 | P0 | PositionLimit |
| 5.3 | 实现交易次数限制 | P0 | TradeCountLimit |
| 5.4 | 实现止损止盈管理 | P1 | StopLossManager |
| 5.5 | 实现异常检测 | P1 | AnomalyDetector |
| 5.6 | 实现紧急止损机制 | P0 | EmergencyStop |

### 5.6 第六阶段：任务调度与自动化（2周）

**目标**: 实现自动交易循环

| 序号 | 任务 | 优先级 | 产出 |
|------|------|--------|------|
| 6.1 | 集成 APScheduler | P0 | 调度器框架 |
| 6.2 | 实现定时扫描任务 | P0 | scan_and_trade |
| 6.3 | 实现交易时间管理 | P0 | TradingHoursManager |
| 6.4 | 实现任务状态监控 | P1 | TaskMonitor |
| 6.5 | 实现失败重试机制 | P1 | RetryPolicy |

### 5.7 第七阶段：前端开发（3-4周）

**目标**: 实现React前端，复用NOFX设计

| 序号 | 任务 | 优先级 | 产出 |
|------|------|--------|------|
| 7.1 | 搭建前端项目结构 | P0 | 项目骨架 |
| 7.2 | 实现登录注册页面 | P0 | AuthPage |
| 7.3 | 实现交易员管理页面 | P0 | TradersPage |
| 7.4 | 实现交易仪表盘 | P0 | Dashboard |
| 7.5 | 实现WebSocket实时更新 | P0 | RealtimeUpdates |
| 7.6 | 实现决策分析页面 | P1 | AnalysisPage |
| 7.7 | 实现持仓管理页面 | P1 | PositionsPage |
| 7.8 | 实现系统配置页面 | P1 | SettingsPage |

### 5.8 第八阶段：测试与优化（2-3周）

**目标**: 全面测试，优化性能

| 序号 | 任务 | 优先级 | 产出 |
|------|------|--------|------|
| 8.1 | 编写单元测试 | P0 | 测试覆盖率>70% |
| 8.2 | 编写集成测试 | P0 | 端到端测试 |
| 8.3 | 性能测试与优化 | P1 | 性能报告 |
| 8.4 | 安全审计 | P0 | 安全报告 |
| 8.5 | 压力测试 | P1 | 并发报告 |
| 8.6 | Bug修复与优化 | P0 | 稳定版本 |

---

## 六、数据库设计

### 6.1 核心表结构

```sql
-- 用户表
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    otp_secret VARCHAR(255),
    otp_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AI模型配置表
CREATE TABLE ai_models (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    provider VARCHAR(50) NOT NULL,  -- 'deepseek', 'qwen', 'custom'
    enabled BOOLEAN DEFAULT TRUE,
    api_key_encrypted TEXT,
    custom_api_url VARCHAR(500),
    custom_model_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 券商配置表
CREATE TABLE brokers (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL,  -- 'futu', 'mock'
    enabled BOOLEAN DEFAULT TRUE,
    api_key_encrypted TEXT,
    secret_key_encrypted TEXT,
    is_testnet BOOLEAN DEFAULT FALSE,
    -- 富途特定字段
    futu_host VARCHAR(100) DEFAULT '127.0.0.1',
    futu_port INTEGER DEFAULT 11111,
    futu_trd_env VARCHAR(20) DEFAULT 'SIMULATE',
    markets VARCHAR(100) DEFAULT 'CN,HK',  -- 支持的市场
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 交易员表
CREATE TABLE traders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    ai_model_id INTEGER REFERENCES ai_models(id) ON DELETE SET NULL,
    broker_id INTEGER REFERENCES brokers(id) ON DELETE SET NULL,
    is_running BOOLEAN DEFAULT FALSE,
    -- 策略配置
    initial_balance DECIMAL(20, 2) DEFAULT 0,
    scan_interval_minutes INTEGER DEFAULT 5,
    max_position_ratio DECIMAL(5, 4) DEFAULT 0.3,  -- 单只股票最大持仓比例
    max_total_position_ratio DECIMAL(5, 4) DEFAULT 0.8,  -- 总持仓比例
    max_daily_trades INTEGER DEFAULT 10,
    -- 股票池配置
    trading_symbols TEXT,  -- JSON数组: ["000001.SZ", "00700.HK"]
    use_stock_pool BOOLEAN DEFAULT TRUE,
    custom_pool_url VARCHAR(500),
    -- AI配置
    system_prompt_template VARCHAR(100) DEFAULT 'a_share',
    custom_prompt TEXT,
    -- 状态统计
    total_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20, 2) DEFAULT 0,
    win_rate DECIMAL(5, 4) DEFAULT 0,
    last_scan_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 订单表
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    trader_id INTEGER REFERENCES traders(id) ON DELETE CASCADE,
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'buy', 'sell'
    order_type VARCHAR(20) NOT NULL,  -- 'limit', 'market'
    quantity INTEGER NOT NULL,
    price DECIMAL(20, 4),
    status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'submitted', 'filled', 'cancelled', 'failed'
    order_id VARCHAR(100),  -- 券商返回的订单ID
    decision_id VARCHAR(100),  -- 关联的决策ID
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 持仓表
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    trader_id INTEGER REFERENCES traders(id) ON DELETE CASCADE,
    symbol VARCHAR(50) NOT NULL,
    quantity INTEGER NOT NULL,
    cost_price DECIMAL(20, 4) NOT NULL,
    current_price DECIMAL(20, 4),
    market_value DECIMAL(20, 2),
    unrealized_pnl DECIMAL(20, 2),
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trader_id, symbol)
);

-- 决策日志表
CREATE TABLE decision_logs (
    id SERIAL PRIMARY KEY,
    trader_id INTEGER REFERENCES traders(id) ON DELETE CASCADE,
    decision_cycle INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    account_snapshot JSONB,
    positions_snapshot JSONB,
    market_data JSONB,
    system_prompt TEXT,
    user_prompt TEXT,
    ai_response TEXT,
    parsed_decisions JSONB,
    executed_orders JSONB,
    ai_request_duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 系统配置表
CREATE TABLE system_config (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 信号源配置表
CREATE TABLE signal_sources (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    source_type VARCHAR(50) NOT NULL,  -- 'url', 'file', 'custom'
    url VARCHAR(500),
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 七、风险控制细则

### 7.1 持仓限额

| 限制类型 | 默认值 | 说明 |
|----------|--------|------|
| 单只股票最大持仓比例 | 30% | 防止过度集中 |
| 总持仓比例上限 | 80% | 保留现金储备 |
| 最大持仓股票数 | 10只 | 分散投资 |
| 同一板块最大比例 | 50% | 行业分散 |

### 7.2 交易限制

| 限制类型 | 默认值 | 说明 |
|----------|--------|------|
| 每日最大交易次数 | 10次 | 防止过度交易 |
| 最小交易间隔 | 60秒 | 防止频繁交易 |
| 单笔最小交易额 | 1000元 | 避免小额交易 |
| 单笔最大交易额 | 总资产的20% | 控制单笔风险 |

### 7.3 止损止盈

| 类型 | 触发条件 | 操作 |
|------|----------|------|
| 止损 | 亏损 > 10% | 平仓 |
| 止盈 | 盈利 > 20% | 平半仓 |
| 移动止损 | 盈利 > 15%后回调5% | 平仓 |
| 时间止损 | 持仓 > 30天无盈利 | 平仓 |

---

## 八、性能优化建议

### 8.1 异步处理

- 使用 FastAPI 的异步特性处理所有IO操作
- 并发获取多只股票的行情数据
- 异步调用AI模型API

### 8.2 缓存策略

```python
# 使用Redis缓存
- 股票基础信息 (1小时)
- 实时行情 (5秒)
- K线数据 (1分钟)
- 技术指标 (1分钟)
```

### 8.3 数据库优化

- 为常用查询字段添加索引
- 使用连接池管理数据库连接
- 定期归档历史决策日志

---

## 九、部署方案

### 9.1 开发环境

```bash
# 使用 Docker Compose
docker-compose -f docker-compose.dev.yml up
```

### 9.2 生产环境

```bash
# 使用 Nginx + Gunicorn + Uvicorn
# 前端: Nginx 静态文件服务
# 后端: Gunicorn 多进程 + Uvicorn 工作进程
# 数据库: PostgreSQL 主从复制
# 缓存: Redis 哨兵模式
```

### 9.3 监控告警

- 使用 Prometheus + Grafana 监控系统指标
- 使用 Sentry 收集错误日志
- 关键指标告警: 持仓亏损、API调用失败、系统异常

---

## 十、总结

本技术方案基于 NOFX 的优秀架构设计，针对A股和港股市场的特性进行了全面适配。主要特点：

1. **保留核心优势**: 多AI模型支持、统一风控、模块化设计
2. **市场特性适配**: T+1限制、涨跌停、交易时间、手数规则
3. **Python生态优势**: 丰富的金融库、易维护、快速开发
4. **可扩展性**: 支持多券商、多AI模型、多策略

预期开发周期: **20-28周** (约5-7个月)

---

*文档版本: 1.0*
*创建时间: 2026-01-05*
