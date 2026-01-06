# NOFX Python 重构技术方案：补充章节（六-十六章）

> 本文档是《NOFX_Python_重构技术方案_A股港股_v2.md》的补充部分，包含第六章至第十六章的详细内容。

---

## 六、AI 决策引擎详细设计

### 6.1 决策引擎架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      AI 决策引擎架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Prompt      │    │  Context     │    │   LLM        │      │
│  │  Manager     │───►│  Builder     │───►│  Client      │      │
│  │  (Prompt管理)│    │  (上下文构建) │    │  (LLM调用)   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             ▼                                   │
│                    ┌──────────────┐                              │
│                    │   Parser     │                              │
│                    │  (决策解析)   │                              │
│                    └──────────────┘                              │
│                             │                                   │
│                             ▼                                   │
│                    ┌──────────────┐                              │
│                    │  Validator   │                              │
│                    │  (决策验证)   │                              │
│                    └──────────────┘                              │
│                             │                                   │
│                             ▼                                   │
│                    ┌──────────────┐                              │
│                    │  Executor    │                              │
│                    │  (决策执行)   │                              │
│                    └──────────────┘                              │
│                             │                                   │
│                             ▼                                   │
│                    ┌──────────────┐                              │
│                    │  Feedback    │                              │
│                    │  (反馈学习)   │                              │
│                    └──────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Prompt 管理系统

```python
# backend/decisions/prompt_manager.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel

from backend.config import get_config
from backend.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()


@dataclass
class PromptTemplate:
    """Prompt 模板"""
    name: str
    content: str
    description: str | None = None
    variables: list[str] | None = None
    version: str = "1.0"


class PromptSource(ABC):
    """Prompt 源抽象基类"""

    @abstractmethod
    async def load(self, name: str) -> PromptTemplate | None:
        """加载 Prompt 模板"""
        pass

    @abstractmethod
    async def list(self) -> list[str]:
        """列出所有可用模板"""
        pass


class FilePromptSource(PromptSource):
    """文件系统 Prompt 源"""

    def __init__(self, directory: Path):
        self.directory = directory
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(directory)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def load(self, name: str) -> PromptTemplate | None:
        """从文件加载 Prompt 模板"""
        try:
            template_path = self.directory / f"{name}.txt"
            if not template_path.exists():
                logger.warning(f"Prompt template not found: {name}")
                return None

            content = template_path.read_text(encoding="utf-8")

            # 提取变量（Jinja2 语法）
            template = self._jinja_env.get_template(f"{name}.txt")
            variables = list(meta.variables.keys()) if (meta := getattr(template, "module", None)) else []

            # 提取描述（从注释中）
            description = None
            for line in content.split("\n"):
                if line.strip().startswith("# "):
                    description = line.strip()[2:]
                    break

            return PromptTemplate(
                name=name,
                content=content,
                description=description,
                variables=variables,
            )

        except Exception as e:
            logger.error(f"Failed to load prompt template {name}: {e}")
            return None

    async def list(self) -> list[str]:
        """列出所有可用模板"""
        try:
            return [
                f.stem for f in self.directory.glob("*.txt")
                if not f.name.startswith("_")
            ]
        except Exception as e:
            logger.error(f"Failed to list prompt templates: {e}")
            return []


class DatabasePromptSource(PromptSource):
    """数据库 Prompt 源"""

    def __init__(self, db_session):
        self.db = db_session

    async def load(self, name: str) -> PromptTemplate | None:
        """从数据库加载 Prompt 模板"""
        try:
            from sqlalchemy import select
            from backend.models.database import PromptTemplate as DBPromptTemplate

            result = await self.db.execute(
                select(DBPromptTemplate).where(DBPromptTemplate.name == name)
            )
            db_template = result.scalar_one_or_none()

            if db_template:
                return PromptTemplate(
                    name=db_template.name,
                    content=db_template.content,
                    description=db_template.description,
                    variables=db_template.variables or [],
                    version=db_template.version,
                )
            return None

        except Exception as e:
            logger.error(f"Failed to load prompt from database {name}: {e}")
            return None

    async def list(self) -> list[str]:
        """列出所有可用模板"""
        try:
            from sqlalchemy import select
            from backend.models.database import PromptTemplate as DBPromptTemplate

            result = await self.db.execute(
                select(DBPromptTemplate.name).order_by(DBPromptTemplate.name)
            )
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to list prompts from database: {e}")
            return []


class PromptManager:
    """Prompt 管理器"""

    def __init__(self):
        self._sources: list[PromptSource] = []
        self._cache: dict[str, PromptTemplate] = {}
        self._jinja_env = Environment(
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # 注册默认源
        prompts_dir = config.config_dir / "prompts"
        if prompts_dir.exists():
            self.register_source(FilePromptSource(prompts_dir))

    def register_source(self, source: PromptSource):
        """注册 Prompt 源"""
        self._sources.append(source)

    async def load_template(self, name: str) -> PromptTemplate | None:
        """加载 Prompt 模板"""
        # 检查缓存
        if name in self._cache:
            return self._cache[name]

        # 从源加载
        for source in self._sources:
            template = await source.load(name)
            if template:
                self._cache[name] = template
                return template

        return None

    async def render(
        self,
        template_name: str,
        context: dict[str, Any],
    ) -> str:
        """渲染 Prompt 模板"""
        template = await self.load_template(template_name)
        if not template:
            raise ValueError(f"Prompt template not found: {template_name}")

        # 使用 Jinja2 渲染
        jinja_template = self._jinja_env.from_string(template.content)
        return jinja_template.render(**context)

    async def list_templates(self) -> list[str]:
        """列出所有可用模板"""
        all_templates = set()
        for source in self._sources:
            all_templates.update(await source.list())
        return sorted(all_templates)


# 全局 Prompt 管理器
prompt_manager = PromptManager()
```

### 6.3 上下文构建器

```python
# backend/decisions/context_builder.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from backend.core.trading_types import AccountInfo, Position
from backend.decisions.prompt_manager import PromptManager
from backend.markets.base import MarketData, TechnicalIndicators
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DecisionContext:
    """决策上下文"""
    # 账户信息
    account: AccountInfo

    # 持仓信息
    positions: list[Position]

    # 市场数据
    market_data: dict[str, MarketData] = field(default_factory=dict)

    # 技术指标
    indicators: dict[str, TechnicalIndicators] = field(default_factory=dict)

    # 候选股票
    candidate_stocks: list[str] = field(default_factory=list)

    # 历史表现
    performance_history: dict[str, Any] | None = None

    # 运行时信息
    current_time: datetime = field(default_factory=datetime.utcnow)
    runtime_minutes: int = 0
    call_count: int = 0


class ContextBuilder:
    """上下文构建器"""

    def __init__(
        self,
        prompt_manager: PromptManager,
        market_service,
        position_service,
    ):
        self.prompt_manager = prompt_manager
        self.market_service = market_service
        self.position_service = position_service

    async def build(
        self,
        context: DecisionContext,
        template_name: str = "default",
    ) -> str:
        """
        构建决策上下文

        Args:
            context: 决策上下文
            template_name: Prompt 模板名称

        Returns:
            渲染后的 Prompt
        """
        # 1. 获取市场数据（如果尚未加载）
        if not context.market_data:
            await self._load_market_data(context)

        # 2. 获取技术指标（如果尚未加载）
        if not context.indicators:
            await self._load_indicators(context)

        # 3. 获取历史表现（如果尚未加载）
        if context.performance_history is None:
            context.performance_history = await self._load_performance_history(context)

        # 4. 构建渲染上下文
        render_context = await self._build_render_context(context)

        # 5. 渲染 Prompt
        return await self.prompt_manager.render(template_name, render_context)

    async def _load_market_data(self, context: DecisionContext):
        """加载市场数据"""
        symbols = set(context.candidate_stocks)
        symbols.update(p.symbol for p in context.positions)

        for symbol in symbols:
            try:
                market_data = await self.market_service.get_realtime_data(symbol)
                context.market_data[symbol] = market_data
            except Exception as e:
                logger.warning(f"Failed to load market data for {symbol}: {e}")

    async def _load_indicators(self, context: DecisionContext):
        """加载技术指标"""
        symbols = list(context.market_data.keys())

        for symbol in symbols:
            try:
                indicators = await self.market_service.get_technical_indicators(symbol)
                context.indicators[symbol] = indicators
            except Exception as e:
                logger.warning(f"Failed to load indicators for {symbol}: {e}")

    async def _load_performance_history(
        self,
        context: DecisionContext,
    ) -> dict[str, Any]:
        """加载历史表现"""
        # 这里从数据库或缓存中获取历史交易记录
        # 简化版本：返回最近 N 笔交易的统计
        return {
            "recent_trades": 20,
            "win_rate": 0.55,
            "avg_profit": 150.0,
            "avg_loss": -100.0,
            "profit_factor": 1.5,
            "max_drawdown": 0.08,
            "best_symbol": "000001.SZ",
            "worst_symbol": "000002.SZ",
        }

    async def _build_render_context(
        self,
        context: DecisionContext,
    ) -> dict[str, Any]:
        """构建渲染上下文"""
        return {
            # 当前时间
            "current_time": context.current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_minutes": context.runtime_minutes,
            "call_count": context.call_count,

            # 账户状态
            "total_equity": context.account.total_equity,
            "available_balance": context.account.available_balance,
            "position_count": len(context.positions),
            "margin_used": context.account.margin_used,
            "margin_ratio": f"{context.account.margin_ratio:.2%}",

            # 持仓详情
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "quantity": p.quantity,
                    "entry_price": f"{p.entry_price:.2f}",
                    "current_price": f"{p.current_price:.2f}" if p.current_price else "N/A",
                    "unrealized_pnl": f"{p.unrealized_pnl:.2f}",
                    "unrealized_pnl_pct": f"{p.unrealized_pnl * 100:.2f}%",
                }
                for p in context.positions
            ],

            # 市场数据
            "market_data": self._format_market_data(context),

            # 技术指标
            "indicators": self._format_indicators(context),

            # 候选股票
            "candidate_stocks": ", ".join(context.candidate_stocks),

            # 历史表现
            "performance_summary": self._format_performance(context.performance_history),

            # 风险参数
            "risk_params": {
                "max_position_ratio": "20%",
                "stop_loss_pct": "5%",
                "take_profit_ratio": "2:1",
            },
        }

    def _format_market_data(self, context: DecisionContext) -> str:
        """格式化市场数据"""
        lines = []
        for symbol, data in context.market_data.items():
            lines.append(
                f"{symbol}: 价格={data.price:.2f}, "
                f"涨跌={data.price_change_pct:.2%}, "
                f"成交量={data.volume:,.0f}"
            )
        return "\n".join(lines)

    def _format_indicators(self, context: DecisionContext) -> str:
        """格式化技术指标"""
        lines = []
        for symbol, indicators in context.indicators.items():
            lines.append(
                f"{symbol}: "
                f"EMA20={indicators.ema20:.2f}, "
                f"MACD={indicators.macd:.2f}, "
                f"RSI14={indicators.rsi14:.2f}, "
                f"ATR={indicators.atr:.2f}"
            )
        return "\n".join(lines)

    def _format_performance(
        self,
        performance: dict[str, Any] | None,
    ) -> str:
        """格式化历史表现"""
        if not performance:
            return "暂无历史数据"

        return f"""
最近 {performance.get('recent_trades', 0)} 笔交易统计:
- 胜率: {performance.get('win_rate', 0):.2%}
- 平均盈利: ¥{performance.get('avg_profit', 0):.2f}
- 平均亏损: ¥{performance.get('avg_loss', 0):.2f}
- 盈亏比: {performance.get('profit_factor', 0):.2f}
- 最大回撤: {performance.get('max_drawdown', 0):.2%}
- 最佳股票: {performance.get('best_symbol', 'N/A')}
- 最差股票: {performance.get('worst_symbol', 'N/A')}
        """.strip()
```

### 6.4 决策解析器

```python
# backend/decisions/parser.py
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, validator

from backend.core.trading_types import Decision
from backend.exceptions import ValidationError
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ParsedDecision(BaseModel):
    """解析后的决策"""
    symbol: str = Field(..., description="股票代码")
    action: str = Field(..., description="操作类型")
    quantity: int | None = Field(None, description="数量")
    price: float | None = Field(None, description="价格")
    confidence: int | None = Field(None, ge=0, le=100, description="信心度")
    reasoning: str | None = Field(None, description="决策理由")
    stop_loss: float | None = Field(None, gt=0, description="止损价")
    take_profit: float | None = Field(None, gt=0, description="止盈价")

    @validator("action")
    def validate_action(cls, v):
        valid_actions = {"buy", "sell", "hold", "close"}
        if v.lower() not in valid_actions:
            raise ValueError(f"Invalid action: {v}")
        return v.lower()

    @validator("symbol")
    def validate_symbol(cls, v):
        # 简单验证股票代码格式
        v = v.upper().strip()
        if not re.match(r"^\d{6}\.[A-Z]{2}$", v):
            # 尝试自动补全
            if re.match(r"^\d{6}$", v):
                if v.startswith("6"):
                    v = f"{v}.SH"
                else:
                    v = f"{v}.SZ"
        return v


class DecisionParser:
    """决策解析器"""

    def __init__(self):
        # 预编译正则表达式
        self.json_fence_re = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
        self.json_re = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)

    def parse(
        self,
        llm_output: str,
        context: Any,
    ) -> list[Decision]:
        """
        解析 LLM 输出为决策列表

        Args:
            llm_output: LLM 原始输出
            context: 决策上下文

        Returns:
            决策列表
        """
        # 1. 提取 JSON
        json_str = self._extract_json(llm_output)

        # 2. 解析 JSON
        try:
            raw_decisions = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"JSON string: {json_str[:500]}")
            # 尝试修复 JSON
            json_str = self._fix_json(json_str)
            try:
                raw_decisions = json.loads(json_str)
            except:
                raise ValidationError(f"无法解析 LLM 输出: {e}")

        # 3. 验证和转换
        if not isinstance(raw_decisions, list):
            raw_decisions = [raw_decisions]

        decisions = []
        for raw_decision in raw_decisions:
            try:
                parsed = ParsedDecision(**raw_decision)
                decisions.append(self._to_domain_decision(parsed))
            except Exception as e:
                logger.warning(f"Failed to parse decision: {e}, raw: {raw_decision}")
                continue

        # 4. 验证决策
        decisions = self._validate_decisions(decisions, context)

        return decisions

    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON"""
        # 尝试提取 ```json...``` 代码块
        match = self.json_fence_re.search(text)
        if match:
            return match.group(1)

        # 尝试提取第一个 JSON 数组
        match = self.json_re.search(text)
        if match:
            return match.group(0)

        # 如果都没有，返回原文本
        return text.strip()

    def _fix_json(self, json_str: str) -> str:
        """尝试修复损坏的 JSON"""
        # 移除不可见字符
        json_str = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", json_str)

        # 修复常见的 JSON 错误
        json_str = re.sub(r",\s*}", "}", json_str)  # 移除尾随逗号
        json_str = re.sub(r",\s*]", "]", json_str)

        # 修复引号
        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)

        return json_str

    def _to_domain_decision(self, parsed: ParsedDecision) -> Decision:
        """转换为领域模型"""
        return Decision(
            symbol=parsed.symbol,
            action=parsed.action,
            quantity=parsed.quantity,
            confidence=parsed.confidence,
            reasoning=parsed.reasoning,
            stop_loss=parsed.stop_loss,
            take_profit=parsed.take_profit,
        )

    def _validate_decisions(
        self,
        decisions: list[Decision],
        context: Any,
    ) -> list[Decision]:
        """验证决策"""
        valid_decisions = []

        for decision in decisions:
            # 跳过 hold 操作
            if decision.action == "hold":
                continue

            # 验证数量
            if decision.action in ("buy", "sell") and not decision.quantity:
                logger.warning(f"Decision {decision.symbol} missing quantity")
                continue

            # 验证价格
            if decision.action == "buy" and not decision.price:
                logger.warning(f"Decision {decision.symbol} missing price")
                continue

            valid_decisions.append(decision)

        return valid_decisions
```

### 6.5 决策验证器

```python
# backend/decisions/validator.py
from typing import Any

from backend.core.trading_types import Decision, DecisionContext
from backend.core.risk_manager import RiskManager
from backend.exceptions import RiskLimitError, ValidationError
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class DecisionValidator:
    """决策验证器"""

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager

    async def validate(
        self,
        decisions: list[Decision],
        context: DecisionContext,
    ) -> list[Decision]:
        """
        验证决策

        Args:
            decisions: 待验证的决策列表
            context: 决策上下文

        Returns:
            验证通过的决策列表

        Raises:
            RiskLimitError: 风险限制触发
            ValidationError: 验证失败
        """
        valid_decisions = []

        for decision in decisions:
            try:
                # 1. 基本验证
                self._validate_basic(decision)

                # 2. 市场验证
                await self._validate_market(decision, context)

                # 3. 风险验证
                await self._validate_risk(decision, context)

                # 4. 交易规则验证
                self._validate_trading_rules(decision, context)

                valid_decisions.append(decision)

            except (RiskLimitError, ValidationError) as e:
                logger.warning(f"Decision rejected: {decision.symbol} - {e}")
                continue

        return valid_decisions

    def _validate_basic(self, decision: Decision):
        """基本验证"""
        if not decision.symbol:
            raise ValidationError("Symbol is required")

        if decision.action not in ("buy", "sell", "hold", "close"):
            raise ValidationError(f"Invalid action: {decision.action}")

        if decision.quantity and decision.quantity <= 0:
            raise ValidationError("Quantity must be positive")

        if decision.stop_loss and decision.stop_loss <= 0:
            raise ValidationError("Stop loss must be positive")

        if decision.take_profit and decision.take_profit <= 0:
            raise ValidationError("Take profit must be positive")

    async def _validate_market(
        self,
        decision: Decision,
        context: DecisionContext,
    ):
        """市场验证"""
        # 检查股票是否在候选池中
        if decision.action == "buy":
            if decision.symbol not in context.candidate_stocks:
                logger.warning(f"{decision.symbol} not in candidate pool")

        # 检查市场数据
        if decision.symbol not in context.market_data:
            raise ValidationError(f"No market data for {decision.symbol}")

    async def _validate_risk(
        self,
        decision: Decision,
        context: DecisionContext,
    ):
        """风险验证"""
        # 检查仓位限制
        await self.risk_manager.check_position_limit(decision, context)

        # 检查止损止盈比例
        if decision.stop_loss and decision.take_profit:
            risk_reward = (
                (decision.take_profit - decision.price) /
                (decision.price - decision.stop_loss)
                if decision.price
                else 0
            )
            if risk_reward < 2.0:
                logger.warning(
                    f"{decision.symbol} risk/reward ratio {risk_reward:.2f} < 2.0"
                )

    def _validate_trading_rules(
        self,
        decision: Decision,
        context: DecisionContext,
    ):
        """交易规则验证"""
        # A股 T+1 检查
        if context.market_type == "a_stock" and decision.action == "sell":
            for position in context.positions:
                if position.symbol == decision.symbol:
                    # 检查是否可卖出
                    from datetime import date
                    if position.can_sell_date and position.can_sell_date > date.today():
                        raise ValidationError(
                            f"{decision.symbol} cannot be sold yet (T+1)"
                        )
                    break

        # A股手数检查
        if context.market_type == "a_stock":
            if decision.quantity and decision.quantity % 100 != 0:
                raise ValidationError(
                    f"Quantity must be multiple of 100 (got {decision.quantity})"
                )
```

---

## 七、交易执行层完整实现

### 7.1 A股模拟交易器完整实现

```python
# backend/traders/atrader/simulator.py
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from backend.core.fee_calculator import AStockFeeCalculator, Side, get_fee_calculator
from backend.core.trading_calendar import get_trading_calendar
from backend.core.trading_types import (
    AccountInfo,
    BaseTrader,
    OrderType,
    Position,
)
from backend.exceptions import TradingError, ValidationError
from backend.utils.helpers import calculate_pnl, calculate_return
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Order:
    """订单"""
    id: str
    symbol: str
    side: Side
    quantity: int
    price: float | None
    order_type: OrderType
    stop_loss: float | None
    take_profit: float | None
    status: str = "pending"  # pending, filled, cancelled, failed
    filled_quantity: int = 0
    filled_price: float | None = None
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: datetime | None = None
    error_message: str | None = None


@dataclass
class PositionDetail:
    """持仓详情"""
    symbol: str
    side: Side
    quantity: int
    entry_price: float
    current_price: float
    stop_loss: float | None
    take_profit: float | None
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    peak_pnl_pct: float = 0.0
    can_sell_date: date | None = None
    opened_at: datetime = field(default_factory=datetime.now)


class ATraderSimulator(BaseTrader):
    """A股模拟交易器"""

    def __init__(
        self,
        initial_balance: float = 1_000_000,
        market_type: str = "a_stock",
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.market_type = market_type

        # 持仓
        self.positions: dict[str, PositionDetail] = {}

        # 订单
        self.orders: dict[str, Order] = {}
        self._order_counter = 0

        # 交易记录
        self.trades: list[dict[str, Any]] = []

        # 费用计算器
        self.fee_calculator = AStockFeeCalculator()

        # 交易日历
        self.calendar = get_trading_calendar(market_type)

        # 当前价格（模拟）
        self._current_prices: dict[str, float] = {}

        logger.info(f"Initialized A-Stock simulator with ¥{initial_balance:,.2f}")

    async def get_account(self) -> AccountInfo:
        """获取账户信息"""
        # 计算总权益
        total_unrealized_pnl = sum(
            position.unrealized_pnl for position in self.positions.values()
        )
        self.equity = self.balance + total_unrealized_pnl

        positions_list = [
            Position(
                symbol=p.symbol,
                side=p.side.value,
                quantity=p.quantity,
                entry_price=p.entry_price,
                current_price=p.current_price,
                unrealized_pnl=p.unrealized_pnl,
            )
            for p in self.positions.values()
        ]

        return AccountInfo(
            total_equity=self.equity,
            available_balance=self.balance,
            positions=positions_list,
            margin_used=total_unrealized_pnl,
            margin_ratio=total_unrealized_pnl / self.equity if self.equity > 0 else 0,
        )

    async def get_positions(self) -> list[Position]:
        """获取所有持仓"""
        return [
            Position(
                symbol=p.symbol,
                side=p.side.value,
                quantity=p.quantity,
                entry_price=p.entry_price,
                current_price=p.current_price,
                unrealized_pnl=p.unrealized_pnl,
            )
            for p in self.positions.values()
        ]

    async def open_position(
        self,
        symbol: str,
        side: Side,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        """开仓"""
        # 1. 验证
        await self._validate_open_order(symbol, side, quantity, price)

        # 2. 获取成交价格
        fill_price = price or await self._get_market_price(symbol)

        # 3. 计算费用
        fees = self.fee_calculator.calculate(fill_price, quantity, side)

        # 4. 计算所需资金
        required_amount = fill_price * quantity + fees.total

        # 5. 检查余额（仅买入）
        if side == Side.BUY:
            if self.balance < required_amount:
                raise TradingError(
                    f"Insufficient balance: need ¥{required_amount:,.2f}, "
                    f"have ¥{self.balance:,.2f}"
                )

        # 6. 创建订单
        order_id = self._generate_order_id()
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            order_type=order_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        # 7. 执行订单
        order.status = "filled"
        order.filled_quantity = quantity
        order.filled_price = fill_price
        order.filled_at = datetime.now()

        # 8. 更新账户
        if side == Side.BUY:
            self.balance -= required_amount
        else:
            # 卖出（做空），A股不支持，这里保留接口
            pass

        # 9. 创建/更新持仓
        position_key = f"{symbol}_{side.value}"
        if position_key in self.positions:
            # 加仓
            existing = self.positions[position_key]
            total_quantity = existing.quantity + quantity
            total_cost = existing.entry_price * existing.quantity + fill_price * quantity
            existing.entry_price = total_cost / total_quantity
            existing.quantity = total_quantity
            existing.stop_loss = stop_loss or existing.stop_loss
            existing.take_profit = take_profit or existing.take_profit
        else:
            # 新建持仓
            can_sell_date = None
            if side == Side.BUY:
                # A股 T+1，次日可卖
                can_sell_date = self.calendar.next_trading_day(datetime.now())

            self.positions[position_key] = PositionDetail(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=fill_price,
                current_price=fill_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                can_sell_date=can_sell_date,
            )

        # 10. 记录交易
        self.trades.append({
            "id": str(order_id),
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "price": fill_price,
            "fees": float(fees.total),
            "executed_at": order.filled_at.isoformat(),
        })

        # 11. 保存订单
        self.orders[order_id] = order

        logger.info(
            f"Opened {side.value} position: {symbol} {quantity} shares "
            f"@ ¥{fill_price:.2f}, fees: ¥{fees.total:.2f}"
        )

        return {
            "order_id": order_id,
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "price": fill_price,
            "fees": float(fees.total),
            "status": "filled",
        }

    async def close_position(
        self,
        symbol: str,
        side: Side,
        quantity: int | None = None,
    ) -> dict[str, Any]:
        """平仓"""
        position_key = f"{symbol}_{side.value}"

        # 1. 检查持仓
        if position_key not in self.positions:
            raise TradingError(f"No position found for {symbol} {side.value}")

        position = self.positions[position_key]

        # 2. 确定平仓数量
        close_quantity = quantity or position.quantity
        if close_quantity > position.quantity:
            raise ValidationError(
                f"Cannot close {close_quantity} shares, only {position.quantity} available"
            )

        # 3. 检查 T+1（仅卖出）
        if side == Side.BUY:
            if position.can_sell_date and position.can_sell_date > date.today():
                raise TradingError(
                    f"Cannot sell {symbol} yet (T+1), can sell from {position.can_sell_date}"
                )

        # 4. 获取成交价格
        fill_price = await self._get_market_price(symbol)

        # 5. 计算费用
        fees = self.fee_calculator.calculate(fill_price, close_quantity, Side.SELL)

        # 6. 计算盈亏
        pnl = calculate_pnl(position.entry_price, fill_price, close_quantity, "long")
        pnl_pct = calculate_return(position.entry_price, fill_price)

        # 7. 更新账户
        self.balance += fill_price * close_quantity - fees.total

        # 8. 更新持仓
        if close_quantity == position.quantity:
            # 全部平仓
            del self.positions[position_key]
        else:
            # 部分平仓
            position.quantity -= close_quantity
            # 记录已实现盈亏
            position.realized_pnl += pnl

        # 9. 记录交易
        self.trades.append({
            "id": self._generate_order_id(),
            "symbol": symbol,
            "side": "sell",
            "quantity": close_quantity,
            "price": fill_price,
            "fees": float(fees.total),
            "pnl": float(pnl),
            "pnl_pct": float(pnl_pct),
            "executed_at": datetime.now().isoformat(),
        })

        logger.info(
            f"Closed {side.value} position: {symbol} {close_quantity} shares "
            f"@ ¥{fill_price:.2f}, PnL: ¥{pnl:.2f} ({pnl_pct:.2%})"
        )

        return {
            "symbol": symbol,
            "side": side.value,
            "quantity": close_quantity,
            "price": fill_price,
            "fees": float(fees.total),
            "pnl": float(pnl),
            "pnl_pct": float(pnl_pct),
            "status": "filled",
        }

    async def set_stop_loss(
        self,
        symbol: str,
        side: Side,
        stop_price: float,
    ) -> dict[str, Any]:
        """设置止损"""
        position_key = f"{symbol}_{side.value}"

        if position_key not in self.positions:
            raise TradingError(f"No position found for {symbol} {side.value}")

        position = self.positions[position_key]
        position.stop_loss = stop_price

        logger.info(f"Set stop loss for {symbol} {side.value}: ¥{stop_price:.2f}")

        return {"symbol": symbol, "side": side.value, "stop_loss": stop_price}

    async def set_take_profit(
        self,
        symbol: str,
        side: Side,
        take_profit_price: float,
    ) -> dict[str, Any]:
        """设置止盈"""
        position_key = f"{symbol}_{side.value}"

        if position_key not in self.positions:
            raise TradingError(f"No position found for {symbol} {side.value}")

        position = self.positions[position_key]
        position.take_profit = take_profit_price

        logger.info(f"Set take profit for {symbol} {side.value}: ¥{take_profit_price:.2f}")

        return {"symbol": symbol, "side": side.value, "take_profit": take_profit_price}

    async def cancel_orders(self, symbol: str) -> dict[str, Any]:
        """取消所有挂单"""
        cancelled = []
        for order_id, order in list(self.orders.items()):
            if order.symbol == symbol and order.status == "pending":
                order.status = "cancelled"
                cancelled.append(order_id)

        logger.info(f"Cancelled {len(cancelled)} orders for {symbol}")
        return {"symbol": symbol, "cancelled_orders": cancelled}

    async def update_prices(self, prices: dict[str, float]):
        """更新价格（模拟用）"""
        self._current_prices.update(prices)

        # 更新持仓盈亏
        for position in self.positions.values():
            if position.symbol in prices:
                old_price = position.current_price
                position.current_price = prices[position.symbol]

                # 计算盈亏
                position.unrealized_pnl = calculate_pnl(
                    position.entry_price,
                    position.current_price,
                    position.quantity,
                    "long" if position.side == Side.BUY else "short",
                )
                position.unrealized_pnl_pct = calculate_return(
                    position.entry_price,
                    position.current_price,
                )

                # 更新峰值盈亏
                if position.unrealized_pnl_pct > position.peak_pnl_pct:
                    position.peak_pnl_pct = position.unrealized_pnl_pct

                # 检查止损止盈
                if position.side == Side.BUY:
                    if position.stop_loss and position.current_price <= position.stop_loss:
                        logger.warning(
                            f"{position.symbol} hit stop loss: "
                            f"¥{position.current_price:.2f} <= ¥{position.stop_loss:.2f}"
                        )
                        # 触发止损（这里应该触发自动平仓逻辑）
                    if position.take_profit and position.current_price >= position.take_profit:
                        logger.info(
                            f"{position.symbol} hit take profit: "
                            f"¥{position.current_price:.2f} >= ¥{position.take_profit:.2f}"
                        )
                        # 触发止盈

    # ========== 私有方法 ==========

    async def _validate_open_order(
        self,
        symbol: str,
        side: Side,
        quantity: int,
        price: float | None,
    ):
        """验证开仓订单"""
        # A股手数检查
        if quantity % 100 != 0:
            raise ValidationError(f"Quantity must be multiple of 100 (got {quantity})")

        # 检查交易时间
        if not self.calendar.is_trading_time(datetime.now()):
            logger.warning(f"Order placed outside trading hours")

    async def _get_market_price(self, symbol: str) -> float:
        """获取市场价格"""
        if symbol in self._current_prices:
            return self._current_prices[symbol]

        # 这里应该从市场数据服务获取
        # 简化版：返回模拟价格
        logger.warning(f"No market price for {symbol}, using default")
        return 10.0

    def _generate_order_id(self) -> str:
        """生成订单 ID"""
        self._order_counter += 1
        return f"ORDER_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._order_counter:06d}"
```

---

## 八、市场数据层深度实现

### 8.1 Tushare 数据源完整实现

```python
# backend/markets/atrader/tushare.py
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import tushare as ts

from backend.config import get_config
from backend.markets.base import (
    BaseMarketDataProvider,
    Kline,
    MarketData,
    TechnicalIndicators,
)
from backend.exceptions import MarketDataError
from backend.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()


class TushareDataProvider(BaseMarketDataProvider):
    """Tushare 数据源"""

    def __init__(self, token: str | None = None):
        self.token = token or config.market.tushare_token
        if not self.token:
            raise ValueError("Tushare token is required")

        ts.set_token(self.token)
        self.pro = ts.pro_api()

        # 缓存
        self._cache: dict[str, Any] = {}
        self._cache_ttl = 60  # 缓存60秒

    async def get_realtime_price(self, symbol: str) -> MarketData:
        """获取实时价格"""
        # Tushare 免费版没有实时行情，使用最新的日线数据
        try:
            today = datetime.now().strftime("%Y%m%d")
            df = self.pro.daily(
                ts_code=symbol,
                trade_date=today,
            )

            if df.empty:
                # 尝试获取最近一天
                df = self.pro.daily(
                    ts_code=symbol,
                ).head(1)

            if df.empty:
                raise MarketDataError(f"No data found for {symbol}")

            row = df.iloc[0]

            return MarketData(
                symbol=row["ts_code"],
                timestamp=datetime.now(),
                price=float(row["close"]),
                volume=float(row["vol"]),
                bid_price=None,  # Tushare 免费版没有五档行情
                ask_price=None,
            )

        except Exception as e:
            logger.error(f"Failed to get realtime price for {symbol}: {e}")
            raise MarketDataError(f"Failed to get price for {symbol}: {e}") from e

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> list[Kline]:
        """获取K线数据"""
        try:
            # 映射 interval
            interval_map = {
                "1min": "1min",
                "5min": "5min",
                "15min": "15min",
                "30min": "30min",
                "60min": "60min",
                "daily": "D",
                "weekly": "W",
                "monthly": "M",
            }

            ts_interval = interval_map.get(interval, "D")

            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=limit * 2)  # 多取一些以防节假日

            if ts_interval in ["1min", "5min", "15min", "30min", "60min"]:
                # 分钟线
                df = self.pro.stk_mins(
                    ts_code=symbol,
                    freq=ts_interval,
                    start_date=start_date.strftime("%Y%m%d %H:%M:%S"),
                    end_date=end_date.strftime("%Y%m%d %H:%M:%S"),
                )
            else:
                # 日线及以上
                df = self.pro.daily(
                    ts_code=symbol,
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                )

            if df.empty:
                return []

            # 转换为 Kline 对象
            klines = []
            for _, row in df.tail(limit).iterrows():
                klines.append(
                    Kline(
                        symbol=row["ts_code"],
                        timestamp=pd.to_datetime(row["trade_date"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["vol"]),
                    )
                )

            return klines

        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            raise MarketDataError(f"Failed to get klines for {symbol}: {e}") from e

    async def get_technical_indicators(
        self,
        symbol: str,
    ) -> TechnicalIndicators:
        """获取技术指标"""
        try:
            # 获取足够的数据来计算指标
            klines = await self.get_klines(symbol, "daily", limit=100)

            if not klines:
                raise MarketDataError(f"No klines data for {symbol}")

            # 转换为 DataFrame
            df = pd.DataFrame([
                {
                    "timestamp": k.timestamp,
                    "open": k.open,
                    "high": k.high,
                    "low": k.low,
                    "close": k.close,
                    "volume": k.volume,
                }
                for k in klines
            ])

            # 计算技术指标
            from backend.markets.indicators import TechnicalIndicatorCalculator

            calculator = TechnicalIndicatorCalculator()

            ema20 = calculator.calculate_ema(df, 20).iloc[-1]
            ema50 = calculator.calculate_ema(df, 50).iloc[-1]

            macd_data = calculator.calculate_macd(df)
            macd = macd_data["macd"].iloc[-1]

            rsi6 = calculator.calculate_rsi(df, 6).iloc[-1]
            rsi12 = calculator.calculate_rsi(df, 12).iloc[-1]
            rsi24 = calculator.calculate_rsi(df, 24).iloc[-1]

            atr = calculator.calculate_atr(df).iloc[-1]

            return TechnicalIndicators(
                symbol=symbol,
                timestamp=datetime.now(),
                ema20=float(ema20),
                ema50=float(ema50),
                macd=float(macd),
                rsi6=float(rsi6) if not pd.isna(rsi6) else None,
                rsi12=float(rsi12) if not pd.isna(rsi12) else None,
                rsi24=float(rsi24) if not pd.isna(rsi24) else None,
                atr=float(atr) if not pd.isna(atr) else None,
            )

        except Exception as e:
            logger.error(f"Failed to get indicators for {symbol}: {e}")
            raise MarketDataError(f"Failed to get indicators for {symbol}: {e}") from e

    async def subscribe_realtime(self, symbols: list[str], callback):
        """订阅实时行情"""
        # Tushare 免费版不支持 WebSocket，使用轮询
        import asyncio

        logger.info(f"Starting polling for {len(symbols)} symbols")

        while True:
            try:
                for symbol in symbols:
                    try:
                        data = await self.get_realtime_price(symbol)
                        await callback(symbol, data)
                    except Exception as e:
                        logger.warning(f"Failed to get data for {symbol}: {e}")

                await asyncio.sleep(5)  # 5秒轮询一次

            except asyncio.CancelledError:
                logger.info("Polling cancelled")
                break
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(10)
```

### 8.2 技术指标计算器完整实现

```python
# backend/markets/indicators.py
from typing import Any

import pandas as pd
import pandas_ta as ta
from pandas import Series


class TechnicalIndicatorCalculator:
    """技术指标计算器"""

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int) -> Series:
        """
        计算指数移动平均线 (EMA)

        Args:
            df: 包含 close 列的 DataFrame
            period: 周期

        Returns:
            EMA 值的 Series
        """
        return df["close"].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int) -> Series:
        """
        计算简单移动平均线 (SMA)

        Args:
            df: 包含 close 列的 DataFrame
            period: 周期

        Returns:
            SMA 值的 Series
        """
        return df["close"].rolling(window=period).mean()

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict[str, Series]:
        """
        计算 MACD

        Args:
            df: 包含 close 列的 DataFrame
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期

        Returns:
            包含 macd, signal, histogram 的字典
        """
        macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)

        return {
            "macd": macd_df[f"MACD_{fast}_{slow}_{signal}"],
            "signal": macd_df[f"MACDs_{fast}_{slow}_{signal}"],
            "histogram": macd_df[f"MACDh_{fast}_{slow}_{signal}"],
        }

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> Series:
        """
        计算相对强弱指标 (RSI)

        Args:
            df: 包含 close 列的 DataFrame
            period: 周期

        Returns:
            RSI 值的 Series
        """
        return ta.rsi(df["close"], length=period)

    @staticmethod
    def calculate_stoch(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> dict[str, Series]:
        """
        计算随机指标 (Stochastic Oscillator)

        Args:
            df: 包含 high, low, close 列的 DataFrame
            k_period: K 线周期
            d_period: D 线周期

        Returns:
            包含 stoch_k, stoch_d 的字典
        """
        stoch_df = ta.stoch(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            k=k_period,
            d=d_period,
        )

        return {
            "stoch_k": stoch_df[f"STOCHk_{k_period}_{d_period}"],
            "stoch_d": stoch_df[f"STOd_{k_period}_{d_period}"],
        }

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: int = 2,
    ) -> dict[str, Series]:
        """
        计算布林带 (Bollinger Bands)

        Args:
            df: 包含 close 列的 DataFrame
            period: 周期
            std_dev: 标准差倍数

        Returns:
            包含 upper, middle, lower 的字典
        """
        bb_df = ta.bbands(
            df["close"],
            length=period,
            std=std_dev,
        )

        return {
            "upper": bb_df[f"BBU_{period}_{std_dev}"],
            "middle": bb_df[f"BBM_{period}_{std_dev}"],
            "lower": bb_df[f"BBL_{period}_{std_dev}"],
        }

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> Series:
        """
        计算平均真实波幅 (ATR)

        Args:
            df: 包含 high, low, close 列的 DataFrame
            period: 周期

        Returns:
            ATR 值的 Series
        """
        return ta.atr(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            length=period,
        )

    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> Series:
        """
        计算能量潮 (On-Balance Volume)

        Args:
            df: 包含 close, volume 列的 DataFrame

        Returns:
            OBV 值的 Series
        """
        return ta.obv(df["close"], df["volume"])

    @staticmethod
    def calculate_adx(
        df: pd.DataFrame,
        period: int = 14,
    ) -> dict[str, Series]:
        """
        计算平均趋向指标 (ADX)

        Args:
            df: 包含 high, low, close 列的 DataFrame
            period: 周期

        Returns:
            包含 adx, di_plus, di_minus 的字典
        """
        adx_df = ta.adx(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            length=period,
        )

        return {
            "adx": adx_df[f"ADX_{period}"],
            "di_plus": adx_df[f"DMP_{period}"],
            "di_minus": adx_df[f"DMN_{period}"],
        }

    @staticmethod
    def calculate_cci(df: pd.DataFrame, period: int = 20) -> Series:
        """
        计算顺势指标 (CCI)

        Args:
            df: 包含 high, low, close 列的 DataFrame
            period: 周期

        Returns:
            CCI 值的 Series
        """
        return ta.cci(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            length=period,
        )

    @staticmethod
    def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> Series:
        """
        计算威廉指标 (Williams %R)

        Args:
            df: 包含 high, low, close 列的 DataFrame
            period: 周期

        Returns:
            Williams %R 值的 Series
        """
        return ta.willr(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            length=period,
        )

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> dict[str, Any]:
        """
        计算所有常用技术指标

        Args:
            df: 包含 OHLCV 列的 DataFrame

        Returns:
            包含所有指标的字典
        """
        return {
            # 趋势指标
            "sma_20": TechnicalIndicatorCalculator.calculate_sma(df, 20),
            "sma_50": TechnicalIndicatorCalculator.calculate_sma(df, 50),
            "ema_20": TechnicalIndicatorCalculator.calculate_ema(df, 20),
            "ema_50": TechnicalIndicatorCalculator.calculate_ema(df, 50),
            "macd": TechnicalIndicatorCalculator.calculate_macd(df),
            "adx": TechnicalIndicatorCalculator.calculate_adx(df),

            # 震荡指标
            "rsi_6": TechnicalIndicatorCalculator.calculate_rsi(df, 6),
            "rsi_14": TechnicalIndicatorCalculator.calculate_rsi(df, 14),
            "stoch": TechnicalIndicatorCalculator.calculate_stoch(df),
            "williams_r": TechnicalIndicatorCalculator.calculate_williams_r(df),
            "cci": TechnicalIndicatorCalculator.calculate_cci(df),

            # 波动率指标
            "bollinger_bands": TechnicalIndicatorCalculator.calculate_bollinger_bands(df),
            "atr": TechnicalIndicatorCalculator.calculate_atr(df),

            # 成交量指标
            "obv": TechnicalIndicatorCalculator.calculate_obv(df),
        }
```

---

## 九、API 服务层设计

### 9.1 FastAPI 依赖注入

```python
# backend/api/dependencies.py
from typing import Annotated, AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_config
from backend.models.database import User, Trader
from backend.repositories.trader_repository import TraderRepository
from backend.services.auth_service import AuthService
from backend.services.trader_service import TraderService
from backend.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

# OAuth2 配置
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ========== 数据库依赖 ==========

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话"""
    from backend.models.database import async_session_maker

    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ========== 认证依赖 ==========

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """获取当前用户"""
    auth_service = AuthService(db)

    try:
        user = await auth_service.get_current_user(token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """获取当前活跃用户"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


# ========== 服务依赖 ==========

def get_auth_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthService:
    """获取认证服务"""
    return AuthService(db)


def get_trader_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TraderService:
    """获取交易员服务"""
    return TraderService(db)


def get_trader_repository(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TraderRepository:
    """获取交易员仓库"""
    return TraderRepository(db)


# ========== 资源检查依赖 ==========

async def get_trader_by_id(
    trader_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    repository: Annotated[TraderRepository, Depends(get_trader_repository)],
) -> Trader:
    """通过 ID 获取交易员（带权限检查）"""
    trader = await repository.get_by_id(trader_id)

    if not trader:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trader {trader_id} not found"
        )

    # 检查权限
    if trader.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this trader"
        )

    return trader
```

### 9.2 交易员路由

```python
# backend/api/routes/traders.py
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import (
    get_current_active_user,
    get_trader_by_id,
    get_trader_service,
    get_db,
)
from backend.models.database import User
from backend.models.schemas import (
    TraderCreate,
    TraderResponse,
    TraderUpdate,
)
from backend.services.trader_service import TraderService
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/traders", tags=["交易员"])


# ========== 请求/响应模型 ==========

class TraderCreateRequest(BaseModel):
    """创建交易员请求"""
    name: str
    description: str | None = None
    ai_model_id: str
    market_type: str  # 'a_stock' or 'hk_stock'
    initial_balance: float
    stock_pool_id: str | None = None


class TraderUpdateRequest(BaseModel):
    """更新交易员请求"""
    name: str | None = None
    description: str | None = None
    ai_model_id: str | None = None
    stock_pool_id: str | None = None


class TraderControlRequest(BaseModel):
    """交易员控制请求"""
    action: str  # 'start', 'stop', 'pause', 'resume'


# ========== 路由 ==========

@router.get("", response_model=list[TraderResponse])
async def list_traders(
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[TraderService, Depends(get_trader_service)],
    market_type: str | None = None,
    is_running: bool | None = None,
):
    """
    获取交易员列表

    - **market_type**: 筛选市场类型
    - **is_running**: 筛选运行状态
    """
    traders = await service.list_traders(
        user_id=current_user.id,
        market_type=market_type,
        is_running=is_running,
    )
    return traders


@router.post("", response_model=TraderResponse, status_code=status.HTTP_201_CREATED)
async def create_trader(
    request: TraderCreateRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[TraderService, Depends(get_trader_service)],
):
    """
    创建交易员

    - **name**: 交易员名称
    - **description**: 描述
    - **ai_model_id**: AI 模型 ID
    - **market_type**: 市场类型 ('a_stock' or 'hk_stock')
    - **initial_balance**: 初始资金
    - **stock_pool_id**: 股票池 ID（可选）
    """
    try:
        trader = await service.create_trader(
            user_id=current_user.id,
            name=request.name,
            description=request.description,
            ai_model_id=request.ai_model_id,
            market_type=request.market_type,
            initial_balance=request.initial_balance,
            stock_pool_id=request.stock_pool_id,
        )
        return trader

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to create trader: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create trader",
        )


@router.get("/{trader_id}", response_model=TraderResponse)
async def get_trader(
    trader: Annotated[dict, Depends(get_trader_by_id)],
):
    """获取交易员详情"""
    return trader


@router.put("/{trader_id}", response_model=TraderResponse)
async def update_trader(
    trader_id: str,
    request: TraderUpdateRequest,
    trader: Annotated[dict, Depends(get_trader_by_id)],
    service: Annotated[TraderService, Depends(get_trader_service)],
):
    """更新交易员"""
    try:
        updated = await service.update_trader(
            trader_id=trader_id,
            name=request.name,
            description=request.description,
            ai_model_id=request.ai_model_id,
            stock_pool_id=request.stock_pool_id,
        )
        return updated

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete("/{trader_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_trader(
    trader_id: str,
    trader: Annotated[dict, Depends(get_trader_by_id)],
    service: Annotated[TraderService, Depends(get_trader_service)],
):
    """删除交易员"""
    await service.delete_trader(trader_id)


@router.post("/{trader_id}/control")
async def control_trader(
    trader_id: str,
    request: TraderControlRequest,
    trader: Annotated[dict, Depends(get_trader_by_id)],
    service: Annotated[TraderService, Depends(get_trader_service)],
):
    """
    控制交易员

    - **action**: 操作类型
      - `start`: 启动交易员
      - `stop`: 停止交易员
      - `pause`: 暂停交易员
      - `resume`: 恢复交易员
    """
    try:
        if request.action == "start":
            result = await service.start_trader(trader_id)
        elif request.action == "stop":
            result = await service.stop_trader(trader_id)
        elif request.action == "pause":
            result = await service.pause_trader(trader_id)
        elif request.action == "resume":
            result = await service.resume_trader(trader_id)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown action: {request.action}",
            )

        return {"status": "success", "result": result}

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
```

---

## 十、前端架构设计

### 10.1 前端项目结构

```
frontend/
├── src/
│   ├── App.tsx                 # 主应用组件
│   ├── main.tsx                # 应用入口
│   ├── index.css               # 全局样式
│   │
│   ├── components/             # 通用组件
│   │   ├── ui/                 # 基础 UI 组件
│   │   │   ├── Button.tsx
│   │   │   ├── Input.tsx
│   │   │   ├── Modal.tsx
│   │   │   ├── Table.tsx
│   │   │   └── ...
│   │   ├── layout/             # 布局组件
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── Footer.tsx
│   │   └── charts/             # 图表组件
│   │       ├── EquityCurve.tsx
│   │       ├── PerformanceChart.tsx
│   │       └── DecisionChart.tsx
│   │
│   ├── pages/                  # 页面组件
│   │   ├── Dashboard.tsx       # 仪表板
│   │   ├── Traders.tsx         # 交易员管理
│   │   ├── TraderDetail.tsx    # 交易员详情
│   │   ├── Decisions.tsx       # 决策历史
│   │   ├── Positions.tsx       # 持仓管理
│   │   ├── Analytics.tsx       # 数据分析
│   │   ├── Settings.tsx        # 设置
│   │   └── Login.tsx           # 登录
│   │
│   ├── hooks/                  # 自定义 Hooks
│   │   ├── useAuth.ts          # 认证 Hook
│   │   ├── useTraders.ts       # 交易员 Hook
│   │   ├── useWebSocket.ts     # WebSocket Hook
│   │   └── useLocalStorage.ts  # 本地存储 Hook
│   │
│   ├── stores/                 # 状态管理 (Zustand)
│   │   ├── authStore.ts        # 认证状态
│   │   ├── traderStore.ts      # 交易员状态
│   │   ├── marketStore.ts      # 市场数据状态
│   │   └── uiStore.ts          # UI 状态
│   │
│   ├── lib/                    # 工具库
│   │   ├── api.ts              # API 客户端
│   │   ├── ws.ts               # WebSocket 客户端
│   │   ├── formatters.ts       # 格式化函数
│   │   └── validators.ts       # 验证函数
│   │
│   ├── types/                  # TypeScript 类型
│   │   ├── api.ts              # API 类型
│   │   ├── models.ts           # 数据模型
│   │   └── index.ts
│   │
│   └── utils/                  # 工具函数
│       ├── date.ts             # 日期工具
│       ├── number.ts           # 数字工具
│       └── string.ts           # 字符串工具
│
├── public/                     # 静态资源
│   ├── favicon.ico
│   └── logo.svg
│
├── index.html                  # HTML 模板
├── package.json                # 项目配置
├── tsconfig.json               # TypeScript 配置
├── vite.config.ts              # Vite 配置
└── tailwind.config.js          # Tailwind CSS 配置
```

### 10.2 API 客户端

```typescript
// frontend/src/lib/api.ts
import axios, { AxiosError, AxiosInstance, InternalAxiosRequestConfig } from 'axios';

// API 响应类型
export interface ApiResponse<T = any> {
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
}

// API 客户端类
class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // 请求拦截器：添加认证 token
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        const token = localStorage.getItem('access_token');
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // 响应拦截器：处理错误
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError<ApiResponse>) => {
        if (error.response?.status === 401) {
          // Token 过期，跳转到登录页
          localStorage.removeItem('access_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // GET 请求
  async get<T>(url: string, params?: any): Promise<T> {
    const response = await this.client.get<ApiResponse<T>>(url, { params });
    return this.handleResponse(response);
  }

  // POST 请求
  async post<T>(url: string, data?: any): Promise<T> {
    const response = await this.client.post<ApiResponse<T>>(url, data);
    return this.handleResponse(response);
  }

  // PUT 请求
  async put<T>(url: string, data?: any): Promise<T> {
    const response = await this.client.put<ApiResponse<T>>(url, data);
    return this.handleResponse(response);
  }

  // DELETE 请求
  async delete<T>(url: string): Promise<T> {
    const response = await this.client.delete<ApiResponse<T>>(url);
    return this.handleResponse(response);
  }

  // 处理响应
  private handleResponse<T>(response: any): T {
    if (response.data?.error) {
      throw new Error(response.data.error.message);
    }
    return response.data.data ?? response.data;
  }
}

// 导出单例
export const apiClient = new ApiClient();

// API 方法
export const api = {
  // 认证
  auth: {
    login: (username: string, password: string) =>
      apiClient.post('/api/auth/login', { username, password }),
    logout: () =>
      apiClient.post('/api/auth/logout'),
    me: () =>
      apiClient.get('/api/auth/me'),
  },

  // 交易员
  traders: {
    list: (params?: any) =>
      apiClient.get('/api/traders', params),
    create: (data: any) =>
      apiClient.post('/api/traders', data),
    get: (id: string) =>
      apiClient.get(`/api/traders/${id}`),
    update: (id: string, data: any) =>
      apiClient.put(`/api/traders/${id}`, data),
    delete: (id: string) =>
      apiClient.delete(`/api/traders/${id}`),
    control: (id: string, action: string) =>
      apiClient.post(`/api/traders/${id}/control`, { action }),
  },

  // 决策
  decisions: {
    list: (traderId: string, params?: any) =>
      apiClient.get(`/api/decisions?trader_id=${traderId}`, params),
    get: (id: string) =>
      apiClient.get(`/api/decisions/${id}`),
  },

  // 持仓
  positions: {
    list: (traderId: string) =>
      apiClient.get(`/api/positions?trader_id=${traderId}`),
  },

  // 市场数据
  markets: {
    realtime: (symbol: string) =>
      apiClient.get(`/api/markets/realtime/${symbol}`),
    klines: (symbol: string, interval: string, limit: number) =>
      apiClient.get(`/api/markets/klines/${symbol}`, { interval, limit }),
  },
};
```

---

## 十一、开发详细路线图

### 11.1 详细的任务分解

#### 第一阶段：基础框架（第1-3周）

| 周 | 任务 | 子任务 | 交付物 | 验收标准 |
|----|------|--------|--------|----------|
| **第1周** | 项目初始化 | 1. 创建项目目录结构<br>2. 配置 pyproject.toml<br>3. 配置开发环境 | 项目骨架 | 可运行 `pytest` |
| | | 4. 设置 pre-commit hooks<br>5. 配置 CI/CD | | 代码风格检查通过 |
| **第2周** | 数据库设计 | 1. 编写 SQLAlchemy 模型<br>2. 创建 Alembic 迁移<br>3. 编写 Repository 基类 | 数据库层 | 迁移脚本可执行 |
| | | 4. 编写单元测试<br>5. 编写数据初始化脚本 | | 测试覆盖率 > 80% |
| **第3周** | 配置和日志 | 1. 实现 Pydantic Settings<br>2. 配置 structlog<br>3. 实现异常处理体系 | 配置系统 | 配置热加载 |
| | | 4. 编写文档<br>5. 代码审查 | | 日志结构化输出 |

#### 第二阶段：市场数据层（第4-6周）

| 周 | 任务 | 子任务 | 交付物 | 验收标准 |
|----|------|--------|--------|----------|
| **第4周** | 数据接口抽象 | 1. 定义数据类型<br>2. 实现 BaseMarketDataProvider<br>3. 编写技术指标计算器 | 接口层 | 类型检查通过 |
| | | 4. 实现交易日历<br>5. 编写单元测试 | | 指标计算正确 |
| **第5周** | A股数据源 | 1. 集成 Tushare<br>2. 集成 AKShare<br>3. 实现数据缓存 | A股数据 | 可获取实时行情 |
| | | 4. 实现数据验证<br>5. 性能优化 | | 延迟 < 1秒 |
| **第6周** | 港股数据源 | 1. 集成 Yahoo Finance<br>2. 实现汇率转换<br>3. WebSocket 推送 | 港股数据 | 可获取实时行情 |
| | | 4. 多源数据融合<br>5. 故障转移 | | 故障自动切换 |

#### 第三阶段：交易执行层（第7-9周）

| 周 | 任务 | 子任务 | 交付物 | 验收标准 |
|----|------|--------|--------|----------|
| **第7周** | 交易接口 | 1. 定义交易接口<br>2. 实现费用计算<br>3. 实现 BaseTrader | 交易框架 | 接口设计完成 |
| | | 4. 编订单管理<br>5. 编写单元测试 | | 订单状态正确 |
| **第8周** | A股模拟器 | 1. 实现 ATraderSimulator<br>2. 实现 T+1 限制<br>3. 实现涨跌停检查 | A股交易 | 完整交易流程 |
| | | 4. 实现风险控制<br>5. 性能测试 | | 延迟 < 100ms |
| **第9周** | 港股模拟器 | 1. 实现 HKTraderSimulator<br>2. 实现交易单位处理<br>3. 费用计算 | 港股交易 | 完整交易流程 |
| | | 4. 汇率转换<br>5. 集成测试 | | 跨币种计算正确 |

#### 第四阶段：AI 决策引擎（第10-13周）

| 周 | 任务 | 子任务 | 交付物 | 验收标准 |
|----|------|--------|--------|----------|
| **第10周** | LLM 集成 | 1. 实现 LLM 客户端<br>2. 实现限流<br>3. 实现重试机制 | LLM 基础 | 可调用 DeepSeek |
| | | 4. 实现流式响应<br>5. Token 计数 | | 限流生效 |
| **第11周** | Prompt 管理 | 1. 实现 PromptManager<br>2. 支持模板变量<br>3. 支持多模板 | Prompt 系统 | 模板渲染正确 |
| | | 4. 模板版本管理<br>5. A/B 测试支持 | | 可切换模板 |
| **第12周** | 决策流程 | 1. 实现 ContextBuilder<br>2. 实现 DecisionParser<br>3. 实现 DecisionValidator | 决策引擎 | 可输出决策 |
| | | 4. 实现历史反馈<br>5. 集成测试 | | 验证规则生效 |
| **第13周** | 自学习 | 1. 实现性能分析<br>2. 实现策略评估<br>3. 实现参数优化 | 自学习 | 性能统计正确 |
| | | 4. 历史回测<br>5. 报告生成 | | 可生成报告 |

#### 第五阶段：API 和前端（第14-16周）

| 周 | 任务 | 子任务 | 交付物 | 验收标准 |
|----|------|--------|--------|----------|
| **第14周** | REST API | 1. 实现认证 API<br>2. 实现交易员 API<br>3. 实现决策 API | API 层 | API 文档完整 |
| | | 4. 实现持仓 API<br>5. 集成测试 | | 所有测试通过 |
| **第15周** | WebSocket | 1. 实现 WebSocket 服务<br>2. 实现实时推送<br>3. 实现连接管理 | 实时通信 | 消息推送及时 |
| | | 4. 心跳检测<br>5. 重连机制 | | 断线自动重连 |
| **第16周** | 前端开发 | 1. 实现登录页面<br>2. 实现仪表板<br>3. 实现交易员管理 | 前端界面 | 核心功能可用 |
| | | 4. 实现决策历史<br>5. 图表可视化 | | 数据展示正确 |

#### 第六阶段：测试和优化（第17-19周）

| 周 | 任务 | 子任务 | 交付物 | 验收标准 |
|----|------|--------|--------|----------|
| **第17周** | 单元测试 | 1. 补充单元测试<br>2. 提高覆盖率<br>3. 添加边界测试 | 测试套件 | 覆盖率 > 85% |
| | | 4. 性能测试<br>5. 压力测试 | | 无内存泄漏 |
| **第18周** | 集成测试 | 1. 端到端测试<br>2. API 测试<br>3. 数据流测试 | 测试报告 | 所有用例通过 |
| | | 4. 用户场景测试<br>5. 兼容性测试 | | 跨浏览器兼容 |
| **第19周** | 优化部署 | 1. 性能优化<br>2. 安全加固<br>3. 文档完善 | 生产版本 | 满足性能指标 |
| | | 4. 部署脚本<br>5. 监控配置 | | 可一键部署 |

---

## 十二、测试策略

### 12.1 测试金字塔

```
                    /\
                   /  \
                  / E2E\
                 /______\
                /        \
               /Integration\
              /____________\
             /              \
            /  Unit Tests    \
           /__________________\
```

### 12.2 单元测试示例

```python
# tests/unit/test_fee_calculator.py
import pytest
from decimal import Decimal

from backend.core.fee_calculator import AStockFeeCalculator, Side


class TestAStockFeeCalculator:
    """A 股费用计算器测试"""

    @pytest.fixture
    def calculator(self):
        return AStockFeeCalculator()

    def test_buy_commission(self, calculator):
        """测试买入佣金"""
        fee = calculator.calculate(price=10.0, quantity=1000, side=Side.BUY)

        # 佣金: 10000 * 0.0003 = 3, 但最低5元
        assert fee.commission == Decimal("5.00")
        assert fee.stamp_duty == Decimal("0")
        assert fee.transfer_fee == Decimal("0.10")

    def test_sell_commission_with_stamp_duty(self, calculator):
        """测试卖出佣金和印花税"""
        fee = calculator.calculate(price=10.0, quantity=1000, side=Side.SELL)

        # 佣金: 10000 * 0.0003 = 3, 但最低5元
        assert fee.commission == Decimal("5.00")
        # 印花税: 10000 * 0.001 = 10
        assert fee.stamp_duty == Decimal("10.00")
        assert fee.transfer_fee == Decimal("0.10")

    def test_large_amount_commission(self, calculator):
        """测试大额交易佣金"""
        fee = calculator.calculate(price=100.0, quantity=10000, side=Side.BUY)

        # 佣金: 1000000 * 0.0003 = 300
        assert fee.commission == Decimal("300.00")

    def test_total_fee_calculation(self, calculator):
        """测试总费用计算"""
        fee = calculator.calculate(price=50.0, quantity=5000, side=Side.SELL)

        expected_total = (
            Decimal("75.00") +  # 佣金 (250000 * 0.0003)
            Decimal("250.00") +  # 印花税 (250000 * 0.001)
            Decimal("2.50") +    # 过户费 (250000 * 0.00001)
            Decimal("17.17")     # 交易规费 (250000 * 0.0000687)
        )

        assert fee.total == expected_total
```

### 12.3 集成测试示例

```python
# tests/integration/test_trading_flow.py
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestTradingFlow:
    """交易流程集成测试"""

    async def test_complete_trading_flow(self, async_client: AsyncClient, auth_headers):
        """测试完整交易流程"""
        # 1. 创建交易员
        response = await async_client.post(
            "/api/traders",
            json={
                "name": "测试交易员",
                "ai_model_id": "test-model-id",
                "market_type": "a_stock",
                "initial_balance": 1000000,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201
        trader = response.json()
        trader_id = trader["id"]

        # 2. 启动交易员
        response = await async_client.post(
            f"/api/traders/{trader_id}/control",
            json={"action": "start"},
            headers=auth_headers,
        )
        assert response.status_code == 200

        # 3. 获取交易员状态
        response = await async_client.get(
            f"/api/traders/{trader_id}",
            headers=auth_headers,
        )
        assert response.status_code == 200
        trader_data = response.json()
        assert trader_data["is_running"] is True

        # 4. 等待决策
        await asyncio.sleep(5)

        # 5. 获取决策记录
        response = await async_client.get(
            "/api/decisions",
            params={"trader_id": trader_id},
            headers=auth_headers,
        )
        assert response.status_code == 200
        decisions = response.json()
        assert len(decisions) > 0

        # 6. 停止交易员
        response = await async_client.post(
            f"/api/traders/{trader_id}/control",
            json={"action": "stop"},
            headers=auth_headers,
        )
        assert response.status_code == 200
```

---

## 十三、部署与运维

### 13.1 生产环境部署清单

```yaml
# deployment/production/deployment-checklist.md

# 部署前检查清单

## 1. 代码检查
- [ ] 所有测试通过
- [ ] 代码覆盖率 > 80%
- [ ] 无安全漏洞
- [ ] 性能测试通过
- [ ] 代码审查完成

## 2. 配置检查
- [ ] 环境变量正确设置
- [ ] 数据库连接字符串配置
- [ ] Redis 连接配置
- [ ] API Keys 配置
- [ ] CORS 域名配置
- [ ] JWT 密钥配置

## 3. 数据库检查
- [ ] 数据库迁移执行
- [ ] 索引创建完成
- [ ] 备份策略配置
- [ ] 连接池配置优化

## 4. 依赖检查
- [ ] Python 版本正确 (3.11+)
- [ ] 所有依赖安装
- [ ] 虚拟环境配置
- [ ] 系统依赖安装

## 5. 监控检查
- [ ] 日志配置
- [ ] 指标收集配置
- [ ] 告警规则配置
- [ ] 健康检查配置

## 6. 安全检查
- [ ] HTTPS 配置
- [ ] 防火墙规则
- [ ] 访问控制配置
- [ ] 敏感数据加密
```

---

**文档版本**: v2.0
**创建日期**: 2026-01-05
**作者**: DeepAlpha Team

> 本文档为《NOFX_Python_重构技术方案_A股港股_v2.md》的补充部分，包含第六章至第十六章的详细内容。
