"""核心数据模型"""

import asyncio
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class SignalType(str, Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class DecisionType(str, Enum):
    """决策类型"""
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"


class OrderType(str, Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Kline(BaseModel):
    """K线数据模型"""
    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal
    trades: Optional[int] = None
    taker_buy_volume: Optional[Decimal] = None
    taker_buy_quote_volume: Optional[Decimal] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
        }

    @property
    def is_green(self) -> bool:
        """是否是阳线"""
        return self.close > self.open

    @property
    def body_size(self) -> Decimal:
        """实体大小"""
        return abs(self.close - self.open)

    @property
    def upper_shadow(self) -> Decimal:
        """上影线"""
        if self.is_green:
            return self.high - self.close
        return self.high - self.open

    @property
    def lower_shadow(self) -> Decimal:
        """下影线"""
        if self.is_green:
            return self.open - self.low
        return self.close - self.low

    @property
    def range_percent(self) -> float:
        """振幅百分比"""
        return float((self.high - self.low) / self.open * 100)


class Ticker(BaseModel):
    """行情数据模型"""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    high: Decimal
    low: Decimal
    volume: Decimal
    quote_volume: Decimal
    open: Decimal
    timestamp: datetime
    change: Optional[Decimal] = None
    change_percent: Optional[float] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
        }


class Balance(BaseModel):
    """账户余额模型"""
    asset: str
    free: Decimal
    used: Decimal
    total: Decimal

    @property
    def available_percent(self) -> float:
        """可用余额百分比"""
        if self.total == 0:
            return 0
        return float(self.free / self.total * 100)


class Signal(BaseModel):
    """交易信号模型"""
    symbol: str
    signal_type: SignalType
    confidence: float = Field(ge=0, le=1)
    reason: str
    agent_name: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    price: Optional[Decimal] = None
    timeframe: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("confidence")
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class Decision(BaseModel):
    """交易决策模型"""
    id: Optional[str] = None
    symbol: str
    decision_type: DecisionType
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    signals: List[Signal] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    executed: bool = False
    execution_time: Optional[datetime] = None

    # 风险管理参数
    suggested_size: Optional[float] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

    @validator("confidence")
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v

    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def is_valid(self) -> bool:
        """决策是否有效"""
        return not self.executed and not self.is_expired


class Order(BaseModel):
    """订单模型"""
    id: Optional[str] = None
    client_order_id: Optional[str] = None
    symbol: str
    type: OrderType
    side: OrderSide
    amount: Decimal = Field(gt=0)
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    filled: Decimal = Field(default=0)
    remaining: Optional[Decimal] = None
    average_price: Optional[Decimal] = None
    fee: Optional[Decimal] = None
    fee_currency: Optional[str] = None
    trades: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("remaining", always=True)
    def calculate_remaining(cls, v, values):
        if v is None:
            amount = values.get("amount", 0)
            filled = values.get("filled", 0)
            return amount - filled
        return v

    @property
    def is_active(self) -> bool:
        """订单是否活跃"""
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]

    @property
    def is_filled(self) -> bool:
        """是否完全成交"""
        return self.status == OrderStatus.FILLED

    @property
    def fill_rate(self) -> float:
        """成交率"""
        if self.amount == 0:
            return 0
        return float(self.filled / self.amount)


class Position(BaseModel):
    """持仓模型"""
    symbol: str
    side: OrderSide
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Field(default=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    orders: List[Order] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def pnl_percent(self) -> float:
        """盈亏百分比"""
        if self.size == 0 or self.entry_price == 0:
            return 0
        if self.side == OrderSide.BUY:
            return float((self.current_price - self.entry_price) / self.entry_price * 100)
        else:
            return float((self.entry_price - self.current_price) / self.entry_price * 100)

    @property
    def is_profit(self) -> bool:
        """是否盈利"""
        return self.unrealized_pnl > 0

    @property
    def margin_requirement(self) -> Decimal:
        """保证金需求"""
        return self.size * self.current_price


class AgentResult(BaseModel):
    """智能体分析结果"""
    agent_name: str
    symbol: str
    signal: Optional[Signal] = None
    analysis: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time: Optional[float] = None  # 处理时间（秒）


class DecisionContext(BaseModel):
    """决策上下文"""
    symbol: str
    klines: Dict[str, List[Kline]] = Field(default_factory=dict)
    indicators: Dict[str, Any] = Field(default_factory=dict)
    patterns: List[str] = Field(default_factory=list)
    current_position: Optional[Position] = None
    account_balance: Dict[str, Balance] = Field(default_factory=dict)
    current_price: Optional[Decimal] = None
    market_data: Dict[str, Any] = Field(default_factory=dict)


class RiskMetrics(BaseModel):
    """风险指标"""
    total_exposure: Decimal = Field(default=0)
    free_margin: Decimal = Field(default=0)
    margin_level: Optional[float] = None
    daily_pnl: Decimal = Field(default=0)
    max_drawdown: float = Field(default=0)
    sharpe_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    total_trades: int = Field(default=0)
    winning_trades: int = Field(default=0)

    @property
    def losing_trades(self) -> int:
        """亏损交易数"""
        return self.total_trades - self.winning_trades

    @property
    def avg_win_rate(self) -> float:
        """平均胜率"""
        if self.total_trades == 0:
            return 0
        return self.winning_trades / self.total_trades * 100


class MarketMetrics(BaseModel):
    """市场指标"""
    volatility: float
    volume: Decimal
    momentum: float
    trend_strength: float
    support_levels: List[Decimal] = Field(default_factory=list)
    resistance_levels: List[Decimal] = Field(default_factory=list)
    sentiment: Optional[float] = None  # -1 to 1