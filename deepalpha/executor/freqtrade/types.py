"""Freqtrade类型定义，对应 Go 版本的 types.go"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TierInfo(BaseModel):
    """层级信息，对应 Go 版本的 TierInfo"""
    target: float
    ratio: float
    done: bool


class APIPosition(BaseModel):
    """API持仓信息，对应 Go 版本的 APIPosition"""
    trade_id: int
    symbol: str
    side: str
    entry_price: float
    amount: float
    initial_amount: Optional[float] = None
    stake: float
    leverage: float
    position_value: Optional[float] = None
    opened_at: int
    holding_ms: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    pnl_ratio: Optional[float] = None
    pnl_usd: Optional[float] = None
    realized_pnl_ratio: Optional[float] = None
    realized_pnl_usd: Optional[float] = None
    unrealized_pnl_ratio: Optional[float] = None
    unrealized_pnl_usd: Optional[float] = None
    remaining_ratio: Optional[float] = None
    tier1: TierInfo
    tier2: TierInfo
    tier3: TierInfo
    tier_notes: Optional[str] = None
    placeholder: bool = False
    tier_logs: List[Dict[str, Any]] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    status: str
    closed_at: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class ForceEnterPayload(BaseModel):
    """强制开仓载荷，对应 Go 版本的 ForceEnterPayload"""
    pair: str
    side: str
    price: Optional[float] = None
    ordertype: Optional[str] = None
    stakeamount: Optional[float] = None
    entry_tag: Optional[str] = None
    leverage: Optional[float] = None


class ForceEnterResponse(BaseModel):
    """强制开仓响应，对应 Go 版本的 ForceEnterResponse"""
    trade_id: int


class ForceExitPayload(BaseModel):
    """强制平仓载荷，对应 Go 版本的 ForceExitPayload"""
    tradeid: str
    ordertype: Optional[str] = None
    amount: Optional[float] = None


class TierUpdateRequest(BaseModel):
    """层级更新请求，对应 Go 版本的 TierUpdateRequest"""
    trade_id: int
    symbol: str
    side: str
    stop_loss: float
    take_profit: float
    tier1_target: Optional[float] = None
    tier1_ratio: Optional[float] = None
    tier2_target: Optional[float] = None
    tier2_ratio: Optional[float] = None
    tier3_target: Optional[float] = None
    tier3_ratio: Optional[float] = None
    reason: Optional[str] = None


class ManualOpenRequest(BaseModel):
    """手动开仓请求，对应 Go 版本的 ManualOpenRequest"""
    symbol: str
    side: str
    position_size_usd: float
    leverage: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    tier1_target: Optional[float] = None
    tier1_ratio: Optional[float] = None
    tier2_target: Optional[float] = None
    tier2_ratio: Optional[float] = None
    tier3_target: Optional[float] = None
    tier3_ratio: Optional[float] = None
    reason: Optional[str] = None


class PositionListOptions(BaseModel):
    """持仓列表选项，对应 Go 版本的 PositionListOptions"""
    symbol: Optional[str] = None
    page: int = 1
    page_size: int = 50
    include_logs: bool = False
    logs_limit: int = 10


class PositionListResult(BaseModel):
    """持仓列表结果，对应 Go 版本的 PositionListResult"""
    total_count: int
    page: int
    page_size: int
    positions: List[APIPosition]


class Balance(BaseModel):
    """余额信息，对应 Go 版本的 Balance"""
    stake_currency: Optional[str] = None
    total: Optional[float] = None
    available: Optional[float] = None
    used: Optional[float] = None
    balance: Optional[float] = None
    stake_balance: Optional[float] = None
    wallets: Dict[str, float] = Field(default_factory=dict)
    raw: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class Trade(BaseModel):
    """交易信息，对应 Go 版本的 Trade"""
    id: int = Field(alias="trade_id")
    pair: str
    side: str
    is_short: bool
    open_date: str
    close_date: Optional[str] = None
    open_rate: float
    close_rate: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    amount: float
    stake_amount: float
    leverage: float
    open_order_id: Optional[str] = None
    close_order_id: Optional[str] = None
    is_open: bool
    current_rate: Optional[float] = None
    close_profit: Optional[float] = None
    close_profit_abs: Optional[float] = None
    profit_ratio: Optional[float] = None
    profit_abs: Optional[float] = None

    class Config:
        allow_population_by_field_name = True


# 辅助函数，对应 Go 版本的辅助函数

def position_pnl_value(stake: float, leverage: float, position_value: float) -> float:
    """计算PnL价值，对应 Go 版本的 PositionPnLValue"""
    if position_value > 0:
        return position_value
    if stake <= 0:
        return 0
    if leverage <= 0:
        leverage = 1
    return stake * leverage


def remaining_position_value(
    stake: float,
    leverage: float,
    position_value: float,
    amount: float,
    initial_amount: float
) -> float:
    """计算剩余仓位价值，对应 Go 版本的 RemainingPositionValue"""
    base = position_pnl_value(stake, leverage, position_value)
    if base <= 0:
        return 0
    if initial_amount <= 0:
        return base
    if amount <= 0:
        return 0
    frac = amount / initial_amount
    if frac <= 0:
        return 0
    if frac > 1:
        frac = 1
    return base * frac


def format_qty(val: float) -> str:
    """格式化数量，对应 Go 版本的 formatQty"""
    if val == 0:
        return "-"
    return f"{val:.4f}"


def format_price(val: float) -> str:
    """格式化价格，对应 Go 版本的 formatPrice"""
    if val == 0:
        return "-"
    return f"{val:.4f}"


def short_reason(desc: str) -> str:
    """缩短原因，对应 Go 版本的 shortReason"""
    desc = desc.strip()
    if not desc:
        return ""
    max_len = 200
    if len(desc) <= max_len:
        return desc
    return desc[:max_len] + "..."


def ptr_float(val: float) -> float:
    """浮点数指针，对应 Go 版本的 ptrFloat"""
    return val


def ptr_bool(val: bool) -> bool:
    """布尔值指针，对应 Go 版本的 ptrBool"""
    return val


def val_or_zero(val: Optional[float]) -> float:
    """值或零，对应 Go 版本的 valOrZero"""
    if val is None:
        return 0
    return val


def time_to_millis(t: Optional[datetime]) -> int:
    """时间转毫秒，对应 Go 版本的 timeToMillis"""
    if not t or t == datetime.min:
        return 0
    return int(t.timestamp() * 1000)


def millis_since(t: Optional[datetime]) -> int:
    """毫秒差，对应 Go 版本的 millisSince"""
    if not t or t == datetime.min:
        return 0
    return int((datetime.utcnow() - t).total_seconds() * 1000)