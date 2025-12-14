"""决策类型定义，对应 Go 版本的 types.go"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class DecisionTiers(BaseModel):
    """决策层级，对应 Go 版本的 DecisionTiers"""
    tier1_target: Optional[float] = Field(None, description="第一目标价格")
    tier1_ratio: Optional[float] = Field(None, description="第一目标比例")
    tier2_target: Optional[float] = Field(None, description="第二目标价格")
    tier2_ratio: Optional[float] = Field(None, description="第二目标比例")
    tier3_target: Optional[float] = Field(None, description="第三目标价格")
    tier3_ratio: Optional[float] = Field(None, description="第三目标比例")

    @validator('tier1_ratio', 'tier2_ratio', 'tier3_ratio')
    def normalize_ratio(cls, v):
        """标准化比例（处理百分比输入）"""
        if v and v > 1:
            return v / 100
        return v


class Decision(BaseModel):
    """单笔AI决策，对应 Go 版本的 Decision"""
    symbol: str = Field(..., description="交易对符号")
    action: str = Field(..., description="动作类型")
    leverage: Optional[int] = Field(None, description="杠杆倍数")
    position_size_usd: Optional[float] = Field(None, description="仓位大小(美元)")
    close_ratio: Optional[float] = Field(None, description="平仓比例")
    stop_loss: Optional[float] = Field(None, description="止损价格")
    take_profit: Optional[float] = Field(None, description="止盈价格")
    confidence: Optional[int] = Field(None, ge=0, le=100, description="置信度(0-100)")
    reasoning: Optional[str] = Field(None, description="决策理由")
    tiers: Optional[DecisionTiers] = Field(None, description="目标层级")
    update_tiers: Optional[DecisionTiers] = Field(None, description="更新层级(兼容字段)")

    @validator('action')
    def normalize_action(cls, v):
        """标准化动作名称"""
        action_map = {
            'long': 'enter_long',
            'short': 'enter_short',
            'buy': 'enter_long',
            'sell': 'enter_short',
            'close': 'close',
            'close_long': 'close_long',
            'close_short': 'close_short',
            'hold': 'hold',
            'neutral': 'hold',
        }
        return action_map.get(v.lower(), v.lower())

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def __init__(self, **data):
        # 处理 update_tiers 兼容性
        if 'update_tiers' in data and data['update_tiers'] and 'tiers' not in data:
            data['tiers'] = data.pop('update_tiers')
        super().__init__(**data)


class DecisionResult(BaseModel):
    """AI决策输出结果，对应 Go 版本的 DecisionResult"""
    decisions: List[Decision] = Field(default_factory=list)
    raw_output: Optional[str] = Field(None, description="原始模型输出")
    raw_json: Optional[str] = Field(None, description="提取的JSON决策")
    symbol_results: List[Dict[str, Any]] = Field(default_factory=list)
    meta_summary: Optional[str] = Field(None, description="元聚合摘要")
    trace_id: Optional[str] = Field(None, description="追踪ID")


class SymbolDecisionOutput(BaseModel):
    """符号决策输出，对应 Go 版本的 SymbolDecisionOutput"""
    symbol: str
    raw_output: Optional[str] = None
    raw_json: Optional[str] = None
    meta_summary: Optional[str] = None
    trace_id: Optional[str] = None


class DecisionMemory(BaseModel):
    """决策记忆，对应 Go 版本的 DecisionMemory"""
    symbol: str
    horizon: str
    decided_at: datetime
    decisions: List[Decision]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class LastDecisionRecord(BaseModel):
    """最后决策记录，对应 Go 版本的 LastDecisionRecord"""
    symbol: str
    horizon: str
    decided_at: datetime
    decisions: List[Decision]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class PositionSnapshot(BaseModel):
    """持仓快照，对应 Go 版本的 PositionSnapshot"""
    symbol: str
    side: str
    entry_price: float
    quantity: float
    stake: Optional[float] = None
    leverage: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pn: Optional[float] = None
    unrealized_pn_pct: Optional[float] = None
    position_value: Optional[float] = None
    account_ratio: Optional[float] = None
    rr: Optional[float] = None
    holding_ms: Optional[int] = None
    remaining_ratio: Optional[float] = None
    tier1_target: Optional[float] = None
    tier1_ratio: Optional[float] = None
    tier1_done: Optional[bool] = None
    tier2_target: Optional[float] = None
    tier2_ratio: Optional[float] = None
    tier2_done: Optional[bool] = None
    tier3_target: Optional[float] = None
    tier3_ratio: Optional[float] = None
    tier3_done: Optional[bool] = None
    tier_notes: Optional[str] = None


class AccountSnapshot(BaseModel):
    """账户快照，对应 Go 版本的 AccountSnapshot"""
    total: float
    available: float
    used: float
    currency: str = "USDT"
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class MarketData(BaseModel):
    """市场数据占位符，对应 Go 版本的 MarketData"""
    pass


class PromptBundle(BaseModel):
    """提示词包，对应 Go 版本的 PromptBundle"""
    system: str
    user: str


class DecisionContext(BaseModel):
    """决策上下文，对应 Go 版本的 Context"""
    candidates: List[str] = Field(default_factory=list)
    market: Dict[str, MarketData] = Field(default_factory=dict)
    positions: List[PositionSnapshot] = Field(default_factory=list)
    account: Optional[AccountSnapshot] = None
    prompt: Optional[PromptBundle] = None
    analysis: List[Dict[str, Any]] = Field(default_factory=list)
    last_decisions: List[DecisionMemory] = Field(default_factory=list)
    last_raw_json: Optional[str] = None


class ModelOutput(BaseModel):
    """模型输出，对应 Go 版本的 ModelOutput"""
    provider_id: str
    raw: Optional[str] = None
    parsed: Optional[DecisionResult] = None
    error: Optional[str] = None
    images: List[Dict[str, Any]] = Field(default_factory=list)
    vision_enabled: bool = False
    image_count: int = 0


# 辅助函数

def normalize_action(action: str) -> str:
    """标准化动作，对应 Go 版本的 NormalizeAction"""
    action_map = {
        'long': 'enter_long',
        'short': 'enter_short',
        'buy': 'enter_long',
        'sell': 'enter_short',
        'close': 'close',
        'close_long': 'close_long',
        'close_short': 'close_short',
        'hold': 'hold',
        'neutral': 'hold',
        'enter': 'enter_long',
        'exit': 'close',
    }
    return action_map.get(action.lower(), action.lower())


def align_close_actions(decisions: List[Decision], positions: List[PositionSnapshot]) -> List[Decision]:
    """对齐平仓动作，对应 Go 版本的 AlignCloseActions"""
    if not decisions or not positions:
        return decisions

    # 构建持仓侧映射
    side_map = {}
    for pos in positions:
        sym = pos.symbol.upper().strip()
        if sym:
            side = pos.side.lower().strip()
            if side and sym not in side_map:
                side_map[sym] = side

    if not side_map:
        return decisions

    # 调整平仓动作
    aligned_decisions = []
    for decision in decisions:
        decision_copy = decision.copy(deep=True)
        sym = decision.symbol.upper().strip()
        side = side_map.get(sym)

        if side:
            if decision.action == "close_long" and side == "short":
                decision_copy.action = "close_short"
            elif decision.action == "close_short" and side == "long":
                decision_copy.action = "close_long"

        aligned_decisions.append(decision_copy)

    return aligned_decisions


def normalize_and_align_decisions(decisions: List[Decision], positions: List[PositionSnapshot]) -> List[Decision]:
    """标准化并对齐决策，对应 Go 版本的 NormalizeAndAlignDecisions"""
    if not decisions:
        return decisions

    # 标准化动作
    normalized = []
    for decision in decisions:
        decision_copy = decision.copy(deep=True)
        decision_copy.action = normalize_action(decision.action)
        normalized.append(decision_copy)

    # 对齐平仓动作
    return align_close_actions(normalized, positions)