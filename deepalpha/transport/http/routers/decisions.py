"""决策相关的API路由，对应 Go 版本的 decision handlers"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from deepalpha.decision.types import Decision, DecisionStatus
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# 响应模型
class DecisionResponse(BaseModel):
    """决策响应模型"""
    id: str
    symbol: str
    action: str
    confidence: int
    status: str
    timestamp: datetime
    position_size_usd: Optional[float] = None
    leverage: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: Optional[str] = None
    agent_name: Optional[str] = None
    metadata: Optional[dict] = None


class DecisionListResponse(BaseModel):
    """决策列表响应"""
    decisions: List[DecisionResponse]
    total: int
    page: int
    page_size: int


# 请求模型
class DecisionQuery(BaseModel):
    """决策查询参数"""
    symbol: Optional[str] = None
    action: Optional[str] = None
    status: Optional[str] = None
    agent_name: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


# 模拟数据存储（实际应该从数据库获取）
_decisions_db: List[Decision] = []


@router.get("/decisions", response_model=DecisionListResponse)
async def list_decisions(
    symbol: Optional[str] = Query(None, description="交易对"),
    action: Optional[str] = Query(None, description="操作类型"),
    status: Optional[str] = Query(None, description="状态"),
    agent_name: Optional[str] = Query(None, description="代理名称"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
):
    """
    获取决策列表
    对应 Go 版本的 GET /api/live/decisions
    """
    logger.info(
        "获取决策列表",
        symbol=symbol,
        action=action,
        status=status,
        page=page,
        page_size=page_size
    )

    # 筛选决策
    filtered_decisions = _decisions_db

    if symbol:
        filtered_decisions = [d for d in filtered_decisions if d.symbol == symbol]
    if action:
        filtered_decisions = [d for d in filtered_decisions if d.action == action]
    if status:
        filtered_decisions = [d for d in filtered_decisions if d.status.value == status]
    if agent_name:
        filtered_decisions = [d for d in filtered_decisions if d.agent_name == agent_name]

    # 分页
    total = len(filtered_decisions)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_decisions = filtered_decisions[start_idx:end_idx]

    # 转换为响应格式
    decision_responses = [
        DecisionResponse(
            id=decision.id,
            symbol=decision.symbol,
            action=decision.action,
            confidence=decision.confidence,
            status=decision.status.value,
            timestamp=decision.timestamp,
            position_size_usd=decision.position_size_usd,
            leverage=decision.leverage,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            reason=decision.reason,
            agent_name=decision.agent_name,
            metadata=decision.metadata,
        )
        for decision in page_decisions
    ]

    return DecisionListResponse(
        decisions=decision_responses,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/decisions/{decision_id}", response_model=DecisionResponse)
async def get_decision(decision_id: str):
    """
    获取决策详情
    对应 Go 版本的 GET /api/live/decisions/:id
    """
    logger.info("获取决策详情", decision_id=decision_id)

    # 查找决策
    decision = next((d for d in _decisions_db if d.id == decision_id), None)
    if not decision:
        raise HTTPException(status_code=404, detail="决策未找到")

    return DecisionResponse(
        id=decision.id,
        symbol=decision.symbol,
        action=decision.action,
        confidence=decision.confidence,
        status=decision.status.value,
        timestamp=decision.timestamp,
        position_size_usd=decision.position_size_usd,
        leverage=decision.leverage,
        stop_loss=decision.stop_loss,
        take_profit=decision.take_profit,
        reason=decision.reason,
        agent_name=decision.agent_name,
        metadata=decision.metadata,
    )


@router.get("/traces", response_model=DecisionListResponse)
async def list_traces(
    # 使用相同的参数
    symbol: Optional[str] = Query(None, description="交易对"),
    action: Optional[str] = Query(None, description="操作类型"),
    status: Optional[str] = Query(None, description="状态"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
):
    """
    获取决策追踪列表（decisions的别名）
    对应 Go 版本的 GET /api/live/traces
    """
    # 直接调用decisions端点
    return await list_decisions(
        symbol=symbol,
        action=action,
        status=status,
        page=page,
        page_size=page_size
    )


@router.post("/decisions/{decision_id}/cancel")
async def cancel_decision(decision_id: str):
    """
    取消决策
    """
    logger.info("取消决策", decision_id=decision_id)

    # 查找决策
    decision = next((d for d in _decisions_db if d.id == decision_id), None)
    if not decision:
        raise HTTPException(status_code=404, detail="决策未找到")

    # 检查状态
    if decision.status != DecisionStatus.PENDING:
        raise HTTPException(status_code=400, detail="只能取消待执行的决策")

    # 更新状态
    decision.status = DecisionStatus.CANCELLED

    return {"message": "决策已取消"}


@router.get("/decisions/stats")
async def get_decision_stats():
    """
    获取决策统计信息
    """
    total = len(_decisions_db)
    pending = sum(1 for d in _decisions_db if d.status == DecisionStatus.PENDING)
    executed = sum(1 for d in _decisions_db if d.status == DecisionStatus.EXECUTED)
    failed = sum(1 for d in _decisions_db if d.status == DecisionStatus.FAILED)
    cancelled = sum(1 for d in _decisions_db if d.status == DecisionStatus.CANCELLED)

    # 按操作类型统计
    actions = {}
    for decision in _decisions_db:
        actions[decision.action] = actions.get(decision.action, 0) + 1

    # 按交易对统计
    symbols = {}
    for decision in _decisions_db:
        symbols[decision.symbol] = symbols.get(decision.symbol, 0) + 1

    return {
        "total": total,
        "by_status": {
            "pending": pending,
            "executed": executed,
            "failed": failed,
            "cancelled": cancelled,
        },
        "by_action": actions,
        "by_symbol": symbols,
    }


# 辅助函数：添加决策到数据库（用于测试）
def add_test_decision(decision: Decision):
    """添加测试决策"""
    _decisions_db.append(decision)


# 辅助函数：清空数据库（用于测试）
def clear_test_decisions():
    """清空测试决策"""
    _decisions_db.clear()


# 导出
__all__ = ["router"]