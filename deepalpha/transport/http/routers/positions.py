"""持仓相关的API路由，对应 Go 版本的 position handlers"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from deepalpha.core.models import OrderSide
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# 响应模型
class PositionResponse(BaseModel):
    """持仓响应模型"""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: datetime
    trade_id: Optional[int] = None
    metadata: Optional[dict] = None


class PositionListResponse(BaseModel):
    """持仓列表响应"""
    positions: List[PositionResponse]
    total: int
    total_exposure: float
    total_pnl: float


class TierInfo(BaseModel):
    """层级信息"""
    target: Optional[float] = None
    ratio: Optional[float] = None
    done: bool = False


class PositionDetailResponse(PositionResponse):
    """持仓详情响应（继承基础响应）"""
    tier1: Optional[TierInfo] = None
    tier2: Optional[TierInfo] = None
    tier3: Optional[TierInfo] = None
    orders_count: int = 0
    profit_ratio: float = 0.0


# 请求模型
class ClosePositionRequest(BaseModel):
    """平仓请求"""
    symbol: str = Field(..., description="交易对")
    amount: Optional[float] = Field(None, description="平仓数量，不填则全部平仓")
    reason: Optional[str] = Field(None, description="平仓原因")


class UpdateTierRequest(BaseModel):
    """更新层级请求"""
    trade_id: int = Field(..., description="交易ID")
    tier: int = Field(..., ge=1, le=3, description="层级（1-3）")
    target: Optional[float] = Field(None, description="目标价格")
    ratio: Optional[float] = Field(None, description="平仓比例")
    done: Optional[bool] = Field(None, description="是否完成")


# 模拟数据存储（实际应该从Freqtrade API获取）
_positions_db: List[dict] = []


@router.get("/positions", response_model=PositionListResponse)
async def list_positions(
    symbol: Optional[str] = Query(None, description="交易对"),
    side: Optional[str] = Query(None, description="方向（long/short）"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
):
    """
    获取持仓列表
    对应 Go 版本的 GET /api/live/positions
    """
    logger.info(
        "获取持仓列表",
        symbol=symbol,
        side=side,
        page=page,
        page_size=page_size
    )

    # 筛选持仓
    filtered_positions = _positions_db

    if symbol:
        filtered_positions = [p for p in filtered_positions if p["symbol"] == symbol]
    if side:
        filtered_positions = [p for p in filtered_positions if p["side"] == side]

    # 分页
    total = len(filtered_positions)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_positions = filtered_positions[start_idx:end_idx]

    # 转换为响应格式
    position_responses = [
        PositionResponse(
            symbol=position["symbol"],
            side=position["side"],
            size=position["size"],
            entry_price=position["entry_price"],
            current_price=position.get("current_price"),
            unrealized_pnl=position.get("unrealized_pnl", 0.0),
            realized_pnl=position.get("realized_pnl", 0.0),
            timestamp=position["timestamp"],
            trade_id=position.get("trade_id"),
            metadata=position.get("metadata"),
        )
        for position in page_positions
    ]

    # 计算总敞口和盈亏
    total_exposure = sum(
        p["size"] * (p.get("current_price") or p["entry_price"])
        for p in filtered_positions
    )
    total_pnl = sum(
        p.get("unrealized_pnl", 0.0) + p.get("realized_pnl", 0.0)
        for p in filtered_positions
    )

    return PositionListResponse(
        positions=position_responses,
        total=total,
        total_exposure=total_exposure,
        total_pnl=total_pnl,
    )


@router.get("/positions/{symbol}", response_model=PositionDetailResponse)
async def get_position(symbol: str):
    """
    获取持仓详情
    对应 Go 版本的 GET /api/live/positions/:id
    """
    logger.info("获取持仓详情", symbol=symbol)

    # 查找持仓
    position = next((p for p in _positions_db if p["symbol"] == symbol), None)
    if not position:
        raise HTTPException(status_code=404, detail="持仓未找到")

    # 计算盈亏比例
    entry_price = position["entry_price"]
    current_price = position.get("current_price", entry_price)
    if position["side"] == "long":
        profit_ratio = (current_price - entry_price) / entry_price
    else:
        profit_ratio = (entry_price - current_price) / entry_price

    # 获取层级信息
    metadata = position.get("metadata", {})

    return PositionDetailResponse(
        symbol=position["symbol"],
        side=position["side"],
        size=position["size"],
        entry_price=position["entry_price"],
        current_price=position.get("current_price"),
        unrealized_pnl=position.get("unrealized_pnl", 0.0),
        realized_pnl=position.get("realized_pnl", 0.0),
        timestamp=position["timestamp"],
        trade_id=position.get("trade_id"),
        metadata=position.get("metadata"),
        tier1=TierInfo(**metadata["tier1"]) if "tier1" in metadata else None,
        tier2=TierInfo(**metadata["tier2"]) if "tier2" in metadata else None,
        tier3=TierInfo(**metadata["tier3"]) if "tier3" in metadata else None,
        orders_count=position.get("orders_count", 0),
        profit_ratio=profit_ratio,
    )


@router.post("/positions/close")
async def close_position(request: ClosePositionRequest):
    """
    快速平仓
    对应 Go 版本的 POST /api/live/positions/close
    """
    logger.info("平仓请求", symbol=request.symbol, amount=request.amount)

    # 查找持仓
    position = next((p for p in _positions_db if p["symbol"] == request.symbol), None)
    if not position:
        raise HTTPException(status_code=404, detail="持仓未找到")

    # 计算平仓数量
    close_amount = request.amount or position["size"]

    if close_amount > position["size"]:
        raise HTTPException(status_code=400, detail="平仓数量超过持仓数量")

    # 更新持仓（这里简化处理，实际应该调用Freqtrade API）
    if close_amount >= position["size"]:
        # 全部平仓
        _positions_db.remove(position)
        message = f"已全部平仓 {request.symbol}"
    else:
        # 部分平仓
        position["size"] -= close_amount
        message = f"已部分平仓 {request.symbol}，数量: {close_amount}"

    return {"message": message, "amount": close_amount}


@router.post("/positions/tiers")
async def update_tier(request: UpdateTierRequest):
    """
    更新持仓层级
    对应 Go 版本的 POST /api/live/positions/tiers
    """
    logger.info("更新层级", trade_id=request.trade_id, tier=request.tier)

    # 查找持仓
    position = next(
        (p for p in _positions_db if p.get("trade_id") == request.trade_id),
        None
    )
    if not position:
        raise HTTPException(status_code=404, detail="持仓未找到")

    # 更新层级信息
    metadata = position.get("metadata", {})
    tier_key = f"tier{request.tier}"
    tier_info = metadata.get(tier_key, {})

    if request.target is not None:
        tier_info["target"] = request.target
    if request.ratio is not None:
        tier_info["ratio"] = request.ratio
    if request.done is not None:
        tier_info["done"] = request.done

    metadata[tier_key] = tier_info
    position["metadata"] = metadata

    return {"message": f"层级{request.tier}更新成功"}


@router.get("/positions/stats")
async def get_position_stats():
    """
    获取持仓统计信息
    """
    total = len(_positions_db)
    long_positions = sum(1 for p in _positions_db if p["side"] == "long")
    short_positions = sum(1 for p in _positions_db if p["side"] == "short")

    total_exposure = sum(
        p["size"] * (p.get("current_price") or p["entry_price"])
        for p in _positions_db
    )

    total_unrealized = sum(p.get("unrealized_pnl", 0.0) for p in _positions_db)
    total_realized = sum(p.get("realized_pnl", 0.0) for p in _positions_db)

    profit_positions = sum(
        1 for p in _positions_db
        if (p.get("unrealized_pnl", 0.0) + p.get("realized_pnl", 0.0)) > 0
    )
    loss_positions = total - profit_positions

    # 按交易对统计
    symbols = {}
    for position in _positions_db:
        symbol = position["symbol"]
        if symbol not in symbols:
            symbols[symbol] = {"count": 0, "exposure": 0.0, "pnl": 0.0}
        symbols[symbol]["count"] += 1
        symbols[symbol]["exposure"] += position["size"] * (
            position.get("current_price") or position["entry_price"]
        )
        symbols[symbol]["pnl"] += (
            position.get("unrealized_pnl", 0.0) + position.get("realized_pnl", 0.0)
        )

    return {
        "total": total,
        "by_side": {
            "long": long_positions,
            "short": short_positions,
        },
        "total_exposure": total_exposure,
        "total_unrealized_pnl": total_unrealized,
        "total_realized_pnl": total_realized,
        "total_pnl": total_unrealized + total_realized,
        "profit_positions": profit_positions,
        "loss_positions": loss_positions,
        "by_symbol": symbols,
    }


# 辅助函数：添加测试持仓
def add_test_position(position: dict):
    """添加测试持仓"""
    _positions_db.append(position)


# 辅助函数：清空测试持仓
def clear_test_positions():
    """清空测试持仓"""
    _positions_db.clear()


# 导出
__all__ = ["router"]