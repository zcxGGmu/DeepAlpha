"""决策日志数据访问对象"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import and_, or_

from deepalpha.db.models import LiveDecisionLog, DecisionStatus
from deepalpha.db.dao.base import BaseDAO
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class DecisionDAO(BaseDAO[LiveDecisionLog]):
    """决策日志DAO"""

    def __init__(self, session):
        super().__init__(session, LiveDecisionLog)

    async def create_decision(
        self,
        decision_id: str,
        symbol: str,
        action: str,
        confidence: int,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        position_size_usd: Optional[float] = None,
        leverage: Optional[int] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        tier1_target: Optional[float] = None,
        tier1_ratio: Optional[float] = None,
        tier2_target: Optional[float] = None,
        tier2_ratio: Optional[float] = None,
        tier3_target: Optional[float] = None,
        tier3_ratio: Optional[float] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LiveDecisionLog:
        """创建决策日志"""
        return await self.create(
            decision_id=decision_id,
            symbol=symbol,
            action=action,
            status=DecisionStatus.PENDING,
            model_provider=model_provider,
            model_name=model_name,
            confidence=confidence,
            position_size_usd=position_size_usd,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            tier1_target=tier1_target,
            tier1_ratio=tier1_ratio,
            tier2_target=tier2_target,
            tier2_ratio=tier2_ratio,
            tier3_target=tier3_target,
            tier3_ratio=tier3_ratio,
            reason=reason,
            metadata=metadata,
        )

    async def get_by_decision_id(self, decision_id: str) -> Optional[LiveDecisionLog]:
        """根据决策ID获取记录"""
        return await self.find_one_by({"decision_id": decision_id})

    async def get_by_symbol(
        self,
        symbol: str,
        status: Optional[DecisionStatus] = None,
        limit: Optional[int] = None,
    ) -> List[LiveDecisionLog]:
        """根据交易对获取决策记录"""
        filters = {"symbol": symbol}
        if status:
            filters["status"] = status

        return await self.find_by(filters, limit=limit, order_by="-created_at")

    async def get_pending_decisions(self, symbol: Optional[str] = None) -> List[LiveDecisionLog]:
        """获取待执行的决策"""
        filters = {"status": DecisionStatus.PENDING}
        if symbol:
            filters["symbol"] = symbol

        return await self.find_by(filters, order_by="-created_at")

    async def get_decisions_in_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        symbol: Optional[str] = None,
        status: Optional[DecisionStatus] = None,
    ) -> List[LiveDecisionLog]:
        """获取时间范围内的决策"""
        query = select(self.model_class).where(
            and_(
                self.model_class.created_at >= start_time,
                self.model_class.created_at <= end_time,
            )
        )

        if symbol:
            query = query.where(self.model_class.symbol == symbol)
        if status:
            query = query.where(self.model_class.status == status)

        query = query.order_by(self.model_class.created_at.desc())

        result = await self.session.execute(query)
        return result.scalars().all()

    async def update_status(
        self,
        decision_id: str,
        status: DecisionStatus,
    ) -> Optional[LiveDecisionLog]:
        """更新决策状态"""
        # 先找到记录
        decision = await self.get_by_decision_id(decision_id)
        if not decision:
            return None

        # 更新状态
        result = await self.session.execute(
            update(self.model_class)
            .where(self.model_class.decision_id == decision_id)
            .values(status=status, updated_at=datetime.utcnow())
            .returning(self.model_class)
        )
        await self.session.commit()
        return result.scalar_one_or_none()

    async def get_decision_stats(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """获取决策统计信息"""
        query = select(
            self.model_class.status,
            func.count(self.model_class.id)
        ).group_by(self.model_class.status)

        if symbol:
            query = query.where(self.model_class.symbol == symbol)
        if start_time:
            query = query.where(self.model_class.created_at >= start_time)
        if end_time:
            query = query.where(self.model_class.created_at <= end_time)

        result = await self.session.execute(query)
        stats = {}
        for status, count in result.all():
            stats[status.value] = count

        return stats

    async def get_recent_decisions(
        self,
        limit: int = 100,
        symbol: Optional[str] = None,
    ) -> List[LiveDecisionLog]:
        """获取最近的决策记录"""
        filters = {}
        if symbol:
            filters["symbol"] = symbol

        return await self.find_by(filters, limit=limit, order_by="-created_at")