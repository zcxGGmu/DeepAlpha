"""决策日志存储实现
对应Go版本的 decision_log_store.go
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from deepalpha.db.models import LiveDecisionLog, DecisionStatus
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class DecisionLogStore:
    """决策日志存储"""

    def __init__(self, db: Session):
        self.db = db

    def save(self, decision: LiveDecisionLog) -> LiveDecisionLog:
        """保存决策日志"""
        try:
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"保存决策日志: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"保存决策日志失败: {e}")
            raise

    def get_by_id(self, decision_id: str) -> Optional[LiveDecisionLog]:
        """根据决策ID获取日志"""
        return self.db.query(LiveDecisionLog).filter(
            LiveDecisionLog.decision_id == decision_id
        ).first()

    def get_by_symbol(
        self,
        symbol: str,
        limit: int = 100,
        status: Optional[DecisionStatus] = None,
    ) -> List[LiveDecisionLog]:
        """获取指定交易对的决策日志"""
        query = self.db.query(LiveDecisionLog).filter(
            LiveDecisionLog.symbol == symbol
        )

        if status:
            query = query.filter(LiveDecisionLog.status == status)

        return query.order_by(desc(LiveDecisionLog.created_at)).limit(limit).all()

    def get_pending(self, symbol: Optional[str] = None) -> List[LiveDecisionLog]:
        """获取待执行的决策"""
        query = self.db.query(LiveDecisionLog).filter(
            LiveDecisionLog.status == DecisionStatus.PENDING
        )

        if symbol:
            query = query.filter(LiveDecisionLog.symbol == symbol)

        return query.order_by(desc(LiveDecisionLog.created_at)).all()

    def update_status(
        self,
        decision_id: str,
        status: DecisionStatus,
        reason: Optional[str] = None,
    ) -> bool:
        """更新决策状态"""
        try:
            decision = self.get_by_id(decision_id)
            if not decision:
                logger.warning(f"决策不存在: {decision_id}")
                return False

            decision.status = status
            decision.updated_at = datetime.utcnow()

            if reason:
                # 将原因添加到现有reason中
                if decision.reason:
                    decision.reason = f"{decision.reason}\n\n{reason}"
                else:
                    decision.reason = reason

            self.db.commit()
            logger.info(f"更新决策状态: {decision_id} -> {status.value}")
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新决策状态失败: {e}")
            return False

    def get_recent(
        self,
        limit: int = 50,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[LiveDecisionLog]:
        """获取最近的决策日志"""
        query = self.db.query(LiveDecisionLog)

        if symbol:
            query = query.filter(LiveDecisionLog.symbol == symbol)
        if start_time:
            query = query.filter(LiveDecisionLog.created_at >= start_time)
        if end_time:
            query = query.filter(LiveDecisionLog.created_at <= end_time)

        return query.order_by(desc(LiveDecisionLog.created_at)).limit(limit).all()

    def count_by_status(self, symbol: Optional[str] = None) -> Dict[str, int]:
        """按状态统计决策数量"""
        query = self.db.query(
            LiveDecisionLog.status,
            func.count(LiveDecisionLog.id)
        ).group_by(LiveDecisionLog.status)

        if symbol:
            query = query.filter(LiveDecisionLog.symbol == symbol)

        result = query.all()
        return {status.value: count for status, count in result}

    def delete_old(self, before: datetime) -> int:
        """删除旧的决策日志"""
        try:
            count = self.db.query(LiveDecisionLog).filter(
                LiveDecisionLog.created_at < before
            ).delete()
            self.db.commit()
            logger.info(f"删除旧决策日志: {count} 条")
            return count
        except Exception as e:
            self.db.rollback()
            logger.error(f"删除旧决策日志失败: {e}")
            return 0

    def get_stats(
        self,
        symbol: Optional[str] = None,
        days: int = 7,
    ) -> Dict[str, Any]:
        """获取决策统计信息"""
        from sqlalchemy import func
        from datetime import timedelta

        start_time = datetime.utcnow() - timedelta(days=days)

        # 基础查询
        base_query = self.db.query(LiveDecisionLog).filter(
            LiveDecisionLog.created_at >= start_time
        )
        if symbol:
            base_query = base_query.filter(LiveDecisionLog.symbol == symbol)

        # 总决策数
        total_count = base_query.count()

        # 按状态统计
        status_counts = base_query.with_entities(
            LiveDecisionLog.status,
            func.count(LiveDecisionLog.id)
        ).group_by(LiveDecisionLog.status).all()

        # 按动作统计
        action_counts = base_query.with_entities(
            LiveDecisionLog.action,
            func.count(LiveDecisionLog.id)
        ).group_by(LiveDecisionLog.action).all()

        # 平均置信度
        avg_confidence = base_query.with_entities(
            func.avg(LiveDecisionLog.confidence)
        ).scalar() or 0

        # 按交易对统计
        symbol_counts = base_query.with_entities(
            LiveDecisionLog.symbol,
            func.count(LiveDecisionLog.id)
        ).group_by(LiveDecisionLog.symbol).all()

        return {
            "total_count": total_count,
            "status_distribution": {
                status.value: count for status, count in status_counts
            },
            "action_distribution": {
                action: count for action, count in action_counts
            },
            "avg_confidence": round(avg_confidence, 2),
            "symbol_distribution": {
                symbol: count for symbol, count in symbol_counts
            },
            "period_days": days
        }