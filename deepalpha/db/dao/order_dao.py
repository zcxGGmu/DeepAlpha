"""订单数据访问对象"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import and_, or_

from deepalpha.db.models import LiveOrder, LiveTier, OrderStatus, OrderSide
from deepalpha.db.dao.base import BaseDAO
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class OrderDAO(BaseDAO[LiveOrder]):
    """订单DAO"""

    def __init__(self, session):
        super().__init__(session, LiveOrder)

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: float,
        price: Optional[float] = None,
        ft_order_id: Optional[int] = None,
        ft_pair: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> LiveOrder:
        """创建订单"""
        return await self.create(
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=OrderStatus.OPEN,
            amount=amount,
            price=price,
            ft_order_id=ft_order_id,
            ft_pair=ft_pair,
            decision_id=decision_id,
        )

    async def get_by_symbol(
        self,
        symbol: str,
        status: Optional[OrderStatus] = None,
        limit: Optional[int] = None,
    ) -> List[LiveOrder]:
        """根据交易对获取订单"""
        filters = {"symbol": symbol}
        if status:
            filters["status"] = status

        return await self.find_by(filters, limit=limit, order_by="-created_at")

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[LiveOrder]:
        """获取未成交订单"""
        filters = {"status": OrderStatus.OPEN}
        if symbol:
            filters["symbol"] = symbol

        return await self.find_by(filters, order_by="-created_at")

    async def get_by_ft_order_id(self, ft_order_id: int) -> Optional[LiveOrder]:
        """根据Freqtrade订单ID获取订单"""
        return await self.find_one_by({"ft_order_id": ft_order_id})

    async def get_by_decision_id(self, decision_id: str) -> List[LiveOrder]:
        """根据决策ID获取相关订单"""
        return await self.find_by({"decision_id": decision_id}, order_by="-created_at")

    async def update_status(
        self,
        order_id: int,
        status: OrderStatus,
        filled_amount: Optional[float] = None,
        filled_price: Optional[float] = None,
        realized_pnl: Optional[float] = None,
        commission: Optional[float] = None,
        closed_at: Optional[datetime] = None,
    ) -> Optional[LiveOrder]:
        """更新订单状态"""
        update_data = {"status": status}

        if filled_amount is not None:
            update_data["filled_amount"] = filled_amount
        if filled_price is not None:
            update_data["filled_price"] = filled_price
        if realized_pnl is not None:
            update_data["realized_pnl"] = realized_pnl
        if commission is not None:
            update_data["commission"] = commission
        if closed_at is not None:
            update_data["closed_at"] = closed_at

        return await self.update(order_id, **update_data)

    async def get_positions(self) -> List[LiveOrder]:
        """获取当前持仓"""
        # 获取已开仓但未平仓的订单
        result = await self.session.execute(
            select(self.model_class)
            .where(
                or_(
                    and_(
                        self.model_class.side.in_([OrderSide.BUY, OrderSide.LONG]),
                        self.model_class.status == OrderStatus.FILLED,
                    ),
                    and_(
                        self.model_class.side.in_([OrderSide.SELL, OrderSide.SHORT]),
                        self.model_class.status == OrderStatus.OPEN,
                    ),
                )
            )
            .order_by(self.model_class.created_at.desc())
        )
        return result.scalars().all()

    async def get_position_for_symbol(self, symbol: str) -> Optional[LiveOrder]:
        """获取指定交易对的持仓"""
        result = await self.session.execute(
            select(self.model_class)
            .where(
                and_(
                    self.model_class.symbol == symbol,
                    self.model_class.status == OrderStatus.FILLED,
                    self.model_class.side.in_([OrderSide.BUY, OrderSide.LONG]),
                )
            )
            .order_by(self.model_class.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def calculate_total_pnl(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> float:
        """计算总盈亏"""
        query = select(func.sum(self.model_class.realized_pnl)).where(
            self.model_class.realized_pnl.isnot(None)
        )

        if symbol:
            query = query.where(self.model_class.symbol == symbol)
        if start_time:
            query = query.where(self.model_class.closed_at >= start_time)
        if end_time:
            query = query.where(self.model_class.closed_at <= end_time)

        result = await self.session.execute(query)
        total = result.scalar()
        return float(total) if total else 0.0

    async def get_order_stats(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """获取订单统计信息"""
        # 基础查询
        base_query = select(self.model_class)
        if symbol:
            base_query = base_query.where(self.model_class.symbol == symbol)
        if start_time:
            base_query = base_query.where(self.model_class.created_at >= start_time)

        # 总订单数
        total_result = await self.session.execute(
            select(func.count()).select_from(base_query.subquery())
        )
        total_orders = total_result.scalar()

        # 成交订单数
        filled_result = await self.session.execute(
            select(func.count()).select_from(
                base_query.where(self.model_class.status == OrderStatus.FILLED).subquery()
            )
        )
        filled_orders = filled_result.scalar()

        # 胜率（盈利订单数 / 已平仓订单数）
        profit_result = await self.session.execute(
            select(func.count()).select_from(
                base_query.where(
                    and_(
                        self.model_class.realized_pnl > 0,
                        self.model_class.realized_pnl.isnot(None),
                    )
                ).subquery()
            )
        )
        profit_orders = profit_result.scalar()

        closed_result = await self.session.execute(
            select(func.count()).select_from(
                base_query.where(
                    and_(
                        self.model_class.realized_pnl.isnot(None),
                        self.model_class.closed_at.isnot(None),
                    )
                ).subquery()
            )
        )
        closed_orders = closed_result.scalar()

        win_rate = (profit_orders / closed_orders * 100) if closed_orders > 0 else 0

        return {
            "total_orders": total_orders,
            "filled_orders": filled_orders,
            "fill_rate": (filled_orders / total_orders * 100) if total_orders > 0 else 0,
            "closed_orders": closed_orders,
            "profit_orders": profit_orders,
            "win_rate": win_rate,
        }


class TierDAO(BaseDAO[LiveTier]):
    """层级DAO"""

    def __init__(self, session):
        super().__init__(session, LiveTier)

    async def create_tiers_for_order(
        self,
        order_id: int,
        tiers_config: List[Dict[str, float]],
    ) -> List[LiveTier]:
        """为订单创建层级"""
        tiers = []
        for i, config in enumerate(tiers_config, 1):
            tier = await self.create(
                order_id=order_id,
                tier=i,
                target_price=config["target_price"],
                ratio=config["ratio"],
                done=False,
            )
            tiers.append(tier)
        return tiers

    async def get_by_order_id(self, order_id: int) -> List[LiveTier]:
        """获取订单的所有层级"""
        return await self.find_by({"order_id": order_id}, order_by="tier")

    async def get_pending_tiers(self, symbol: Optional[str] = None) -> List[LiveTier]:
        """获取待执行的层级"""
        query = select(self.model_class).where(self.model_class.done == False)

        if symbol:
            query = query.join(LiveOrder).where(LiveOrder.symbol == symbol)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def mark_as_done(self, tier_id: int) -> Optional[LiveTier]:
        """标记层级为已完成"""
        return await self.update(tier_id, done=True, executed_at=datetime.utcnow())

    async def get_next_tier(self, order_id: int) -> Optional[LiveTier]:
        """获取下一个待执行的层级"""
        result = await self.session.execute(
            select(self.model_class)
            .where(
                and_(
                    self.model_class.order_id == order_id,
                    self.model_class.done == False,
                )
            )
            .order_by(self.model_class.tier)
            .limit(1)
        )
        return result.scalar_one_or_none()