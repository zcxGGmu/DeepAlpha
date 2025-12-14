"""实时持仓存储实现
对应Go版本的 live_positions.go
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from deepalpha.db.models import LiveOrder, LiveTier, OrderStatus, OrderSide
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class LivePositionsStore:
    """实时持仓存储"""

    def __init__(self, db: Session):
        self.db = db

    def get_open_positions(self, symbol: Optional[str] = None) -> List[LiveOrder]:
        """获取当前开仓"""
        query = self.db.query(LiveOrder).filter(
            LiveOrder.status == OrderStatus.FILLED,
            LiveOrder.side.in_([OrderSide.BUY, OrderSide.LONG])
        )

        if symbol:
            query = query.filter(LiveOrder.symbol == symbol)

        # 按交易对分组，只获取最新的开仓
        subquery = query.with_entities(
            LiveOrder.symbol,
            func.max(LiveOrder.created_at).label('max_created')
        ).group_by(LiveOrder.symbol).subquery()

        positions = self.db.query(LiveOrder).join(
            subquery,
            and_(
                LiveOrder.symbol == subquery.c.symbol,
                LiveOrder.created_at == subquery.c.max_created
            )
        ).all()

        return positions

    def get_position_by_symbol(self, symbol: str) -> Optional[LiveOrder]:
        """获取指定交易对的持仓"""
        return self.db.query(LiveOrder).filter(
            LiveOrder.symbol == symbol,
            LiveOrder.status == OrderStatus.FILLED,
            LiveOrder.side.in_([OrderSide.BUY, OrderSide.LONG])
        ).order_by(desc(LiveOrder.created_at)).first()

    def update_position(
        self,
        order_id: int,
        mark_price: float,
        unrealized_pnl: float,
        unrealized_percent: float,
    ) -> bool:
        """更新持仓信息"""
        try:
            order = self.db.query(LiveOrder).filter(
                LiveOrder.id == order_id
            ).first()

            if not order:
                logger.warning(f"订单不存在: {order_id}")
                return False

            # 更新实时价格和盈亏（这里可能需要添加额外的字段）
            # 当前模型中没有mark_price字段，需要扩展
            order.updated_at = datetime.utcnow()
            self.db.commit()

            logger.info(f"更新持仓信息: {order.symbol}")
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新持仓信息失败: {e}")
            return False

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        realized_pnl: float,
    ) -> Optional[LiveOrder]:
        """平仓"""
        try:
            # 获取持仓
            position = self.get_position_by_symbol(symbol)
            if not position:
                logger.warning(f"没有找到持仓: {symbol}")
                return None

            # 创建平仓订单
            close_order = LiveOrder(
                symbol=symbol,
                side=OrderSide.SELL if position.side == OrderSide.LONG else OrderSide.BUY,
                order_type=position.order_type,
                status=OrderStatus.FILLED,
                amount=position.filled_amount,
                price=exit_price,
                filled_amount=position.filled_amount,
                filled_price=exit_price,
                entry_price=position.entry_price,
                exit_price=exit_price,
                realized_pnl=realized_pnl,
                commission=0,  # 需要计算
                decision_id=position.decision_id,
                closed_at=datetime.utcnow(),
            )

            self.db.add(close_order)

            # 更新原持仓状态
            position.status = OrderStatus.FILLED
            position.closed_at = datetime.utcnow()

            self.db.commit()
            logger.info(f"平仓成功: {symbol} @ {exit_price}, PnL: {realized_pnl}")
            return close_order
        except Exception as e:
            self.db.rollback()
            logger.error(f"平仓失败: {e}")
            return None

    def get_position_tiers(self, position_id: int) -> List[LiveTier]:
        """获取持仓的层级设置"""
        return self.db.query(LiveTier).filter(
            LiveTier.order_id == position_id
        ).order_by(LiveTier.tier).all()

    def execute_tier(self, tier_id: int, exit_price: float) -> bool:
        """执行层级平仓"""
        try:
            tier = self.db.query(LiveTier).filter(
                LiveTier.id == tier_id
            ).first()

            if not tier:
                logger.warning(f"层级不存在: {tier_id}")
                return False

            if tier.done:
                logger.warning(f"层级已执行: {tier_id}")
                return False

            # 标记为已执行
            tier.done = True
            tier.executed_at = datetime.utcnow()

            self.db.commit()
            logger.info(f"执行层级平仓: L{tier.tier} @ {exit_price}")
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"执行层级平仓失败: {e}")
            return False

    def get_pending_tiers(self, symbol: Optional[str] = None) -> List[LiveTier]:
        """获取待执行的层级"""
        query = self.db.query(LiveTier).join(LiveOrder).filter(
            LiveTier.done == False
        )

        if symbol:
            query = query.filter(LiveOrder.symbol == symbol)

        return query.order_by(LiveTier.tier).all()

    def calculate_pnl(self, position: LiveOrder, current_price: float) -> Dict[str, float]:
        """计算持仓盈亏"""
        if position.side in [OrderSide.BUY, OrderSide.LONG]:
            # 多头持仓
            unrealized_pnl = (current_price - position.entry_price) * position.filled_amount
            unrealized_percent = ((current_price - position.entry_price) / position.entry_price) * 100
        else:
            # 空头持仓
            unrealized_pnl = (position.entry_price - current_price) * position.filled_amount
            unrealized_percent = ((position.entry_price - current_price) / position.entry_price) * 100

        return {
            "unrealized_pnl": unrealized_pnl,
            "unrealized_percent": unrealized_percent
        }

    def get_position_summary(self) -> Dict[str, Any]:
        """获取持仓汇总"""
        positions = self.get_open_positions()

        total_value = 0
        total_pnl = 0
        total_margin = 0
        position_count = len(positions)

        for position in positions:
            # 需要从交易所获取当前价格
            # 这里使用entry_price作为示例
            current_price = position.entry_price
            pnl_info = self.calculate_pnl(position, current_price)

            total_value += position.filled_amount * current_price
            total_pnl += pnl_info["unrealized_pnl"]
            total_margin += position.filled_amount * current_price / position.leverage

        return {
            "position_count": position_count,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "total_margin": total_margin,
            "avg_leverage": sum(p.leverage for p in positions) / position_count if position_count > 0 else 0,
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "size": p.filled_amount,
                    "entry_price": p.entry_price,
                    "leverage": p.leverage,
                    "unrealized_pnl": self.calculate_pnl(p, p.entry_price)["unrealized_pnl"]
                }
                for p in positions
            ]
        }

    def check_risk_limits(self) -> List[Dict[str, Any]]:
        """检查风险限制"""
        warnings = []
        positions = self.get_open_positions()

        for position in positions:
            # 检查单个持仓风险
            current_price = position.entry_price  # 应该从市场数据获取
            pnl_info = self.calculate_pnl(position, current_price)

            # 检查亏损超过10%
            if pnl_info["unrealized_percent"] < -10:
                warnings.append({
                    "type": "high_loss",
                    "symbol": position.symbol,
                    "message": f"持仓亏损过大: {pnl_info['unrealized_percent']:.2f}%",
                    "severity": "high"
                })

            # 检查杠杆过高
            if position.leverage > 10:
                warnings.append({
                    "type": "high_leverage",
                    "symbol": position.symbol,
                    "message": f"杠杆过高: {position.leverage}x",
                    "severity": "medium"
                })

        # 检查总持仓风险
        summary = self.get_position_summary()
        if summary["total_pnl"] < -1000:  # 总亏损超过1000
            warnings.append({
                "type": "total_loss",
                "message": f"总亏损过大: ${summary['total_pnl']:.2f}",
                "severity": "high"
            })

        return warnings