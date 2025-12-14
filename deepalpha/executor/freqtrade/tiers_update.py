"""层级更新逻辑
对应Go版本的 tiers_update.go
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_

from deepalpha.executor.freqtrade.client import FreqtradeClient
from deepalpha.db.models import LiveOrder, LiveTier, OrderStatus
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class TiersUpdater:
    """层级更新管理器"""

    def __init__(self, client: FreqtradeClient, db: Session):
        self.client = client
        self.db = db
        self.check_interval = 5  # 5秒检查一次

    async def start_monitoring(self):
        """启动层级监控"""
        logger.info("启动层级监控任务")

        while True:
            try:
                await self.check_and_execute_tiers()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"层级检查失败: {e}")
                await asyncio.sleep(5)

    async def check_and_execute_tiers(self):
        """检查并执行层级"""
        # 获取所有待执行的层级
        pending_tiers = self._get_pending_tiers()

        for tier in pending_tiers:
            try:
                should_execute = await self._should_execute_tier(tier)
                if should_execute:
                    await self._execute_tier(tier)
            except Exception as e:
                logger.error(f"处理层级失败 {tier.id}: {e}")

    def _get_pending_tiers(self) -> List[LiveTier]:
        """获取待执行的层级"""
        query = self.db.query(LiveTier).join(LiveOrder).filter(
            LiveTier.done == False,
            LiveOrder.status == OrderStatus.FILLED,
            LiveOrder.closed_at.is_(None)
        )

        # 只获取每个订单的下一个待执行层级
        tiers = []
        for tier in query.all():
            # 检查是否是下一个待执行的层级
            if self._is_next_tier(tier):
                tiers.append(tier)

        return tiers

    def _is_next_tier(self, tier: LiveTier) -> bool:
        """检查是否是下一个待执行的层级"""
        if tier.tier == 1:
            return True

        # 检查前一层是否已执行
        prev_tier = self.db.query(LiveTier).filter(
            LiveTier.order_id == tier.order_id,
            LiveTier.tier == tier.tier - 1
        ).first()

        return prev_tier and prev_tier.done

    async def _should_execute_tier(self, tier: LiveTier) -> bool:
        """判断是否应该执行层级"""
        # 获取订单和当前价格
        order = self.db.query(LiveOrder).filter(LiveOrder.id == tier.order_id).first()
        if not order:
            return False

        # 获取当前价格
        try:
            ticker = await self.client.get_ticker(order.symbol)
            current_price = float(ticker.last)
        except Exception as e:
            logger.error(f"获取价格失败 {order.symbol}: {e}")
            return False

        # 检查是否达到目标价格
        if order.side.value in ["buy", "long"]:
            # 多头：价格 >= 目标价格
            return current_price >= tier.target_price
        else:
            # 空头：价格 <= 目标价格
            return current_price <= tier.target_price

    async def _execute_tier(self, tier: LiveTier):
        """执行层级平仓"""
        order = self.db.query(LiveOrder).filter(LiveOrder.id == tier.order_id).first()
        if not order:
            logger.error(f"订单不存在: {tier.order_id}")
            return

        # 计算要平仓的数量
        close_amount = order.filled_amount * tier.ratio

        logger.info(
            f"执行层级平仓: {order.symbol} L{tier.tier} "
            f"数量: {close_amount} 比例: {tier.ratio * 100}%"
        )

        try:
            # 创建平仓订单
            close_order = await self.client.create_order(
                pair=order.symbol,
                ordertype="market",
                side="sell" if order.side.value in ["buy", "long"] else "buy",
                amount=close_amount
            )

            if close_order and close_order.get("id"):
                # 记录层级执行
                tier.done = True
                tier.executed_at = datetime.utcnow()

                # 保存Freqtrade订单ID
                tier.ft_order_id = close_order["id"]

                logger.info(f"层级平仓订单已创建: {close_order['id']}")

                # 检查是否所有层级都已完成
                await self._check_all_tiers_done(order)

            self.db.commit()

        except Exception as e:
            self.db.rollback()
            logger.error(f"执行层级平仓失败: {e}")
            raise

    async def _check_all_tiers_done(self, order: LiveOrder):
        """检查是否所有层级都已完成"""
        all_tiers = self.db.query(LiveTier).filter(
            LiveTier.order_id == order.id
        ).all()

        if all(t.done for t in all_tiers):
            # 所有层级都已完成，关闭订单
            order.status = OrderStatus.CLOSED
            order.closed_at = datetime.utcnow()
            logger.info(f"订单所有层级已完成: {order.symbol}")

    async def update_tiers(self, order_id: int, tiers_config: List[Dict[str, Any]]) -> bool:
        """更新层级设置"""
        try:
            # 删除现有层级
            self.db.query(LiveTier).filter(
                LiveTier.order_id == order_id
            ).delete()

            # 创建新层级
            for i, config in enumerate(tiers_config, 1):
                tier = LiveTier(
                    order_id=order_id,
                    tier=i,
                    target_price=float(config["target_price"]),
                    ratio=float(config["ratio"]),
                    done=False
                )
                self.db.add(tier)

            self.db.commit()
            logger.info(f"更新层级设置成功: {len(tiers_config)} 层")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"更新层级设置失败: {e}")
            return False

    async def get_tiers_status(self, order_id: int) -> List[Dict[str, Any]]:
        """获取层级状态"""
        tiers = self.db.query(LiveTier).filter(
            LiveTier.order_id == order_id
        ).order_by(LiveTier.tier).all()

        # 获取当前价格
        order = self.db.query(LiveOrder).filter(LiveOrder.id == order_id).first()
        current_price = None

        if order:
            try:
                ticker = await self.client.get_ticker(order.symbol)
                current_price = float(ticker.last)
            except:
                pass

        return [
            {
                "tier": t.tier,
                "target_price": t.target_price,
                "ratio": t.ratio,
                "done": t.done,
                "executed_at": t.executed_at.isoformat() if t.executed_at else None,
                "progress": self._calculate_tier_progress(t, current_price) if current_price else 0,
                "current_price": current_price
            }
            for t in tiers
        ]

    def _calculate_tier_progress(self, tier: LiveTier, current_price: float) -> float:
        """计算层级进度"""
        order = self.db.query(LiveOrder).filter(LiveOrder.id == tier.order_id).first()
        if not order:
            return 0

        if order.side.value in ["buy", "long"]:
            # 多头
            entry_price = order.entry_price or order.price
            target_distance = tier.target_price - entry_price
            current_distance = current_price - entry_price
            return max(0, min(100, (current_distance / target_distance) * 100)) if target_distance != 0 else 0
        else:
            # 空头
            entry_price = order.entry_price or order.price
            target_distance = entry_price - tier.target_price
            current_distance = entry_price - current_price
            return max(0, min(100, (current_distance / target_distance) * 100)) if target_distance != 0 else 0

    async def force_execute_tier(self, tier_id: int) -> bool:
        """强制执行层级"""
        try:
            tier = self.db.query(LiveTier).filter(LiveTier.id == tier_id).first()
            if not tier:
                logger.error(f"层级不存在: {tier_id}")
                return False

            if tier.done:
                logger.warning(f"层级已执行: {tier_id}")
                return True

            await self._execute_tier(tier)
            logger.info(f"强制执行层级成功: {tier_id}")
            return True

        except Exception as e:
            logger.error(f"强制执行层级失败: {e}")
            return False

    def get_pending_tiers_count(self) -> int:
        """获取待执行的层级数量"""
        return self.db.query(LiveTier).join(LiveOrder).filter(
            LiveTier.done == False,
            LiveOrder.status == OrderStatus.FILLED,
            LiveOrder.closed_at.is_(None)
        ).count()