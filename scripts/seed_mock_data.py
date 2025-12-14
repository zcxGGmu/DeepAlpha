#!/usr/bin/env python3
"""生成模拟数据脚本"""

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepalpha.db.connection import init_database, get_async_db
from deepalpha.db.models import (
    LiveDecisionLog, LiveOrder, LiveTier, LiveOrderLog,
    LiveModificationLog, TradeOperationLog, LastDecision,
    DecisionStatus, OrderStatus, OrderSide, OrderType
)
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class MockDataGenerator:
    """模拟数据生成器"""

    def __init__(self):
        self.symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
        self.actions = ["enter_long", "enter_short", "close_long", "close_short", "hold"]
        self.model_providers = ["openai", "anthropic", "deepseek"]
        self.model_names = ["gpt-4", "claude-3", "deepseek-chat"]

    async def generate_decisions(self, count: int = 100):
        """生成决策数据"""
        logger.info(f"生成 {count} 条决策记录...")

        async for db in get_async_db():
            for i in range(count):
                symbol = random.choice(self.symbols)
                action = random.choice(self.actions)

                # 只为交易动作创建订单关联
                decision_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"

                decision = LiveDecisionLog(
                    decision_id=decision_id,
                    symbol=symbol,
                    action=action,
                    status=random.choice(list(DecisionStatus)),
                    model_provider=random.choice(self.model_providers),
                    model_name=random.choice(self.model_names),
                    confidence=random.randint(60, 95),
                    position_size_usd=random.uniform(100, 1000),
                    leverage=random.choice([1, 2, 3, 5]),
                    stop_loss=round(random.uniform(0.02, 0.05), 4),
                    take_profit=round(random.uniform(0.06, 0.12), 4),
                    tier1_target=round(random.uniform(1.02, 1.05), 4),
                    tier1_ratio=round(random.uniform(0.3, 0.4), 2),
                    tier2_target=round(random.uniform(1.05, 1.08), 4),
                    tier2_ratio=round(random.uniform(0.3, 0.4), 2),
                    tier3_target=round(random.uniform(1.08, 1.12), 4),
                    tier3_ratio=round(random.uniform(0.2, 0.4), 2),
                    reason=f"Mock decision reason {i}",
                    metadata={"test": True, "index": i}
                )

                db.add(decision)

                # 为部分决策创建订单
                if action in ["enter_long", "enter_short"] and random.random() > 0.3:
                    await self._create_order_for_decision(db, decision, i)

            await db.commit()
            logger.info("决策记录生成完成")

    async def _create_order_for_decision(self, db, decision: LiveDecisionLog, index: int):
        """为决策创建订单"""
        order_id = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}"

        side = OrderSide.LONG if "long" in decision.action else OrderSide.SHORT
        order_type = random.choice([OrderType.MARKET, OrderType.LIMIT])
        status = random.choice([OrderStatus.OPEN, OrderStatus.FILLED, OrderStatus.CANCELLED])

        order = LiveOrder(
            symbol=decision.symbol,
            side=side,
            order_type=order_type,
            status=status,
            amount=random.uniform(0.01, 0.1) if "BTC" in decision.symbol else random.uniform(1, 10),
            price=round(random.uniform(30000, 70000) if "BTC" in decision.symbol else random.uniform(1500, 3500), 2),
            ft_order_id=random.randint(10000, 99999),
            ft_pair=decision.symbol.replace("/", ""),
            decision_id=decision.decision_id,
        )

        # 如果订单已成交，填写成交信息
        if status == OrderStatus.FILLED:
            order.filled_amount = order.amount
            order.filled_price = order.price
            order.realized_pnl = round(random.uniform(-50, 200), 2)
            order.commission = round(order.amount * order.price * 0.001, 2)
            order.closed_at = datetime.utcnow() + timedelta(minutes=random.randint(1, 60))

        db.add(order)
        await db.flush()  # 获取order.id

        # 创建层级
        await self._create_tiers_for_order(db, order)

    async def _create_tiers_for_order(self, db, order: LiveOrder):
        """为订单创建层级"""
        tiers = [
            {"tier": 1, "target_price": order.price * 1.03, "ratio": 0.4},
            {"tier": 2, "target_price": order.price * 1.06, "ratio": 0.4},
            {"tier": 3, "target_price": order.price * 1.10, "ratio": 0.2},
        ]

        for tier_data in tiers:
            tier = LiveTier(
                order_id=order.id,
                tier=tier_data["tier"],
                target_price=tier_data["target_price"],
                ratio=tier_data["ratio"],
                done=random.choice([True, False]),
                executed_at=datetime.utcnow() if tier_data["tier"] == 1 and random.random() > 0.5 else None
            )
            db.add(tier)

    async def generate_order_logs(self, count: int = 200):
        """生成订单日志"""
        logger.info(f"生成 {count} 条订单日志...")

        async for db in get_async_db():
            # 获取所有订单ID
            result = await db.execute("SELECT id FROM live_orders")
            order_ids = [row[0] for row in result]

            if not order_ids:
                logger.warning("没有找到订单，跳过生成订单日志")
                return

            for i in range(count):
                order_id = random.choice(order_ids)
                action = random.choice(["create", "update", "fill", "cancel"])
                message = f"Order {action} action - Mock log {i}"

                log = LiveOrderLog(
                    order_id=order_id,
                    action=action,
                    message=message,
                    from_status=random.choice(list(OrderStatus)) if action in ["update", "cancel"] else None,
                    to_status=random.choice(list(OrderStatus)) if action in ["update", "fill", "cancel"] else None,
                    metadata={"test": True, "index": i}
                )

                db.add(log)

            await db.commit()
            logger.info("订单日志生成完成")

    async def generate_trade_operations(self, count: int = 150):
        """生成交易操作日志"""
        logger.info(f"生成 {count} 条交易操作日志...")

        async for db in get_async_db():
            for i in range(count):
                operation = TradeOperationLog(
                    operation_type=random.choice(["enter", "exit", "modify"]),
                    symbol=random.choice(self.symbols),
                    side=random.choice(["long", "short"]),
                    amount=random.uniform(0.01, 0.1) if random.random() > 0.5 else random.uniform(1, 10),
                    price=round(random.uniform(30000, 70000), 2) if random.random() > 0.5 else None,
                    status=random.choice(["success", "failed"]),
                    message=f"Trade operation {i}",
                    ft_order_id=random.randint(10000, 99999),
                    error_code=random.choice(["ERR_001", "ERR_002", None]) if random.random() > 0.8 else None
                )

                db.add(operation)

            await db.commit()
            logger.info("交易操作日志生成完成")

    async def generate_last_decisions(self):
        """生成最后决策缓存"""
        logger.info("生成最后决策缓存...")

        async for db in get_async_db():
            for symbol in self.symbols:
                # 检查是否已存在
                existing = await db.get(LastDecision, symbol)
                if existing:
                    continue

                decision = LastDecision(
                    symbol=symbol,
                    decision_id=f"last_decision_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    action=random.choice(self.actions),
                    status=DecisionStatus.PENDING,
                    confidence=random.randint(70, 90),
                    position_size_usd=random.uniform(100, 500),
                    metadata={"cache": True, "timestamp": datetime.utcnow().isoformat()}
                )

                db.add(decision)

            await db.commit()
            logger.info("最后决策缓存生成完成")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="生成模拟数据")
    parser.add_argument("--decisions", type=int, default=100, help="决策数量")
    parser.add_argument("--order-logs", type=int, default=200, help="订单日志数量")
    parser.add_argument("--trade-ops", type=int, default=150, help="交易操作数量")
    parser.add_argument("--all", action="store_true", help="生成所有类型的数据")

    args = parser.parse_args()

    # 初始化数据库
    init_database()

    generator = MockDataGenerator()

    if args.all:
        await generator.generate_decisions(args.decisions)
        await generator.generate_order_logs(args.order_logs)
        await generator.generate_trade_operations(args.trade_ops)
        await generator.generate_last_decisions()
    else:
        if args.decisions > 0:
            await generator.generate_decisions(args.decisions)
        if args.order_logs > 0:
            await generator.generate_order_logs(args.order_logs)
        if args.trade_ops > 0:
            await generator.generate_trade_operations(args.trade_ops)

    logger.info("模拟数据生成完成！")


if __name__ == "__main__":
    asyncio.run(main())