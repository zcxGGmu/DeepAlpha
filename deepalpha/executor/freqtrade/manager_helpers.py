"""Freqtrade管理器辅助函数
对应Go版本的 manager_helpers.go
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
import json

from deepalpha.executor.freqtrade.client import FreqtradeClient
from deepalpha.executor.freqtrade.types import FreqtradeStatus, FreqtradeBalance
from deepalpha.executor.freqtrade.sync import FreqtradeSync
from deepalpha.executor.freqtrade.tiers_update import TiersUpdater
from deepalpha.executor.freqtrade.pending import PendingOrderManager
from deepalpha.executor.freqtrade.price import PriceProcessor
from deepalpha.db.models import LiveOrder
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class FreqtradeManagerHelpers:
    """Freqtrade管理器辅助类"""

    def __init__(self, client: FreqtradeClient, db_session):
        self.client = client
        self.db = db_session
        self.sync_manager = None
        self.tiers_updater = None
        self.pending_manager = PendingOrderManager()
        self.price_processor = PriceProcessor(client)

    async def initialize(self):
        """初始化辅助管理器"""
        from deepalpha.executor.freqtrade.sync import FreqtradeSync
        from deepalpha.executor.freqtrade.tiers_update import TiersUpdater

        self.sync_manager = FreqtradeSync(self.client, self.db)
        self.tiers_updater = TiersUpdater(self.client, self.db)

        logger.info("Freqtrade管理器辅助模块初始化完成")

    async def start_background_tasks(self):
        """启动后台任务"""
        # 启动同步任务
        sync_task = asyncio.create_task(self.sync_manager.start_sync())

        # 启动层级监控任务
        tiers_task = asyncio.create_task(self.tiers_updater.start_monitoring())

        # 启动待处理订单管理器
        pending_task = asyncio.create_task(
            self.pending_manager.start_processing(self._process_pending_order)
        )

        # 启动定期清理任务
        cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("所有后台任务已启动")

        return [sync_task, tiers_task, pending_task, cleanup_task]

    async def stop_background_tasks(self):
        """停止后台任务"""
        if self.tiers_updater:
            # TiersUpdater 没有停止方法，这里只是记录
            logger.info("停止层级监控")

        if self.pending_manager:
            self.pending_manager.stop_processing()
            logger.info("停止待处理订单管理器")

        if self.sync_manager:
            # Sync 没有停止方法，这里只是记录
            logger.info("停止同步任务")

    async def _process_pending_order(self, order) -> Optional[Dict]:
        """处理待处理订单"""
        try:
            logger.info(f"处理待处理订单: {order.id} - {order.pair} {order.side} {order.amount}")

            # 调用Freqtrade创建订单
            result = await self.client.create_order(
                pair=order.pair,
                ordertype=order.ordertype,
                side=order.side,
                amount=order.amount,
                price=order.price
            )

            if result:
                logger.info(f"订单创建成功: {result.get('id')}")
            else:
                logger.error(f"订单创建失败: {order.id}")

            return result
        except Exception as e:
            logger.error(f"处理订单失败: {e}")
            return None

    async def _cleanup_loop(self):
        """定期清理循环"""
        while True:
            try:
                # 每小时清理一次旧订单
                await asyncio.sleep(3600)
                await self.pending_manager.cleanup_old_orders()
                await self._cleanup_old_data()
            except Exception as e:
                logger.error(f"清理任务异常: {e}")

    async def _cleanup_old_data(self):
        """清理旧数据"""
        cutoff_date = datetime.utcnow() - timedelta(days=30)

        # 清理旧的订单日志
        from deepalpha.db.models import LiveOrderLog

        deleted_count = self.db.query(LiveOrderLog).filter(
            LiveOrderLog.created_at < cutoff_date
        ).delete()

        if deleted_count > 0:
            self.db.commit()
            logger.info(f"清理了 {deleted_count} 条旧订单日志")

    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {}

        # Freqtrade状态
        try:
            ft_status = await self.client.get_status()
            status["freqtrade"] = {
                "state": ft_status.state,
                "strategy": getattr(ft_status, "strategy", None),
                "bot_name": getattr(ft_status, "bot_name", None),
                "running": ft_status.state == "running"
            }
        except Exception as e:
            status["freqtrade"] = {"error": str(e)}

        # 版本信息
        try:
            version = await self.client.get_version()
            status["version"] = version
        except:
            status["version"] = {"error": "Failed to get version"}

        # 同步状态
        if self.sync_manager:
            sync_status = await self.sync_manager.get_sync_status()
            status["sync"] = sync_status

        # 待处理订单统计
        pending_stats = await self.pending_manager.get_stats()
        status["pending_orders"] = pending_stats

        # 层级统计
        if self.tiers_updater:
            status["tiers"] = {
                "pending_count": self.tiers_updater.get_pending_tiers_count()
            }

        return status

    async def execute_with_retry(
        self,
        func: Callable,
        max_retries: int = 3,
        delay: float = 1.0
    ):
        """带重试的执行函数"""
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"执行失败，重试 {attempt + 1}/{max_retries}: {e}")
                await asyncio.sleep(delay * (2 ** attempt))

    async def validate_order_params(
        self,
        pair: str,
        amount: float,
        price: Optional[float] = None,
        ordertype: str = "market"
    ) -> Dict[str, Any]:
        """验证订单参数"""
        errors = []

        # 检查交易对格式
        if "/" not in pair:
            errors.append("Invalid pair format, should be BASE/QUOTE")

        # 检查数量
        if amount <= 0:
            errors.append("Amount must be positive")

        # 检查限价单价格
        if ordertype == "limit" and (price is None or price <= 0):
            errors.append("Limit order must have a positive price")

        # 获取市场信息验证
        try:
            ticker = await self.client.get_ticker(pair)
            current_price = float(ticker.last)

            # 检查价格是否合理
            if price:
                price_diff = abs(price - current_price) / current_price
                if price_diff > 0.1:  # 10%
                    errors.append(f"Price deviates too much from market price: {price_diff:.2%}")

            # 检查最小订单金额
            min_amount = 10 / current_price  # 假设最小10 USDT
            if amount < min_amount:
                errors.append(f"Amount too small, minimum: {min_amount:.6f}")

        except Exception as e:
            errors.append(f"Failed to get market info: {e}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "current_price": current_price if 'current_price' in locals() else None
        }

    async def calculate_order_metrics(self, order_id: int) -> Dict[str, Any]:
        """计算订单指标"""
        order = self.db.query(LiveOrder).filter(LiveOrder.id == order_id).first()
        if not order:
            return {"error": "Order not found"}

        metrics = {
            "order_id": order_id,
            "symbol": order.symbol,
            "status": order.status.value,
            "created_at": order.created_at.isoformat(),
        }

        # 计算执行时间
        if order.closed_at:
            execution_time = order.closed_at - order.created_at
            metrics["execution_time_seconds"] = execution_time.total_seconds()

        # 计算盈亏
        if order.realized_pnl is not None:
            metrics["realized_pnl"] = float(order.realized_pnl)

            # 计算盈亏率
            if order.entry_price and order.filled_amount:
                pnl_rate = order.realized_pnl / (order.entry_price * order.filled_amount) * 100
                metrics["pnl_percent"] = pnl_rate

        # 获取当前价格（如果订单未关闭）
        if not order.closed_at:
            try:
                current_price = await self.price_processor.get_current_price(order.symbol)
                unrealized_pnl = self.price_processor.calculate_unrealized_pnl(
                    Decimal(str(order.entry_price or 0)),
                    current_price,
                    order.filled_amount or 0,
                    order.side.value
                )
                metrics["current_price"] = float(current_price)
                metrics["unrealized_pnl"] = float(unrealized_pnl)
            except:
                pass

        return metrics

    async def get_daily_stats(self) -> Dict[str, Any]:
        """获取每日统计"""
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        # 查询今日订单
        orders = self.db.query(LiveOrder).filter(
            LiveOrder.created_at >= today_start
        ).all()

        stats = {
            "date": today_start.date().isoformat(),
            "total_orders": len(orders),
            "filled_orders": 0,
            "cancelled_orders": 0,
            "total_pnl": 0,
            "total_volume": 0,
            "symbols": {}
        }

        symbol_stats = {}

        for order in orders:
            symbol = order.symbol

            # 初始化交易对统计
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    "orders": 0,
                    "volume": 0,
                    "pnl": 0
                }

            symbol_stats[symbol]["orders"] += 1

            if order.status.value == "filled":
                stats["filled_orders"] += 1
            elif order.status.value == "cancelled":
                stats["cancelled_orders"] += 1

            # 计算成交量
            if order.filled_amount and order.price:
                volume = order.filled_amount * order.price
                stats["total_volume"] += volume
                symbol_stats[symbol]["volume"] += volume

            # 累计盈亏
            if order.realized_pnl:
                stats["total_pnl"] += order.realized_pnl
                symbol_stats[symbol]["pnl"] += order.realized_pnl

        stats["symbols"] = symbol_stats
        return stats

    async def export_orders_to_csv(self, date_from: datetime = None, date_to: datetime = None) -> str:
        """导出订单到CSV"""
        import csv
        import io

        query = self.db.query(LiveOrder)

        if date_from:
            query = query.filter(LiveOrder.created_at >= date_from)
        if date_to:
            query = query.filter(LiveOrder.created_at <= date_to)

        orders = query.all()

        # 创建CSV内容
        output = io.StringIO()
        writer = csv.writer(output)

        # 写入标题
        writer.writerow([
            "ID", "Symbol", "Side", "Type", "Amount", "Price",
            "Filled Amount", "Filled Price", "Status", "PnL",
            "Created At", "Closed At"
        ])

        # 写入数据
        for order in orders:
            writer.writerow([
                order.id,
                order.symbol,
                order.side.value,
                order.order_type.value,
                order.amount,
                order.price,
                order.filled_amount,
                order.filled_price,
                order.status.value,
                order.realized_pnl,
                order.created_at.isoformat(),
                order.closed_at.isoformat() if order.closed_at else ""
            ])

        return output.getvalue()

    async def backup_data(self, backup_dir: str = "./backups"):
        """备份数据"""
        import os

        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        backup_files = {}

        # 备份订单
        orders_csv = await self.export_orders_to_csv()
        orders_file = os.path.join(backup_dir, f"orders_{timestamp}.csv")
        with open(orders_file, "w") as f:
            f.write(orders_csv)
        backup_files["orders"] = orders_file

        # 备份配置
        try:
            config = await self.client.get_config()
            config_file = os.path.join(backup_dir, f"freqtrade_config_{timestamp}.json")
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            backup_files["config"] = config_file
        except:
            pass

        logger.info(f"数据备份完成: {backup_files}")
        return backup_files

    async def check_health(self) -> Dict[str, Any]:
        """健康检查"""
        health = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.utcnow().isoformat()
        }

        # 检查Freqtrade连接
        try:
            status = await self.client.get_status()
            health["checks"]["freqtrade"] = {
                "status": "healthy",
                "state": status.state
            }
        except Exception as e:
            health["checks"]["freqtrade"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "unhealthy"

        # 检查数据库连接
        try:
            count = self.db.query(LiveOrder).count()
            health["checks"]["database"] = {
                "status": "healthy",
                "orders_count": count
            }
        except Exception as e:
            health["checks"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "unhealthy"

        # 检查待处理队列
        pending_stats = await self.pending_manager.get_stats()
        if pending_stats["failed"] > 10:
            health["checks"]["pending_orders"] = {
                "status": "warning",
                "failed_count": pending_stats["failed"]
            }
            if health["status"] == "healthy":
                health["status"] = "warning"
        else:
            health["checks"]["pending_orders"] = {
                "status": "healthy",
                "stats": pending_stats
            }

        return health