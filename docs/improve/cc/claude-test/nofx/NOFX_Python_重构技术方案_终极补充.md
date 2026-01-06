# NOFX Python 重构技术方案：终极补充（设计模式与最佳实践）

> 本文档是前三个文档的最终补充，包含设计模式、错误处理、测试策略、性能调优参数、故障恢复、数据迁移、CI/CD 配置等生产级实现细节。

---

## 十八、设计模式与架构最佳实践

### 18.1 核心设计模式应用

#### 18.1.1 仓储模式 (Repository Pattern)

```python
# backend/repositories/base.py
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Optional, Any
from uuid import UUID
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


class BaseRepository(Generic[T], ABC):
    """基础仓储类"""

    def __init__(self, session: AsyncSession, model: Type[T]):
        self.session = session
        self.model = model

    async def get_by_id(self, id: UUID | str) -> Optional[T]:
        """通过 ID 获取实体"""
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        offset: int = 0,
        limit: int = 100,
        **filters,
    ) -> list[T]:
        """获取所有实体"""
        query = select(self.model)

        # 应用过滤条件
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        # 应用分页
        query = query.offset(offset).limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def create(self, entity: T) -> T:
        """创建实体"""
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def create_batch(self, entities: list[T]) -> list[T]:
        """批量创建实体"""
        self.session.add_all(entities)
        await self.session.flush()
        return entities

    async def update(self, entity: T) -> T:
        """更新实体"""
        await self.session.merge(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def update_by_id(
        self,
        id: UUID | str,
        **kwargs,
    ) -> Optional[T]:
        """通过 ID 更新实体"""
        # 检查实体是否存在
        entity = await self.get_by_id(id)
        if not entity:
            return None

        # 更新字段
        for key, value in kwargs.items():
            if hasattr(entity, key):
                setattr(entity, key, value)

        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def delete(self, entity: T) -> bool:
        """删除实体"""
        await self.session.delete(entity)
        await self.session.flush()
        return True

    async def delete_by_id(self, id: UUID | str) -> bool:
        """通过 ID 删除实体"""
        entity = await self.get_by_id(id)
        if not entity:
            return False

        return await self.delete(entity)

    async def exists(self, **filters) -> bool:
        """检查实体是否存在"""
        query = select(self.model.id)

        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        result = await self.session.execute(query)
        return result.scalar_one_or_none() is not None

    async def count(self, **filters) -> int:
        """统计实体数量"""
        from sqlalchemy import func

        query = select(func.count(self.model.id))

        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        result = await self.session.execute(query)
        return result.scalar_one()


# ========== 具体仓储实现 ==========

class TraderRepository(BaseRepository):
    """交易员仓储"""

    def __init__(self, session: AsyncSession):
        from backend.models.database import Trader
        super().__init__(session, Trader)

    async def get_by_user_id(
        self,
        user_id: str,
        market_type: Optional[str] = None,
        is_running: Optional[bool] = None,
    ) -> list:
        """获取用户的交易员列表"""
        from backend.models.database import Trader

        query = select(Trader).where(Trader.user_id == user_id)

        if market_type:
            query = query.where(Trader.market_type == market_type)

        if is_running is not None:
            query = query.where(Trader.is_running == is_running)

        query = query.order_by(Trader.created_at.desc())

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_running_traders(self) -> list:
        """获取所有运行中的交易员"""
        from backend.models.database import Trader

        result = await self.session.execute(
            select(Trader).where(Trader.is_running == True)
        )
        return result.scalars().all()

    async def update_performance(
        self,
        trader_id: str,
        total_pnl: float,
        total_pnl_pct: float,
        win_rate: float,
    ):
        """更新交易员性能指标"""
        from backend.models.database import Trader

        await self.session.execute(
            update(Trader)
            .where(Trader.id == trader_id)
            .values(
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                win_rate=win_rate,
            )
        )
        await self.session.flush()


class DecisionRepository(BaseRepository):
    """决策仓储"""

    def __init__(self, session: AsyncSession):
        from backend.models.database import Decision
        super().__init__(session, Decision)

    async def get_recent_decisions(
        self,
        trader_id: str,
        limit: int = 20,
    ) -> list:
        """获取最近的决策记录"""
        from backend.models.database import Decision

        result = await self.session.execute(
            select(Decision)
            .where(Decision.trader_id == trader_id)
            .order_by(Decision.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def get_decisions_by_date_range(
        self,
        trader_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list:
        """获取日期范围内的决策"""
        from backend.models.database import Decision

        result = await self.session.execute(
            select(Decision)
            .where(
                Decision.trader_id == trader_id,
                Decision.created_at >= start_date,
                Decision.created_at <= end_date,
            )
            .order_by(Decision.created_at.desc())
        )
        return result.scalars().all()

    async def get_decision_stats(
        self,
        trader_id: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """获取决策统计信息"""
        from datetime import timedelta
        from sqlalchemy import func, case
        from backend.models.database import Decision

        start_date = datetime.now() - timedelta(days=days)

        # 总决策数
        total_result = await self.session.execute(
            select(func.count(Decision.id))
            .where(
                Decision.trader_id == trader_id,
                Decision.created_at >= start_date,
            )
        )
        total_count = total_result.scalar_one()

        # 按状态分组统计
        status_result = await self.session.execute(
            select(
                Decision.status,
                func.count(Decision.id).label('count'),
            )
            .where(
                Decision.trader_id == trader_id,
                Decision.created_at >= start_date,
            )
            .group_by(Decision.status)
        )
        status_counts = {row.status: row.count for row in status_result.all()}

        # 按操作类型分组
        action_stats = await self.session.execute(
            select(
                func.json_array_elements_text(Decision.decisions).label('action'),
                func.count().label('count'),
            )
            .where(
                Decision.trader_id == trader_id,
                Decision.created_at >= start_date,
            )
            .group_by('action')
        )
        # 处理 action 统计...

        return {
            "total_decisions": total_count,
            "status_counts": status_counts,
            "avg_duration": 0,  # 可以添加平均执行时间
        }
```

#### 18.1.2 工厂模式 (Factory Pattern)

```python
# backend/factories/__init__.py
from abc import ABC, abstractmethod
from typing import Any

from backend.llm.base import BaseLLMClient
from backend.llm.deepseek import DeepSeekClient
from backend.llm.qwen import QwenClient
from backend.markets.base import BaseMarketDataProvider
from backend.markets.atrader.tushare import TushareDataProvider
from backend.markets.atrader.akshare import AkShareDataProvider
from backend.traders.base import BaseTrader
from backend.traders.atrader.simulator import ATraderSimulator
from backend.traders.hkstock.simulator import HKStockSimulator


class LLMClientFactory:
    """LLM 客户端工厂"""

    _clients = {
        "deepseek": DeepSeekClient,
        "qwen": QwenClient,
    }

    @classmethod
    def create(
        cls,
        provider: str,
        **kwargs,
    ) -> BaseLLMClient:
        """创建 LLM 客户端"""
        client_class = cls._clients.get(provider)

        if not client_class:
            raise ValueError(f"Unknown LLM provider: {provider}")

        return client_class(**kwargs)

    @classmethod
    def register(cls, provider: str, client_class: type):
        """注册新的 LLM 客户端"""
        cls._clients[provider] = client_class


class MarketDataProviderFactory:
    """市场数据提供者工厂"""

    _providers = {
        "tushare": TushareDataProvider,
        "akshare": AkShareDataProvider,
    }

    @classmethod
    def create(
        cls,
        provider: str,
        **kwargs,
    ) -> BaseMarketDataProvider:
        """创建市场数据提供者"""
        provider_class = cls._providers.get(provider)

        if not provider_class:
            raise ValueError(f"Unknown market data provider: {provider}")

        return provider_class(**kwargs)

    @classmethod
    def create_chain(
        cls,
        providers: list[str],
        **kwargs,
    ) -> BaseMarketDataProvider:
        """创建数据源链（主备切换）"""
        from backend.markets.chain import ChainedDataProvider

        providers_list = []
        for provider_name in providers:
            provider = cls.create(provider_name, **kwargs)
            providers_list.append(provider)

        return ChainedDataProvider(providers_list)


class TraderFactory:
    """交易员工厂"""

    @classmethod
    def create_trader(
        cls,
        market_type: str,
        trader_type: str = "simulator",
        **kwargs,
    ) -> BaseTrader:
        """创建交易员"""
        market_map = {
            "a_stock": ATraderSimulator,
            "hk_stock": HKStockSimulator,
        }

        trader_class = market_map.get(market_type)

        if not trader_class:
            raise ValueError(f"Unknown market type: {market_type}")

        return trader_class(**kwargs)
```

#### 18.1.3 策略模式 (Strategy Pattern)

```python
# backend/strategies/__init__.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from backend.core.trading_types import DecisionContext, Decision


class ExecutionStrategy(ABC):
    """执行策略抽象基类"""

    @abstractmethod
    async def execute(
        self,
        decisions: list[Decision],
        context: DecisionContext,
    ) -> list[dict]:
        """执行决策"""
        pass


class ImmediateExecutionStrategy(ExecutionStrategy):
    """立即执行策略"""

    async def execute(
        self,
        decisions: list[Decision],
        context: DecisionContext,
    ) -> list[dict]:
        """立即执行所有决策"""
        results = []

        for decision in decisions:
            try:
                if decision.action == "buy":
                    result = await self._execute_buy(decision, context)
                elif decision.action == "sell":
                    result = await self._execute_sell(decision, context)
                elif decision.action == "close":
                    result = await self._execute_close(decision, context)
                else:
                    result = {"status": "skipped", "reason": f"Action {decision.action} not executed"}

                results.append(result)

            except Exception as e:
                results.append({
                    "status": "error",
                    "decision": decision.symbol,
                    "error": str(e),
                })

        return results

    async def _execute_buy(self, decision: Decision, context: DecisionContext):
        """执行买入"""
        # 获取交易员
        trader = context.trader

        # 开仓
        result = await trader.open_position(
            symbol=decision.symbol,
            side="long",
            quantity=decision.quantity,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
        )

        return {
            "status": "success",
            "action": "buy",
            "symbol": decision.symbol,
            "order_id": result.get("order_id"),
        }

    async def _execute_sell(self, decision: Decision, context: DecisionContext):
        """执行卖出"""
        trader = context.trader

        # 平仓
        result = await trader.close_position(
            symbol=decision.symbol,
            side="long",
            quantity=decision.quantity,
        )

        return {
            "status": "success",
            "action": "sell",
            "symbol": decision.symbol,
            "pnl": result.get("pnl"),
        }

    async def _execute_close(self, decision: Decision, context: DecisionContext):
        """执行平仓"""
        # 与卖出类似
        return await self._execute_sell(decision, context)


class CautiousExecutionStrategy(ExecutionStrategy):
    """谨慎执行策略（需要确认）"""

    def __init__(self, approval_threshold: float = 0.8):
        self.approval_threshold = approval_threshold

    async def execute(
        self,
        decisions: list[Decision],
        context: DecisionContext,
    ) -> list[dict]:
        """谨慎执行决策（需要高置信度）"""
        results = []

        for decision in decisions:
            # 检查置信度
            if decision.confidence and decision.confidence < self.approval_threshold * 100:
                results.append({
                    "status": "skipped",
                    "symbol": decision.symbol,
                    "reason": f"置信度 {decision.confidence}% 低于阈值 {self.approval_threshold * 100}%",
                })
                continue

            # 执行决策
            if decision.action == "buy":
                result = await self._execute_buy(decision, context)
            elif decision.action == "sell":
                result = await self._execute_sell(decision, context)
            else:
                result = {"status": "skipped", "reason": f"Action {decision.action} not executed"}

            results.append(result)

        return results


class StagedExecutionStrategy(ExecutionStrategy):
    """分批执行策略"""

    def __init__(self, batch_size: float = 0.3, batch_interval: int = 60):
        """
        Args:
            batch_size: 每批执行比例（0-1）
            batch_interval: 批次间隔（秒）
        """
        self.batch_size = batch_size
        self.batch_interval = batch_interval

    async def execute(
        self,
        decisions: list[Decision],
        context: DecisionContext,
    ) -> list[dict]:
        """分批执行决策"""
        results = []
        immediate = ImmediateExecutionStrategy()

        for decision in decisions:
            if decision.action in ("buy", "sell"):
                # 计算批次
                total_quantity = decision.quantity
                batch_quantity = int(total_quantity * self.batch_size)

                # 确保是100的整数倍（A股）
                batch_quantity = (batch_quantity // 100) * 100

                executed_quantity = 0
                batch_results = []

                while executed_quantity < total_quantity:
                    # 计算本批次数量
                    current_batch = min(
                        batch_quantity,
                        total_quantity - executed_quantity,
                    )

                    if current_batch == 0:
                        break

                    # 创建批次决策
                    batch_decision = Decision(
                        symbol=decision.symbol,
                        action=decision.action,
                        quantity=current_batch,
                        confidence=decision.confidence,
                        reasoning=decision.reasoning,
                    )

                    # 执行批次
                    batch_result = await immediate.execute(
                        [batch_decision],
                        context,
                    )
                    batch_results.append(batch_result)

                    executed_quantity += current_batch

                    # 如果不是最后一批，等待间隔时间
                    if executed_quantity < total_quantity:
                        await asyncio.sleep(self.batch_interval)

                results.append({
                    "status": "success",
                    "action": decision.action,
                    "symbol": decision.symbol,
                    "total_quantity": executed_quantity,
                    "batches": len(batch_results),
                })

            else:
                # 其他操作直接执行
                result = await immediate.execute([decision], context)
                results.extend(result)

        return results
```

#### 18.1.4 观察者模式 (Observer Pattern)

```python
# backend/events/__init__.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Awaitable
from enum import Enum
import asyncio
from datetime import datetime

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """事件类型"""
    # 交易员事件
    TRADER_CREATED = "trader.created"
    TRADER_STARTED = "trader.started"
    TRADER_STOPPED = "trader.stopped"
    TRADER_PAUSED = "trader.paused"
    TRADER_RESUMED = "trader.resumed"
    TRADER_DELETED = "trader.deleted"

    # 决策事件
    DECISION_REQUESTED = "decision.requested"
    DECISION_MAKING = "decision.making"
    DECISION_COMPLETED = "decision.completed"
    DECISION_FAILED = "decision.failed"

    # 交易事件
    ORDER_PLACED = "order.placed"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_FAILED = "order.failed"

    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_UPDATED = "position.updated"

    # 风险事件
    RISK_LIMIT_EXCEEDED = "risk.limit_exceeded"
    DRAWDOWN_WARNING = "risk.drawdown_warning"
    MARGIN_CALL = "risk.margin_call"

    # 市场事件
    MARKET_DATA_UPDATED = "market.data_updated"
    MARKET_OPENED = "market.opened"
    MARKET_CLOSED = "market.closed"

    # 系统事件
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"


@dataclass
class Event:
    """事件对象"""
    type: EventType
    data: dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventListener(ABC):
    """事件监听器抽象基类"""

    @abstractmethod
    async def handle(self, event: Event) -> None:
        """处理事件"""
        pass


class EventEmitter:
    """事件发射器"""

    def __init__(self):
        self._listeners: dict[EventType, list[EventListener]] = {}
        self._event_queue: asyncio.Queue[Event] | None = None
        self._processor_task: asyncio.Task | None = None

    def subscribe(self, event_type: EventType, listener: EventListener):
        """订阅事件"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []

        self._listeners[event_type].append(listener)
        logger.debug(f"Subscribed to {event_type.value}: {listener.__class__.__name__}")

    def unsubscribe(self, event_type: EventType, listener: EventListener):
        """取消订阅"""
        if event_type in self._listeners:
            try:
                self._listeners[event_type].remove(listener)
                logger.debug(f"Unsubscribed from {event_type.value}: {listener.__class__.__name__}")
            except ValueError:
                pass

    async def emit(self, event: Event):
        """发射事件"""
        logger.debug(f"Emitting event: {event.type.value}")

        # 异步处理事件
        if self._event_queue:
            await self._event_queue.put(event)
        else:
            await self._process_event(event)

    async def _process_event(self, event: Event):
        """处理事件"""
        listeners = self._listeners.get(event.type, [])

        if not listeners:
            return

        # 并发通知所有监听器
        tasks = [
            self._safe_handle(listener, event)
            for listener in listeners
        ]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_handle(self, listener: EventListener, event: Event):
        """安全地调用监听器"""
        try:
            await listener.handle(event)
        except Exception as e:
            logger.error(
                f"Error in {listener.__class__.__name__} handling {event.type.value}: {e}",
                exc_info=True,
            )

    async def start_processor(self):
        """启动事件处理器"""
        if self._processor_task is not None:
            return

        self._event_queue = asyncio.Queue()
        self._processor_task = asyncio.create_task(self._process_loop())

        logger.info("Event processor started")

    async def stop_processor(self):
        """停止事件处理器"""
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

            self._processor_task = None
            logger.info("Event processor stopped")

    async def _process_loop(self):
        """事件处理循环"""
        while True:
            try:
                event = await self._event_queue.get()
                await self._process_event(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}", exc_info=True)


# 全局事件发射器
event_emitter = EventEmitter()


# ========== 具体监听器实现 ==========

class DecisionEventListener(EventListener):
    """决策事件监听器"""

    async def handle(self, event: Event) -> None:
        """处理决策事件"""
        if event.type == EventType.DECISION_REQUESTED:
            await self._on_decision_requested(event)
        elif event.type == EventType.DECISION_COMPLETED:
            await self._on_decision_completed(event)
        elif event.type == EventType.DECISION_FAILED:
            await self._on_decision_failed(event)

    async def _on_decision_requested(self, event: Event):
        """决策请求事件"""
        trader_id = event.data.get("trader_id")
        logger.info(f"Decision requested for trader: {trader_id}")

    async def _on_decision_completed(self, event: Event):
        """决策完成事件"""
        trader_id = event.data.get("trader_id")
        decision_count = event.data.get("decision_count", 0)
        logger.info(f"Decision completed for trader: {trader_id}, decisions: {decision_count}")

    async def _on_decision_failed(self, event: Event):
        """决策失败事件"""
        trader_id = event.data.get("trader_id")
        error = event.data.get("error")
        logger.error(f"Decision failed for trader: {trader_id}, error: {error}")


class RiskEventListener(EventListener):
    """风险事件监听器"""

    async def handle(self, event: Event) -> None:
        """处理风险事件"""
        if event.type == EventType.RISK_LIMIT_EXCEEDED:
            await self._on_risk_limit_exceeded(event)
        elif event.type == EventType.DRAWDOWN_WARNING:
            await self._on_drawdown_warning(event)

    async def _on_risk_limit_exceeded(self, event: Event):
        """风险限制触发事件"""
        trader_id = event.data.get("trader_id")
        limit_type = event.data.get("limit_type")
        current_value = event.data.get("current_value")

        logger.warning(
            f"Risk limit exceeded for trader: {trader_id}, "
            f"type: {limit_type}, value: {current_value}"
        )

        # 发送告警
        from backend.monitoring.notifier import alert_manager, Alert, AlertSeverity

        alert = Alert(
            title=f"风险限制触发: {limit_type}",
            description=f"交易员 {trader_id} 触发 {limit_type} 限制",
            severity=AlertSeverity.WARNING,
            labels={"trader_id": trader_id, "limit_type": limit_type},
            value=current_value,
        )

        await alert_manager.send_alert(alert)

    async def _on_drawdown_warning(self, event: Event):
        """回撤警告事件"""
        trader_id = event.data.get("trader_id")
        drawdown = event.data.get("drawdown")

        logger.warning(
            f"Drawdown warning for trader: {trader_id}, "
            f"drawdown: {drawdown:.2%}"
        )


class MetricsEventListener(EventListener):
    """指标事件监听器"""

    def __init__(self):
        from backend.monitoring.metrics import (
            decisions_total,
            decision_duration_seconds,
            trades_total,
            account_equity,
        )

        self.decisions_total = decisions_total
        self.decision_duration = decision_duration_seconds
        self.trades_total = trades_total
        self.account_equity = account_equity

    async def handle(self, event: Event) -> None:
        """处理指标事件"""
        if event.type == EventType.DECISION_COMPLETED:
            self._on_decision_completed(event)
        elif event.type == EventType.ORDER_FILLED:
            self._on_order_filled(event)
        elif event.type == EventType.POSITION_UPDATED:
            self._on_position_updated(event)

    def _on_decision_completed(self, event: Event):
        """决策完成 - 更新指标"""
        trader_id = event.data.get("trader_id")
        duration_ms = event.data.get("duration_ms", 0)
        action_counts = event.data.get("action_counts", {})

        # 更新决策计数
        for action, count in action_counts.items():
            self.decisions_total.labels(
                trader_id=trader_id,
                action=action,
            ).inc(count)

        # 更新决策延迟
        self.decision_duration.labels(
            trader_id=trader_id,
        ).observe(duration_ms / 1000)

    def _on_order_filled(self, event: Event):
        """订单成交 - 更新指标"""
        trader_id = event.data.get("trader_id")
        symbol = event.data.get("symbol")
        side = event.data.get("side")
        quantity = event.data.get("quantity")
        value = event.data.get("value")

        # 更新交易计数
        self.trades_total.labels(
            trader_id=trader_id,
            symbol=symbol,
            side=side,
        ).inc()

        # 更新交易金额
        if value:
            from backend.monitoring.metrics import trades_value_total
            trades_value_total.labels(
                trader_id=trader_id,
                symbol=symbol,
            ).inc(value)

    def _on_position_updated(self, event: Event):
        """持仓更新 - 更新指标"""
        trader_id = event.data.get("trader_id")
        total_equity = event.data.get("total_equity")

        if total_equity:
            self.account_equity.labels(
                trader_id=trader_id,
            ).set(total_equity)
```

#### 18.1.5 责任链模式 (Chain of Responsibility)

```python
# backend/chains/__init__.py
from abc import ABC, abstractmethod
from typing import Optional, Any

from backend.core.trading_types import Decision, DecisionContext
from backend.exceptions import ValidationError, RiskLimitError


class DecisionHandler(ABC):
    """决策处理器抽象基类"""

    def __init__(self):
        self._next_handler: Optional[DecisionHandler] = None

    def set_next(self, handler: 'DecisionHandler') -> 'DecisionHandler':
        """设置下一个处理器"""
        self._next_handler = handler
        return handler

    @abstractmethod
    async def handle(
        self,
        decision: Decision,
        context: DecisionContext,
    ) -> tuple[bool, Optional[str]]:
        """
        处理决策

        Returns:
            (是否继续处理, 拒绝原因)
        """
        pass

    async def process(
        self,
        decision: Decision,
        context: DecisionContext,
    ) -> tuple[bool, Optional[str]]:
        """处理决策（通过责任链）"""
        # 当前处理器处理
        should_continue, reason = await self.handle(decision, context)

        # 如果被拒绝，返回
        if not should_continue:
            return False, reason

        # 如果有下一个处理器，继续传递
        if self._next_handler:
            return await self._next_handler.process(decision, context)

        # 所有处理器都通过
        return True, None


class ValidationHandler(DecisionHandler):
    """验证处理器"""

    async def handle(
        self,
        decision: Decision,
        context: DecisionContext,
    ) -> tuple[bool, Optional[str]]:
        """验证决策"""
        # 检查必填字段
        if not decision.symbol:
            return False, "缺少股票代码"

        if not decision.action:
            return False, "缺少操作类型"

        # 检查操作类型
        valid_actions = {"buy", "sell", "hold", "close"}
        if decision.action not in valid_actions:
            return False, f"无效的操作类型: {decision.action}"

        # 检查数量
        if decision.action in ("buy", "sell"):
            if not decision.quantity or decision.quantity <= 0:
                return False, "数量必须大于0"

        # 检查价格
        if decision.action == "buy":
            if not hasattr(decision, 'price') or not decision.price:
                return False, "缺少买入价格"

        return True, None


class MarketRuleHandler(DecisionHandler):
    """市场规则处理器"""

    async def handle(
        self,
        decision: Decision,
        context: DecisionContext,
    ) -> tuple[bool, Optional[str]]:
        """检查市场规则"""
        market_type = context.market_type  # 需要从 context 获取

        if market_type == "a_stock":
            return await self._check_a_stock_rules(decision, context)
        elif market_type == "hk_stock":
            return await self._check_hk_stock_rules(decision, context)

        return True, None

    async def _check_a_stock_rules(
        self,
        decision: Decision,
        context: DecisionContext,
    ) -> tuple[bool, Optional[str]]:
        """检查 A股规则"""
        from backend.core.trading_calendar import get_trading_calendar

        calendar = get_trading_calendar("a_stock")

        # 检查交易时间
        if not calendar.is_trading_time(datetime.now()):
            return False, "不在交易时间内"

        # 检查手数（A股100股=1手）
        if decision.action in ("buy", "sell"):
            if decision.quantity % 100 != 0:
                return False, f"数量必须是100的整数倍（当前：{decision.quantity}）"

        # 检查 T+1（卖出）
        if decision.action == "sell":
            from datetime import date

            for position in context.positions:
                if position.symbol == decision.symbol:
                    if position.can_sell_date and position.can_sell_date > date.today():
                        return False, f"T+1限制：{decision.symbol} 当日买入次日才能卖出"
                    break

        # 检查涨跌停
        if decision.action == "buy":
            market_data = context.market_data.get(decision.symbol)
            if market_data:
                # 计算涨跌停价
                prev_close = market_data.get("prev_close", market_data.price)
                limit_type = self._get_stock_limit_type(decision.symbol)

                if limit_type == "main_board":
                    limit_pct = 0.10
                elif limit_type in ("chi_next", "star_market"):
                    limit_pct = 0.20
                elif limit_type == "st":
                    limit_pct = 0.05
                else:
                    limit_pct = 0.10

                upper_limit = prev_close * (1 + limit_pct)

                if decision.price > upper_limit:
                    return False, f"买入价格超过涨停价 ¥{upper_limit:.2f}"

        return True, None

    async def _check_hk_stock_rules(
        self,
        decision: Decision,
        context: DecisionContext,
    ) -> tuple[bool, Optional[str]]:
        """检查港股规则"""
        # 港股没有涨跌停，T+0
        # 但需要检查交易单位
        return True, None

    def _get_stock_limit_type(self, symbol: str) -> str:
        """获取股票涨跌停类型"""
        # ST 股
        if "ST" in symbol or "st" in symbol:
            return "st"

        # 科创板
        if symbol.startswith("688"):
            return "star_market"

        # 创业板
        if symbol.startswith("30"):
            return "chi_next"

        # 主板
        return "main_board"


class RiskLimitHandler(DecisionHandler):
    """风险限制处理器"""

    def __init__(self, max_position_ratio: float = 0.2):
        self.max_position_ratio = max_position_ratio

    async def handle(
        self,
        decision: Decision,
        context: DecisionContext,
    ) -> tuple[bool, Optional[str]]:
        """检查风险限制"""
        account = context.account

        if decision.action == "buy":
            # 检查单只股票持仓限制
            position_value = decision.quantity * decision.price
            position_ratio = position_value / account.total_equity

            if position_ratio > self.max_position_ratio:
                return False, (
                    f"持仓比例超限: {position_ratio:.2%} > {self.max_position_ratio:.2%}"
                )

            # 检查现有持仓
            for position in context.positions:
                if position.symbol == decision.symbol and position.side == "long":
                    existing_value = position.quantity * position.current_price
                    total_value = existing_value + position_value
                    total_ratio = total_value / account.total_equity

                    if total_ratio > self.max_position_ratio * 1.5:
                        return False, (
                            f"单只股票总持仓比例超限: {total_ratio:.2%} > {self.max_position_ratio * 1.5:.2%}"
                        )

        # 检查止损止盈比例
        if decision.stop_loss and decision.take_profit:
            risk_reward = (
                (decision.take_profit - decision.price)
                / (decision.price - decision.stop_loss)
                if decision.price
                else 0
            )

            if risk_reward < 2.0:
                return False, f"风险收益比不足: {risk_reward:.2f} < 2.0"

        return True, None


class MarketDataHandler(DecisionHandler):
    """市场数据处理器"""

    async def handle(
        self,
        decision: Decision,
        context: DecisionContext,
    ) -> tuple[bool, Optional[str]]:
        """检查市场数据"""
        # 确保有市场数据
        if decision.symbol not in context.market_data:
            return False, f"缺少 {decision.symbol} 的市场数据"

        market_data = context.market_data[decision.symbol]

        # 检查数据新鲜度
        if market_data.timestamp:
            data_age = (datetime.now() - market_data.timestamp).total_seconds()

            if data_age > 300:  # 5分钟
                return False, f"市场数据过期: {data_age:.0f} 秒前"

        return True, None


class ExecutionHandler(DecisionHandler):
    """执行处理器（链的末端）"""

    def __init__(self, trader_service):
        self.trader_service = trader_service

    async def handle(
        self,
        decision: Decision,
        context: DecisionContext,
    ) -> tuple[bool, Optional[str]]:
        """执行决策"""
        try:
            if decision.action == "buy":
                result = await self._execute_buy(decision, context)
            elif decision.action == "sell":
                result = await self._execute_sell(decision, context)
            elif decision.action == "close":
                result = await self._execute_close(decision, context)
            else:
                return True, None  # hold 等，跳过

            logger.info(f"Decision executed: {decision.action} {decision.symbol}")
            return True, None

        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            return False, str(e)

    async def _execute_buy(self, decision: Decision, context: DecisionContext):
        """执行买入"""
        return await self.trader_service.execute_buy(
            trader_id=context.trader_id,
            symbol=decision.symbol,
            quantity=decision.quantity,
            price=decision.price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
        )

    async def _execute_sell(self, decision: Decision, context: DecisionContext):
        """执行卖出"""
        return await self.trader_service.execute_sell(
            trader_id=context.trader_id,
            symbol=decision.symbol,
            quantity=decision.quantity,
        )

    async def _execute_close(self, decision: Decision, context: DecisionContext):
        """执行平仓"""
        return await self.trader_service.execute_close(
            trader_id=context.trader_id,
            symbol=decision.symbol,
        )


# ========== 责任链工厂 ==========

class DecisionChainFactory:
    """决策责任链工厂"""

    @staticmethod
    def create_chain(
        trader_service,
        max_position_ratio: float = 0.2,
    ) -> DecisionHandler:
        """创建决策处理责任链"""
        # 创建处理器
        validation = ValidationHandler()
        market_rules = MarketRuleHandler()
        risk_limit = RiskLimitHandler(max_position_ratio=max_position_ratio)
        market_data = MarketDataHandler()
        execution = ExecutionHandler(trader_service)

        # 构建责任链
        validation.set_next(market_rules)
        market_rules.set_next(risk_limit)
        risk_limit.set_next(market_data)
        market_data.set_next(execution)

        return validation
```

---

## 十九、错误处理和重试机制

### 19.1 完整的错误处理体系

```python
# backend/errors/handlers.py
from typing import Any, Callable, TypeVar, Optional
from functools import wraps
import asyncio
import random
from datetime import datetime

from backend.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RetryConfig:
    """重试配置"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: tuple[type] = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on


def retry(
    config: RetryConfig | None = None,
    on_retry: Callable[[Exception, int], Any] | None = None,
):
    """
    重试装饰器

    Args:
        config: 重试配置
        on_retry: 重试时的回调函数
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)

                except config.retry_on as e:
                    last_exception = e

                    # 最后一次尝试失败，抛出异常
                    if attempt == config.max_attempts - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {config.max_attempts} attempts"
                        )
                        raise

                    # 计算延迟时间
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay,
                    )

                    # 添加随机抖动
                    if config.jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{config.max_attempts}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )

                    # 调用重试回调
                    if on_retry:
                        await on_retry(e, attempt + 1)

                    # 等待后重试
                    await asyncio.sleep(delay)

            # 理论上不会到这里
            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)

                except config.retry_on as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {config.max_attempts} attempts"
                        )
                        raise

                    # 同步函数使用 time.sleep
                    import time

                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay,
                    )

                    if config.jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{config.max_attempts}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )

                    time.sleep(delay)

            raise last_exception

        # 根据函数类型返回对应的包装器
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ========== 断路器模式 ==========

class CircuitBreaker:
    """断路器"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        """
        Args:
            failure_threshold: 失败阈值
            recovery_timeout: 恢复超时（秒）
            expected_exception: 预期的异常类型
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """通过断路器调用函数"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置"""
        if self.last_failure_time is None:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def _on_success(self):
        """成功时调用"""
        self.failure_count = 0

        if self.state == "half-open":
            self.state = "closed"
            logger.info("Circuit breaker recovered, entering closed state")

    def _on_failure(self):
        """失败时调用"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


# ========== 超时处理 ==========

class Timeout:
    """超时上下文管理器"""

    def __init__(self, timeout: float):
        self.timeout = timeout

    async def __aenter__(self):
        self._task = asyncio.current_task()
        self._handle = asyncio.get_event_loop().call_later(
            self.timeout,
            self._timeout_callback,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._handle.cancel()
        if exc_type is asyncio.TimeoutError:
            self._task.cancel()

    def _timeout_callback(self):
        self._task.cancel()
        raise asyncio.TimeoutError(f"Operation timed out after {self.timeout} seconds")


async def with_timeout(coro, timeout: float):
    """带超时的异步执行"""
    async with Timeout(timeout):
        return await coro


# ========== 错误聚合 ==========

class ErrorAggregator:
    """错误聚合器"""

    def __init__(self):
        self.errors: list[dict[str, Any]] = []
        self.start_time = datetime.now()

    def add_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
    ):
        """添加错误"""
        self.errors.append({
            "type": type(error).__name__,
            "message": str(error),
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        })

    def get_summary(self) -> dict[str, Any]:
        """获取错误摘要"""
        from collections import Counter

        error_types = Counter(err["type"] for err in self.errors)

        return {
            "total_errors": len(self.errors),
            "error_types": dict(error_types),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "first_error": self.errors[0] if self.errors else None,
            "last_error": self.errors[-1] if self.errors else None,
        }

    def has_errors(self) -> bool:
        """是否有错误"""
        return len(self.errors) > 0

    def clear(self):
        """清空错误"""
        self.errors.clear()
        self.start_time = datetime.now()
```

### 19.2 具体的重试策略

```python
# backend/errors/retry_strategies.py

# ========== 数据库重试策略 ==========

db_retry_config = RetryConfig(
    max_attempts=3,
    base_delay=0.5,
    max_delay=5.0,
    exponential_base=2.0,
    retry_on=(
        ConnectionError,
        TimeoutError,
    ),
)

# ========== LLM API 重试策略 ==========

llm_retry_config = RetryConfig(
    max_attempts=5,
    base_delay=2.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    retry_on=(
        ConnectionError,
        TimeoutError,
        # API 返回的特定错误
        lambda e: "rate limit" in str(e).lower(),
        lambda e: "too many requests" in str(e).lower(),
    ),
)

# ========== 市场数据重试策略 ==========

market_data_retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    exponential_base=2.0,
    retry_on=(
        ConnectionError,
        TimeoutError,
        lambda e: "temporarily unavailable" in str(e).lower(),
    ),
)

# ========== 交易执行重试策略 ==========

trade_retry_config = RetryConfig(
    max_attempts=2,  # 交易重试次数不宜过多
    base_delay=0.1,
    max_delay=1.0,
    exponential_base=1.5,
    retry_on=(
        TimeoutError,
        lambda e: "order book" in str(e).lower(),
    ),
)


# ========== 使用示例 ==========

@retry(config=db_retry_config)
async def fetch_trader_data(trader_id: str):
    """获取交易员数据（带重试）"""
    from backend.repositories.trader_repository import TraderRepository
    from backend.models.database import async_session_maker

    async with async_session_maker() as session:
        repo = TraderRepository(session)
        return await repo.get_by_id(trader_id)


@retry(config=llm_retry_config)
async def call_llm_with_retry(prompt: str, model: str):
    """调用 LLM（带重试）"""
    from backend.llm.provider import LLMClientFactory

    client = LLMClientFactory.create("deepseek")
    return await client.complete(prompt)


# ========== 断路器示例 ==========

class LLMCircuitBreaker:
    """LLM 断路器管理器"""

    def __init__(self):
        self.breakers = {}

    def get_breaker(self, provider: str) -> CircuitBreaker:
        """获取断路器"""
        if provider not in self.breakers:
            self.breakers[provider] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=300,  # 5分钟恢复时间
                expected_exception=ConnectionError,
            )
        return self.breakers[provider]

    async def call(
        self,
        provider: str,
        func: Callable,
        *args,
        **kwargs,
    ):
        """通过断路器调用"""
        breaker = self.get_breaker(provider)
        return await breaker.call(func, *args, **kwargs)


# 全局 LLM 断路器
llm_circuit_breaker = LLMCircuitBreaker()
```

---

## 二十、完整的测试策略

### 20.1 测试配置和 Fixtures

```python
# tests/conftest.py
import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from httpx import AsyncClient, ASGITransport

from backend.config import get_config
from backend.models.database import Base, User, Trader, AIModel
from backend.main import app


# ========== 测试配置 ==========

config = get_config()

# 使用测试数据库
TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/deepalpha_test"


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """创建测试数据库会话"""
    # 创建引擎
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
    )

    # 创建表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 创建会话工厂
    async_session_maker = async_sessionmaker(
        engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )

    async with async_session_maker() as session:
        yield session

    # 清理
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_user(db_session: AsyncSession) -> User:
    """创建测试用户"""
    from backend.security.auth import PasswordManager

    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=PasswordManager.hash_password("Test1234!"),
        full_name="Test User",
        is_active=True,
    )

    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    return user


@pytest_asyncio.fixture(scope="function")
async def test_ai_model(db_session: AsyncSession) -> AIModel:
    """创建测试 AI 模型"""
    ai_model = AIModel(
        user_id="default",
        name="test-deepseek",
        provider="deepseek",
        model_name="deepseek-chat",
        api_key_encrypted="encrypted_test_key",
        base_url="https://api.test.com",
        max_tokens=4096,
        temperature=0.7,
        is_active=True,
    )

    db_session.add(ai_model)
    await db_session.commit()
    await db_session.refresh(ai_model)

    return ai_model


@pytest_asyncio.fixture(scope="function")
async def test_trader(
    db_session: AsyncSession,
    test_user: User,
    test_ai_model: AIModel,
) -> Trader:
    """创建测试交易员"""
    trader = Trader(
        user_id=test_user.id,
        name="测试交易员",
        description="用于测试的交易员",
        ai_model_id=test_ai_model.id,
        market_type="a_stock",
        initial_balance=1_000_000,
        is_running=False,
    )

    db_session.add(trader)
    await db_session.commit()
    await db_session.refresh(trader)

    return trader


@pytest.fixture
def mock_llm_client():
    """模拟 LLM 客户端"""
    client = AsyncMock()

    # 模拟成功响应
    client.complete.return_value = """
    ```json
    [
      {
        "symbol": "000001.SZ",
        "action": "buy",
        "quantity": 1000,
        "price": 10.50,
        "confidence": 85,
        "reasoning": "技术指标显示上涨趋势",
        "stop_loss": 10.00,
        "take_profit": 11.50
      }
    ]
    ```
    """

    return client


@pytest.fixture
def mock_market_data_provider():
    """模拟市场数据提供者"""
    provider = MagicMock()

    # 模拟实时价格
    provider.get_realtime_price.return_value = MarketData(
        symbol="000001.SZ",
        timestamp=datetime.now(),
        price=10.50,
        volume=1000000,
    )

    # 模拟 K线
    provider.get_klines.return_value = [
        Kline(
            symbol="000001.SZ",
            timestamp=datetime.now(),
            open=10.30,
            high=10.60,
            low=10.20,
            close=10.50,
            volume=500000,
        )
    ]

    # 模拟技术指标
    provider.get_technical_indicators.return_value = TechnicalIndicators(
        symbol="000001.SZ",
        timestamp=datetime.now(),
        ema20=10.40,
        ema50=10.20,
        macd=0.15,
        rsi6=55.0,
        rsi12=58.0,
        rsi24=60.0,
        atr=0.30,
    )

    return provider


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """创建测试 HTTP 客户端"""
    from backend.api.dependencies import get_current_user

    # 覆盖依赖
    async def override_get_current_user():
        return test_user

    app.dependency_overrides[get_current_user] = override_get_current_user

    async with AsyncClient(
        app=app,
        base_url="http://test",
        transport=ASGITransport(),
    ) as client:
        yield client

    # 清理
    app.dependency_overrides.clear()
```

### 20.2 单元测试示例

```python
# tests/unit/test_fee_calculator.py
import pytest
from decimal import Decimal

from backend.core.fee_calculator import (
    AStockFeeCalculator,
    HKStockFeeCalculator,
    Side,
)


class TestAStockFeeCalculator:
    """A 股费用计算器测试"""

    @pytest.fixture
    def calculator(self):
        return AStockFeeCalculator()

    def test_buy_commission_minimum(self, calculator):
        """测试买入佣金最低收费"""
        fee = calculator.calculate(price=1.0, quantity=100, side=Side.BUY)

        # 100 * 0.0003 = 0.03，但最低5元
        assert fee.commission == Decimal("5.00")

    def test_buy_large_amount_commission(self, calculator):
        """测试大额买入佣金"""
        fee = calculator.calculate(price=100.0, quantity=10000, side=Side.BUY)

        # 1,000,000 * 0.0003 = 300
        assert fee.commission == Decimal("300.00")

    def test_buy_no_stamp_duty(self, calculator):
        """测试买入无印花税"""
        fee = calculator.calculate(price=10.0, quantity=1000, side=Side.BUY)

        assert fee.stamp_duty == Decimal("0")

    def test_sell_with_stamp_duty(self, calculator):
        """测试卖出有印花税"""
        fee = calculator.calculate(price=10.0, quantity=1000, side=Side.SELL)

        # 10,000 * 0.001 = 10
        assert fee.stamp_duty == Decimal("10.00")

    def test_transfer_fee(self, calculator):
        """测试过户费"""
        fee = calculator.calculate(price=10.0, quantity=1000, side=Side.BUY)

        # 10,000 * 0.00001 = 0.1
        assert fee.transfer_fee == Decimal("0.10")

    def test_total_fee_calculation(self, calculator):
        """测试总费用计算"""
        fee = calculator.calculate(
            price=50.0,
            quantity=5000,
            side=Side.SELL,
        )

        # 计算各项费用
        # 佣金: 250,000 * 0.0003 = 75，最低5元
        # 印花税: 250,000 * 0.001 = 250
        # 过户费: 250,000 * 0.00001 = 2.5
        # 交易费: 250,000 * 0.0000687 ≈ 17.17
        expected_total = Decimal("75") + Decimal("250") + Decimal("2.50") + Decimal("17.17")

        assert fee.total == expected_total

    @pytest.mark.parametrize("price,quantity,expected_min", [
        (1.0, 100, 5.00),   # 小额交易，最低5元
        (10.0, 1000, 5.00),  # 中额交易，最低5元
        (100.0, 1000, 30.00), # 大额交易
        (1.0, 10000, 30.00), # 大量交易
    ])
    def test_commission_minimum enforcement(
        self, calculator, price, quantity, expected_min
    ):
        """测试佣金最低收费强制执行"""
        fee = calculator.calculate(price=price, quantity=quantity, side=Side.BUY)

        assert fee.commission == Decimal(str(expected_min))


class TestHKStockFeeCalculator:
    """港股费用计算器测试"""

    @pytest.fixture
    def calculator(self):
        return HKStockFeeCalculator()

    def test_buy_commission(self, calculator):
        """测试买入佣金"""
        fee = calculator.calculate(
            price=300.0,
            quantity=1000,
            side=Side.BUY,
        )

        # 300,000 * 0.001 = 300
        assert fee.commission == Decimal("300.00")

    def test_stamp_duty_both_sides(self, calculator):
        """测试双向印花税"""
        buy_fee = calculator.calculate(
            price=300.0,
            quantity=1000,
            side=Side.BUY,
        )
        sell_fee = calculator.calculate(
            price=300.0,
            quantity=1000,
            side=Side.SELL,
        )

        # 买卖都有印花税
        assert buy_fee.stamp_duty == Decimal("300.00")  # 300,000 * 0.001
        assert sell_fee.stamp_duty == Decimal("300.00")

    def test_trading_fee(self, calculator):
        """测试交易费"""
        fee = calculator.calculate(
            price=300.0,
            quantity=1000,
            side=Side.BUY,
        )

        # 300,000 * 0.00005 = 15
        assert fee.trading_fee == Decimal("15.00")

    def test_ccass_fee(self, calculator):
        """测试结算费"""
        fee = calculator.calculate(
            price=300.0,
            quantity=1000,
            side=Side.BUY,
        )

        # 300,000 * 0.00002 = 6
        assert fee.transfer_fee == Decimal("6.00")


# tests/unit/test_technical_indicators.py
import pytest
import pandas as pd
import numpy as np

from backend.markets.indicators import TechnicalIndicatorCalculator


class TestTechnicalIndicators:
    """技术指标计算器测试"""

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        # 生成模拟价格数据
        close_prices = 10 + np.cumsum(np.random.randn(100) * 0.1)
        high_prices = close_prices + np.random.rand(100) * 0.5
        low_prices = close_prices - np.random.rand(100) * 0.5
        open_prices = close_prices + np.random.randn(100) * 0.1
        volumes = np.random.randint(1000000, 10000000, 100)

        return pd.DataFrame({
            "timestamp": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        })

    def test_ema_calculation(self, sample_data):
        """测试 EMA 计算"""
        calculator = TechnicalIndicatorCalculator()

        ema20 = calculator.calculate_ema(sample_data, 20)
        ema50 = calculator.calculate_ema(sample_data, 50)

        # 检查结果长度
        assert len(ema20) == len(sample_data)
        assert len(ema50) == len(sample_data)

        # 检查前50个EMA50值应该是NaN
        assert ema50[:50].isna().all()

        # EMA20应该比EMA50更接近当前价格
        current_price = sample_data["close"].iloc[-1]
        last_ema20 = ema20.iloc[-1]
        last_ema50 = ema50.iloc[-1]

        assert abs(last_ema20 - current_price) < abs(last_ema50 - current_price)

    def test_rsi_range(self, sample_data):
        """测试 RSI 范围"""
        calculator = TechnicalIndicatorCalculator()

        rsi14 = calculator.calculate_rsi(sample_data, 14)

        # RSI 应该在 0-100 之间
        valid_rsi = rsi14.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_macd_components(self, sample_data):
        """测试 MACD 组件"""
        calculator = TechnicalIndicatorCalculator()

        macd_data = calculator.calculate_macd(sample_data)

        # 检查组件
        assert "macd" in macd_data
        assert "signal" in macd_data
        assert "histogram" in macd_data

        # MACD - Signal 应该等于 Histogram
        diff = macd_data["macd"] - macd_data["signal"]
        np.testing.assert_array_almost_equal(
            diff.fillna(0).values,
            macd_data["histogram"].fillna(0).values,
            decimal=5,
        )

    def test_atr_positive(self, sample_data):
        """测试 ATR 为正数"""
        calculator = TechnicalIndicatorCalculator()

        atr = calculator.calculate_atr(sample_data)

        # ATR 应该都是正数
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()
```

### 20.3 集成测试示例

```python
# tests/integration/test_trading_flow.py
import pytest
from datetime import datetime

from backend.services.trader_service import TraderService
from backend.services.decision_service import DecisionService
from backend.core.trading_types import DecisionContext, Decision


@pytest.mark.asyncio
class TestTradingFlow:
    """交易流程集成测试"""

    async def test_complete_trading_cycle(
        self,
        db_session,
        test_trader,
        mock_llm_client,
        mock_market_data_provider,
    ):
        """测试完整交易周期"""
        # 准备上下文
        context = DecisionContext(
            account=AccountInfo(
                total_equity=1000000,
                available_balance=1000000,
                positions=[],
                margin_used=0,
                margin_ratio=0,
            ),
            positions=[],
            market_data={},
            indicators={},
            candidate_stocks=["000001.SZ"],
        )

        # 模拟市场数据
        context.market_data["000001.SZ"] = MarketData(
            symbol="000001.SZ",
            timestamp=datetime.now(),
            price=10.50,
            volume=1000000,
        )

        # 创建服务
        trader_service = TraderService(db_session)
        decision_service = DecisionService(db_session)

        # 注入模拟对象
        decision_service.llm_client = mock_llm_client
        decision_service.market_service.get_realtime_data = mock_market_data_provider.get_realtime_price

        # 1. 启动交易员
        await trader_service.start_trader(test_trader.id)
        assert test_trader.is_running is True

        # 2. 执行决策
        decisions = await decision_service.make_decision(
            trader_id=test_trader.id,
            context=context,
        )

        assert len(decisions) > 0
        assert decisions[0].symbol == "000001.SZ"

        # 3. 执行交易
        results = await trader_service.execute_decisions(
            trader_id=test_trader.id,
            decisions=decisions,
        )

        assert len(results) > 0
        assert results[0]["status"] == "success"

        # 4. 检查持仓
        trader = await trader_service.get_trader(test_trader.id)
        assert len(trader.positions) > 0

        # 5. 停止交易员
        await trader_service.stop_trader(test_trader.id)
        assert test_trader.is_running is False
```

### 20.4 性能测试

```python
# tests/performance/test_database_performance.py
import pytest
import asyncio
import time
from sqlalchemy import text

from backend.models.database import async_session_maker, Trader, Decision


@pytest.mark.asyncio
@pytest.mark.performance
class TestDatabasePerformance:
    """数据库性能测试"""

    async def test_bulk_insert_performance(self, db_session):
        """测试批量插入性能"""
        # 创建1000条决策记录
        traders = [Trader(...) for _ in range(10)]

        for trader in traders:
            db_session.add(trader)

        await db_session.commit()

        # 批量插入决策
        start_time = time.time()

        decisions = [
            Decision(
                trader_id=trader.id,
                symbol="000001.SZ",
                action="buy",
                decisions=[{}],
                status="pending",
            )
            for trader in traders
            for _ in range(100)
        ]

        for decision in decisions:
            db_session.add(decision)

        await db_session.commit()

        elapsed = time.time() - start_time

        # 性能断言：1000条记录应该在1秒内插入完成
        assert elapsed < 1.0
        print(f"Bulk inserted 1000 decisions in {elapsed:.3f} seconds")

    async def test_query_performance_with_index(self, db_session):
        """测试带索引的查询性能"""
        from sqlalchemy import select, func

        # 创建测试数据
        await self._create_test_data(db_session, 1000)

        # 测试查询性能
        start_time = time.time()

        result = await db_session.execute(
            select(Decision)
            .where(
                Decision.trader_id == "test-trader-id",
                Decision.created_at >= datetime.now() - timedelta(days=1),
            )
            .order_by(Decision.created_at.desc())
            .limit(100)
        )

        decisions = result.scalars().all()
        elapsed = time.time() - start_time

        assert len(decisions) > 0
        # 性能断言：查询应该在100ms内完成
        assert elapsed < 0.1
        print(f"Queried {len(decisions)} decisions in {elapsed*1000:.1f}ms")

    async def _create_test_data(self, db_session, count: int):
        """创建测试数据"""
        trader = Trader(
            user_id="test-user",
            name="Performance Test Trader",
            ai_model_id="test-model",
            market_type="a_stock",
            initial_balance=1000000,
        )
        db_session.add(trader)
        await db_session.flush()

        # 批量插入
        batch_size = 100
        for i in range(0, count, batch_size):
            decisions = [
                Decision(
                    trader_id=trader.id,
                    symbol=f"00000{(i % 10)}",
                    action="buy",
                    decisions=[{}],
                    status="completed",
                )
                for _ in range(min(batch_size, count - i))
            ]

            for decision in decisions:
                db_session.add(decision)

            await db_session.commit()
```

---

**文档版本**: v4.0
**创建日期**: 2026-01-05
**作者**: DeepAlpha Team
