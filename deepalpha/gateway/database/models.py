"""数据库模型"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from deepalpha.core.models import DecisionType, OrderSide, OrderStatus, OrderType, SignalType

Base = declarative_base()


class TimestampMixin:
    """时间戳混入类"""
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class KlineModel(Base, TimestampMixin):
    """K线数据表"""
    __tablename__ = "klines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    interval = Column(String(16), nullable=False)
    open_time = Column(DateTime(timezone=True), nullable=False)
    close_time = Column(DateTime(timezone=True), nullable=False)
    open = Column(Numeric(precision=18, scale=8), nullable=False)
    high = Column(Numeric(precision=18, scale=8), nullable=False)
    low = Column(Numeric(precision=18, scale=8), nullable=False)
    close = Column(Numeric(precision=18, scale=8), nullable=False)
    volume = Column(Numeric(precision=18, scale=8), nullable=False)
    quote_volume = Column(Numeric(precision=18, scale=8), nullable=False)
    trades = Column(Integer)
    taker_buy_volume = Column(Numeric(precision=18, scale=8))
    taker_buy_quote_volume = Column(Numeric(precision=18, scale=8))

    # 索引
    __table_args__ = (
        Index("idx_klines_symbol_interval", "symbol", "interval"),
        Index("idx_klines_symbol_time", "symbol", "open_time"),
        Index("idx_klines_time", "open_time"),
    )


class DecisionModel(Base, TimestampMixin):
    """决策记录表"""
    __tablename__ = "decisions"

    id = Column(String(64), primary_key=True)
    symbol = Column(String(32), nullable=False)
    decision_type = Column(Enum(DecisionType), nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text)
    signals = Column(JSONB)
    expires_at = Column(DateTime(timezone=True))
    executed = Column(Boolean, default=False)
    execution_time = Column(DateTime(timezone=True))
    suggested_size = Column(Float)
    stop_loss = Column(Numeric(precision=18, scale=8))
    take_profit = Column(Numeric(precision=18, scale=8))

    # 关联订单
    orders = relationship("OrderModel", back_populates="decision")

    # 索引
    __table_args__ = (
        Index("idx_decisions_symbol", "symbol"),
        Index("idx_decisions_created", "created_at"),
        Index("idx_decisions_executed", "executed"),
        Index("idx_decisions_symbol_type", "symbol", "decision_type"),
    )


class OrderModel(Base, TimestampMixin):
    """订单记录表"""
    __tablename__ = "orders"

    id = Column(String(64), primary_key=True)
    client_order_id = Column(String(64), unique=True)
    symbol = Column(String(32), nullable=False)
    type = Column(Enum(OrderType), nullable=False)
    side = Column(Enum(OrderSide), nullable=False)
    amount = Column(Numeric(precision=18, scale=8), nullable=False)
    price = Column(Numeric(precision=18, scale=8))
    stop_price = Column(Numeric(precision=18, scale=8))
    status = Column(Enum(OrderStatus), nullable=False)
    filled = Column(Numeric(precision=18, scale=8), default=0)
    remaining = Column(Numeric(precision=18, scale=8))
    average_price = Column(Numeric(precision=18, scale=8))
    fee = Column(Numeric(precision=18, scale=8))
    fee_currency = Column(String(16))
    trades = Column(JSONB)
    metadata = Column(JSONB)

    # 关联决策
    decision_id = Column(String(64), ForeignKey("decisions.id"))
    decision = relationship("DecisionModel", back_populates="orders")

    # 索引
    __table_args__ = (
        Index("idx_orders_symbol", "symbol"),
        Index("idx_orders_status", "status"),
        Index("idx_orders_created", "created_at"),
        Index("idx_orders_symbol_status", "symbol", "status"),
    )


class PositionModel(Base, TimestampMixin):
    """持仓记录表"""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False, unique=True)
    side = Column(Enum(OrderSide), nullable=False)
    size = Column(Numeric(precision=18, scale=8), nullable=False)
    entry_price = Column(Numeric(precision=18, scale=8), nullable=False)
    current_price = Column(Numeric(precision=18, scale=8), nullable=False)
    unrealized_pnl = Column(Numeric(precision=18, scale=8), nullable=False)
    realized_pnl = Column(Numeric(precision=18, scale=8), default=0)
    metadata = Column(JSONB)

    # 索引
    __table_args__ = (
        Index("idx_positions_symbol", "symbol"),
        Index("idx_positions_side", "side"),
    )


class BalanceModel(Base, TimestampMixin):
    """余额记录表"""
    __tablename__ = "balances"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset = Column(String(16), nullable=False, unique=True)
    free = Column(Numeric(precision=18, scale=8), nullable=False)
    used = Column(Numeric(precision=18, scale=8), nullable=False)
    total = Column(Numeric(precision=18, scale=8), nullable=False)

    # 索引
    __table_args__ = (
        Index("idx_balances_asset", "asset"),
    )


class AgentResultModel(Base, TimestampMixin):
    """智能体结果表"""
    __tablename__ = "agent_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_name = Column(String(64), nullable=False)
    symbol = Column(String(32), nullable=False)
    signal_type = Column(Enum(SignalType))
    confidence = Column(Float)
    reason = Column(Text)
    price = Column(Numeric(precision=18, scale=8))
    timeframe = Column(String(16))
    analysis = Column(JSONB)
    processing_time = Column(Float)

    # 索引
    __table_args__ = (
        Index("idx_agent_results_symbol", "symbol"),
        Index("idx_agent_results_agent", "agent_name"),
        Index("idx_agent_results_created", "created_at"),
    )


class TradeModel(Base, TimestampMixin):
    """成交记录表"""
    __tablename__ = "trades"

    id = Column(String(64), primary_key=True)
    order_id = Column(String(64), nullable=False)
    symbol = Column(String(32), nullable=False)
    side = Column(Enum(OrderSide), nullable=False)
    amount = Column(Numeric(precision=18, scale=8), nullable=False)
    price = Column(Numeric(precision=18, scale=8), nullable=False)
    fee = Column(Numeric(precision=18, scale=8))
    fee_currency = Column(String(16))
    timestamp = Column(DateTime(timezone=True), nullable=False)

    # 索引
    __table_args__ = (
        Index("idx_trades_order", "order_id"),
        Index("idx_trades_symbol", "symbol"),
        Index("idx_trades_timestamp", "timestamp"),
    )


class PerformanceMetricModel(Base, TimestampMixin):
    """性能指标表"""
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    total_pnl = Column(Numeric(precision=18, scale=2), default=0)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    max_drawdown = Column(Float, default=0)
    sharpe_ratio = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    metrics = Column(JSONB)

    # 索引
    __table_args__ = (
        Index("idx_performance_date", "date"),
        Index("idx_performance_total_pnl", "total_pnl"),
    )


class SystemLogModel(Base, TimestampMixin):
    """系统日志表"""
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(16), nullable=False)
    logger = Column(String(64), nullable=False)
    message = Column(Text, nullable=False)
    module = Column(String(64))
    function = Column(String(64))
    line = Column(Integer)
    exception = Column(Text)
    extra = Column(JSONB)

    # 索引
    __table_args__ = (
        Index("idx_logs_level", "level"),
        Index("idx_logs_created", "created_at"),
        Index("idx_logs_logger", "logger"),
    )