"""数据库模型，对应 Go 版本的数据库结构"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    Text, Index, ForeignKey, JSON, Enum as SQLEnum
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class DecisionStatus(SQLEnum):
    """决策状态枚举"""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    HOLD = "hold"


class OrderStatus(SQLEnum):
    """订单状态枚举"""
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(SQLEnum):
    """订单方向枚举"""
    BUY = "buy"
    SELL = "sell"
    LONG = "long"
    SHORT = "short"


class OrderType(SQLEnum):
    """订单类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class LiveDecisionLog(Base):
    """实时决策日志表"""
    __tablename__ = "live_decision_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 决策信息
    decision_id = Column(String(64), index=True, nullable=False)
    symbol = Column(String(32), index=True, nullable=False)
    action = Column(String(64), nullable=False)  # enter_long, enter_short, close_long, close_short
    status = Column(Enum(DecisionStatus), default=DecisionStatus.PENDING, index=True)

    # AI模型信息
    model_provider = Column(String(32), nullable=True)
    model_name = Column(String(64), nullable=True)

    # 决策参数
    confidence = Column(Integer, nullable=False)  # 0-100
    position_size_usd = Column(Float, nullable=True)
    leverage = Column(Integer, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Float(nullable=True)

    # 层级设置
    tier1_target = Column(Float, nullable=True)
    tier1_ratio = Column(Float, nullable=True)
    tier2_target = Column(Float, nullable=True)
    tier2_ratio = Column(Float, nullable=True)
    tier3_target = Column(Float, nullable=True)
    tier3_ratio = Column(Float, nullable=True)

    # 元数据
    reason = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # 索引
    __table_args__ = (
        Index('idx_decision_logs_symbol_status', 'symbol', 'status'),
        Index('idx_decision_logs_created_at', 'created_at'),
    )


class LiveOrder(Base):
    """实时订单表"""
    __tablename__ = "live_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 基本信息
    symbol = Column(String(32), index=True, nullable=False)
    side = Column(Enum(OrderSide), index=True, nullable=False)
    order_type = Column(Enum(OrderType), default=OrderType.MARKET, nullable=False)
    status = Column(Enum(OrderStatus), default=OrderStatus.OPEN, index=True, nullable=False)

    # 订单参数
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    filled_amount = Column(Float, default=0, nullable=False)
    filled_price = Column(Float, nullable=True)

    # Freqtrade关联
    ft_order_id = Column(Integer, nullable=True)
    ft_pair = Column(String(32, nullable=True)

    # 盈亏计算
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    realized_pnl = Column(Float, default=0, nullable=False)
    commission = Column(Float, default=0, nullable=False)

    # 决策关联
    decision_id = Column(String(64), nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    closed_at = Column(DateTime, nullable=True)

    # 索引
    __table_args__ = (
        Index('idx_orders_symbol_status', 'symbol', 'status'),
        Index('idx_orders_decision_id', 'decision_id'),
        Index('idx_orders_created_at', 'created_at'),
    )


class LiveTier(Base):
    """实时层级表"""
    __tablename__ = "live_tiers"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 关联订单
    order_id = Column(Integer, ForeignKey('live_orders.id'), nullable=False)

    # 层级信息
    tier = Column(Integer, nullable=False)  # 1, 2, 3
    target_price = Column(Float, nullable=False)
    ratio = Column(Float, nullable=False)  # 平仓比例
    done = Column(Boolean, default=False, nullable=False)

    # 执行信息
    executed_at = Column(DateTime, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # 关系
    order = relationship("LiveOrder", back_populates="tiers")


# 为LiveOrder添加关系
LiveOrder.tiers = relationship("LiveTier", back_populates="order", cascade="all, delete-orphan")


class LiveOrderLog(Base):
    """实时订单日志表"""
    __tablename__ = "live_order_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 关联订单
    order_id = Column(Integer, ForeignKey('live_orders.id'), nullable=True)

    # 日志信息
    action = Column(String(64), nullable=False)  # create, update, fill, cancel
    message = Column(Text, nullable=True)

    # 状态变更
    from_status = Column(Enum(OrderStatus), nullable=True)
    to_status = Column(Enum(OrderStatus), nullable=True)

    # 额外信息
    metadata = Column(JSON, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 关系
    order = relationship("LiveOrder", back_populates="logs")


class LiveModificationLog(Base):
    """实时修改日志表"""
    __tablename__ = "live_modification_log"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 关联订单
    order_id = Column(Integer, ForeignKey('live_orders.id'), nullable=True)

    # 修改信息
    field_name = Column(String(64), nullable=False)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    reason = Column(Text, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 关系
    order = relationship("LiveOrder", back_populates="modification_logs")


class TradeOperationLog(Base):
    """交易操作日志表"""
    __tablename__ = "trade_operation_log"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 操作信息
    operation_type = Column(String(64), nullable=False)  # enter, exit, modify
    symbol = Column(String(32), index=True, nullable=False)
    side = Column(String(8), nullable=False)  # long, short

    # 订单信息
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=True)

    # 结果
    status = Column(String(32), nullable=False)  # success, failed
    message = Column(Text, nullable=True)

    # 系统信息
    ft_order_id = Column(Integer, nullable=True)
    error_code = Column(String(32), nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 索引
    __table_args__ = (
        Index('idx_trade_logs_symbol', 'symbol'),
        Index('idx_trade_logs_created_at', 'created_at'),
    )


class LastDecision(Base):
    """最后决策缓存表"""
    __tablename__ = "last_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 决策信息
    symbol = Column(String(32), primary_key=True, nullable=False)
    decision_id = Column(String(64, unique=True, nullable=False)

    # 决策内容
    action = Column(String(64), nullable=False)
    status = Column(Enum(DecisionStatus), default=Status.PENDING, nullable=False)
    confidence = Column(Integer, nullable=False)
    position_size_usd = Column(Float, nullable=True)

    # 元数据
    metadata = Column(JSON, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # 索引
    __table_args__ = (
        Index('idx_last_decisions_updated_at', 'updated_at'),
    )


# 验证器
@validates('filled_amount')
def validate_filled_amount(cls, v):
    if v is not None and v < 0:
        raise ValueError("filled_amount cannot be negative")


@validates('price')
def validate_price(cls, v):
    if v is not None and v <= 0:
        raise ValueError("price must be positive")


@validates('amount')
def validate_amount(cls, v):
    if v is not None and v <= 0:
        raise ValueError("amount must be positive")


# 为LiveOrder添加验证
LiveOrder.filled_amount = property(LiveOrder.filled_amount, validate_filled_amount)
LiveOrder.price = property(LiveOrder.price, validate_price)
LiveOrder.amount = property(LiveOrder.amount, validate_amount)


# 创建所有表
def create_tables(engine):
    """创建所有数据库表"""
    Base.metadata.create_all(engine)

    # 创建索引
    engine.execute(
        "CREATE INDEX IF NOT EXISTS idx_decision_logs_symbol_status_created_at "
        "ON live_decision_logs (symbol, status, created_at)"
    )

    engine.execute(
        "CREATE INDEX IF NOT EXISTS idx_orders_symbol_status_created_at "
        "ON live_orders (symbol, status, created_at)"
    )

    engine.execute(
        "CREATE INDEX IF NOT EXISTS idx_tiers_order_id_tier "
        "ON live_tiers (order_id, tier)"
    )