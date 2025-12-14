"""数据库连接管理"""

import asyncio
from typing import AsyncGenerator
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from deepalpha.config import config
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)

# SQLAlchemy engine
engine = None
async_engine = None

# Session makers
SessionLocal = None
AsyncSessionLocal = None


def init_database():
    """初始化数据库连接"""
    global engine, async_engine, SessionLocal, AsyncSessionLocal

    # 获取数据库URL
    db_url = config.database.get_url()
    async_db_url = config.database.get_async_url()

    logger.info(f"连接数据库: {db_url}")

    # 创建同步引擎（用于Alembic迁移）
    engine = create_engine(
        db_url,
        echo=config.app.debug,
        pool_pre_ping=True,
        pool_recycle=3600,
    )

    # 创建异步引擎（用于应用）
    async_engine = create_async_engine(
        async_db_url,
        echo=config.app.debug,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=20,
        max_overflow=0,
    )

    # 创建session工厂
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )

    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    logger.info("数据库连接初始化完成")


def get_db() -> SessionLocal:
    """获取数据库会话（同步）"""
    if not SessionLocal:
        init_database()
    return SessionLocal()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话（异步）"""
    if not AsyncSessionLocal:
        init_database()

    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"数据库会话错误: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


# 数据库依赖注入函数
async def get_db_session() -> AsyncSession:
    """FastAPI依赖注入：获取数据库会话"""
    async for session in get_async_db():
        return session


def create_tables():
    """创建所有表"""
    from deepalpha.db.models import Base

    if not engine:
        init_database()

    logger.info("创建数据库表...")
    Base.metadata.create_all(bind=engine)
    logger.info("数据库表创建完成")


async def close_database():
    """关闭数据库连接"""
    global async_engine

    if async_engine:
        logger.info("关闭数据库连接...")
        await async_engine.dispose()
        logger.info("数据库连接已关闭")