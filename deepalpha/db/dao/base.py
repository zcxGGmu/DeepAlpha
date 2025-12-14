"""基础数据访问对象"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Any
from datetime import datetime

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class BaseDAO(ABC, Generic[T]):
    """基础DAO抽象类"""

    def __init__(self, session: AsyncSession, model_class: type):
        self.session = session
        self.model_class = model_class

    async def create(self, **kwargs) -> T:
        """创建记录"""
        obj = self.model_class(**kwargs)
        self.session.add(obj)
        await self.session.commit()
        await self.session.refresh(obj)
        return obj

    async def get_by_id(self, id: int) -> Optional[T]:
        """根据ID获取记录"""
        result = await self.session.execute(
            select(self.model_class).where(self.model_class.id == id)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> List[T]:
        """获取所有记录"""
        query = select(self.model_class)

        # 添加排序
        if order_by:
            if order_by.startswith('-'):
                field = order_by[1:]
                query = query.order_by(getattr(self.model_class, field).desc())
            else:
                query = query.order_by(getattr(self.model_class, order_by))

        # 添加分页
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def update(self, id: int, **kwargs) -> Optional[T]:
        """更新记录"""
        # 添加updated_at字段（如果存在）
        if hasattr(self.model_class, 'updated_at'):
            kwargs['updated_at'] = datetime.utcnow()

        result = await self.session.execute(
            update(self.model_class)
            .where(self.model_class.id == id)
            .values(**kwargs)
            .returning(self.model_class)
        )
        await self.session.commit()
        return result.scalar_one_or_none()

    async def delete(self, id: int) -> bool:
        """删除记录"""
        result = await self.session.execute(
            delete(self.model_class).where(self.model_class.id == id)
        )
        await self.session.commit()
        return result.rowcount > 0

    async def count(self) -> int:
        """获取记录总数"""
        result = await self.session.execute(
            select(func.count(self.model_class.id))
        )
        return result.scalar()

    async def exists(self, id: int) -> bool:
        """检查记录是否存在"""
        result = await self.session.execute(
            select(func.count(self.model_class.id))
            .where(self.model_class.id == id)
        )
        return result.scalar() > 0

    async def find_by(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> List[T]:
        """根据条件查找记录"""
        query = select(self.model_class)

        # 添加过滤条件
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                query = query.where(getattr(self.model_class, key) == value)

        # 添加排序
        if order_by:
            if order_by.startswith('-'):
                field = order_by[1:]
                query = query.order_by(getattr(self.model_class, field).desc())
            else:
                query = query.order_by(getattr(self.model_class, order_by))

        # 添加分页
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def find_one_by(self, filters: Dict[str, Any]) -> Optional[T]:
        """根据条件查找单条记录"""
        query = select(self.model_class)

        # 添加过滤条件
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                query = query.where(getattr(self.model_class, key) == value)

        result = await self.session.execute(query)
        return result.scalar_one_or_none()