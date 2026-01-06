# NOFX Python重构技术方案 - 实现细节补充篇

> 本文档是《NOFX_Python_重构技术方案_A股港股》系列的第七部分
> 覆盖第26-30章：API设计、交易策略实现、实时数据处理、风险管理、回测系统

---

## 第26章 API设计详细规范

### 26.1 RESTful API完整设计

#### 26.1.1 API版本控制

```python
# src/api/v1/__init__.py
"""
API v1 路由注册
"""

from fastapi import APIRouter
from src.api.v1.endpoints import (
    auth,
    traders,
    positions,
    orders,
    trades,
    market_data,
    analytics,
    system,
)

api_router = APIRouter()

# 认证相关（无需token）
api_router.include_router(auth.router, prefix="/auth", tags=["认证"])

# 交易员管理
api_router.include_router(traders.router, prefix="/traders", tags=["交易员"])

# 持仓管理
api_router.include_router(positions.router, prefix="/positions", tags=["持仓"])

# 订单管理
api_router.include_router(orders.router, prefix="/orders", tags=["订单"])

# 成交记录
api_router.include_router(trades.router, prefix="/trades", tags=["成交"])

# 市场数据
api_router.include_router(market_data.router, prefix="/market", tags=["市场数据"])

# 数据分析
api_router.include_router(analytics.router, prefix="/analytics", tags=["数据分析"])

# 系统管理
api_router.include_router(system.router, prefix="/system", tags=["系统管理"])
```

#### 26.1.2 通用响应模型

```python
# src/api/v1/schemas/common.py
"""
通用API响应模型
"""

from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar('T')


class APIResponse(BaseModel, Generic[T]):
    """通用API响应"""
    code: int = Field(default=200, description="状态码")
    message: str = Field(default="success", description="响应消息")
    data: Optional[T] = Field(default=None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应"""
    code: int = Field(default=200, description="状态码")
    message: str = Field(default="success", description="响应消息")
    data: list[T] = Field(default_factory=list, description="数据列表")
    pagination: PaginationMeta = Field(description="分页信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")


class PaginationMeta(BaseModel):
    """分页元数据"""
    page: int = Field(ge=1, description="当前页码")
    page_size: int = Field(ge=1, le=100, description="每页数量")
    total: int = Field(ge=0, description="总记录数")
    total_pages: int = Field(ge=0, description="总页数")
    has_next: bool = Field(description="是否有下一页")
    has_prev: bool = Field(description="是否有上一页")

    @classmethod
    def create(cls, page: int, page_size: int, total: int) -> 'PaginationMeta':
        """创建分页元数据"""
        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
        return cls(
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )


class ErrorDetail(BaseModel):
    """错误详情"""
    field: Optional[str] = Field(None, description="错误字段")
    message: str = Field(description="错误消息")
    type: Optional[str] = Field(None, description="错误类型")


class ErrorResponse(BaseModel):
    """错误响应"""
    code: int = Field(description="错误码")
    message: str = Field(description="错误消息")
    errors: Optional[list[ErrorDetail]] = Field(default_factory=list, description="错误详情列表")
    timestamp: datetime = Field(default_factory=datetime.now, description="错误时间戳")
    path: Optional[str] = Field(None, description="请求路径")


# 预定义错误响应
class ErrorResponseFactory:
    """错误响应工厂"""

    @staticmethod
    def validation_error(errors: list[ErrorDetail], path: str = None) -> ErrorResponse:
        """验证错误 (400)"""
        return ErrorResponse(
            code=400,
            message="Validation failed",
            errors=errors,
            path=path,
        )

    @staticmethod
    def unauthorized(message: str = "Unauthorized", path: str = None) -> ErrorResponse:
        """未授权 (401)"""
        return ErrorResponse(
            code=401,
            message=message,
            path=path,
        )

    @staticmethod
    def forbidden(message: str = "Forbidden", path: str = None) -> ErrorResponse:
        """禁止访问 (403)"""
        return ErrorResponse(
            code=403,
            message=message,
            path=path,
        )

    @staticmethod
    def not_found(resource: str = "Resource", path: str = None) -> ErrorResponse:
        """未找到 (404)"""
        return ErrorResponse(
            code=404,
            message=f"{resource} not found",
            path=path,
        )

    @staticmethod
    def conflict(message: str = "Conflict", path: str = None) -> ErrorResponse:
        """冲突 (409)"""
        return ErrorResponse(
            code=409,
            message=message,
            path=path,
        )

    @staticmethod
    def rate_limit_exceeded(retry_after: int = None, path: str = None) -> ErrorResponse:
        """速率限制 (429)"""
        return ErrorResponse(
            code=429,
            message="Rate limit exceeded",
            path=path,
        )

    @staticmethod
    def internal_error(message: str = "Internal server error", path: str = None) -> ErrorResponse:
        """内部错误 (500)"""
        return ErrorResponse(
            code=500,
            message=message,
            path=path,
        )
```

#### 26.1.3 依赖注入

```python
# src/api/v1/dependencies.py
"""
API依赖注入
"""

from typing import Annotated
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.database import get_db
from src.models.user import UserModel
from src.services.auth_service import AuthService


security = HTTPBearer(auto_error=False)


async def get_db_session() -> AsyncSession:
    """获取数据库会话"""
    async for session in get_db():
        yield session


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: AsyncSession = Depends(get_db_session),
) -> UserModel:
    """获取当前用户"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    token = credentials.credentials
    auth_service = AuthService(db)

    try:
        user = await auth_service.get_user_by_token(token)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled",
            )

        return user

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


async def get_current_active_user(
    current_user: UserModel = Depends(get_current_user),
) -> UserModel:
    """获取当前活跃用户"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    return current_user


async def get_current_admin(
    current_user: UserModel = Depends(get_current_user),
) -> UserModel:
    """获取当前管理员用户"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user


async def validate_trader_access(
    trader_id: str,
    current_user: UserModel = Depends(get_current_user),
) -> None:
    """验证交易员访问权限"""
    # 管理员可以访问所有交易员
    if current_user.role == "admin":
        return

    # 其他用户只能访问自己的交易员
    # 这里需要根据实际业务逻辑实现
    # 例如：检查trader是否属于current_user
    pass


async def rate_limit_check(
    x_forwarded_for: Annotated[str | None, Header()] = None,
    x_real_ip: Annotated[str | None, Header()] = None,
) -> None:
    """速率限制检查"""
    import time
    import redis

    # 获取客户端IP
    client_ip = x_forwarded_for or x_real_ip or "unknown"

    # 连接Redis
    redis_client = redis.from_url(str(settings.REDIS_URL))

    # 速率限制：每分钟100次请求
    key = f"rate_limit:{client_ip}"
    current = redis_client.get(key)

    if current is None:
        redis_client.setex(key, 60, 1)
    else:
        count = int(current)
        if count >= 100:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
        redis_client.incr(key)


class QueryParams:
    """通用查询参数"""

    def __init__(
        self,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ):
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page must be >= 1",
            )

        if page_size < 1 or page_size > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page size must be between 1 and 100",
            )

        if sort_order not in ("asc", "desc"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sort order must be 'asc' or 'desc'",
            )

        self.page = page
        self.page_size = page_size
        self.sort_by = sort_by
        self.sort_order = sort_order

    @property
    def offset(self) -> int:
        """计算偏移量"""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """获取限制数量"""
        return self.page_size
```

#### 26.1.4 认证API端点

```python
# src/api/v1/endpoints/auth.py
"""
认证相关API端点
"""

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from src.api.v1.dependencies import get_db_session
from src.api.v1.schemas.common import APIResponse, ErrorResponse, ErrorResponseFactory
from src.schemas.auth import TokenResponse, LoginRequest, RegisterRequest
from src.services.auth_service import AuthService
from src.models.user import UserModel

router = APIRouter()


@router.post("/register", response_model=APIResponse[TokenResponse], status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    用户注册

    - **email**: 邮箱地址
    - **password**: 密码（至少8位）
    - **name**: 用户名
    """
    auth_service = AuthService(db)

    # 检查邮箱是否已存在
    existing_user = await auth_service.get_user_by_email(request.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # 创建用户
    user = await auth_service.create_user(
        email=request.email,
        password=request.password,
        name=request.name,
    )

    # 生成Token
    token = await auth_service.create_token(user)

    return APIResponse(
        data=TokenResponse(
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            token_type="bearer",
            expires_in=token.expires_in,
        )
    )


@router.post("/login", response_model=APIResponse[TokenResponse])
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    用户登录

    - **email**: 邮箱地址
    - **password**: 密码
    """
    auth_service = AuthService(db)

    # 验证用户
    user = await auth_service.authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # 生成Token
    token = await auth_service.create_token(user)

    return APIResponse(
        data=TokenResponse(
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            token_type="bearer",
            expires_in=token.expires_in,
        )
    )


@router.post("/refresh", response_model=APIResponse[TokenResponse])
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    刷新访问令牌

    - **refresh_token**: 刷新令牌
    """
    auth_service = AuthService(db)

    # 验证刷新令牌
    token = await auth_service.refresh_access_token(refresh_token)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    return APIResponse(
        data=TokenResponse(
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            token_type="bearer",
            expires_in=token.expires_in,
        )
    )


@router.post("/logout", response_model=APIResponse[dict])
async def logout(
    current_user: UserModel = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    用户登出

    将当前用户的Token加入黑名单
    """
    auth_service = AuthService(db)
    await auth_service.revoke_token(current_user)

    return APIResponse(data={"message": "Successfully logged out"})


@router.get("/me", response_model=APIResponse[UserResponse])
async def get_current_user_info(
    current_user: UserModel = Depends(get_current_active_user),
):
    """
    获取当前用户信息

    返回当前登录用户的详细信息
    """
    return APIResponse(data=UserResponse.from_orm(current_user))


@router.put("/me", response_model=APIResponse[UserResponse])
async def update_current_user(
    request: UserUpdateRequest,
    current_user: UserModel = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    更新当前用户信息

    - **name**: 新用户名（可选）
    - **avatar**: 头像URL（可选）
    """
    auth_service = AuthService(db)

    updated_user = await auth_service.update_user(
        user_id=current_user.id,
        name=request.name,
        avatar=request.avatar,
    )

    return APIResponse(data=UserResponse.from_orm(updated_user))


@router.post("/me/change-password", response_model=APIResponse[dict])
async def change_password(
    request: ChangePasswordRequest,
    current_user: UserModel = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    修改密码

    - **old_password**: 旧密码
    - **new_password**: 新密码（至少8位）
    """
    auth_service = AuthService(db)

    # 验证旧密码
    if not await auth_service.verify_password(current_user, request.old_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid old password",
        )

    # 更新密码
    await auth_service.change_password(
        user_id=current_user.id,
        new_password=request.new_password,
    )

    return APIResponse(data={"message": "Password changed successfully"})
```

#### 26.1.5 交易员API端点

```python
# src/api/v1/endpoints/traders.py
"""
交易员管理API端点
"""

from typing import Annotated, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.v1.dependencies import (
    get_db_session,
    get_current_user,
    get_current_admin,
    QueryParams,
    validate_trader_access,
)
from src.api.v1.schemas.common import APIResponse, PaginatedResponse, PaginationMeta
from src.api.v1.schemas.traders import (
    TraderResponse,
    TraderCreateRequest,
    TraderUpdateRequest,
    TraderStatsResponse,
)
from src.models.user import UserModel
from src.models.trader import TraderModel
from src.services.trader_service import TraderService

router = APIRouter()


@router.get("", response_model=PaginatedResponse[TraderResponse])
async def list_traders(
    query: QueryParams = Depends(),
    is_active: bool | None = None,
    trader_type: str | None = None,
    search: str | None = None,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    获取交易员列表

    - **page**: 页码（默认1）
    - **page_size**: 每页数量（默认20，最大100）
    - **sort_by**: 排序字段（默认created_at）
    - **sort_order**: 排序方向（asc/desc，默认desc）
    - **is_active**: 是否活跃（可选）
    - **trader_type**: 交易员类型（可选）
    - **search**: 搜索关键词（可选）
    """

    trader_service = TraderService(db)

    # 构建查询条件
    filters = {}
    if is_active is not None:
        filters["is_active"] = is_active
    if trader_type is not None:
        filters["type"] = trader_type

    # 非管理员只能查看自己的交易员
    if current_user.role != "admin":
        filters["user_id"] = current_user.id

    # 查询数据
    traders, total = await trader_service.list_traders(
        filters=filters,
        search=search,
        offset=query.offset,
        limit=query.limit,
        sort_by=query.sort_by,
        sort_order=query.sort_order,
    )

    # 转换为响应模型
    trader_responses = [TraderResponse.from_orm(t) for t in traders]

    return PaginatedResponse(
        data=trader_responses,
        pagination=PaginationMeta.create(query.page, query.page_size, total),
    )


@router.post("", response_model=APIResponse[TraderResponse], status_code=status.HTTP_201_CREATED)
async def create_trader(
    request: TraderCreateRequest,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    创建交易员

    - **name**: 交易员名称
    - **type**: 交易员类型（discretionary/ai/hybrid）
    - **initial_capital**: 初始资金
    - **config**: 配置参数（可选）
    """

    trader_service = TraderService(db)

    # 创建交易员
    trader = await trader_service.create_trader(
        user_id=current_user.id,
        name=request.name,
        type=request.type,
        initial_capital=request.initial_capital,
        config=request.config,
    )

    return APIResponse(data=TraderResponse.from_orm(trader))


@router.get("/{trader_id}", response_model=APIResponse[TraderResponse])
async def get_trader(
    trader_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    获取交易员详情

    - **trader_id**: 交易员ID
    """

    trader_service = TraderService(db)

    trader = await trader_service.get_trader_by_id(trader_id)
    if not trader:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trader not found",
        )

    # 权限验证
    if trader.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return APIResponse(data=TraderResponse.from_orm(trader))


@router.put("/{trader_id}", response_model=APIResponse[TraderResponse])
async def update_trader(
    trader_id: str,
    request: TraderUpdateRequest,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    更新交易员信息

    - **trader_id**: 交易员ID
    - **name**: 新名称（可选）
    - **is_active**: 是否活跃（可选）
    - **config**: 配置参数（可选）
    """

    trader_service = TraderService(db)

    # 验证权限
    await validate_trader_access(trader_id, current_user)

    # 更新交易员
    trader = await trader_service.update_trader(
        trader_id=trader_id,
        name=request.name,
        is_active=request.is_active,
        config=request.config,
    )

    if not trader:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trader not found",
        )

    return APIResponse(data=TraderResponse.from_orm(trader))


@router.delete("/{trader_id}", response_model=APIResponse[dict], status_code=status.HTTP_204_NO_CONTENT)
async def delete_trader(
    trader_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    删除交易员

    - **trader_id**: 交易员ID
    """

    trader_service = TraderService(db)

    # 验证权限
    await validate_trader_access(trader_id, current_user)

    # 删除交易员
    success = await trader_service.delete_trader(trader_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trader not found",
        )

    return APIResponse(data={"message": "Trader deleted successfully"})


@router.get("/{trader_id}/stats", response_model=APIResponse[TraderStatsResponse])
async def get_trader_stats(
    trader_id: str,
    period: str = Query("1m", description="统计周期: 1d, 1w, 1m, 3m, 6m, 1y, all"),
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    获取交易员统计信息

    - **trader_id**: 交易员ID
    - **period**: 统计周期（默认1m）
    """

    trader_service = TraderService(db)

    # 验证权限
    await validate_trader_access(trader_id, current_user)

    # 获取统计数据
    stats = await trader_service.get_trader_stats(
        trader_id=trader_id,
        period=period,
    )

    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trader not found",
        )

    return APIResponse(data=stats)


@router.post("/{trader_id}/start", response_model=APIResponse[dict])
async def start_trader(
    trader_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    启动交易员

    - **trader_id**: 交易员ID
    """

    trader_service = TraderService(db)

    # 验证权限
    await validate_trader_access(trader_id, current_user)

    # 启动交易员
    success = await trader_service.start_trader(trader_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to start trader",
        )

    return APIResponse(data={"message": "Trader started successfully"})


@router.post("/{trader_id}/stop", response_model=APIResponse[dict])
async def stop_trader(
    trader_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    停止交易员

    - **trader_id**: 交易员ID
    """

    trader_service = TraderService(db)

    # 验证权限
    await validate_trader_access(trader_id, current_user)

    # 停止交易员
    success = await trader_service.stop_trader(trader_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to stop trader",
        )

    return APIResponse(data={"message": "Trader stopped successfully"})


@router.get("/{trader_id}/positions", response_model=PaginatedResponse[PositionResponse])
async def get_trader_positions(
    trader_id: str,
    query: QueryParams = Depends(),
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    获取交易员持仓列表

    - **trader_id**: 交易员ID
    - **page**: 页码
    - **page_size**: 每页数量
    """

    trader_service = TraderService(db)

    # 验证权限
    await validate_trader_access(trader_id, current_user)

    # 获取持仓
    positions, total = await trader_service.get_trader_positions(
        trader_id=trader_id,
        offset=query.offset,
        limit=query.limit,
        sort_by=query.sort_by,
        sort_order=query.sort_order,
    )

    position_responses = [PositionResponse.from_orm(p) for p in positions]

    return PaginatedResponse(
        data=position_responses,
        pagination=PaginationMeta.create(query.page, query.page_size, total),
    )


@router.get("/{trader_id}/orders", response_model=PaginatedResponse[OrderResponse])
async def get_trader_orders(
    trader_id: str,
    query: QueryParams = Depends(),
    status: str | None = None,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    获取交易员订单列表

    - **trader_id**: 交易员ID
    - **status**: 订单状态过滤（可选）
    - **page**: 页码
    - **page_size**: 每页数量
    """

    trader_service = TraderService(db)

    # 验证权限
    await validate_trader_access(trader_id, current_user)

    # 获取订单
    orders, total = await trader_service.get_trader_orders(
        trader_id=trader_id,
        status=status,
        offset=query.offset,
        limit=query.limit,
        sort_by=query.sort_by,
        sort_order=query.sort_order,
    )

    order_responses = [OrderResponse.from_orm(o) for o in orders]

    return PaginatedResponse(
        data=order_responses,
        pagination=PaginationMeta.create(query.page, query.page_size, total),
    )
```

#### 26.1.6 订单API端点

```python
# src/api/v1/endpoints/orders.py
"""
订单管理API端点
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.v1.dependencies import (
    get_db_session,
    get_current_user,
    QueryParams,
)
from src.api.v1.schemas.common import APIResponse, PaginatedResponse, PaginationMeta
from src.api.v1.schemas.orders import (
    OrderResponse,
    OrderCreateRequest,
    OrderUpdateRequest,
    OrderCancelResponse,
)
from src.models.user import UserModel
from src.services.order_service import OrderService

router = APIRouter()


@router.post("", response_model=APIResponse[OrderResponse], status_code=status.HTTP_201_CREATED)
async def create_order(
    request: OrderCreateRequest,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    创建订单

    - **trader_id**: 交易员ID
    - **symbol**: 股票代码
    - **exchange**: 交易所代码
    - **side**: 买卖方向（buy/sell）
    - **type**: 订单类型（market/limit）
    - **quantity**: 数量
    - **price**: 价格（限价单必填）
    """

    order_service = OrderService(db)

    # 验证交易员权限
    trader = await order_service.get_trader(request.trader_id)
    if trader.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    # 验证交易员状态
    if not trader.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Trader is not active",
        )

    # 创建订单
    try:
        order = await order_service.create_order(
            trader_id=request.trader_id,
            symbol=request.symbol,
            exchange=request.exchange,
            side=request.side,
            order_type=request.type,
            quantity=request.quantity,
            price=request.price,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return APIResponse(data=OrderResponse.from_orm(order))


@router.get("/{order_id}", response_model=APIResponse[OrderResponse])
async def get_order(
    order_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    获取订单详情

    - **order_id**: 订单ID
    """

    order_service = OrderService(db)

    order = await order_service.get_order_by_id(order_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found",
        )

    # 验证权限
    trader = await order_service.get_trader(order.trader_id)
    if trader.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return APIResponse(data=OrderResponse.from_orm(order))


@router.patch("/{order_id}", response_model=APIResponse[OrderResponse])
async def update_order(
    order_id: str,
    request: OrderUpdateRequest,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    修改订单

    只能修改未成交的订单

    - **order_id**: 订单ID
    - **quantity**: 新数量（可选）
    - **price**: 新价格（可选，仅限价单）
    """

    order_service = OrderService(db)

    order = await order_service.get_order_by_id(order_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found",
        )

    # 验证权限
    trader = await order_service.get_trader(order.trader_id)
    if trader.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    # 验证订单状态
    if order.status not in ("pending", "open"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Order cannot be modified",
        )

    # 修改订单
    try:
        updated_order = await order_service.update_order(
            order_id=order_id,
            quantity=request.quantity,
            price=request.price,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return APIResponse(data=OrderResponse.from_orm(updated_order))


@router.post("/{order_id}/cancel", response_model=APIResponse[OrderCancelResponse])
async def cancel_order(
    order_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    取消订单

    - **order_id**: 订单ID
    """

    order_service = OrderService(db)

    order = await order_service.get_order_by_id(order_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found",
        )

    # 验证权限
    trader = await order_service.get_trader(order.trader_id)
    if trader.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    # 取消订单
    success = await order_service.cancel_order(order_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Order cannot be cancelled",
        )

    # 重新获取订单
    cancelled_order = await order_service.get_order_by_id(order_id)

    return APIResponse(
        data=OrderCancelResponse(
            order_id=order_id,
            status="cancelled",
            message="Order cancelled successfully"
        )
    )


@router.get("", response_model=PaginatedResponse[OrderResponse])
async def list_orders(
    query: QueryParams = Depends(),
    trader_id: str | None = None,
    symbol: str | None = None,
    exchange: str | None = None,
    status: str | None = None,
    side: str | None = None,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    获取订单列表

    - **trader_id**: 交易员ID过滤（可选）
    - **symbol**: 股票代码过滤（可选）
    - **exchange**: 交易所过滤（可选）
    - **status**: 订单状态过滤（可选）
    - **side**: 买卖方向过滤（可选）
    - **page**: 页码
    - **page_size**: 每页数量
    """

    order_service = OrderService(db)

    # 构建过滤条件
    filters = {}
    if trader_id is not None:
        filters["trader_id"] = trader_id
    if symbol is not None:
        filters["symbol"] = symbol
    if exchange is not None:
        filters["exchange"] = exchange
    if status is not None:
        filters["status"] = status
    if side is not None:
        filters["side"] = side

    # 非管理员只能查看自己的订单
    if current_user.role != "admin":
        # 获取用户的交易员ID列表
        user_trader_ids = await order_service.get_user_trader_ids(current_user.id)
        if trader_id and trader_id not in user_trader_ids:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )
        filters["trader_id__in"] = user_trader_ids

    # 查询订单
    orders, total = await order_service.list_orders(
        filters=filters,
        offset=query.offset,
        limit=query.limit,
        sort_by=query.sort_by,
        sort_order=query.sort_order,
    )

    order_responses = [OrderResponse.from_orm(o) for o in orders]

    return PaginatedResponse(
        data=order_responses,
        pagination=PaginationMeta.create(query.page, query.page_size, total),
    )
```

#### 26.1.7 WebSocket实时推送

```python
# src/api/v1/websocket.py
"""
WebSocket实时推送
"""

from typing import Set
from fastapi import WebSocket, WebSocketDisconnect, Depends, Query
from jose import jwt, JWTError

from src.core.config import settings
from src.services.websocket_manager import WebSocketManager
from src.services.auth_service import AuthService

# WebSocket管理器
manager = WebSocketManager()


async def get_ws_user(token: str = Query(...)):
    """验证WebSocket Token"""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except JWTError:
        return None


@router.websocket("/ws/market")
async def market_data_websocket(
    websocket: WebSocket,
    token: str = Query(...),
):
    """
    市场数据WebSocket连接

    订阅实时市场数据推送
    """
    # 验证用户
    user_id = await get_ws_user(token)
    if user_id is None:
        await websocket.close(code=1008, reason="Unauthorized")
        return

    # 接受连接
    await manager.connect(websocket, user_id)

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()

            message_type = data.get("type")

            if message_type == "subscribe":
                # 订阅股票
                symbols = data.get("symbols", [])
                await manager.subscribe(user_id, symbols)

            elif message_type == "unsubscribe":
                # 取消订阅
                symbols = data.get("symbols", [])
                await manager.unsubscribe(user_id, symbols)

            elif message_type == "ping":
                # 心跳
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)


@router.websocket("/ws/orders")
async def orders_websocket(
    websocket: WebSocket,
    token: str = Query(...),
):
    """
    订单状态WebSocket连接

    订阅订单状态变更推送
    """
    # 验证用户
    user_id = await get_ws_user(token)
    if user_id is None:
        await websocket.close(code=1008, reason="Unauthorized")
        return

    # 接受连接
    await manager.connect(websocket, user_id, channel="orders")

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()

            message_type = data.get("type")

            if message_type == "subscribe_trader":
                # 订阅交易员订单
                trader_id = data.get("trader_id")
                await manager.subscribe_trader_orders(user_id, trader_id)

            elif message_type == "unsubscribe_trader":
                # 取消订阅
                trader_id = data.get("trader_id")
                await manager.unsubscribe_trader_orders(user_id, trader_id)

            elif message_type == "ping":
                # 心跳
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id, channel="orders")
```

---

## 第27章 交易策略详细实现

### 27.1 策略基类架构

```python
# src/strategies/base.py
"""
交易策略基类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from src.models.trader import TraderModel
from src.models.position import PositionModel
from src.models.market_data import MarketData
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    exchange: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1
    reason: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PositionSize:
    """仓位大小"""
    quantity: int
    price: float
    value: float
    risk_percent: float  # 风险百分比


class BaseStrategy(ABC):
    """策略基类"""

    def __init__(
        self,
        trader: TraderModel,
        config: Dict[str, Any] = None,
    ):
        """
        初始化策略

        Args:
            trader: 交易员实例
            config: 策略配置
        """
        self.trader = trader
        self.config = config or {}
        self.positions: Dict[str, PositionModel] = {}
        self.market_data: Dict[str, MarketData] = {}
        self.is_running = False

        # 从config获取参数
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)  # 每笔交易风险
        self.max_positions = self.config.get('max_positions', 5)  # 最大持仓数
        self.position_size_method = self.config.get('position_size_method', 'fixed')  # fixed, kelly, risk_parity

        logger.info(f"策略初始化: {self.__class__.__name__}", extra={
            'trader_id': trader.id,
            'config': self.config,
        })

    @abstractmethod
    async def initialize(self) -> None:
        """策略初始化（加载历史数据、预热模型等）"""
        pass

    @abstractmethod
    async def on_bar(self, bar: MarketData) -> List[Signal]:
        """
        K线数据更新回调

        Args:
            bar: K线数据

        Returns:
            交易信号列表
        """
        pass

    @abstractmethod
    async def on_tick(self, tick: MarketData) -> List[Signal]:
        """
        Tick数据更新回调

        Args:
            tick: Tick数据

        Returns:
            交易信号列表
        """
        pass

    @abstractmethod
    async def on_order_update(self, order: OrderModel) -> None:
        """
        订单状态更新回调

        Args:
            order: 订单对象
        """
        pass

    @abstractmethod
    async def on_position_update(self, position: PositionModel) -> None:
        """
        持仓更新回调

        Args:
            position: 持仓对象
        """
        pass

    async def start(self) -> None:
        """启动策略"""
        logger.info(f"策略启动: {self.__class__.__name__}", extra={
            'trader_id': self.trader.id,
        })

        await self.initialize()
        self.is_running = True

    async def stop(self) -> None:
        """停止策略"""
        logger.info(f"策略停止: {self.__class__.__name__}", extra={
            'trader_id': self.trader.id,
        })

        self.is_running = False

    async def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
    ) -> PositionSize:
        """
        计算仓位大小

        Args:
            signal: 交易信号
            current_price: 当前价格

        Returns:
            仓位大小
        """
        available_capital = self._get_available_capital()

        if self.position_size_method == 'fixed':
            # 固定金额
            fixed_amount = self.config.get('fixed_amount', 10000)
            quantity = int(fixed_amount / current_price / 100) * 100  # 整手
            value = quantity * current_price
            risk_percent = value / self.trader.equity

        elif self.position_size_method == 'kelly':
            # 凯利公式
            win_rate = signal.metadata.get('win_rate', 0.5)
            avg_win = signal.metadata.get('avg_win', 0.05)
            avg_loss = signal.metadata.get('avg_loss', 0.03)

            if avg_loss == 0:
                kelly_ratio = 0
            else:
                kelly_ratio = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss

            kelly_ratio = max(0, min(kelly_ratio, 0.25))  # 限制在0-25%

            value = available_capital * kelly_ratio
            quantity = int(value / current_price / 100) * 100
            risk_percent = kelly_ratio

        elif self.position_size_method == 'risk_parity':
            # 风险平价
            stop_loss_percent = self.config.get('stop_loss_percent', 0.02)
            risk_amount = self.trader.equity * self.risk_per_trade

            value = risk_amount / stop_loss_percent
            quantity = int(value / current_price / 100) * 100
            risk_percent = self.risk_per_trade

        else:
            # 默认：风险百分比
            value = available_capital * self.risk_per_trade
            quantity = int(value / current_price / 100) * 100
            risk_percent = self.risk_per_trade

        # 确保数量是100的整数倍（A股）
        quantity = max(100, (quantity // 100) * 100)

        return PositionSize(
            quantity=quantity,
            price=current_price,
            value=quantity * current_price,
            risk_percent=risk_percent,
        )

    def _get_available_capital(self) -> float:
        """获取可用资金"""
        # 计算已使用的资金
        used_capital = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )

        return self.trader.equity - used_capital

    async def validate_signal(self, signal: Signal) -> bool:
        """
        验证信号有效性

        Args:
            signal: 交易信号

        Returns:
            是否有效
        """
        # 检查策略是否运行
        if not self.is_running:
            logger.warning("策略未运行，忽略信号")
            return False

        # 检查交易员状态
        if not self.trader.is_active:
            logger.warning("交易员未激活，忽略信号")
            return False

        # 检查信号置信度
        if signal.confidence < self.config.get('min_confidence', 0.6):
            logger.info(f"信号置信度不足: {signal.confidence}")
            return False

        # 检查持仓数量限制
        if signal.action in ('buy', 'sell'):
            if len(self.positions) >= self.max_positions:
                logger.info(f"持仓数量已达上限: {len(self.positions)}")
                return False

        return True

    async def risk_check(self, signal: Signal, position_size: PositionSize) -> bool:
        """
        风险检查

        Args:
            signal: 交易信号
            position_size: 仓位大小

        Returns:
            是否通过风险检查
        """
        # 检查单笔风险
        if position_size.risk_percent > self.config.get('max_risk_per_trade', 0.05):
            logger.warning(f"单笔风险超限: {position_size.risk_percent:.2%}")
            return False

        # 检查总持仓风险
        total_risk = sum(
            pos.market_value * self.config.get('position_risk_factor', 0.02)
            for pos in self.positions.values()
        )
        total_risk += position_size.value * self.config.get('position_risk_factor', 0.02)

        if total_risk > self.trader.equity * self.config.get('max_portfolio_risk', 0.2):
            logger.warning(f"总持仓风险超限: {total_risk / self.trader.equity:.2%}")
            return False

        # 检查单只股票持仓比例
        max_stock_ratio = self.config.get('max_stock_ratio', 0.2)
        for pos in self.positions.values():
            if pos.symbol == signal.symbol:
                current_value = pos.market_value + position_size.value
                if current_value / self.trader.equity > max_stock_ratio:
                    logger.warning(f"单只股票持仓比例超限: {current_value / self.trader.equity:.2%}")
                    return False

        return True

    def get_state(self) -> Dict[str, Any]:
        """获取策略状态"""
        return {
            'strategy_name': self.__class__.__name__,
            'is_running': self.is_running,
            'positions_count': len(self.positions),
            'config': self.config,
        }
```

### 27.2 双均线策略实现

```python
# src/strategies/dual_moving_average.py
"""
双均线策略
"""

from typing import List
from datetime import datetime, timedelta
import numpy as np

from src.strategies.base import BaseStrategy, Signal
from src.models.market_data import MarketData
from src.core.logging import get_logger

logger = get_logger(__name__)


class DualMovingAverageStrategy(BaseStrategy):
    """
    双均线策略

    策略逻辑：
    1. 计算快速均线（短期）和慢速均线（长期）
    2. 快线上穿慢线（金叉）买入
    3. 快线下穿慢线（死叉）卖出
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 策略参数
        self.fast_period = self.config.get('fast_period', 5)  # 快线周期
        self.slow_period = self.config.get('slow_period', 20)  # 慢线周期
        self.stop_loss_percent = self.config.get('stop_loss_percent', 0.05)  # 止损百分比
        self.take_profit_percent = self.config.get('take_profit_percent', 0.15)  # 止盈百分比

        # 历史数据缓存
        self.price_history: Dict[str, List[float]] = {}

    async def initialize(self) -> None:
        """初始化策略"""
        logger.info("双均线策略初始化", extra={
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
        })

        # 这里可以加载历史数据
        # 例如：从数据库加载最近N天的K线数据

    async def on_bar(self, bar: MarketData) -> List[Signal]:
        """
        K线数据回调

        Args:
            bar: K线数据

        Returns:
            交易信号列表
        """
        # 更新市场数据
        self.market_data[bar.symbol] = bar

        # 更新价格历史
        if bar.symbol not in self.price_history:
            self.price_history[bar.symbol] = []

        self.price_history[bar.symbol].append(bar.close)

        # 保持历史数据长度
        max_history = max(self.fast_period, self.slow_period) * 2
        if len(self.price_history[bar.symbol]) > max_history:
            self.price_history[bar.symbol] = self.price_history[bar.symbol][-max_history:]

        # 检查是否有足够数据
        if len(self.price_history[bar.symbol]) < self.slow_period:
            return []

        # 计算均线
        fast_ma = self._calculate_ma(self.price_history[bar.symbol], self.fast_period)
        slow_ma = self._calculate_ma(self.price_history[bar.symbol], self.slow_period)

        # 计算上一根K线的均线
        prev_fast_ma = self._calculate_ma(
            self.price_history[bar.symbol][:-1],
            self.fast_period
        )
        prev_slow_ma = self._calculate_ma(
            self.price_history[bar.symbol][:-1],
            self.slow_period
        )

        signals = []

        # 判断交易信号
        if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
            # 金叉：买入信号
            signal = Signal(
                symbol=bar.symbol,
                exchange=bar.exchange,
                action='buy',
                confidence=self._calculate_buy_confidence(bar),
                reason=f"金叉: MA{self.fast_period} > MA{self.slow_period}",
                timestamp=bar.timestamp,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'current_price': bar.close,
                }
            )
            signals.append(signal)

        elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
            # 死叉：卖出信号
            signal = Signal(
                symbol=bar.symbol,
                exchange=bar.exchange,
                action='sell',
                confidence=self._calculate_sell_confidence(bar),
                reason=f"死叉: MA{self.fast_period} < MA{self.slow_period}",
                timestamp=bar.timestamp,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'current_price': bar.close,
                }
            )
            signals.append(signal)

        # 检查止损止盈
        if bar.symbol in self.positions:
            position = self.positions[bar.symbol]
            stop_loss_signal = self._check_stop_loss(position, bar)
            if stop_loss_signal:
                signals.append(stop_loss_signal)

            take_profit_signal = self._check_take_profit(position, bar)
            if take_profit_signal:
                signals.append(take_profit_signal)

        return signals

    async def on_tick(self, tick: MarketData) -> List[Signal]:
        """Tick数据回调（双均线策略通常不使用Tick）"""
        return []

    async def on_order_update(self, order) -> None:
        """订单更新回调"""
        logger.info("订单更新", extra={
            'order_id': order.id,
            'status': order.status,
        })

    async def on_position_update(self, position) -> None:
        """持仓更新回调"""
        self.positions[position.symbol] = position

        logger.debug("持仓更新", extra={
            'symbol': position.symbol,
            'quantity': position.quantity,
            'unrealized_pnl': position.unrealized_pnl,
        })

    def _calculate_ma(self, prices: List[float], period: int) -> float:
        """计算移动平均线"""
        if len(prices) < period:
            return 0.0
        return sum(prices[-period:]) / period

    def _calculate_buy_confidence(self, bar: MarketData) -> float:
        """计算买入置信度"""
        # 基础置信度
        confidence = 0.7

        # 成交量确认
        if bar.volume > 0:
            # 计算量比
            avg_volume = sum(
                self.market_data.get(bar.symbol, MarketData()).volume
                for _ in range(5)
            ) / 5

            if bar.volume > avg_volume * 1.5:
                confidence += 0.15  # 放量确认

        # 价格趋势确认
        prices = self.price_history.get(bar.symbol, [])
        if len(prices) >= 3:
            # 连续上涨
            if prices[-1] > prices[-2] > prices[-3]:
                confidence += 0.1

        return min(confidence, 1.0)

    def _calculate_sell_confidence(self, bar: MarketData) -> float:
        """计算卖出置信度"""
        # 基础置信度
        confidence = 0.7

        # 成交量确认
        if bar.volume > 0:
            avg_volume = sum(
                self.market_data.get(bar.symbol, MarketData()).volume
                for _ in range(5)
            ) / 5

            if bar.volume > avg_volume * 1.2:
                confidence += 0.15  # 放量确认

        return min(confidence, 1.0)

    def _check_stop_loss(self, position, bar: MarketData) -> Signal | None:
        """检查止损"""
        if position.side == 'long':
            loss_percent = (bar.close - position.entry_price) / position.entry_price

            if loss_percent <= -self.stop_loss_percent:
                return Signal(
                    symbol=position.symbol,
                    exchange=position.exchange,
                    action='sell',
                    confidence=1.0,
                    reason=f"止损触发: 亏损{-loss_percent:.2%}",
                    timestamp=bar.timestamp,
                )

        return None

    def _check_take_profit(self, position, bar: MarketData) -> Signal | None:
        """检查止盈"""
        if position.side == 'long':
            profit_percent = (bar.close - position.entry_price) / position.entry_price

            if profit_percent >= self.take_profit_percent:
                return Signal(
                    symbol=position.symbol,
                    exchange=position.exchange,
                    action='sell',
                    confidence=0.8,
                    reason=f"止盈触发: 盈利{profit_percent:.2%}",
                    timestamp=bar.timestamp,
                )

        return None
```

### 27.3 布林带策略实现

```python
# src/strategies/bollinger_bands.py
"""
布林带策略
"""

from typing import List
import numpy as np
from datetime import datetime

from src.strategies.base import BaseStrategy, Signal
from src.models.market_data import MarketData
from src.core.logging import get_logger

logger = get_logger(__name__)


class BollingerBandsStrategy(BaseStrategy):
    """
    布林带策略

    策略逻辑：
    1. 计算中轨（N日移动平均）
    2. 计算上下轨（中轨 ± K倍标准差）
    3. 价格触及下轨买入，触及上轨卖出
    4. 价格突破中轨加仓，跌破中轨减仓
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 策略参数
        self.period = self.config.get('period', 20)  # 均线周期
        self.std_multiplier = self.config.get('std_multiplier', 2)  # 标准差倍数
        self.stop_loss_atr_multiplier = self.config.get('stop_loss_atr_multiplier', 2)  # ATR止损倍数
        self.take_profit_bb = self.config.get('take_profit_bb', True)  # 是否使用布林带止盈

        # 历史数据
        self.price_history: Dict[str, List[float]] = {}

    async def initialize(self) -> None:
        """初始化策略"""
        logger.info("布林带策略初始化", extra={
            'period': self.period,
            'std_multiplier': self.std_multiplier,
        })

    async def on_bar(self, bar: MarketData) -> List[Signal]:
        """K线数据回调"""
        # 更新历史数据
        if bar.symbol not in self.price_history:
            self.price_history[bar.symbol] = []

        self.price_history[bar.symbol].append(bar.close)

        # 保持数据长度
        max_history = self.period * 2
        if len(self.price_history[bar.symbol]) > max_history:
            self.price_history[bar.symbol] = self.price_history[bar.symbol][-max_history:]

        # 检查数据充足性
        if len(self.price_history[bar.symbol]) < self.period:
            return []

        # 计算布林带
        bb = self._calculate_bollinger_bands(self.price_history[bar.symbol])
        current_price = bar.close

        signals = []

        # 触及下轨买入
        if current_price <= bb['lower']:
            signal = Signal(
                symbol=bar.symbol,
                exchange=bar.exchange,
                action='buy',
                confidence=self._calculate_buy_confidence(bb, current_price),
                reason=f"触及下轨: 价格 {current_price:.2f} <= 下轨 {bb['lower']:.2f}",
                timestamp=bar.timestamp,
                metadata={
                    'upper': bb['upper'],
                    'middle': bb['middle'],
                    'lower': bb['lower'],
                    'bb_width': bb['upper'] - bb['lower'],
                }
            )
            signals.append(signal)

        # 触及上轨卖出
        elif current_price >= bb['upper']:
            signal = Signal(
                symbol=bar.symbol,
                exchange=bar.exchange,
                action='sell',
                confidence=self._calculate_sell_confidence(bb, current_price),
                reason=f"触及上轨: 价格 {current_price:.2f} >= 上轨 {bb['upper']:.2f}",
                timestamp=bar.timestamp,
                metadata=bb,
            )
            signals.append(signal)

        # 检查止损止盈
        if bar.symbol in self.positions:
            position = self.positions[bar.symbol]

            # ATR止损
            atr = self._calculate_atr(self.price_history[bar.symbol], 14)
            stop_loss_signal = self._check_atr_stop_loss(position, bar, atr)
            if stop_loss_signal:
                signals.append(stop_loss_signal)

            # 布林带止盈
            if self.take_profit_bb:
                take_profit_signal = self._check_bb_take_profit(position, bb, bar)
                if take_profit_signal:
                    signals.append(take_profit_signal)

        return signals

    async def on_tick(self, tick: MarketData) -> List[Signal]:
        """Tick数据回调"""
        return []

    async def on_order_update(self, order) -> None:
        """订单更新回调"""
        pass

    async def on_position_update(self, position) -> None:
        """持仓更新回调"""
        self.positions[position.symbol] = position

    def _calculate_bollinger_bands(self, prices: List[float]) -> Dict[str, float]:
        """计算布林带"""
        prices_array = np.array(prices[-self.period:])

        middle = np.mean(prices_array)
        std = np.std(prices_array)

        upper = middle + self.std_multiplier * std
        lower = middle - self.std_multiplier * std

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'std': std,
        }

    def _calculate_atr(self, prices: List[float], period: int) -> float:
        """计算ATR（平均真实波幅）"""
        if len(prices) < period + 1:
            return 0.0

        # 简化版：使用价格标准差代替真实波幅
        prices_array = np.array(prices[-period:])
        return float(np.std(prices_array))

    def _calculate_buy_confidence(self, bb: Dict, current_price: float) -> float:
        """计算买入置信度"""
        # 基础置信度
        confidence = 0.6

        # 计算价格偏离下轨的程度
        deviation = (bb['lower'] - current_price) / bb['lower']

        # 偏离越大，置信度越高
        confidence += min(abs(deviation) * 2, 0.3)

        # 布林带收窄（低波动）时提高置信度
        bb_width = bb['upper'] - bb['lower']
        bb_width_percent = bb_width / bb['middle']

        if bb_width_percent < 0.05:  # 带宽小于5%
            confidence += 0.1

        return min(confidence, 1.0)

    def _calculate_sell_confidence(self, bb: Dict, current_price: float) -> float:
        """计算卖出置信度"""
        # 基础置信度
        confidence = 0.6

        # 计算价格偏离上轨的程度
        deviation = (current_price - bb['upper']) / bb['upper']

        # 偏离越大，置信度越高
        confidence += min(abs(deviation) * 2, 0.3)

        return min(confidence, 1.0)

    def _check_atr_stop_loss(self, position, bar: MarketData, atr: float) -> Signal | None:
        """ATR止损检查"""
        if position.side == 'long':
            stop_price = position.entry_price - atr * self.stop_loss_atr_multiplier

            if bar.close <= stop_price:
                return Signal(
                    symbol=position.symbol,
                    exchange=position.exchange,
                    action='sell',
                    confidence=1.0,
                    reason=f"ATR止损: 价格 {bar.close:.2f} <= 止损价 {stop_price:.2f}",
                    timestamp=bar.timestamp,
                )

        return None

    def _check_bb_take_profit(self, position, bb: Dict, bar: MarketData) -> Signal | None:
        """布林带止盈检查"""
        if position.side == 'long':
            # 持仓触及上轨止盈
            if bar.close >= bb['upper']:
                return Signal(
                    symbol=position.symbol,
                    exchange=position.exchange,
                    action='sell',
                    confidence=0.9,
                    reason=f"布林带止盈: 价格 {bar.close:.2f} >= 上轨 {bb['upper']:.2f}",
                    timestamp=bar.timestamp,
                )

        return None
```

### 27.4 AI驱动的动态策略

```python
# src/strategies/ai_driven.py
"""
AI驱动的动态策略
"""

from typing import List, Dict, Any
from datetime import datetime

from src.strategies.base import BaseStrategy, Signal
from src.models.market_data import MarketData
from src.services.llm_service import LLMService
from src.core.logging import get_logger

logger = get_logger(__name__)


class AIDrivenStrategy(BaseStrategy):
    """
    AI驱动策略

    使用LLM分析市场数据并生成交易信号
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # LLM服务
        self.llm_service = LLMService()

        # 策略参数
        self.llm_model = self.config.get('llm_model', 'deepseek-chat')
        self.temperature = self.config.get('temperature', 0.7)
        self.max_tokens = self.config.get('max_tokens', 2000)

        # 分析历史
        self.analysis_history: List[Dict] = []

    async def initialize(self) -> None:
        """初始化策略"""
        logger.info("AI驱动策略初始化", extra={
            'llm_model': self.llm_model,
        })

    async def on_bar(self, bar: MarketData) -> List[Signal]:
        """K线数据回调"""
        # 收集分析所需数据
        market_context = await self._build_market_context(bar)

        # 调用LLM进行分析
        analysis = await self._analyze_market(market_context)

        # 解析分析结果生成信号
        signals = await self._parse_analysis_to_signals(bar, analysis)

        return signals

    async def on_tick(self, tick: MarketData) -> List[Signal]:
        """Tick数据回调"""
        return []

    async def on_order_update(self, order) -> None:
        """订单更新回调"""
        # 记录订单状态供AI学习
        self.analysis_history.append({
            'type': 'order_update',
            'order_id': order.id,
            'status': order.status,
            'timestamp': datetime.now(),
        })

    async def on_position_update(self, position) -> None:
        """持仓更新回调"""
        self.positions[position.symbol] = position

        # 记录持仓变化供AI学习
        self.analysis_history.append({
            'type': 'position_update',
            'symbol': position.symbol,
            'unrealized_pnl': position.unrealized_pnl,
            'timestamp': datetime.now(),
        })

    async def _build_market_context(self, bar: MarketData) -> Dict[str, Any]:
        """构建市场分析上下文"""
        context = {
            'symbol': bar.symbol,
            'exchange': bar.exchange,
            'current_price': bar.close,
            'volume': bar.volume,
            'change': bar.change,
            'change_percent': bar.change_percent,
            'timestamp': bar.timestamp,
        }

        # 添加历史数据
        if bar.symbol in self.market_data:
            prev_data = self.market_data[bar.symbol]
            context['prev_close'] = prev_data.close
            context['prev_volume'] = prev_data.volume

        # 添加持仓信息
        if bar.symbol in self.positions:
            position = self.positions[bar.symbol]
            context['position'] = {
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_percent': position.unrealized_pnl / (position.entry_price * position.quantity),
            }

        # 添加历史分析
        if self.analysis_history:
            context['recent_analysis'] = self.analysis_history[-10:]

        return context

    async def _analyze_market(self, context: Dict[str, Any]) -> str:
        """使用LLM分析市场"""
        prompt = self._build_analysis_prompt(context)

        try:
            analysis = await self.llm_service.complete(
                prompt=prompt,
                model=self.llm_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return analysis

        except Exception as e:
            logger.error(f"AI分析失败: {e}")
            return ""

    def _build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """构建分析提示词"""
        prompt = f"""
你是一位专业的股票交易分析师。请基于以下市场数据进行分析，并给出交易建议。

## 市场数据

股票代码: {context['symbol']}
交易所: {context['exchange']}
当前价格: {context['current_price']:.2f}
成交量: {context['volume']}
涨跌幅: {context.get('change_percent', 0):.2%}

"""

        # 添加持仓信息
        if 'position' in context:
            pos = context['position']
            prompt += f"""
## 当前持仓

持仓数量: {pos['quantity']}
持仓成本: {pos['entry_price']:.2f}
未实现盈亏: {pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_percent']:.2%})

"""

        prompt += """
## 分析要求

请从以下维度进行分析：

1. **技术分析**: 判断当前价格趋势、支撑位、压力位
2. **市场情绪**: 判断市场情绪（乐观/悲观/中性）
3. **风险收益比**: 评估潜在收益与风险
4. **交易建议**: 明确给出建议（买入/卖出/持有）

请按照以下JSON格式输出：

```json
{
    "trend": "uptrend/downtrend/neutral",
    "sentiment": "bullish/bearish/neutral",
    "support_level": 0.00,
    "resistance_level": 0.00,
    "action": "buy/sell/hold",
    "confidence": 0.00,
    "reason": "原因说明",
    "stop_loss": 0.00,
    "take_profit": 0.00
}
```

请确保输出有效的JSON格式。
"""

        return prompt

    async def _parse_analysis_to_signals(
        self,
        bar: MarketData,
        analysis: str
    ) -> List[Signal]:
        """解析AI分析结果为交易信号"""
        import json

        try:
            # 提取JSON
            start_idx = analysis.find('{')
            end_idx = analysis.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                logger.warning("AI分析结果未找到JSON")
                return []

            json_str = analysis[start_idx:end_idx]
            result = json.loads(json_str)

            # 转换为信号
            if result.get('action') in ('buy', 'sell'):
                signal = Signal(
                    symbol=bar.symbol,
                    exchange=bar.exchange,
                    action=result['action'],
                    confidence=result.get('confidence', 0.7),
                    reason=result.get('reason', 'AI分析建议'),
                    timestamp=bar.timestamp,
                    metadata={
                        'trend': result.get('trend'),
                        'sentiment': result.get('sentiment'),
                        'support_level': result.get('support_level'),
                        'resistance_level': result.get('resistance_level'),
                        'stop_loss': result.get('stop_loss'),
                        'take_profit': result.get('take_profit'),
                        'analysis': analysis,
                    }
                )

                return [signal]

        except json.JSONDecodeError as e:
            logger.error(f"解析AI分析结果失败: {e}")
        except Exception as e:
            logger.error(f"处理AI分析结果失败: {e}")

        return []
```

---

## 第28章 实时数据处理系统

### 28.1 数据流处理架构

```python
# src/realtime/data_processor.py
"""
实时数据处理器
"""

import asyncio
from typing import Dict, Callable, Any, List
from collections import deque
from datetime import datetime, timedelta
import numpy as np

from src.models.market_data import MarketData, BarData, TickData
from src.core.logging import get_logger
from src.realtime.indicators import IndicatorCalculator

logger = get_logger(__name__)


class DataProcessor:
    """
    实时数据处理器

    功能：
    1. 接收Tick数据
    2. 聚合成K线数据
    3. 计算技术指标
    4. 触发策略回调
    """

    def __init__(self, bar_interval: int = 60):
        """
        初始化数据处理器

        Args:
            bar_interval: K线周期（秒），默认60秒（1分钟）
        """
        self.bar_interval = bar_interval

        # Tick数据缓存（用于合成K线）
        self.tick_buffers: Dict[str, deque] = {}

        # K线数据缓存
        self.bar_cache: Dict[str, List[BarData]] = {}

        # 订阅者
        self.subscribers: Dict[str, List[Callable]] = {
            'on_tick': [],
            'on_bar': [],
            'on_indicator': [],
        }

        # 指标计算器
        self.indicator_calculator = IndicatorCalculator()

        # 运行状态
        self.is_running = False

        logger.info(f"数据处理器初始化: K线周期={bar_interval}秒")

    async def start(self) -> None:
        """启动数据处理器"""
        if self.is_running:
            return

        self.is_running = True

        # 启动K线聚合任务
        asyncio.create_task(self._aggregate_bars())

        logger.info("数据处理器启动")

    async def stop(self) -> None:
        """停止数据处理器"""
        self.is_running = False
        logger.info("数据处理器停止")

    async def process_tick(self, tick: TickData) -> None:
        """
        处理Tick数据

        Args:
            tick: Tick数据
        """
        # 添加到Tick缓存
        symbol_key = f"{tick.symbol}.{tick.exchange}"

        if symbol_key not in self.tick_buffers:
            self.tick_buffers[symbol_key] = deque(maxlen=1000)

        self.tick_buffers[symbol_key].append(tick)

        # 触发Tick订阅者
        await self._notify_subscribers('on_tick', tick)

    async def _aggregate_bars(self) -> None:
        """聚合K线数据"""
        while self.is_running:
            try:
                now = datetime.now()

                # 处理每个股票的Tick缓存
                for symbol_key, ticks in self.tick_buffers.items():
                    if not ticks:
                        continue

                    # 获取当前K线的时间窗口
                    latest_tick = ticks[-1]
                    bar_time = self._get_bar_time(latest_tick.timestamp)

                    # 筛选当前时间窗口内的Tick
                    bar_ticks = [
                        t for t in ticks
                        if self._get_bar_time(t.timestamp) == bar_time
                    ]

                    if not bar_ticks:
                        continue

                    # 合成K线
                    bar = self._create_bar_from_ticks(bar_ticks, bar_time)

                    # 检查是否是新K线
                    if symbol_key not in self.bar_cache:
                        self.bar_cache[symbol_key] = []

                    if not self.bar_cache[symbol_key] or self.bar_cache[symbol_key][-1].timestamp != bar_time:
                        # 新K线
                        self.bar_cache[symbol_key].append(bar)

                        # 限制缓存大小
                        if len(self.bar_cache[symbol_key]) > 1000:
                            self.bar_cache[symbol_key] = self.bar_cache[symbol_key][-1000:]

                        # 计算指标
                        indicators = await self._calculate_indicators(symbol_key, bar)

                        # 触发K线订阅者
                        await self._notify_subscribers('on_bar', bar)

                        # 触发指标订阅者
                        await self._notify_subscribers('on_indicator', indicators)

                # 每秒检查一次
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"聚合K线失败: {e}")
                await asyncio.sleep(1)

    def _get_bar_time(self, timestamp: datetime) -> datetime:
        """获取K线时间（对齐到周期边界）"""
        # 向下取整到周期边界
        epoch_seconds = int(timestamp.timestamp())
        bar_epoch = (epoch_seconds // self.bar_interval) * self.bar_interval
        return datetime.fromtimestamp(bar_epoch)

    def _create_bar_from_ticks(self, ticks: List[TickData], bar_time: datetime) -> BarData:
        """从Tick数据合成K线"""
        # 按时间排序
        sorted_ticks = sorted(ticks, key=lambda t: t.timestamp)

        # OHLCV
        open_price = sorted_ticks[0].price
        high_price = max(t.price for t in ticks)
        low_price = min(t.price for t in ticks)
        close_price = sorted_ticks[-1].price
        volume = sum(t.volume for t in ticks)
        amount = sum(t.amount for t in ticks)

        # 提取symbol和exchange
        symbol = ticks[0].symbol
        exchange = ticks[0].exchange

        # 计算涨跌
        prev_bar = self._get_previous_bar(symbol, exchange)
        if prev_bar:
            change = close_price - prev_bar.close
            change_percent = (change / prev_bar.close) * 100
        else:
            change = 0.0
            change_percent = 0.0

        return BarData(
            symbol=symbol,
            exchange=exchange,
            timestamp=bar_time,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            amount=amount,
            change=change,
            change_percent=change_percent,
        )

    def _get_previous_bar(self, symbol: str, exchange: str) -> BarData | None:
        """获取上一根K线"""
        symbol_key = f"{symbol}.{exchange}"

        if symbol_key not in self.bar_cache or not self.bar_cache[symbol_key]:
            return None

        bars = self.bar_cache[symbol_key]
        if len(bars) < 2:
            return None

        return bars[-2]

    async def _calculate_indicators(
        self,
        symbol_key: str,
        bar: BarData
    ) -> Dict[str, Any]:
        """计算技术指标"""
        symbol, exchange = symbol_key.split('.')

        bars = self.bar_cache.get(symbol_key, [])

        if len(bars) < 20:
            return {}

        # 提取价格序列
        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [b.volume for b in bars]

        # 计算指标
        indicators = {
            'symbol': symbol,
            'exchange': exchange,
            'timestamp': bar.timestamp,
        }

        # MA指标
        for period in [5, 10, 20, 60]:
            ma = self.indicator_calculator.sma(closes, period)
            indicators[f'ma_{period}'] = ma

        # EMA指标
        for period in [12, 26]:
            ema = self.indicator_calculator.ema(closes, period)
            indicators[f'ema_{period}'] = ema

        # MACD
        macd = self.indicator_calculator.macd(closes)
        indicators.update(macd)

        # RSI
        rsi = self.indicator_calculator.rsi(closes, 14)
        indicators['rsi'] = rsi

        # 布林带
        bb = self.indicator_calculator.bollinger_bands(closes, 20, 2)
        indicators.update({
            'bb_upper': bb['upper'],
            'bb_middle': bb['middle'],
            'bb_lower': bb['lower'],
            'bb_width': bb['upper'] - bb['lower'],
        })

        # ATR
        atr = self.indicator_calculator.atr(highs, lows, closes, 14)
        indicators['atr'] = atr

        # 成交量指标
        indicators['volume_ma_5'] = self.indicator_calculator.sma(volumes, 5)

        return indicators

    def subscribe(self, event: str, callback: Callable) -> None:
        """
        订阅事件

        Args:
            event: 事件名称（on_tick/on_bar/on_indicator）
            callback: 回调函数
        """
        if event not in self.subscribers:
            raise ValueError(f"未知事件类型: {event}")

        self.subscribers[event].append(callback)

    def unsubscribe(self, event: str, callback: Callable) -> None:
        """取消订阅"""
        if event in self.subscribers and callback in self.subscribers[event]:
            self.subscribers[event].remove(callback)

    async def _notify_subscribers(self, event: str, data: Any) -> None:
        """通知订阅者"""
        for callback in self.subscribers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"订阅者回调失败: {e}")

    def get_latest_bar(self, symbol: str, exchange: str) -> BarData | None:
        """获取最新K线"""
        symbol_key = f"{symbol}.{exchange}"

        if symbol_key not in self.bar_cache or not self.bar_cache[symbol_key]:
            return None

        return self.bar_cache[symbol_key][-1]

    def get_bars(
        self,
        symbol: str,
        exchange: str,
        limit: int = 100
    ) -> List[BarData]:
        """获取K线序列"""
        symbol_key = f"{symbol}.{exchange}"

        if symbol_key not in self.bar_cache:
            return []

        bars = self.bar_cache[symbol_key]
        return bars[-limit:] if len(bars) > limit else bars
```

### 28.2 技术指标计算器

```python
# src/realtime/indicators.py
"""
技术指标计算器
"""

from typing import Dict, List, Optional
import numpy as np

from src.core.logging import get_logger

logger = get_logger(__name__)


class IndicatorCalculator:
    """技术指标计算器"""

    @staticmethod
    def sma(prices: List[float], period: int) -> Optional[float]:
        """
        简单移动平均（SMA）

        Args:
            prices: 价格序列
            period: 周期

        Returns:
            SMA值
        """
        if len(prices) < period:
            return None

        return float(np.mean(prices[-period:]))

    @staticmethod
    def ema(prices: List[float], period: int) -> Optional[float]:
        """
        指数移动平均（EMA）

        Args:
            prices: 价格序列
            period: 周期

        Returns:
            EMA值
        """
        if len(prices) < period:
            return None

        # 计算平滑系数
        multiplier = 2 / (period + 1)

        # 初始化EMA（使用第一个周期的SMA）
        ema = np.mean(prices[:period])

        # 递推计算
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return float(ema)

    @staticmethod
    def macd(
        prices: List[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, Optional[float]]:
        """
        MACD指标

        Args:
            prices: 价格序列
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期

        Returns:
            MACD相关指标字典
        """
        if len(prices) < slow_period:
            return {
                'macd': None,
                'macd_signal': None,
                'macd_hist': None,
            }

        # 计算快速和慢速EMA
        ema_fast = IndicatorCalculator.ema(prices, fast_period)
        ema_slow = IndicatorCalculator.ema(prices, slow_period)

        # MACD线
        macd = ema_fast - ema_slow

        # 计算MACD历史用于信号线
        # 这里简化处理，实际应该维护完整的历史
        macd_signal = macd * 0.2  # 简化
        macd_hist = macd - macd_signal

        return {
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
        }

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """
        相对强弱指标（RSI）

        Args:
            prices: 价格序列
            period: 周期

        Returns:
            RSI值（0-100）
        """
        if len(prices) < period + 1:
            return None

        # 计算价格变化
        deltas = np.diff(prices[-period-1:])

        # 分离上涨和下跌
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # 计算平均涨跌
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        # 计算RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    @staticmethod
    def bollinger_bands(
        prices: List[float],
        period: int = 20,
        std_multiplier: float = 2
    ) -> Dict[str, float]:
        """
        布林带

        Args:
            prices: 价格序列
            period: 周期
            std_multiplier: 标准差倍数

        Returns:
            布林带指标
        """
        if len(prices) < period:
            return {
                'upper': 0.0,
                'middle': 0.0,
                'lower': 0.0,
                'std': 0.0,
            }

        prices_array = np.array(prices[-period:])

        middle = float(np.mean(prices_array))
        std = float(np.std(prices_array))

        upper = middle + std_multiplier * std
        lower = middle - std_multiplier * std

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'std': std,
        }

    @staticmethod
    def atr(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> Optional[float]:
        """
        平均真实波幅（ATR）

        Args:
            highs: 最高价序列
            lows: 最低价序列
            closes: 收盘价序列
            period: 周期

        Returns:
            ATR值
        """
        if len(closes) < period + 1:
            return None

        true_ranges = []

        for i in range(1, len(closes)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )

            true_ranges.append(tr)

        return float(np.mean(true_ranges[-period:]))

    @staticmethod
    def stochastic(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, Optional[float]]:
        """
        随机指标（Stochastic）

        Args:
            highs: 最高价序列
            lows: 最低价序列
            closes: 收盘价序列
            k_period: K值周期
            d_period: D值周期

        Returns:
            K值和D值
        """
        if len(closes) < k_period:
            return {
                'k': None,
                'd': None,
            }

        # 计算K值
        recent_highs = highs[-k_period:]
        recent_lows = lows[-k_period:]
        current_close = closes[-1]

        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)

        if highest_high == lowest_low:
            k = 50.0
        else:
            k = 100 * (current_close - lowest_low) / (highest_high - lowest_low)

        # 计算D值（K的简单移动平均）
        # 这里简化处理
        d = k  # 简化

        return {
            'k': k,
            'd': d,
        }

    @staticmethod
    def obv(prices: List[float], volumes: List[float]) -> Optional[float]:
        """
        能量潮（OBV）

        Args:
            prices: 价格序列
            volumes: 成交量序列

        Returns:
            OBV值
        """
        if len(prices) != len(volumes) or len(prices) < 2:
            return None

        obv = 0

        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]

        return float(obv)
```

---

## 第29章 风险管理系统

### 29.1 实时风险监控

```python
# src/risk/realtime_monitor.py
"""
实时风险监控系统
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.models.trader import TraderModel
from src.models.position import PositionModel
from src.core.logging import get_logger
from src.services.notification_service import NotificationService

logger = get_logger(__name__)


@dataclass
class RiskAlert:
    """风险告警"""
    level: str  # 'info', 'warning', 'critical'
    type: str  # 'position_limit', 'loss_limit', 'correlation', etc.
    message: str
    trader_id: str
    timestamp: datetime
    metadata: Dict = None


class RiskMonitor:
    """实时风险监控器"""

    def __init__(
        self,
        notification_service: NotificationService,
    ):
        """
        初始化风险监控器

        Args:
            notification_service: 通知服务
        """
        self.notification_service = notification_service

        # 风险阈值配置
        self.thresholds = {
            'max_position_ratio': 0.2,  # 单个持仓最大比例
            'max_daily_loss': 0.05,  # 最大单日亏损
            'max_drawdown': 0.15,  # 最大回撤
            'max_correlation': 0.8,  # 最大相关性
        }

        # 监控状态
        self.is_running = False

        # 风险记录
        self.risk_history: Dict[str, List] = {}

        # 告警回调
        self.alert_callbacks: List = []

    async def start(self) -> None:
        """启动风险监控"""
        if self.is_running:
            return

        self.is_running = True

        # 启动监控任务
        asyncio.create_task(self._monitor_positions())
        asyncio.create_task(self._monitor_portfolio())
        asyncio.create_task(self._monitor_correlation())

        logger.info("风险监控系统启动")

    async def stop(self) -> None:
        """停止风险监控"""
        self.is_running = False
        logger.info("风险监控系统停止")

    async def _monitor_positions(self) -> None:
        """监控持仓风险"""
        while self.is_running:
            try:
                # 这里应该从数据库获取所有活跃的持仓
                # 简化示例

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"持仓监控失败: {e}")
                await asyncio.sleep(5)

    async def _monitor_portfolio(self) -> None:
        """监控组合风险"""
        while self.is_running:
            try:
                # 计算组合指标

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"组合监控失败: {e}")
                await asyncio.sleep(10)

    async def _monitor_correlation(self) -> None:
        """监控相关性风险"""
        while self.is_running:
            try:
                # 计算持仓相关性

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"相关性监控失败: {e}")
                await asyncio.sleep(60)

    async def check_position_risk(
        self,
        trader: TraderModel,
        position: PositionModel
    ) -> List[RiskAlert]:
        """
        检查持仓风险

        Args:
            trader: 交易员
            position: 持仓

        Returns:
            风险告警列表
        """
        alerts = []

        # 检查持仓比例
        position_ratio = position.market_value / trader.equity

        if position_ratio > self.thresholds['max_position_ratio']:
            alert = RiskAlert(
                level='warning',
                type='position_limit',
                message=f"持仓比例超限: {position_ratio:.2%}",
                trader_id=trader.id,
                timestamp=datetime.now(),
                metadata={
                    'symbol': position.symbol,
                    'ratio': position_ratio,
                    'limit': self.thresholds['max_position_ratio'],
                }
            )
            alerts.append(alert)

        # 检查未实现亏损
        if position.unrealized_pnl < 0:
            loss_ratio = abs(position.unrealized_pnl) / position.market_value

            if loss_ratio > self.thresholds['max_daily_loss']:
                alert = RiskAlert(
                    level='critical',
                    type='loss_limit',
                    message=f"持仓亏损超限: {loss_ratio:.2%}",
                    trader_id=trader.id,
                    timestamp=datetime.now(),
                    metadata={
                        'symbol': position.symbol,
                        'loss': position.unrealized_pnl,
                        'loss_ratio': loss_ratio,
                    }
                )
                alerts.append(alert)

        # 发送告警
        for alert in alerts:
            await self._send_alert(alert)

        return alerts

    async def check_portfolio_risk(
        self,
        trader: TraderModel,
        positions: List[PositionModel],
        current_equity: float,
        high_watermark: float
    ) -> List[RiskAlert]:
        """
        检查组合风险

        Args:
            trader: 交易员
            positions: 持仓列表
            current_equity: 当前权益
            high_watermark: 历史最高权益

        Returns:
            风险告警列表
        """
        alerts = []

        # 计算回撤
        drawdown = (high_watermark - current_equity) / high_watermark

        if drawdown > self.thresholds['max_drawdown']:
            alert = RiskAlert(
                level='critical',
                type='drawdown_limit',
                message=f"回撤超限: {drawdown:.2%}",
                trader_id=trader.id,
                timestamp=datetime.now(),
                metadata={
                    'current_equity': current_equity,
                    'high_watermark': high_watermark,
                    'drawdown': drawdown,
                }
            )
            alerts.append(alert)

        # 计算集中度风险
        total_value = sum(p.market_value for p in positions)

        for position in positions:
            concentration = position.market_value / total_value

            if concentration > self.thresholds['max_position_ratio']:
                alert = RiskAlert(
                    level='warning',
                    type='concentration',
                    message=f"持仓集中度过高: {concentration:.2%}",
                    trader_id=trader.id,
                    timestamp=datetime.now(),
                    metadata={
                        'symbol': position.symbol,
                        'concentration': concentration,
                    }
                )
                alerts.append(alert)

        # 发送告警
        for alert in alerts:
            await self._send_alert(alert)

        return alerts

    async def check_correlation_risk(
        self,
        positions: List[PositionModel],
        price_history: Dict[str, List[float]]
    ) -> List[RiskAlert]:
        """
        检查相关性风险

        Args:
            positions: 持仓列表
            price_history: 价格历史数据

        Returns:
            风险告警列表
        """
        alerts = []

        if len(positions) < 2:
            return alerts

        # 计算收益率相关系数
        symbols = [p.symbol for p in positions]

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                if symbol1 not in price_history or symbol2 not in price_history:
                    continue

                prices1 = price_history[symbol1]
                prices2 = price_history[symbol2]

                if len(prices1) < 20 or len(prices2) < 20:
                    continue

                # 计算收益率
                returns1 = [prices1[k] / prices1[k-1] - 1 for k in range(1, len(prices1))]
                returns2 = [prices2[k] / prices2[k-1] - 1 for k in range(1, len(prices2))]

                # 计算相关系数
                correlation = self._calculate_correlation(returns1, returns2)

                if abs(correlation) > self.thresholds['max_correlation']:
                    alert = RiskAlert(
                        level='warning',
                        type='high_correlation',
                        message=f"持仓相关性过高: {symbol1} - {symbol2} = {correlation:.2f}",
                        trader_id=positions[0].trader_id,
                        timestamp=datetime.now(),
                        metadata={
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': correlation,
                        }
                    )
                    alerts.append(alert)

        # 发送告警
        for alert in alerts:
            await self._send_alert(alert)

        return alerts

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """计算相关系数"""
        import numpy as np

        min_len = min(len(x), len(y))
        x_array = np.array(x[:min_len])
        y_array = np.array(y[:min_len])

        return float(np.corrcoef(x_array, y_array)[0, 1])

    async def _send_alert(self, alert: RiskAlert) -> None:
        """发送告警"""
        # 记录告警
        if alert.trader_id not in self.risk_history:
            self.risk_history[alert.trader_id] = []

        self.risk_history[alert.trader_id].append(alert)

        # 限制历史记录长度
        if len(self.risk_history[alert.trader_id]) > 1000:
            self.risk_history[alert.trader_id] = self.risk_history[alert.trader_id][-1000:]

        # 发送通知
        await self.notification_service.send_alert(alert)

        # 触发回调
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"告警回调失败: {e}")

        logger.warning(f"风险告警: {alert.message}", extra={
            'level': alert.level,
            'type': alert.type,
            'trader_id': alert.trader_id,
        })

    def subscribe_alerts(self, callback) -> None:
        """订阅告警"""
        self.alert_callbacks.append(callback)
```

### 29.2 风险限额管理

```python
# src/risk/limits.py
"""
风险限额管理
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

from src.models.trader import TraderModel
from src.models.order import OrderModel
from src.core.logging import get_logger

logger = get_logger(__name__)


class LimitType(Enum):
    """限额类型"""
    DAILY_LOSS = "daily_loss"  # 单日亏损限额
    MAX_POSITION = "max_position"  # 最大持仓限额
    MAX_POSITION_RATIO = "max_position_ratio"  # 单股持仓比例
    MAX_CORRELATION = "max_correlation"  # 相关性限额
    TURNOVER_RATE = "turnover_rate"  # 换手率限额


@dataclass
class RiskLimit:
    """风险限额"""
    type: LimitType
    value: float
    period: Optional[timedelta] = None  # 限额周期
    current_usage: float = 0.0  # 当前使用量
    last_reset: datetime = None  # 上次重置时间

    def is_exceeded(self) -> bool:
        """检查是否超限"""
        return self.current_usage >= self.value

    def usage_percent(self) -> float:
        """使用百分比"""
        if self.value == 0:
            return 0.0
        return self.current_usage / self.value


class RiskLimitManager:
    """风险限额管理器"""

    def __init__(self):
        """初始化风险限额管理器"""
        self.limits: Dict[str, Dict[LimitType, RiskLimit]] = {}

    def set_limit(
        self,
        trader_id: str,
        limit_type: LimitType,
        value: float,
        period: Optional[timedelta] = None
    ) -> None:
        """
        设置风险限额

        Args:
            trader_id: 交易员ID
            limit_type: 限额类型
            value: 限额值
            period: 限额周期
        """
        if trader_id not in self.limits:
            self.limits[trader_id] = {}

        self.limits[trader_id][limit_type] = RiskLimit(
            type=limit_type,
            value=value,
            period=period,
            current_usage=0.0,
            last_reset=datetime.now(),
        )

        logger.info(f"设置风险限额", extra={
            'trader_id': trader_id,
            'limit_type': limit_type.value,
            'value': value,
        })

    def check_order(
        self,
        trader: TraderModel,
        order: OrderModel,
        current_positions: Dict[str, PositionModel]
    ) -> tuple[bool, Optional[str]]:
        """
        检查订单是否符合风险限额

        Args:
            trader: 交易员
            order: 订单
            current_positions: 当前持仓

        Returns:
            (是否通过, 拒绝原因)
        """
        # 检查最大持仓数
        if LimitType.MAX_POSITION in self.limits.get(trader.id, {}):
            limit = self.limits[trader.id][LimitType.MAX_POSITION]

            if len(current_positions) >= limit.value:
                return False, f"超过最大持仓数限制: {len(current_positions)}/{limit.value}"

        # 检查单股持仓比例
        if order.symbol in current_positions:
            position = current_positions[order.symbol]

            if LimitType.MAX_POSITION_RATIO in self.limits.get(trader.id, {}):
                limit = self.limits[trader.id][LimitType.MAX_POSITION_RATIO]

                # 计算订单成交后的持仓比例
                new_value = position.market_value + (order.quantity * order.price)
                new_ratio = new_value / trader.equity

                if new_ratio > limit.value:
                    return False, f"超过单股持仓比例限制: {new_ratio:.2%}/{limit.value:.2%}"

        # 检查单日亏损限额
        if LimitType.DAILY_LOSS in self.limits.get(trader.id, {}):
            limit = self.limits[trader.id][LimitType.DAILY_LOSS]

            # 重置每日限额
            if limit.last_reset.date() != datetime.now().date():
                limit.current_usage = 0.0
                limit.last_reset = datetime.now()

            if limit.is_exceeded():
                return False, f"超过单日亏损限额: {limit.current_usage:.2f}/{limit.value:.2f}"

        return True, None

    def update_daily_loss(
        self,
        trader_id: str,
        realized_pnl: float
    ) -> None:
        """
        更新单日亏损

        Args:
            trader_id: 交易员ID
            realized_pnl: 已实现盈亏
        """
        if trader_id not in self.limits:
            return

        if LimitType.DAILY_LOSS not in self.limits[trader_id]:
            return

        limit = self.limits[trader_id][LimitType.DAILY_LOSS]

        # 只计算亏损
        if realized_pnl < 0:
            limit.current_usage += abs(realized_pnl)

            # 记录日志
            if limit.is_exceeded():
                logger.error(f"触发单日亏损限额", extra={
                    'trader_id': trader_id,
                    'loss': limit.current_usage,
                    'limit': limit.value,
                })

    def check_and_reset_daily_limits(self, trader_id: str) -> None:
        """
        检查并重置每日限额

        Args:
            trader_id: 交易员ID
        """
        if trader_id not in self.limits:
            return

        for limit_type, limit in self.limits[trader_id].items():
            if limit.period and limit.last_reset:
                # 检查是否需要重置
                time_since_reset = datetime.now() - limit.last_reset

                if time_since_reset >= limit.period:
                    limit.current_usage = 0.0
                    limit.last_reset = datetime.now()

                    logger.info(f"重置风险限额", extra={
                        'trader_id': trader_id,
                        'limit_type': limit_type.value,
                    })
```

---

## 第30章 回测系统详细实现

### 30.1 回测引擎核心

```python
# src/backtest/engine.py
"""
回测引擎
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

from src.strategies.base import BaseStrategy
from src.models.trader import TraderModel
from src.models.order import OrderModel
from src.models.trade import TradeModel
from src.backtest.broker import BacktestBroker
from src.backtest.data_feed import BacktestDataFeed
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float = 0.0003  # 万三佣金
    slippage_rate: float = 0.001  # 滑点
    benchmark: Optional[str] = None  # 基准指数


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: pd.Series
    trades: List[TradeModel]


class BacktestEngine:
    """
    回测引擎

    功能：
    1. 加载历史数据
    2. 模拟交易执行
    3. 计算绩效指标
    4. 生成分析报告
    """

    def __init__(self, config: BacktestConfig):
        """
        初始化回测引擎

        Args:
            config: 回测配置
        """
        self.config = config

        # 数据源
        self.data_feed = BacktestDataFeed(
            start_date=config.start_date,
            end_date=config.end_date,
        )

        # 模拟券商
        self.broker = BacktestBroker(
            initial_capital=config.initial_capital,
            commission_rate=config.commission_rate,
            slippage_rate=config.slippage_rate,
        )

        # 策略实例
        self.strategy: Optional[BaseStrategy] = None

        logger.info("回测引擎初始化", extra={
            'start_date': config.start_date,
            'end_date': config.end_date,
            'initial_capital': config.initial_capital,
        })

    async def run(self, strategy: BaseStrategy) -> BacktestResult:
        """
        运行回测

        Args:
            strategy: 交易策略

        Returns:
            回测结果
        """
        logger.info(f"开始回测: {strategy.__class__.__name__}")

        self.strategy = strategy

        # 加载数据
        await self.data_feed.load_data()

        # 初始化策略
        await self.strategy.initialize()

        # 获取所有交易日期
        trading_dates = self.data_feed.get_trading_dates()

        # 逐日回放
        for date in trading_dates:
            await self._process_day(date)

        # 平仓
        await self._close_all_positions()

        # 计算结果
        result = await self._calculate_result()

        logger.info("回测完成", extra={
            'total_return': f"{result.total_return:.2%}",
            'sharpe_ratio': f"{result.sharpe_ratio:.2f}",
            'max_drawdown': f"{result.max_drawdown:.2%}",
        })

        return result

    async def _process_day(self, date: datetime) -> None:
        """
        处理单个交易日

        Args:
            date: 交易日期
        """
        # 获取当日K线数据
        bars = await self.data_feed.get_bars(date)

        for bar in bars:
            # 更新市场数据
            self.broker.update_market_data(bar)

            # 策略回调
            signals = await self.strategy.on_bar(bar)

            # 处理信号
            for signal in signals:
                if not await self.strategy.validate_signal(signal):
                    continue

                # 计算仓位
                position_size = await self.strategy.calculate_position_size(
                    signal,
                    bar.close,
                )

                # 风险检查
                if not await self.strategy.risk_check(signal, position_size):
                    continue

                # 执行交易
                await self._execute_signal(signal, position_size, bar)

        # 日终结算
        await self.broker.settle(date)

    async def _execute_signal(
        self,
        signal,
        position_size,
        bar
    ) -> None:
        """
        执行交易信号

        Args:
            signal: 交易信号
            position_size: 仓位大小
            bar: K线数据
        """
        if signal.action == 'buy':
            # 创建买单
            order = await self.broker.create_order(
                symbol=signal.symbol,
                exchange=signal.exchange,
                side='buy',
                order_type='market',
                quantity=position_size.quantity,
                price=bar.close,
            )

            await self.strategy.on_order_update(order)

        elif signal.action == 'sell':
            # 创建卖单
            order = await self.broker.create_order(
                symbol=signal.symbol,
                exchange=signal.exchange,
                side='sell',
                order_type='market',
                quantity=position_size.quantity,
                price=bar.close,
            )

            await self.strategy.on_order_update(order)

    async def _close_all_positions(self) -> None:
        """平掉所有持仓"""
        positions = await self.broker.get_all_positions()

        for position in positions:
            # 获取最新价格
            latest_price = await self.data_feed.get_latest_price(
                position.symbol,
                position.exchange,
            )

            if latest_price:
                # 平仓
                await self.broker.create_order(
                    symbol=position.symbol,
                    exchange=position.exchange,
                    side='sell',
                    order_type='market',
                    quantity=position.quantity,
                    price=latest_price,
                )

    async def _calculate_result(self) -> BacktestResult:
        """计算回测结果"""
        # 获取权益曲线
        equity_curve = await self.broker.get_equity_curve()

        # 计算收益率
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # 计算年化收益率
        days = (self.config.end_date - self.config.start_date).days
        annual_return = (1 + total_return) ** (365 / days) - 1

        # 计算夏普比率
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0

        # 计算最大回撤
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 获取成交记录
        trades = await self.broker.get_all_trades()

        # 计算胜率
        winning_trades = [t for t in trades if t.realized_pnl > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0

        # 计算盈亏比
        total_profit = sum(t.realized_pnl for t in winning_trades) if winning_trades else 0
        total_loss = sum(abs(t.realized_pnl) for t in trades if t.realized_pnl < 0)
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            equity_curve=equity_curve,
            trades=trades,
        )
```

### 30.2 模拟券商实现

```python
# src/backtest/broker.py
"""
模拟券商
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from src.models.order import OrderModel, OrderStatus
from src.models.trade import TradeModel
from src.models.position import PositionModel
from src.core.logging import get_logger

logger = get_logger(__name__)


class BacktestBroker:
    """
    模拟券商

    功能：
    1. 订单管理
    2. 持仓管理
    3. 成交模拟
    4. 资金管理
    """

    def __init__(
        self,
        initial_capital: float,
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.001,
    ):
        """
        初始化模拟券商

        Args:
            initial_capital: 初始资金
            commission_rate: 佣金费率
            slippage_rate: 滑点率
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # 资金
        self.cash = initial_capital
        self.equity = initial_capital

        # 订单
        self.orders: List[OrderModel] = []
        self.trades: List[TradeModel] = []

        # 持仓
        self.positions: Dict[str, PositionModel] = {}

        # 市场数据缓存
        self.market_data: Dict[str, any] = {}

        # 权益曲线
        self.equity_history: List[tuple[datetime, float]] = []

        logger.info("模拟券商初始化", extra={
            'initial_capital': initial_capital,
            'commission_rate': commission_rate,
            'slippage_rate': slippage_rate,
        })

    def update_market_data(self, bar) -> None:
        """更新市场数据"""
        key = f"{bar.symbol}.{bar.exchange}"
        self.market_data[key] = bar

    async def create_order(
        self,
        symbol: str,
        exchange: str,
        side: str,
        order_type: str,
        quantity: int,
        price: float,
    ) -> OrderModel:
        """
        创建订单

        Args:
            symbol: 股票代码
            exchange: 交易所
            side: 买卖方向
            order_type: 订单类型
            quantity: 数量
            price: 价格

        Returns:
            订单对象
        """
        # 创建订单
        order = OrderModel(
            id=self._generate_order_id(),
            symbol=symbol,
            exchange=exchange,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.OPEN,
            created_at=datetime.now(),
        )

        self.orders.append(order)

        # 模拟成交
        await self._execute_order(order)

        return order

    async def _execute_order(self, order: OrderModel) -> None:
        """
        执行订单（模拟成交）

        Args:
            order: 订单对象
        """
        # 获取市场数据
        key = f"{order.symbol}.{order.exchange}"
        market_data = self.market_data.get(key)

        if not market_data:
            order.status = OrderStatus.REJECTED
            order.rejected_reason = "No market data"
            return

        # 计算成交价格（考虑滑点）
        if order.side == 'buy':
            execution_price = market_data.close * (1 + self.slippage_rate)
        else:
            execution_price = market_data.close * (1 - self.slippage_rate)

        # 计算成交金额
        trade_value = order.quantity * execution_price

        # 计算佣金
        commission = trade_value * self.commission_rate

        # 检查资金（买单）
        if order.side == 'buy':
            required_cash = trade_value + commission

            if self.cash < required_cash:
                order.status = OrderStatus.REJECTED
                order.rejected_reason = "Insufficient funds"
                return

            self.cash -= required_cash

        # 检查持仓（卖单）
        else:
            key = f"{order.symbol}.{order.exchange}"
            position = self.positions.get(key)

            if not position or position.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                order.rejected_reason = "Insufficient position"
                return

        # 创建成交记录
        trade = TradeModel(
            id=self._generate_trade_id(),
            order_id=order.id,
            symbol=order.symbol,
            exchange=order.exchange,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            commission=commission,
            timestamp=datetime.now(),
        )

        self.trades.append(trade)

        # 更新持仓
        await self._update_position(trade)

        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = execution_price
        order.filled_at = datetime.now()

    async def _update_position(self, trade: TradeModel) -> None:
        """
        更新持仓

        Args:
            trade: 成交记录
        """
        key = f"{trade.symbol}.{trade.exchange}"

        if key not in self.positions:
            # 新建持仓
            if trade.side == 'buy':
                self.positions[key] = PositionModel(
                    id=self._generate_position_id(),
                    trader_id="backtest",
                    symbol=trade.symbol,
                    exchange=trade.exchange,
                    quantity=trade.quantity,
                    entry_price=trade.price,
                    current_price=trade.price,
                    market_value=trade.quantity * trade.price,
                    unrealized_pnl=0.0,
                    created_at=datetime.now(),
                )
        else:
            # 更新持仓
            position = self.positions[key]

            if trade.side == 'buy':
                # 加仓
                total_cost = position.quantity * position.entry_price + trade.quantity * trade.price
                total_quantity = position.quantity + trade.quantity

                position.entry_price = total_cost / total_quantity
                position.quantity = total_quantity
            else:
                # 减仓/平仓
                position.quantity -= trade.quantity

                if position.quantity == 0:
                    # 完全平仓，删除持仓
                    del self.positions[key]
                    return

            # 更新当前价格和市值
            key = f"{trade.symbol}.{trade.exchange}"
            market_data = self.market_data.get(key)

            if market_data:
                position.current_price = market_data.close
            else:
                position.current_price = trade.price

            position.market_value = position.quantity * position.current_price
            position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity

    async def settle(self, date: datetime) -> None:
        """
        日终结算

        Args:
            date: 结算日期
        """
        # 更新持仓市值
        for position in self.positions.values():
            key = f"{position.symbol}.{position.exchange}"
            market_data = self.market_data.get(key)

            if market_data:
                position.current_price = market_data.close
                position.market_value = position.quantity * market_data.close
                position.unrealized_pnl = (market_data.close - position.entry_price) * position.quantity

        # 计算总权益
        total_position_value = sum(p.market_value for p in self.positions.values())
        self.equity = self.cash + total_position_value

        # 记录权益曲线
        self.equity_history.append((date, self.equity))

    async def get_equity_curve(self) -> pd.Series:
        """获取权益曲线"""
        if not self.equity_history:
            return pd.Series([self.initial_capital])

        dates, values = zip(*self.equity_history)
        return pd.Series(values, index=dates)

    async def get_all_positions(self) -> List[PositionModel]:
        """获取所有持仓"""
        return list(self.positions.values())

    async def get_all_trades(self) -> List[TradeModel]:
        """获取所有成交"""
        return self.trades

    def _generate_order_id(self) -> str:
        """生成订单ID"""
        return f"order_{len(self.orders)}"

    def _generate_trade_id(self) -> str:
        """生成成交ID"""
        return f"trade_{len(self.trades)}"

    def _generate_position_id(self) -> str:
        """生成持仓ID"""
        return f"position_{len(self.positions)}"
```

---

## 总结

本补充文档涵盖更详细的实现内容：

**第26章 - API设计详细规范**
- RESTful API完整设计（版本控制、通用响应模型、依赖注入）
- 认证API端点（注册、登录、刷新、登出）
- 交易员API端点（CRUD操作、统计信息、启停控制）
- 订单API端点（创建、查询、修改、取消）
- WebSocket实时推送（市场数据、订单状态）

**第27章 - 交易策略详细实现**
- 策略基类架构（信号生成、仓位计算、风险检查）
- 双均线策略（金叉死叉、止损止盈）
- 布林带策略（上下轨突破、ATR止损）
- AI驱动策略（LLM分析、动态决策）

**第28章 - 实时数据处理**
- 数据流处理架构（Tick聚合、K线合成、指标计算）
- 技术指标计算器（SMA、EMA、MACD、RSI、布林带、ATR、Stochastic、OBV）

**第29章 - 风险管理系统**
- 实时风险监控（持仓风险、组合风险、相关性风险）
- 风险限额管理（单日亏损、持仓比例、相关性限额）
- 风险告警机制

**第30章 - 回测系统**
- 回测引擎核心（数据加载、策略执行、绩效计算）
- 模拟券商实现（订单管理、持仓管理、成交模拟）

---

*本文档持续更新中...*

*最后更新: 2026-01-05*
