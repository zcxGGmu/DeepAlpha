"""Freqtrade客户端，对应 Go 版本的 client.go"""

import ssl
from typing import Any, Dict, List, Optional

import aiohttp

from deepalpha.executor.freqtrade.types import (
    ForceEnterPayload,
    ForceEnterResponse,
    ForceExitPayload,
    Trade,
    Balance,
    APIPosition,
    PositionListOptions,
    PositionListResult,
)
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class FreqtradeClient:
    """Freqtrade客户端，对应 Go 版本的 Client"""

    def __init__(
        self,
        api_url: str,
        username: str = "",
        password: str = "",
        token: str = "",
        timeout: int = 15,
        insecure_skip_verify: bool = False
    ):
        if not api_url:
            raise ValueError("api_url cannot be empty")

        # 标准化URL
        if not api_url.endswith('/'):
            api_url += '/'

        self.base_url = api_url
        self.username = username
        self.password = password
        self.token = token
        self.timeout = timeout
        self.insecure_skip_verify = insecure_skip_verify

        # 创建会话
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if not self._session:
            # 配置SSL
            ssl_context = ssl.create_default_context()
            if self.insecure_skip_verify:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            # 配置连接器
            connector = aiohttp.TCPConnector(ssl=ssl_context)

            # 配置超时
            timeout = aiohttp.ClientTimeout(total=self.timeout)

            # 配置认证头
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            elif self.username and self.password:
                import base64
                credentials = base64.b64encode(
                    f"{self.username}:{self.password}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"

            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers
            )
        return self._session

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        response_model: Optional[type] = None
    ) -> Any:
        """执行HTTP请求"""
        url = f"{self.base_url}{endpoint}"
        session = await self._get_session()

        try:
            async with session.request(
                method=method,
                url=url,
                json=data if isinstance(data, (dict, list)) else None,
                data=data if isinstance(data, (str, bytes)) else None,
                params=params
            ) as response:
                response.raise_for_status()

                if response_model:
                    result = await response.json()
                    return response_model(**result) if isinstance(result, dict) else result
                else:
                    return await response.json()

        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP请求失败: {e.status} {e.message}")
            raise
        except Exception as e:
            logger.error(f"请求异常: {e}")
            raise

    async def force_enter(
        self,
        payload: ForceEnterPayload
    ) -> ForceEnterResponse:
        """强制开仓，对应 Go 版本的 ForceEnter"""
        response = await self._request(
            "POST",
            "/api/v1/forceenter",
            data=payload.dict(exclude_none=True),
            response_model=ForceEnterResponse
        )

        if response.trade_id == 0:
            raise ValueError("freqtrade 未返回 trade_id")

        logger.info(f"成功开仓, trade_id: {response.trade_id}")
        return response

    async def force_exit(
        self,
        payload: ForceExitPayload
    ) -> None:
        """强制平仓，对应 Go 版本的 ForceExit"""
        await self._request(
            "POST",
            "/api/v1/forceexit",
            data=payload.dict(exclude_none=True)
        )
        logger.info(f"成功平仓, tradeid: {payload.tradeid}")

    async def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return await self._request("GET", "/api/v1/status")

    async def get_balance(self) -> Balance:
        """获取余额"""
        response = await self._request("GET", "/api/v1/balance")
        # 转换为Balance模型
        return Balance(**response)

    async def get_positions(
        self,
        options: Optional[PositionListOptions] = None
    ) -> PositionListResult:
        """获取持仓列表"""
        params = {}
        if options:
            if options.symbol:
                params["symbol"] = options.symbol
            if options.page:
                params["page"] = options.page
            if options.page_size:
                params["page_size"] = options.page_size

        response = await self._request(
            "GET",
            "/api/v1/positions",
            params=params
        )

        # 转换为PositionListResult
        if "positions" in response:
            # 转换持仓数据
            positions = [APIPosition(**pos) for pos in response["positions"]]
            response["positions"] = positions

        return PositionListResult(**response)

    async def get_position(self, trade_id: int) -> Optional[APIPosition]:
        """获取指定持仓"""
        positions = await self.get_positions()
        for pos in positions.positions:
            if pos.trade_id == trade_id:
                return pos
        return None

    async def get_trades(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Trade]:
        """获取交易历史"""
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol

        response = await self._request(
            "GET",
            "/api/v1/trades",
            params=params
        )

        if isinstance(response, list):
            return [Trade(**trade) for trade in response]
        elif "trades" in response:
            return [Trade(**trade) for trade in response["trades"]]
        return []

    async def get_profit(
        self,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取收益信息"""
        params = {}
        if symbol:
            params["symbol"] = symbol

        return await self._request(
            "GET",
            "/api/v1/profit",
            params=params
        )

    async def update_tiers(
        self,
        request: TierUpdateRequest
    ) -> None:
        """更新层级"""
        await self._request(
            "POST",
            "/api/v1/tiers",
            data=request.dict(exclude_none=True)
        )
        logger.info(f"成功更新层级, trade_id: {request.trade_id}")

    async def close(self):
        """关闭客户端"""
        if self._session:
            await self._session.close()
            self._session = None