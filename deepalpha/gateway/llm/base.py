"""LLM Provider基类，对应 Go 版本的 provider 相关功能"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from deepalpha.decision.types import DecisionResult
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class ImagePayload:
    """图像载荷，对应 Go 版本的 ImagePayload"""

    def __init__(
        self,
        media_type: str,
        data: bytes,
        filename: Optional[str] = None
    ):
        self.media_type = media_type
        self.data = data
        self.filename = filename


class ModelProvider(ABC):
    """模型提供方接口，对应 Go 版本的 ModelProvider"""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        images: Optional[List[ImagePayload]] = None,
        expect_json: bool = False
    ) -> str:
        """完成对话"""
        pass

    @abstractmethod
    def supports_vision(self) -> bool:
        """是否支持视觉"""
        pass

    @abstractmethod
    def get_id(self) -> str:
        """获取提供方ID"""
        pass

    @abstractmethod
    async def close(self):
        """关闭连接"""
        pass


class BaseModelProvider(ModelProvider):
    """基础模型提供方实现"""

    def __init__(
        self,
        provider_id: str,
        enabled: bool = True,
        supports_vision: bool = False,
        expect_json: bool = False
    ):
        self.provider_id = provider_id
        self.enabled = enabled
        self.supports_vision_flag = supports_vision
        self.expect_json_flag = expect_json

    def supports_vision(self) -> bool:
        return self.supports_vision_flag

    def get_id(self) -> str:
        return self.provider_id

    async def close(self):
        """默认实现，什么都不做"""
        pass

    async def parse_response(self, response: str) -> DecisionResult:
        """解析响应为决策结果"""
        # 简化实现，返回原始响应
        return DecisionResult(
            raw_output=response,
            raw_json=response,
            meta_summary=f"Provider: {self.provider_id}"
        )

    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self.enabled

    def enable(self):
        """启用"""
        self.enabled = True

    def disable(self):
        """禁用"""
        self.enabled = False