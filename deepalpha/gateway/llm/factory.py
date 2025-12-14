"""Provider工厂，对应 Go 版本的 provider_factory.go"""

import os
from typing import Any, Dict, List, Optional

from deepalpha.config.settings import LLMProviderSettings
from deepalpha.gateway.llm.base import ModelProvider
from deepalpha.gateway.llm.openai import OpenAIProvider
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class ModelConfig:
    """模型配置，对应 Go 版本的 ModelCfg"""

    def __init__(
        self,
        id: str = "",
        provider: str = "",
        api_url: str = "",
        api_key: str = "",
        model: str = "",
        enabled: bool = True,
        headers: Optional[Dict[str, str]] = None,
        supports_vision: bool = False,
        expect_json: bool = False
    ):
        self.id = id
        self.provider = provider
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.enabled = enabled
        self.headers = headers or {}
        self.supports_vision = supports_vision
        self.expect_json = expect_json


class ProviderFactory:
    """Provider工厂，对应 Go 版本的 BuildProvidersFromConfig"""

    @staticmethod
    def build_from_settings(
        providers_settings: List[LLMProviderSettings],
        timeout: int = 30
    ) -> List[ModelProvider]:
        """从设置构建Provider列表"""
        providers = []

        for settings in providers_settings:
            # 转换为ModelConfig
            config = ModelConfig(
                id=settings.name,
                provider=settings.name,
                api_key=settings.api_key,
                model=settings.model,
                enabled=True,
                supports_vision=False,  # 默认不支持视觉
                expect_json=False
            )

            # 根据provider类型创建对应的Provider
            if config.provider.lower() == "openai":
                provider = OpenAIProvider(
                    provider_id=config.id or f"openai:{config.model}",
                    base_url="https://api.openai.com/v1",
                    api_key=config.api_key,
                    model=config.model,
                    timeout=timeout,
                    supports_vision=config.supports_vision,
                    expect_json=config.expect_json
                )
                providers.append(provider)
            elif config.provider.lower() == "anthropic":
                # TODO: 实现Anthropic Provider
                logger.warning(f"Anthropic provider not implemented yet: {config.id}")
            elif config.provider.lower() == "deepseek":
                # TODO: 实现DeepSeek Provider
                logger.warning(f"DeepSeek provider not implemented yet: {config.id}")
            else:
                # 默认使用OpenAI兼容的接口
                base_url = "https://api.openai.com/v1"
                if config.api_url:
                    base_url = config.api_url

                provider = OpenAIProvider(
                    provider_id=config.id or f"{config.provider}:{config.model}",
                    base_url=base_url,
                    api_key=config.api_key,
                    model=config.model,
                    timeout=timeout,
                    headers=config.headers,
                    supports_vision=config.supports_vision,
                    expect_json=config.expect_json
                )
                providers.append(provider)

        logger.info(f"构建了 {len(providers)} 个Provider")
        return providers

    @staticmethod
    def create_openai_provider(
        provider_id: str,
        api_key: str,
        model: str = "gpt-4",
        base_url: str = None,
        timeout: int = 30
    ) -> OpenAIProvider:
        """创建OpenAI Provider的便捷方法"""
        return OpenAIProvider(
            provider_id=provider_id,
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            model=model,
            timeout=timeout
        )

    @staticmethod
    def create_from_env() -> List[ModelProvider]:
        """从环境变量创建Provider"""
        providers = []

        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            openai = ProviderFactory.create_openai_provider(
                provider_id="openai:gpt-4",
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-4")
            )
            providers.append(openai)

        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            # TODO: 实现Anthropic Provider
            logger.warning("Anthropic provider not implemented yet")

        # DeepSeek
        if os.getenv("DEEPSEEK_API_KEY"):
            # TODO: 实现DeepSeek Provider
            logger.warning("DeepSeek provider not implemented yet")

        return providers


# 导出
__all__ = [
    "ModelConfig",
    "ProviderFactory"
]