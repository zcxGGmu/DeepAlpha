"""OpenAI Provider实现，对应 Go 版本的 OpenAI 实现"""

import json
from typing import Any, Dict, List, Optional

import aiohttp

from deepalpha.gateway.llm.base import BaseModelProvider, ImagePayload, ModelProvider
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIProvider(BaseModelProvider):
    """OpenAI模型提供方，对应 Go 版本的 OpenAIChatClient"""

    def __init__(
        self,
        provider_id: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4",
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        super().__init__(
            provider_id=provider_id,
            enabled=kwargs.get('enabled', True),
            supports_vision=kwargs.get('supports_vision', False),
            expect_json=kwargs.get('expect_json', False)
        )
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.extra_headers = headers or {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if not self._session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            headers.update(self.extra_headers)

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
        return self._session

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        images: Optional[List[ImagePayload]] = None,
        expect_json: bool = False
    ) -> str:
        """完成对话，对应 Go 版本的聊天逻辑"""
        if not self.is_enabled():
            raise ValueError(f"Provider {self.provider_id} is disabled")

        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 处理图片
        if images and self.supports_vision():
            content = []
            # 添加文本内容
            content.append({"type": "text", "text": prompt})

            # 添加图片内容
            for img in images:
                import base64
                img_data = base64.b64encode(img.data).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.media_type};base64,{img_data}"
                    }
                })

            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        # 构建请求
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4000
        }

        # 如果期望JSON输出
        if expect_json or self.expect_json_flag:
            payload["response_format"] = {"type": "json_object"}

        try:
            session = await self._get_session()
            url = f"{self.base_url}/chat/completions"

            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()

                if "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"]
                else:
                    raise ValueError("Invalid response format from OpenAI API")

        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            raise

    async def close(self):
        """关闭会话"""
        if self._session:
            await self._session.close()
            self._session = None