"""决策引擎，对应 Go 版本的 engine.go"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from deepalpha.decision.types import Decision, DecisionContext, DecisionResult
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class Decider(ABC):
    """决策器接口，对应 Go 版本的 Decider"""

    @abstractmethod
    async def decide(self, context: DecisionContext) -> DecisionResult:
        """执行决策"""
        pass


class BaseEngine(Decider):
    """基础引擎实现"""

    def __init__(self, name: str):
        self.name = name

    async def decide(self, context: DecisionContext) -> DecisionResult:
        """执行决策"""
        trace_id = str(uuid.uuid4())
        logger.info(f"开始决策", engine=self.name, trace_id=trace_id)

        try:
            # 构建提示词
            if context.prompt:
                prompt = context.prompt.system + "\n\n" + context.prompt.user
            else:
                prompt = self._build_prompt(context)

            # 执行决策逻辑
            result = await self._do_decide(context, prompt)

            # 设置追踪ID
            result.trace_id = trace_id

            logger.info(f"决策完成",
                       engine=self.name,
                       trace_id=trace_id,
                       decisions=len(result.decisions))

            return result

        except Exception as e:
            logger.error(f"决策失败",
                        engine=self.name,
                        trace_id=trace_id,
                        error=str(e))
            raise

    def _build_prompt(self, context: DecisionContext) -> str:
        """构建提示词"""
        # 默认实现，子类可以覆盖
        return ""

    async def _do_decide(self, context: DecisionContext, prompt: str) -> DecisionResult:
        """执行决策逻辑"""
        raise NotImplementedError("子类需要实现 _do_decide 方法")


class LegacyEngineAdapter(BaseEngine):
    """遗留引擎适配器，兼容旧版本接口"""

    def __init__(self, name: str):
        super().__init__(name)

    async def decide(self, context: DecisionContext) -> DecisionResult:
        """兼容旧版本的决策方法"""
        # 转换为新格式
        new_context = self._convert_context(context)

        # 调用基础引擎
        result = await super().decide(new_context)

        # 转换回旧格式
        return self._convert_result(result)

    def _convert_context(self, context: DecisionContext) -> DecisionContext:
        """转换上下文格式"""
        # 添加必要的转换逻辑
        return context

    def _convert_result(self, result: DecisionResult) -> DecisionResult:
        """转换结果格式"""
        # 添加必要的转换逻辑
        return result


class EngineFactory:
    """引擎工厂"""

    @staticmethod
    def create_engine(engine_type: str, **kwargs) -> Decider:
        """创建决策引擎"""
        if engine_type == "legacy":
            return LegacyEngineAdapter("legacy")
        else:
            raise ValueError(f"未知的引擎类型: {engine_type}")


# 导出
__all__ = [
    "Decider",
    "BaseEngine",
    "LegacyEngineAdapter",
    "EngineFactory"
]