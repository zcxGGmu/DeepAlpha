"""决策聚合器，对应 Go 版本的 aggregator.go"""

from abc import ABC, abstractmethod
from typing import List

from deepalpha.decision.types import ModelOutput
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class Aggregator(ABC):
    """聚合器接口，对应 Go 版本的 Aggregator"""

    @abstractmethod
    async def aggregate(self, outputs: List[ModelOutput]) -> ModelOutput:
        """聚合多个模型输出"""
        pass

    @abstractmethod
    def name(self) -> str:
        """获取聚合器名称"""
        pass


class FirstWinsAggregator(Aggregator):
    """第一个成功的聚合器，对应 Go 版本的 FirstWinsAggregator"""

    def name(self) -> str:
        return "first-wins"

    async def aggregate(self, outputs: List[ModelOutput]) -> ModelOutput:
        """取第一个成功的输出"""
        for output in outputs:
            if not output.error and output.parsed and output.parsed.decisions:
                logger.info(f"使用第一个成功输出", provider=output.provider_id)
                return output

        # 如果没有成功输出，返回空结果
        logger.error("无可用的模型输出")
        return ModelOutput(
            provider_id="aggregator",
            error="无可用的模型输出"
        )


class MajorityVoteAggregator(Aggregator):
    """多数投票聚合器（根据Go版本添加）"""

    def name(self) -> str:
        return "majority-vote"

    async def aggregate(self, outputs: List[ModelOutput]) -> ModelOutput:
        """基于多数投票聚合决策"""
        # 收集所有有效输出
        valid_outputs = [
            output for output in outputs
            if not output.error and output.parsed and output.parsed.decisions
        ]

        if not valid_outputs:
            logger.error("无可用的模型输出进行投票")
            return ModelOutput(
                provider_id="aggregator",
                error="无可用的模型输出"
            )

        # TODO: 实现投票逻辑
        # 这里简化处理，取第一个
        logger.info(f"多数投票聚合，有效输出数: {len(valid_outputs)}")
        return valid_outputs[0]


class WeightedAggregator(Aggregator):
    """加权聚合器（根据Go版本添加）"""

    def name(self) -> str:
        return "weighted"

    async def aggregate(self, outputs: List[ModelOutput]) -> ModelOutput:
        """基于权重聚合决策"""
        # TODO: 实现加权逻辑
        # 这里简化处理，取第一个
        logger.info(f"加权聚合，输出数: {len(outputs)}")
        return outputs[0] if outputs else ModelOutput(provider_id="aggregator")


class AggregatorFactory:
    """聚合器工厂"""

    @staticmethod
    def create_aggregator(aggregator_type: str) -> Aggregator:
        """创建聚合器"""
        if aggregator_type == "first-wins":
            return FirstWinsAggregator()
        elif aggregator_type == "majority-vote":
            return MajorityVoteAggregator()
        elif aggregator_type == "weighted":
            return WeightedAggregator()
        else:
            raise ValueError(f"未知的聚合器类型: {aggregator_type}")


# 导出
__all__ = [
    "Aggregator",
    "FirstWinsAggregator",
    "MajorityVoteAggregator",
    "WeightedAggregator",
    "AggregatorFactory"
]