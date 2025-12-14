#!/usr/bin/env python3
"""测试决策引擎"""

import asyncio
import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deepalpha.decision.types import (
    Decision, DecisionResult, DecisionContext, ModelOutput
)
from deepalpha.decision.aggregator import (
    FirstWinsAggregator, MajorityVoteAggregator, AggregatorFactory
)
from deepalpha.decision.validate import DecisionValidator, parse_json_decisions
from deepalpha.decision.cache import DecisionCache, SimpleCache
from deepalpha.gateway.llm.factory import ProviderFactory
from deepalpha.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


class MockLLMProvider:
    """模拟LLM Provider"""

    def __init__(self, provider_id: str):
        self.provider_id = provider_id
        self.response_index = 0

    async def complete(self, prompt, system_prompt=None, images=None, expect_json=False):
        """模拟响应"""
        await asyncio.sleep(0.1)

        responses = [
            """{"decisions": [
                {"symbol": "BTCUSDT", "action": "enter_long", "confidence": 75,
                 "reasoning": "技术指标显示上涨趋势", "position_size_usd": 1000}
            ]}""",
            """{"decisions": [
                {"symbol": "ETHUSDT", "action": "hold", "confidence": 60,
                 "reasoning": "市场横盘整理", "position_size_usd": 0}
            ]}""",
            """{"decisions": [
                {"symbol": "BTCUSDT", "action": "enter_short", "confidence": 80,
                 "reasoning": "阻力位承压", "position_size_usd": 800}
            ]}"""
        ]

        response = responses[self.response_index % len(responses)]
        self.response_index += 1

        if expect_json:
            return response

        return f"分析完成: {response}"

    def supports_vision(self):
        return False

    def get_id(self):
        return self.provider_id

    async def close(self):
        pass


async def test_decision_types():
    """测试决策类型"""
    logger.info("测试决策类型...")

    # 测试决策创建
    decision = Decision(
        symbol="BTCUSDT",
        action="enter_long",
        confidence=75,
        reasoning="技术指标看涨",
        position_size_usd=1000.0,
        stop_loss=45000.0,
        take_profit=50000.0
    )
    logger.info(f"创建决策: {decision.symbol} {decision.action}")

    # 测试决策结果
    result = DecisionResult(
        decisions=[decision],
        raw_output="原始输出",
        raw_json=json.dumps({"decisions": [decision.dict()]}),
        meta_summary="测试摘要"
    )
    logger.info(f"决策结果包含 {len(result.decisions)} 个决策")

    # 测试层级设置
    from deepalpha.decision.types import DecisionTiers
    tiers = DecisionTiers(
        tier1_target=50000.0,
        tier1_ratio=0.3,
        tier2_target=52000.0,
        tier2_ratio=0.4
    )
    decision.tiers = tiers
    logger.info(f"决策层级: Tier1={tiers.tier1_ratio}, Tier2={tiers.tier2_ratio}")


async def test_aggregator():
    """测试聚合器"""
    logger.info("\n测试聚合器...")

    # 创建模拟输出
    outputs = [
        ModelOutput(
            provider_id="provider1",
            parsed=DecisionResult(
                decisions=[
                    Decision(symbol="BTCUSDT", action="enter_long", confidence=80)
                ]
            )
        ),
        ModelOutput(
            provider_id="provider2",
            parsed=DecisionResult(
                decisions=[
                    Decision(symbol="BTCUSDT", action="hold", confidence=60)
                ]
            )
        ),
        ModelOutput(
            provider_id="provider3",
            error="处理失败"
        )
    ]

    # 测试FirstWins聚合器
    first_wins = FirstWinsAggregator()
    result = await first_wins.aggregate(outputs)
    logger.info(f"FirstWins结果: {result.provider_id}")

    # 测试多数投票聚合器
    majority_vote = MajorityVoteAggregator()
    result = await majority_vote.aggregate(outputs)
    logger.info(f"多数投票结果: {result.provider_id}")

    # 测试工厂模式
    aggregator = AggregatorFactory.create_aggregator("first-wins")
    logger.info(f"工厂创建的聚合器: {aggregator.name()}")


async def test_validator():
    """测试决策验证器"""
    logger.info("\n测试决策验证器...")

    validator = DecisionValidator()

    # 测试有效决策
    valid_decision = Decision(
        symbol="BTCUSDT",
        action="enter_long",
        confidence=75,
        reasoning="有效决策"
    )

    # 测试无效决策
    invalid_decision = Decision(
        symbol="",
        action="invalid_action",
        confidence=150,  # 超出范围
        position_size_usd=-100  # 负值
    )

    result = DecisionResult(
        decisions=[valid_decision, invalid_decision]
    )

    # 验证决策
    validated = validator.validate(result)
    logger.info(f"验证前: {len(result.decisions)}, 验证后: {len(validated.decisions)}")

    # 测试JSON解析
    json_str = '''{"decisions": [
        {"symbol": "BTCUSDT", "action": "enter_long", "confidence": 75}
    ]}'''
    parsed = parse_json_decisions(json_str)
    logger.info(f"解析出 {len(parsed)} 个决策")


async def test_cache():
    """测试决策缓存"""
    logger.info("\n测试决策缓存...")

    # 创建缓存
    cache = DecisionCache(ttl=None)  # 不过期

    # 测试设置和获取
    from deepalpha.decision.types import DecisionMemory
    memory = DecisionMemory(
        symbol="BTCUSDT",
        horizon="1h",
        decided_at=datetime.utcnow(),
        decisions=[
            Decision(symbol="BTCUSDT", action="enter_long", confidence=75)
        ]
    )

    await cache.set(memory)

    # 获取缓存
    cached = await cache.get("BTCUSDT")
    if cached:
        logger.info(f"缓存命中: {cached.symbol}, 决策数: {len(cached.decisions)}")

    # 测试快照
    snapshot = await cache.snapshot()
    logger.info(f"快照包含 {len(snapshot)} 条记录")

    # 测试清理
    await cache.reset()
    logger.info("缓存已重置")


async def test_llm_providers():
    """测试LLM Provider"""
    logger.info("\n测试LLM Provider...")

    # 创建模拟Provider
    providers = [
        MockLLMProvider("openai"),
        MockLLMProvider("anthropic")
    ]

    # 测试调用
    for provider in providers:
        response = await provider.complete("测试提示")
        logger.info(f"Provider {provider.get_id()} 响应: {response[:50]}...")

    # 测试工厂
    factory_providers = ProviderFactory.create_from_env()
    logger.info(f"从环境变量创建的Provider数量: {len(factory_providers)}")


async def main():
    """主测试函数"""
    setup_logging(level="INFO", format_type="text", enable_rich=True)

    logger.info("=" * 50)
    logger.info("DeepAlpha 决策引擎测试")
    logger.info("=" * 50)

    try:
        await test_decision_types()
        await test_aggregator()
        await test_validator()
        await test_cache()
        await test_llm_providers()

        logger.info("\n" + "=" * 50)
        logger.info("所有测试完成！")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    from datetime import datetime
    exit_code = asyncio.run(main())
    sys.exit(exit_code)