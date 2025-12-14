#!/usr/bin/env python3
"""测试智能体系统"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deepalpha.decision.analysis_context import AnalysisContext
from deepalpha.decision.multi_agent import MultiAgentManager
from deepalpha.config.loader import ConfigLoader
from deepalpha.market.data import Candle
from deepalpha.analysis.indicators import compute_all, IndicatorSettings
from deepalpha.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


class MockLLMClient:
    """模拟LLM客户端"""

    async def chat(self, prompt: str) -> str:
        """模拟聊天响应"""
        await asyncio.sleep(0.1)  # 模拟延迟

        # 根据提示词类型返回不同的模拟响应
        if "Indicator Agent" in prompt:
            return "【Indicator Summary】当前多头动能较强，量价配合良好，波动率适中。\n- BTCUSDT 1h: RSI=65接近超买，MACD柱线翻红，EMA呈多头排列\n- ETHUSDT 4h: RSI=45中性，MACD零轴附近，EMA交织\n- BTCUSDT 15m: 动量最强，RSI超买后回落\n- ETHUSDT 15m: 最弱，动能不足"
        elif "Pattern Agent" in prompt:
            return "【Pattern Summary】双顶形态显现，警惕向下突破风险。\n- BTCUSDT 1h: 形成潜在双顶，颈线支撑在47000\n- ETHUSDT 4h: 三角整理接近末端，即将选择方向\n- BTCUSDT 15m: 小时级别看跌，建议等待确认"
        elif "Trend Agent" in prompt:
            return "【Trend Summary】多数周期处于上升通道，但短期有回调压力。\n- BTCUSDT 1h: 支撑47500，阻力48500，上涨动能减弱\n- ETHUSDT 4h: 支撑2800，阻力2900，趋势偏多\n- BTCUSDT 15m: 短期看跌，跌破支撑可能加速下行"
        else:
            return "中性观点，建议观望。"


async def test_multi_agent():
    """测试多智能体系统"""
    logger.info("测试多智能体系统...")

    # 加载配置
    config_loader = ConfigLoader()
    settings = config_loader.load()

    # 创建模拟LLM客户端
    llm_clients = {
        "openai": MockLLMClient(),
        "anthropic": MockLLMClient(),
    }

    # 创建多智能体管理器
    manager = MultiAgentManager(settings, llm_clients)

    if not manager.is_enabled():
        logger.warning("多智能体未启用")
        return

    # 创建测试数据
    candles = []
    base_price = 47000.0
    for i in range(100):
        price = base_price + (i * 10) + (i % 3 * 5)
        candles.append(Candle(
            open_time=1640995200000 + i * 60000,
            close_time=1640995259999 + i * 60000,
            open=price,
            high=price + 50,
            low=price - 50,
            close=price + (i % 2 * 20 - 10),
            volume=1000 + i * 10,
            trades=5000 + i * 5
        ))

    # 计算指标
    indicator_settings = IndicatorSettings(
        symbol="BTCUSDT",
        interval="1h"
    )
    indicator_report = compute_all(candles, indicator_settings)

    # 创建分析上下文
    contexts = [
        AnalysisContext(
            symbol="BTCUSDT",
            interval="1h",
            forecast_horizon="24h",
            candles=candles[-50:],  # 最近50根K线
            indicator_report=indicator_report._asdict(),
            pattern_report="潜在双顶形态，颈线在47000",
            trend_report="短期上升趋势",
            image_note="价格接近阻力位"
        ),
        AnalysisContext(
            symbol="ETHUSDT",
            interval="4h",
            forecast_horizon="24h",
            candles=candles[-30:],
            indicator_report={
                "rsi": {"latest": 45.5, "state": "neutral"},
                "macd": {"latest": 0.01, "state": "flat"}
            },
            pattern_report="三角整理",
            trend_report="横盘整理",
            image_note="波动率收敛"
        )
    ]

    # 运行分析
    results = await manager.analyze_all(contexts)

    # 输出结果
    logger.info(f"分析完成，共 {len(results)} 个结果:")
    for result in results:
        logger.info(f"- {result.agent_name}: {result.signal}")
        response = result.analysis.get("response", "")
        if response:
            # 只显示第一行
            first_line = response.split('\n')[0]
            logger.info(f"  响应: {first_line}")

    # 获取洞察
    insights = manager.get_insights()
    logger.info(f"共生成 {len(insights)} 个洞察")

    return True


async def test_individual_agents():
    """测试单个智能体"""
    logger.info("测试单个智能体...")

    from deepalpha.agents.technical import TechnicalAgent
    from deepalpha.agents.pattern import PatternAgent
    from deepalpha.agents.trend import TrendAgent

    llm_client = MockLLMClient()

    # 创建智能体
    tech_agent = TechnicalAgent("openai", llm_client, "agent_indicator")
    pattern_agent = PatternAgent("openai", llm_client, "agent_pattern")
    trend_agent = TrendAgent("openai", llm_client, "agent_trend")

    # 创建测试上下文
    contexts = [
        AnalysisContext(
            symbol="BTCUSDT",
            interval="1h",
            forecast_horizon="24h",
            indicator_report={
                "rsi": {"latest": 65.2, "state": "overbought"},
                "macd": {"latest": 0.05, "state": "bullish"},
                "ema_fast": {"latest": 47200, "state": "above"},
                "ema_mid": {"latest": 46800, "state": "above"},
                "atr": {"latest": 150, "state": "volatility"}
            }
        )
    ]

    # 测试技术智能体
    result = await tech_agent.analyze(contexts)
    logger.info(f"Technical Agent: {result.signal}")
    logger.info(f"响应前缀: {result.analysis.get('response', '').split()[0] if result.analysis.get('response') else 'None'}")

    # 测试形态智能体
    result = await pattern_agent.analyze(contexts)
    logger.info(f"Pattern Agent: {result.signal}")

    # 测试趋势智能体
    result = await trend_agent.analyze(contexts)
    logger.info(f"Trend Agent: {result.signal}")


async def main():
    """主测试函数"""
    setup_logging(level="INFO", format_type="text", enable_rich=True)

    logger.info("=" * 50)
    logger.info("DeepAlpha 智能体系统测试")
    logger.info("=" * 50)

    try:
        # 测试单个智能体
        await test_individual_agents()

        logger.info("\n" + "=" * 50)

        # 测试多智能体管理器
        await test_multi_agent()

        logger.info("=" * 50)
        logger.info("所有测试完成！")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)