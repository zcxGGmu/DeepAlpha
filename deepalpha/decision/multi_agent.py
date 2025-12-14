"""多智能体管理器，对应 Go 版本的 multi_agent.go"""

import asyncio
from typing import Any, Dict, List, Optional

from deepalpha.agents.base import AgentInsight, AGENT_STAGE_INDICATOR, AGENT_STAGE_PATTERN, AGENT_STAGE_TREND
from deepalpha.agents.pattern import PatternAgent
from deepalpha.agents.technical import TechnicalAgent
from deepalpha.agents.trend import TrendAgent
from deepalpha.config.settings import Settings
from deepalpha.core.models import AgentResult
from deepalpha.decision.analysis_context import AnalysisContext
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class MultiAgentManager:
    """多智能体管理器"""

    def __init__(self, settings: Settings, llm_clients: Dict[str, Any]):
        self.settings = settings
        self.llm_clients = llm_clients
        self.agents: Dict[str, Any] = {}
        self.insights: List[AgentInsight] = []

        # 初始化智能体
        self._initialize_agents()

    def _initialize_agents(self):
        """初始化智能体"""
        multi_agent_config = self.settings.multi_agent if hasattr(self.settings, 'multi_agent') else None

        if not multi_agent_config or not multi_agent_config.enabled:
            logger.info("多智能体功能未启用")
            return

        # 创建Technical Agent
        indicator_provider = multi_agent_config.indicator_provider
        if indicator_provider and indicator_provider in self.llm_clients:
            indicator_template = multi_agent_config.indicator_template or "agent_indicator"
            self.agents[AGENT_STAGE_INDICATOR] = TechnicalAgent(
                provider_id=indicator_provider,
                llm_client=self.llm_clients[indicator_provider],
                template=indicator_template
            )
            logger.info(f"Technical Agent 已初始化，使用 Provider: {indicator_provider}")

        # 创建Pattern Agent
        pattern_provider = multi_agent_config.pattern_provider
        if pattern_provider and pattern_provider in self.llm_clients:
            pattern_template = multi_agent_config.pattern_template or "agent_pattern"
            self.agents[AGENT_STAGE_PATTERN] = PatternAgent(
                provider_id=pattern_provider,
                llm_client=self.llm_clients[pattern_provider],
                template=pattern_template
            )
            logger.info(f"Pattern Agent 已初始化，使用 Provider: {pattern_provider}")

        # 创建Trend Agent
        trend_provider = multi_agent_config.trend_provider
        if trend_provider and trend_provider in self.llm_clients:
            trend_template = multi_agent_config.trend_template or "agent_trend"
            self.agents[AGENT_STAGE_TREND] = TrendAgent(
                provider_id=trend_provider,
                llm_client=self.llm_clients[trend_provider],
                template=trend_template
            )
            logger.info(f"Trend Agent 已初始化，使用 Provider: {trend_provider}")

    async def analyze_all(self, contexts: List[AnalysisContext]) -> List[AgentResult]:
        """运行所有智能体分析"""
        if not self.agents:
            logger.warning("没有可用的智能体")
            return []

        logger.info(f"运行多智能体分析，共 {len(self.agents)} 个智能体")

        # 并发运行所有智能体
        tasks = []
        for stage, agent in self.agents.items():
            task = asyncio.create_task(self._run_agent_with_catch(agent, stage, contexts))
            tasks.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                stage = list(self.agents.keys())[i]
                logger.error(f"智能体 {stage} 运行失败", error=str(result))
                # 创建错误结果
                error_result = AgentResult(
                    agent_name=f"Agent_{stage}",
                    symbol=contexts[0].symbol if contexts else "",
                    analysis={"error": str(result)},
                    processing_time=0
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)

        logger.info(f"多智能体分析完成，有效结果: {len(valid_results)}")
        return valid_results

    async def _run_agent_with_catch(self, agent: Any, stage: str, contexts: List[AnalysisContext]) -> AgentResult:
        """运行智能体并捕获异常"""
        try:
            insight = AgentInsight(
                stage=stage,
                provider_id=agent.provider_id,
                output="",
                system="processing"
            )

            # 运行分析
            result = await agent.analyze(contexts)

            # 记录洞察
            insight.output = result.analysis.get("response", "")
            insight.system = "completed"
            self.insights.append(insight)

            return result

        except Exception as e:
            # 记录错误洞察
            error_insight = AgentInsight(
                stage=stage,
                provider_id=agent.provider_id,
                output="",
                error=str(e),
                system="failed"
            )
            self.insights.append(error_insight)

            logger.error(f"智能体 {stage} 分析失败", error=str(e))
            raise

    def get_agent(self, stage: str) -> Optional[Any]:
        """获取指定阶段的智能体"""
        return self.agents.get(stage)

    def get_insights(self) -> List[AgentInsight]:
        """获取所有洞察记录"""
        return self.insights.copy()

    def clear_insights(self):
        """清除洞察记录"""
        self.insights.clear()

    def is_enabled(self) -> bool:
        """检查多智能体是否启用"""
        return len(self.agents) > 0

    def get_available_stages(self) -> List[str]:
        """获取可用的智能体阶段"""
        return list(self.agents.keys())


# 辅助函数，对应 Go 版本的提示词构建函数

def build_indicator_agent_prompt(contexts: List[AnalysisContext], max_blocks: int = 4) -> str:
    """构建指标智能体提示词"""
    if not contexts:
        return ""

    limit = max_blocks if max_blocks > 0 else 4
    lines = []

    lines.append("# Technical Indicator Blocks")

    count = 0
    for ctx in contexts:
        data = ctx.indicator_json.strip()
        if not data:
            continue

        lines.append(f"## {ctx.symbol} {ctx.interval} ({ctx.forecast_horizon})")
        lines.append(data)
        lines.append("")
        count += 1

        if count >= limit:
            break

    if count == 0:
        return ""

    lines.append("请总结动能、量价与波动率，并点名最强与最弱周期。")
    return "\n".join(lines)


def build_pattern_agent_prompt(contexts: List[AnalysisContext], max_blocks: int = 4) -> str:
    """构建形态智能体提示词"""
    if not contexts:
        return ""

    limit = max_blocks if max_blocks > 0 else 4
    lines = []

    lines.append("# Pattern & Narrative Blocks")

    count = 0
    for ctx in contexts:
        pattern = ctx.pattern_report.strip()
        trend = ctx.trend_report.strip()
        note = ctx.image_note.strip()

        if not pattern and not trend and not note:
            continue

        lines.append(f"## {ctx.symbol} {ctx.interval} ({ctx.forecast_horizon})")

        if pattern:
            lines.append(f"- Pattern: {pattern}")
        if trend:
            lines.append(f"- Trend: {trend}")
        if note:
            lines.append(f"- Visual: {note}")

        lines.append("")
        count += 1

        if count >= limit:
            break

    if count == 0:
        return ""

    lines.append("识别多空冲突、图形触发点与SMC叙事，并按优先级输出。")
    return "\n".join(lines)


def build_trend_agent_prompt(contexts: List[AnalysisContext], max_blocks: int = 4) -> str:
    """构建趋势智能体提示词"""
    if not contexts:
        return ""

    limit = max_blocks if max_blocks > 0 else 4
    lines = []

    lines.append("# Trend CSV Windows")
    lines.append("每个数据块包含 Date/Time(UTC), Open(O), High(H), Low(L), Close(C), Volume(V), Trades 列，按行给出最近的 K 线。")
    lines.append("")

    count = 0
    for ctx in contexts:
        csv_data = ctx.kline_csv.strip()
        raw = ctx.kline_json.strip()
        note = ctx.image_note.strip()
        trend = ctx.trend_report.strip()

        if not csv_data and not raw and not note and not trend:
            continue

        lines.append(f"## {ctx.symbol} {ctx.interval} ({ctx.forecast_horizon})")

        if csv_data:
            from deepalpha.agents.base import build_kline_block_tag, write_csv_data_block
            tag = build_kline_block_tag(ctx.interval)
            csv_block = write_csv_data_block(csv_data, tag)
            lines.append(csv_block)
        elif raw:
            lines.append("Raw:")
            lines.append(raw)
            lines.append("")

        if note:
            lines.append(f"Visual: {note}")

        if trend:
            lines.append(f"Trend: {trend}")

        lines.append("")
        count += 1

        if count >= limit:
            break

    if count == 0:
        return ""

    lines.append("请找出关键支撑/阻力、动量加速或背离。")
    return "\n".join(lines)


def format_agent_stage_title(stage: str) -> str:
    """格式化智能体阶段标题"""
    if stage == AGENT_STAGE_INDICATOR:
        return "Indicator Agent"
    elif stage == AGENT_STAGE_PATTERN:
        return "Pattern Agent"
    elif stage == AGENT_STAGE_TREND:
        return "Trend Agent"
    else:
        stage = stage.strip()
        if not stage:
            return "Agent"
        if len(stage) == 1:
            return stage.upper() + " Agent"
        return stage[0].upper() + stage[1:] + " Agent"