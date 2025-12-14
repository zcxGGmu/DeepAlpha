"""Technical Agent - 技术指标分析智能体"""

from typing import Any, Dict, List

from deepalpha.agents.base import LLMAgent, AGENT_STAGE_INDICATOR, agent_block_limit
from deepalpha.decision.analysis_context import AnalysisContext
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class TechnicalAgent(LLMAgent):
    """技术指标分析智能体"""

    def __init__(self, provider_id: str, llm_client, template: str):
        super().__init__(
            name="TechnicalAgent",
            provider_id=provider_id,
            llm_client=llm_client,
            template=template,
            stage=AGENT_STAGE_INDICATOR
        )

    def _build_prompt(self, contexts: List[AnalysisContext]) -> str:
        """构建技术指标智能体的提示词，对应 Go 版本的 buildIndicatorAgentPrompt"""
        if not contexts:
            return ""

        limit = agent_block_limit(4)  # 默认限制4个块
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

        # 添加指示性提示
        lines.append("请总结动能、量价与波动率，并点名最强与最弱周期。")

        # 在开头添加模板
        full_prompt = self.template + "\n\n" + "\n".join(lines)
        return full_prompt