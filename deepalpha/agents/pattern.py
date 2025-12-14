"""Pattern Agent - 形态识别智能体"""

from typing import Any, Dict, List

from deepalpha.agents.base import LLMAgent, AGENT_STAGE_PATTERN, agent_block_limit
from deepalpha.decision.analysis_context import AnalysisContext
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class PatternAgent(LLMAgent):
    """形态识别智能体"""

    def __init__(self, provider_id: str, llm_client, template: str):
        super().__init__(
            name="PatternAgent",
            provider_id=provider_id,
            llm_client=llm_client,
            template=template,
            stage=AGENT_STAGE_PATTERN
        )

    def _build_prompt(self, contexts: List[AnalysisContext]) -> str:
        """构建形态智能体的提示词，对应 Go 版本的 buildPatternAgentPrompt"""
        if not contexts:
            return ""

        limit = agent_block_limit(4)  # 默认限制4个块
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

        # 添加指示性提示
        lines.append("识别多空冲突、图形触发点与SMC叙事，并按优先级输出。")

        # 在开头添加模板
        full_prompt = self.template + "\n\n" + "\n".join(lines)
        return full_prompt