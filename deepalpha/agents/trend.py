"""Trend Agent - 趋势分析智能体"""

from typing import Any, Dict, List

from deepalpha.agents.base import LLMAgent, AGENT_STAGE_TREND, agent_block_limit, build_kline_block_tag, write_csv_data_block
from deepalpha.decision.analysis_context import AnalysisContext
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class TrendAgent(LLMAgent):
    """趋势分析智能体"""

    def __init__(self, provider_id: str, llm_client, template: str):
        super().__init__(
            name="TrendAgent",
            provider_id=provider_id,
            llm_client=llm_client,
            template=template,
            stage=AGENT_STAGE_TREND
        )

    def _build_prompt(self, contexts: List[AnalysisContext]) -> str:
        """构建趋势智能体的提示词，对应 Go 版本的 buildTrendAgentPrompt"""
        if not contexts:
            return ""

        limit = agent_block_limit(4)  # 默认限制4个块
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

        # 添加指示性提示
        lines.append("请找出关键支撑/阻力、动量加速或背离。")

        # 在开头添加模板
        full_prompt = self.template + "\n\n" + "\n".join(lines)
        return full_prompt