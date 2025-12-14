"""智能体基类，对应 Go 版本的 Agent 相关功能"""

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from deepalpha.core.models import AgentResult, Signal, SignalType
from deepalpha.decision.analysis_context import AnalysisContext
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class AgentInsight:
    """Agent洞察记录，对应 Go 版本的 AgentInsight"""
    def __init__(
        self,
        stage: str,
        provider_id: str,
        output: str,
        error: Optional[str] = None,
        warned: bool = False,
        system: Optional[str] = None,
        user: Optional[str] = None
    ):
        self.stage = stage
        self.provider_id = provider_id
        self.output = output
        self.error = error
        self.warned = warned
        self.system = system
        self.user = user
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stage": self.stage,
            "provider_id": self.provider_id,
            "output": self.output,
            "error": self.error,
            "warned": self.warned,
            "system": self.system,
            "user": self.user,
            "timestamp": self.timestamp.isoformat()
        }


class BaseAgent(ABC):
    """智能体基类"""

    def __init__(self, name: str, provider_id: str):
        self.name = name
        self.provider_id = provider_id

    @abstractmethod
    async def analyze(self, contexts: List[AnalysisContext]) -> AgentResult:
        """分析市场数据"""
        pass

    @abstractmethod
    def get_stage(self) -> str:
        """获取智能体阶段"""
        pass

    def format_stage_title(self) -> str:
        """格式化阶段标题，对应 Go 版本的 formatAgentStageTitle"""
        stage = self.get_stage()
        if stage == "indicator":
            return "Indicator Agent"
        elif stage == "pattern":
            return "Pattern Agent"
        elif stage == "trend":
            return "Trend Agent"
        else:
            stage = stage.strip()
            if not stage:
                return "Agent"
            if len(stage) == 1:
                return stage.upper() + " Agent"
            return stage[0].upper() + stage[1:] + " Agent"


class LLMAgent(BaseAgent):
    """基于LLM的智能体"""

    def __init__(
        self,
        name: str,
        provider_id: str,
        llm_client,
        template: str,
        stage: str
    ):
        super().__init__(name, provider_id)
        self.llm_client = llm_client
        self.template = template
        self.stage = stage

    def get_stage(self) -> str:
        return self.stage

    async def analyze(self, contexts: List[AnalysisContext]) -> AgentResult:
        """分析市场数据"""
        start_time = time.time()

        try:
            # 构建提示词
            prompt = self._build_prompt(contexts)

            if not prompt.strip():
                return AgentResult(
                    agent_name=self.name,
                    symbol=contexts[0].symbol if contexts else "",
                    analysis={},
                    processing_time=time.time() - start_time
                )

            # 调用LLM
            response = await self._call_llm(prompt)

            # 创建结果
            result = AgentResult(
                agent_name=self.name,
                symbol=contexts[0].symbol if contexts else "",
                analysis={
                    "prompt": prompt,
                    "response": response,
                    "stage": self.stage
                },
                processing_time=time.time() - start_time
            )

            # 尝试解析信号
            signal = self._parse_signal(response)
            if signal:
                result.signal = signal

            logger.info(f"{self.name} 分析完成",
                       symbol=result.symbol,
                       processing_time=result.processing_time)

            return result

        except Exception as e:
            logger.error(f"{self.name} 分析失败", error=str(e))
            return AgentResult(
                agent_name=self.name,
                symbol=contexts[0].symbol if contexts else "",
                analysis={"error": str(e)},
                processing_time=time.time() - start_time
            )

    def _build_prompt(self, contexts: List[AnalysisContext]) -> str:
        """构建提示词"""
        raise NotImplementedError("子类需要实现 _build_prompt 方法")

    async def _call_llm(self, prompt: str) -> str:
        """调用LLM"""
        try:
            response = await self.llm_client.chat(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM调用失败", error=str(e))
            raise

    def _parse_signal(self, response: str) -> Optional[Signal]:
        """从响应中解析信号"""
        # 简单的关键词匹配，可以后续优化
        response_lower = response.lower()

        # 查找买入信号
        buy_keywords = ["买入", "做多", "看涨", "bull", "long", "buy"]
        sell_keywords = ["卖出", "做空", "看跌", "bear", "short", "sell"]

        buy_score = sum(1 for kw in buy_keywords if kw in response_lower)
        sell_score = sum(1 for kw in sell_keywords if kw in response_lower)

        if buy_score > sell_score and buy_score > 0:
            confidence = min(buy_score / 5.0, 1.0)
            return Signal(
                symbol="",  # 需要上层设置
                signal_type=SignalType.BUY,
                confidence=confidence,
                reason=response[:200],  # 取前200字符作为原因
                agent_name=self.name
            )
        elif sell_score > buy_score and sell_score > 0:
            confidence = min(sell_score / 5.0, 1.0)
            return Signal(
                symbol="",
                signal_type=SignalType.SELL,
                confidence=confidence,
                reason=response[:200],
                agent_name=self.name
            )

        return None


# 常量定义，对应 Go 版本的常量
AGENT_STAGE_INDICATOR = "indicator"
AGENT_STAGE_PATTERN = "pattern"
AGENT_STAGE_TREND = "trend"


def agent_block_limit(max_blocks: int) -> int:
    """计算智能体块限制，对应 Go 版本的 agentBlockLimit"""
    if max_blocks <= 0:
        return 4
    if max_blocks > 8:
        return 8
    return max_blocks


def build_kline_block_tag(interval: str) -> str:
    """构建K线块标签，对应 Go 版本的 buildKlineBlockTag"""
    iv = interval.upper().strip()
    if not iv:
        return "DATA"

    tag_parts = ["DATA_"]
    for r in iv:
        if ('A' <= r <= 'Z') or ('0' <= r <= '9'):
            tag_parts.append(r)

    tag = "".join(tag_parts)
    if len(tag) <= len("DATA_"):
        return "DATA"
    return tag


def write_csv_data_block(csv: str, tag: Optional[str] = None) -> str:
    """写入CSV数据块，对应 Go 版本的 writeCSVDataBlock"""
    if not tag:
        tag = "DATA"
    tag = tag.upper().strip()

    lines = []
    lines.append(f"[{tag}_START]")
    lines.append(csv)
    if not csv.endswith('\n'):
        lines.append("")
    lines.append(f"[{tag}_END]")
    lines.append("")

    return "\n".join(lines)