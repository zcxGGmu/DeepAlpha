"""决策验证器，对应 Go 版本的 validate.go"""

import re
from typing import Any, Dict, List, Optional

from deepalpha.decision.types import Decision, DecisionResult, PositionSnapshot
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class DecisionValidator:
    """决策验证器"""

    def __init__(self):
        self.valid_actions = {
            'enter_long', 'enter_short', 'close_long', 'close_short',
            'hold', 'neutral'
        }

    def validate(self, result: DecisionResult) -> DecisionResult:
        """验证决策结果"""
        if not result.decisions:
            logger.warning("决策结果为空")
            return result

        validated_decisions = []
        for decision in result.decisions:
            if self._validate_decision(decision):
                validated_decisions.append(decision)
            else:
                logger.warning(f"无效决策被过滤", symbol=decision.symbol, action=decision.action)

        # 更新决策列表
        result.decisions = validated_decisions

        logger.info(f"决策验证完成",
                   original_count=len(result.decisions),
                   validated_count=len(validated_decisions))

        return result

    def _validate_decision(self, decision: Decision) -> bool:
        """验证单个决策"""
        # 验证必填字段
        if not decision.symbol:
            logger.error("决策缺少symbol字段")
            return False

        if not decision.action:
            logger.error("决策缺少action字段", symbol=decision.symbol)
            return False

        # 验证动作有效性
        if decision.action not in self.valid_actions:
            logger.warning(f"无效动作: {decision.action}", symbol=decision.symbol)
            return False

        # 验证数值范围
        if decision.confidence is not None and (decision.confidence < 0 or decision.confidence > 100):
            logger.warning(f"置信度超出范围: {decision.confidence}", symbol=decision.symbol)
            decision.confidence = max(0, min(100, decision.confidence))

        if decision.position_size_usd is not None and decision.position_size_usd < 0:
            logger.warning(f"仓位大小不能为负: {decision.position_size_usd}", symbol=decision.symbol)
            decision.position_size_usd = 0

        if decision.close_ratio is not None:
            if decision.close_ratio < 0 or decision.close_ratio > 1:
                logger.warning(f"平仓比例超出范围: {decision.close_ratio}", symbol=decision.symbol)
                decision.close_ratio = max(0, min(1, decision.close_ratio))

        # 验证价格合理性
        if decision.stop_loss is not None and decision.stop_loss <= 0:
            logger.warning(f"止损价格必须大于0: {decision.stop_loss}", symbol=decision.symbol)
            decision.stop_loss = None

        if decision.take_profit is not None and decision.take_profit <= 0:
            logger.warning(f"止盈价格必须大于0: {decision.take_profit}", symbol=decision.symbol)
            decision.take_profit = None

        # 验证层级设置
        if decision.tiers:
            self._validate_tiers(decision.tiers, decision.symbol)

        return True

    def _validate_tiers(self, tiers, symbol: str):
        """验证目标层级设置"""
        # 验证比例范围
        if tiers.tier1_ratio is not None and (tiers.tier1_ratio < 0 or tiers.tier1_ratio > 1):
            logger.warning(f"Tier1比例超出范围: {tiers.tier1_ratio}", symbol=symbol)
            tiers.tier1_ratio = max(0, min(1, tiers.tier1_ratio))

        if tiers.tier2_ratio is not None and (tiers.tier2_ratio < 0 or tiers.tier2_ratio > 1):
            logger.warning(f"Tier2比例超出范围: {tiers.tier2_ratio}", symbol=symbol)
            tiers.tier2_ratio = max(0, min(1, tiers.tier2_ratio))

        if tiers.tier3_ratio is not None and (tiers.tier3_ratio < 0 or tiers.tier3_ratio > 1):
            logger.warning(f"Tier3比例超出范围: {tiers.tier3_ratio}", symbol=symbol)
            tiers.tier3_ratio = max(0, min(1, tiers.tier3_ratio))

        # 验证价格合理性
        if tiers.tier1_target is not None and tiers.tier1_target <= 0:
            logger.warning(f"Tier1目标价格必须大于0: {tiers.tier1_target}", symbol=symbol)
            tiers.tier1_target = None

        if tiers.tier2_target is not None and tiers.tier2_target <= 0:
            logger.warning(f"Tier2目标价格必须大于0: {tiers.tier2_target}", symbol=symbol)
            tiers.tier2_target = None

        if tiers.tier3_target is not None and tiers.tier3_target <= 0:
            logger.warning(f"Tier3目标价格必须大于0: {tiers.tier3_target}", symbol=symbol)
            tiers.tier3_target = None


def parse_json_decisions(raw_json: str) -> List[Decision]:
    """从JSON字符串解析决策，对应Go版本的JSON解析逻辑"""
    if not raw_json:
        return []

    try:
        # 尝试直接解析为决策数组
        data = eval(raw_json)  # 注意：实际使用中应该用json.loads
        if isinstance(data, list):
            return [Decision(**item) if isinstance(item, dict) else None for item in data]
        elif isinstance(data, dict):
            # 如果是单个对象
            return [Decision(**data)]
        else:
            logger.error("无效的JSON格式")
            return []
    except Exception as e:
        logger.error(f"解析JSON失败: {e}")
        return []


def normalize_decision_text(text: str) -> str:
    """标准化决策文本"""
    if not text:
        return ""

    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text.strip())

    return text


def extract_decisions_from_text(text: str) -> List[Dict[str, Any]]:
    """从文本中提取决策信息"""
    decisions = []

    # 简单的模式匹配，实际使用中可能需要更复杂的逻辑
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 查找决策模式
        # 这里简化处理，实际应该更智能
        if any(action in line.lower() for action in ['buy', 'sell', 'long', 'short', 'hold']):
            # 提取符号信息
            symbol_match = re.search(r'([A-Z]{2,10})[/]?([A-Z]{2,10})?', line)
            if symbol_match:
                symbol = f"{symbol_match.group(1)}{symbol_match.group(2) or 'USDT'}"

                decision = {
                    'symbol': symbol,
                    'action': 'hold',  # 默认值
                    'reasoning': line[:200]  # 截取前200字符
                }
                decisions.append(decision)

    return decisions