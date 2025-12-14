"""执行器核心，对应 Go 版本的 Manager"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from deepalpha.decision.types import Decision, DecisionContext
from deepalpha.executor.freqtrade.client import FreqtradeClient
from deepalpha.executor.freqtrade.types import ForceEnterPayload, ForceExitPayload
from deepalpha.executor.position import PositionManager
from deepalpha.executor.risk import RiskManager
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class ExecutionEngine:
    """执行引擎，对应 Go 版本的 Manager"""

    def __init__(
        self,
        freqtrade_client: FreqtradeClient,
        horizon: str = "24h",
        risk_manager: Optional[RiskManager] = None
    ):
        self.client = freqtrade_client
        self.horizon = horizon
        self.risk_manager = risk_manager or RiskManager()
        self.position_manager = PositionManager()

        # 执行状态
        self._executing_decisions: Dict[str, Decision] = {}
        self._execution_lock = asyncio.Lock()

        # 跟踪映射
        self._decision_to_trade: Dict[str, int] = {}
        self._trade_to_decision: Dict[int, str] = {}

    async def execute_decision(
        self,
        decision: Decision,
        current_price: float,
        balance: float
    ) -> Tuple[bool, str, Optional[int]]:
        """执行决策"""
        trace_id = str(uuid.uuid4())

        try:
            logger.info(f"开始执行决策", trace_id=trace_id, symbol=decision.symbol, action=decision.action)

            # 风险检查
            positions = await self.position_manager.get_all_positions()
            risk_pass, risk_reason = await self.risk_manager.check_decision_risk(
                decision, current_price, positions, balance
            )

            if not risk_pass:
                logger.warning(f"风险检查失败", trace_id=trace_id, reason=risk_reason)
                return False, risk_reason, None

            # 记录执行状态
            async with self._execution_lock:
                self._executing_decisions[trace_id] = decision

            try:
                if decision.action in ["enter_long", "open_long"]:
                    # 开仓
                    trade_id = await self._execute_enter(
                        decision, current_price, trace_id
                    )
                elif decision.action in ["enter_short", "open_short"]:
                    # 开空
                    decision.action = "enter_short"
                    trade_id = await self._execute_enter(
                        decision, current_price, trace_id
                    )
                elif decision.action in ["close_long", "close_short"]:
                    # 平仓
                    trade_id = await self._execute_close(
                        decision, current_price, trace_id
                    )
                elif decision.action == "hold":
                    # 持有，不需要执行
                    logger.info(f"决策为持有，无需执行", trace_id=trace_id)
                    return True, "持有", None
                else:
                    # 不支持的行动
                    logger.warning(f"不支持的行动类型: {decision.action}", trace_id=trace_id)
                    return False, f"不支持的行动: {decision.action}", None

                # 记录映射关系
                if trade_id:
                    self._decision_to_trade[trace_id] = trade_id
                    self._trade_to_decision[trade_id] = trace_id

                logger.info(f"决策执行成功", trace_id=trace_id, trade_id=trade_id)
                return True, "执行成功", trade_id

            finally:
                # 清理执行状态
                async with self._execution_lock:
                    self._executing_decisions.pop(trace_id, None)

        except Exception as e:
            logger.error(f"决策执行异常", trace_id=trace_id, error=str(e))
            async with self._execution_lock:
                self._executing_decisions.pop(trace_id, None)
            return False, f"执行异常: {e}", None

    async def _execute_enter(
        self,
        decision: Decision,
        current_price: float,
        trace_id: str
    ) -> int:
        """执行开仓"""
        # 计算参数
        stake_amount = decision.position_size_usd or 0
        if stake_amount <= 0:
            raise ValueError("仓位大小必须大于0")

        # 转换为杠杆倍数
        leverage = decision.leverage or 1

        # 确定方向
        side = "short" if decision.action in ["enter_short", "open_short"] else "long"

        # 构建载荷
        payload = ForceEnterPayload(
            pair=decision.symbol,
            side=side,
            stakeamount=stake_amount,
            leverage=leverage,
            entry_tag=trace_id
        )

        # 设置止损止盈
        if decision.stop_loss:
            payload.stop_loss = decision.stop_loss
        if decision.take_profit:
            payload.take_profit = decision.take_profit

        # 执行开仓
        response = await self.client.force_enter(payload)
        return response.trade_id

    async def _execute_close(
        self,
        decision: Decision,
        current_price: float,
        trace_id: str
    ) -> int:
        """执行平仓"""
        # 查找对应的持仓
        position = await self.position_manager.get_position(decision.symbol)
        if not position:
            raise ValueError(f"未找到持仓: {decision.symbol}")

        # 获取trade_id
        trade_id = None
        for tid, sym in self._trade_to_decision.items():
            if sym == decision.symbol:
                trade_id = tid
                break

        if not trade_id:
            raise ValueError(f"未找到对应的trade_id: {decision.symbol}")

        # 计算平仓数量
        amount = float(position.size)
        if decision.close_ratio:
            amount = amount * decision.close_ratio

        if amount <= 0:
            raise ValueError("平仓数量必须大于0")

        # 构建载荷
        payload = ForceExitPayload(
            tradeid=str(trade_id),
            amount=amount
        )

        # 执行平仓
        await self.client.force_exit(payload)
        return trade_id

    async def update_tiers(
        self,
        trade_id: int,
        decision: Decision
    ) -> bool:
        """更新层级"""
        try:
            if not decision.tiers:
                return True

            symbol = decision.symbol
            position = await self.position_manager.get_position(symbol)
            if not position:
                return False

            # 更新层级1
            if decision.tiers.tier1_target or decision.tiers.tier1_ratio:
                await self.position_manager.update_tier(
                    trade_id, 1,
                    decision.tiers.tier1_target,
                    decision.tiers.tier1_ratio
                )

            # 更新层级2
            if decision.tiers.tier2_target or decision.tiers.tier2_ratio:
                await self.position_manager.update_tier(
                    trade_id, 2,
                    decision.tiers.tier2_target,
                    decision.tiers.tier2_ratio
                )

            # 更新层级3
            if decision.tiers.tier3_target or decision.tiers.tiers3_ratio:
                await self.position_manager.update_tier(
                    trade_id, 3,
                    decision.tiers.tier3_target,
                    decision.tiers.tiers3_ratio
                )

            logger.info(f"更新层级成功", trade_id=trade_id, symbol=symbol)
            return True

        except Exception as e:
            logger.error(f"更新层级失败", trade_id=trade_id, error=str(e))
            return False

    async def get_positions(self) -> List[Dict[str, Any]]:
        """获取所有持仓"""
        positions = await self.position_manager.get_all_positions()
        return [pos.dict() for pos in positions.values()]

    async def get_risk_metrics(self) -> Dict[str, Any]:
        """获取风险指标"""
        return await self.risk_manager.get_risk_metrics()

    async def sync_positions(self) -> int:
        """同步持仓信息"""
        try:
            # 从API获取持仓
            result = await self.client.get_positions()
            await self.position_manager.update_from_api(result.positions)
            return len(result.positions)
        except Exception as e:
            logger.error(f"同步持仓失败: {e}")
            return 0

    async def get_balance(self) -> Dict[str, Any]:
        """获取余额"""
        try:
            balance = await self.client.get_balance()
            return balance.dict()
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return {}

    def get_execution_status(self) -> Dict[str, Any]:
        """获取执行状态"""
        return {
            "executing_decisions": {
                trace_id: dec.dict() for trace_id, dec in self._executing_decisions.items()
            },
            "decision_to_trade": self._decision_to_trade.copy(),
            "trade_to_decision": self._trade_to_decision.copy()
        }

    async def cleanup_completed_trades(self):
        """清理已完成交易"""
        # TODO: 实现清理逻辑
        pass

    async def close(self):
        """关闭执行引擎"""
        if self.client:
            await self.client.close()
        logger.info("执行引擎已关闭")


# 导出
__all__ = ["ExecutionEngine"]