"""决策缓存，对应 Go 版本的 last_decision_cache.go"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from deepalpha.decision.types import Decision, DecisionMemory, LastDecisionRecord
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class DecisionCache:
    """决策缓存，对应 Go 版本的 lastDecisionCache"""

    def __init__(self, ttl: timedelta = None):
        self._cache: Dict[str, DecisionMemory] = {}
        self._lock = asyncio.Lock()
        self.ttl = ttl or timedelta(hours=1)

    async def load(self, records: List[LastDecisionRecord]):
        """加载记录，对应 Go 版本的 Load"""
        async with self._lock:
            for rec in records:
                sym = rec.symbol.upper().strip()
                if not sym:
                    continue

                self._cache[sym] = DecisionMemory(
                    symbol=sym,
                    horizon=rec.horizon,
                    decided_at=rec.decided_at,
                    decisions=rec.decisions.copy()
                )

        logger.info(f"加载了 {len(records)} 条决策缓存记录")

    async def set(self, memory: DecisionMemory):
        """设置缓存，对应 Go 版本的 Set"""
        sym = memory.symbol.upper().strip()
        if not sym:
            return

        async with self._lock:
            memory.symbol = sym
            self._cache[sym] = memory

        logger.debug(f"缓存决策: {sym}")

    async def delete(self, symbol: str):
        """删除缓存，对应 Go 版本的 Delete"""
        sym = symbol.upper().strip()
        if not sym:
            return

        async with self._lock:
            self._cache.pop(sym, None)

        logger.debug(f"删除缓存: {sym}")

    async def reset(self):
        """重置缓存，对应 Go 版本的 Reset"""
        async with self._lock:
            self._cache.clear()

        logger.info("重置决策缓存")

    async def snapshot(self, now: Optional[datetime] = None) -> List[DecisionMemory]:
        """获取快照，对应 Go 版本的 Snapshot"""
        if now is None:
            now = datetime.utcnow()

        async with self._lock:
            if not self._cache:
                return []

            # 过滤过期数据
            valid_memories = []
            for memory in self._cache.values():
                if self.ttl.total_seconds() > 0 and (now - memory.decided_at) > self.ttl:
                    continue

                # 创建副本避免外部修改
                valid_memories.append(DecisionMemory(
                    symbol=memory.symbol,
                    horizon=memory.horizon,
                    decided_at=memory.decided_at,
                    decisions=memory.decisions.copy()
                ))

            return valid_memories

    async def get(self, symbol: str) -> Optional[DecisionMemory]:
        """获取指定符号的决策缓存"""
        sym = symbol.upper().strip()
        if not sym:
            return None

        async with self._lock:
            memory = self._cache.get(sym)
            if not memory:
                return None

            # 检查是否过期
            if self.ttl.total_seconds() > 0:
                if (datetime.utcnow() - memory.decided_at) > self.ttl:
                    del self._cache[sym]
                    return None

            return memory

    async def get_all(self) -> Dict[str, DecisionMemory]:
        """获取所有缓存"""
        async with self._lock:
            return {k: v for k, v in self._cache.items()}

    async def cleanup_expired(self):
        """清理过期缓存"""
        now = datetime.utcnow()
        expired_symbols = []

        async with self._lock:
            for symbol, memory in self._cache.items():
                if self.ttl.total_seconds() > 0 and (now - memory.decided_at) > self.ttl:
                    expired_symbols.append(symbol)

            for symbol in expired_symbols:
                del self._cache[symbol]

        if expired_symbols:
            logger.info(f"清理了 {len(expired_symbols)} 条过期缓存")

    def __len__(self):
        """返回缓存大小"""
        return len(self._cache)


class SimpleCache:
    """简单内存缓存实现"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, tuple] = {}  # value, timestamp
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                return value
        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存值"""
        async with self._lock:
            # 如果缓存满了，移除最旧的项
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(),
                                key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[key] = (value, datetime.utcnow().timestamp())

    async def delete(self, key: str):
        """删除缓存"""
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self):
        """清空缓存"""
        async with self._lock:
            self._cache.clear()


# 导出
__all__ = [
    "DecisionCache",
    "SimpleCache"
]