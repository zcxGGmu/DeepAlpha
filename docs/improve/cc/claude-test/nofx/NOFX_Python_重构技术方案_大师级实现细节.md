# NOFX Python重构技术方案 - 大师级实现细节篇

> 本文档是《NOFX_Python_重构技术方案_A股港股》系列的第十部分
> 覆盖第41-45章：分布式系统架构、高级机器学习技术、超低延迟HFT、安全攻防实战、生产事故管理

---

## 第41章 分布式系统架构深度实现

### 41.1 一致性哈希与数据分片

```python
# src/distributed/consistent_hashing.py
"""
一致性哈希实现

用于分布式缓存和数据分片
"""

import hashlib
import bisect
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sortedcontainers import SortedDict
import asyncio

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VirtualNode:
    """虚拟节点"""
    node_id: str
    physical_node: str
    position: int


class ConsistentHash:
    """
    一致性哈希环

    特性：
    1. 最小化节点增减时的数据迁移
    2. 自动负载均衡
    3. 虚拟节点提高均匀性
    """

    def __init__(
        self,
        virtual_nodes: int = 150,
        replication_factor: int = 3,
    ):
        """
        初始化一致性哈希

        Args:
            virtual_nodes: 每个物理节点的虚拟节点数
            replication_factor: 数据副本数
        """
        self.virtual_nodes = virtual_nodes
        self.replication_factor = replication_factor

        # 哈希环 (排序的节点位置)
        self.ring: SortedDict = SortedDict()

        # 物理节点映射
        self.physical_nodes: Dict[str, List[str]] = {}

        # 节点锁
        self.lock = asyncio.Lock()

    def _hash(self, key: str) -> int:
        """
        计算哈希值

        Args:
            key: 键

        Returns:
            哈希值 (0 ~ 2^32-1)
        """
        # 使用MurmurHash3
        hash_int = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_int % (2 ** 32)

    async def add_node(self, node: str) -> List[Tuple[str, str]]:
        """
        添加节点

        Args:
            node: 节点标识

        Returns:
            需要迁移的键列表 (key, target_node)
        """
        async with self.lock:
            migrations = []

            # 创建虚拟节点
            for i in range(self.virtual_nodes):
                vnode_id = f"{node}#{i}"
                position = self._hash(vnode_id)

                # 找到该位置负责的键
                # (这些键需要迁移到新节点)

                self.ring[position] = vnode_id

            # 记录物理节点
            self.physical_nodes[node] = [
                f"{node}#{i}"
                for i in range(self.virtual_nodes)
            ]

            logger.info(f"添加节点: {node}, 虚拟节点数: {self.virtual_nodes}")

            return migrations

    async def remove_node(self, node: str) -> List[Tuple[str, str]]:
        """
        移除节点

        Args:
            node: 节点标识

        Returns:
            需要迁移的键列表 (key, target_node)
        """
        async with self.lock:
            migrations = []

            if node not in self.physical_nodes:
                return migrations

            # 移除所有虚拟节点
            for vnode_id in self.physical_nodes[node]:
                position = self._hash(vnode_id)

                if position in self.ring:
                    del self.ring[position]

            del self.physical_nodes[node]

            logger.info(f"移除节点: {node}")

            return migrations

    def get_node(self, key: str) -> str:
        """
        获取键对应的节点

        Args:
            key: 键

        Returns:
            节点标识
        """
        if not self.ring:
            raise ValueError("哈希环为空")

        # 计算键的哈希值
        key_hash = self._hash(key)

        # 在环上找到顺时针第一个节点
        positions = self.ring.keys()
        index = bisect.bisect_left(positions, key_hash)

        if index == len(positions):
            # 环绕到第一个节点
            index = 0

        vnode_id = self.ring[positions[index]]
        physical_node = vnode_id.split('#')[0]

        return physical_node

    def get_replication_nodes(self, key: str) -> List[str]:
        """
        获取键的所有副本节点

        Args:
            key: 键

        Returns:
            节点列表 (按优先级排序)
        """
        if not self.ring:
            raise ValueError("哈希环为空")

        nodes = []
        seen = set()

        key_hash = self._hash(key)
        positions = self.ring.keys()

        # 从键的位置开始，顺时针查找N个不同物理节点
        index = bisect.bisect_left(positions, key_hash)
        if index == len(positions):
            index = 0

        while len(nodes) < self.replication_factor and len(seen) < len(self.physical_nodes):
            vnode_id = self.ring[positions[index]]
            physical_node = vnode_id.split('#')[0]

            if physical_node not in seen:
                nodes.append(physical_node)
                seen.add(physical_node)

            index = (index + 1) % len(positions)

        return nodes

    def get_node_keys(self, node: str) -> List[str]:
        """
        获取节点负责的所有键

        Args:
            node: 节点标识

        Returns:
            键列表
        """
        keys = []

        if node not in self.physical_nodes:
            return keys

        # 获取该节点的所有虚拟节点位置
        vnode_positions = []
        for vnode_id in self.physical_nodes[node]:
            vnode_positions.append(self._hash(vnode_id))

        vnode_positions.sort()

        # 找出每个虚拟节点负责的键范围
        for i, position in enumerate(vnode_positions):
            next_position = vnode_positions[(i + 1) % len(vnode_positions)]

            # 当前虚拟节点负责从position到next_position之间的所有键
            # (这里简化实现，实际需要遍历所有可能的键)

            keys.append(f"range_{position}_{next_position}")

        return keys


class DistributedCache:
    """
    分布式缓存

    基于一致性哈希的分布式缓存系统
    """

    def __init__(
        self,
        nodes: List[str],
        virtual_nodes: int = 150,
    ):
        """
        初始化分布式缓存

        Args:
            nodes: 缓存节点列表
            virtual_nodes: 虚拟节点数
        """
        self.consistent_hash = ConsistentHash(virtual_nodes)
        self.node_clients: Dict[str, any] = {}

        # 添加所有节点
        for node in nodes:
            self.consistent_hash.add_node(node)

    async def get(self, key: str) -> Optional[any]:
        """
        获取缓存值

        Args:
            key: 键

        Returns:
            值
        """
        # 获取主节点
        primary_node = self.consistent_hash.get_node(key)

        # 从主节点读取
        client = self.node_clients.get(primary_node)
        if client:
            return await client.get(key)

        return None

    async def set(self, key: str, value: any, ttl: int = 300):
        """
        设置缓存值

        Args:
            key: 键
            value: 值
            ttl: 过期时间
        """
        # 获取所有副本节点
        nodes = self.consistent_hash.get_replication_nodes(key)

        # 写入所有副本
        for node in nodes:
            client = self.node_clients.get(node)
            if client:
                await client.set(key, value, ttl)

    async def delete(self, key: str):
        """
        删除缓存值

        Args:
            key: 键
        """
        nodes = self.consistent_hash.get_replication_nodes(key)

        for node in nodes:
            client = self.node_clients.get(node)
            if client:
                await client.delete(key)
```

### 41.2 分布式事务管理

```python
# src/distributed/distributed_transaction.py
"""
分布式事务管理

实现Saga模式和TCC模式
"""

import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from src.core.logging import get_logger

logger = get_logger(__name__)


class TransactionStatus(Enum):
    """事务状态"""
    PENDING = "pending"
    ACTIVE = "active"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class SagaStep:
    """Saga步骤"""
    name: str
    execute: Callable
    compensate: Callable
    status: TransactionStatus = TransactionStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None


class SagaOrchestrator:
    """
    Saga编排器

    实现长事务的最终一致性
    """

    def __init__(self, saga_id: str = None):
        """
        初始化Saga编排器

        Args:
            saga_id: Saga事务ID
        """
        self.saga_id = saga_id or str(uuid.uuid4())
        self.steps: List[SagaStep] = []
        self.status = TransactionStatus.PENDING
        self.current_step = 0
        self.lock = asyncio.Lock()

    def add_step(
        self,
        name: str,
        execute: Callable,
        compensate: Callable,
    ) -> 'SagaOrchestrator':
        """
        添加步骤

        Args:
            name: 步骤名称
            execute: 执行函数
            compensate: 补偿函数

        Returns:
            self
        """
        step = SagaStep(
            name=name,
            execute=execute,
            compensate=compensate,
        )
        self.steps.append(step)
        return self

    async def execute(self) -> bool:
        """
        执行Saga事务

        Returns:
            是否成功
        """
        async with self.lock:
            self.status = TransactionStatus.ACTIVE

            for i, step in enumerate(self.steps):
                self.current_step = i

                try:
                    logger.info(f"执行步骤: {step.name}", extra={
                        'saga_id': self.saga_id,
                        'step': i,
                    })

                    # 执行步骤
                    result = await step.execute()
                    step.result = result
                    step.status = TransactionStatus.COMMITTED

                except Exception as e:
                    logger.error(f"步骤执行失败: {step.name}", extra={
                        'saga_id': self.saga_id,
                        'step': i,
                        'error': str(e),
                    })

                    step.error = e
                    step.status = TransactionStatus.FAILED

                    # 开始补偿
                    await self._compensate(i)
                    return False

            self.status = TransactionStatus.COMMITTED
            return True

    async def _compensate(self, failed_step: int):
        """
        补偿已执行的步骤

        Args:
            failed_step: 失败步骤索引
        """
        self.status = TransactionStatus.ROLLING_BACK

        # 从失败的步骤前一个开始回滚
        for i in range(failed_step - 1, -1, -1):
            step = self.steps[i]

            if step.status != TransactionStatus.COMMITTED:
                continue

            try:
                logger.info(f"补偿步骤: {step.name}", extra={
                    'saga_id': self.saga_id,
                    'step': i,
                })

                await step.compensate(step.result)
                step.status = TransactionStatus.ROLLED_BACK

            except Exception as e:
                logger.error(f"补偿失败: {step.name}", extra={
                    'saga_id': self.saga_id,
                    'step': i,
                    'error': str(e),
                })

        self.status = TransactionStatus.ROLLED_BACK


class TccTransaction:
    """
    TCC (Try-Confirm-Cancel) 事务

    两阶段提交的变种
    """

    def __init__(self, transaction_id: str = None):
        """
        初始化TCC事务

        Args:
            transaction_id: 事务ID
        """
        self.transaction_id = transaction_id or str(uuid.uuid4())
        self.participants: Dict[str, Any] = {}
        self.status = TransactionStatus.PENDING
        self.lock = asyncio.Lock()

    async def register(
        self,
        participant_id: str,
        try_action: Callable,
        confirm_action: Callable,
        cancel_action: Callable,
    ):
        """
        注册参与者

        Args:
            participant_id: 参与者ID
            try_action: Try操作
            confirm_action: Confirm操作
            cancel_action: Cancel操作
        """
        self.participants[participant_id] = {
            'try': try_action,
            'confirm': confirm_action,
            'cancel': cancel_action,
            'status': TransactionStatus.PENDING,
        }

    async def execute(self) -> bool:
        """
        执行TCC事务

        Returns:
            是否成功
        """
        async with self.lock:
            # 阶段1: Try
            try:
                if not await self._try_phase():
                    # Try失败，执行Cancel
                    await self._cancel_phase()
                    return False
            except Exception as e:
                logger.error(f"Try阶段异常: {str(e)}")
                await self._cancel_phase()
                return False

            # 阶段2: Confirm
            try:
                await self._confirm_phase()
                return True
            except Exception as e:
                logger.error(f"Confirm阶段异常: {str(e)}")
                # Confirm失败需要人工介入
                self.status = TransactionStatus.FAILED
                return False

    async def _try_phase(self) -> bool:
        """
        Try阶段

        Returns:
            是否所有Try都成功
        """
        self.status = TransactionStatus.ACTIVE

        for participant_id, participant in self.participants.items():
            try:
                logger.info(f"Try阶段: {participant_id}")

                result = await participant['try']()
                participant['result'] = result
                participant['status'] = TransactionStatus.COMMITTED

            except Exception as e:
                logger.error(f"Try失败: {participant_id}, error: {str(e)}")
                participant['status'] = TransactionStatus.FAILED
                return False

        return True

    async def _confirm_phase(self):
        """
        Confirm阶段
        """
        self.status = TransactionStatus.COMMITTING

        for participant_id, participant in self.participants.items():
            try:
                logger.info(f"Confirm阶段: {participant_id}")

                await participant['confirm'](participant['result'])
                participant['status'] = TransactionStatus.COMMITTED

            except Exception as e:
                logger.error(f"Confirm失败: {participant_id}, error: {str(e)}")
                participant['status'] = TransactionStatus.FAILED
                # Confirm失败需要记录，不能回滚
                raise

        self.status = TransactionStatus.COMMITTED

    async def _cancel_phase(self):
        """
        Cancel阶段
        """
        self.status = TransactionStatus.ROLLING_BACK

        for participant_id, participant in self.participants.items():
            # 只Cancel已Try成功的参与者
            if participant['status'] != TransactionStatus.COMMITTED:
                continue

            try:
                logger.info(f"Cancel阶段: {participant_id}")

                await participant['cancel'](participant['result'])
                participant['status'] = TransactionStatus.ROLLED_BACK

            except Exception as e:
                logger.error(f"Cancel失败: {participant_id}, error: {str(e)}")
                # Cancel失败需要重试或人工介入

        self.status = TransactionStatus.ROLLED_BACK


class TwoPhaseCommitCoordinator:
    """
    两阶段提交协调器

    实现经典的2PC协议
    """

    def __init__(self, timeout: float = 30.0):
        """
        初始化协调器

        Args:
            timeout: 超时时间
        """
        self.timeout = timeout
        self.participants: Dict[str, Any] = {}
        self.transaction_id: str = None
        self.lock = asyncio.Lock()

    async def begin_transaction(self) -> str:
        """
        开始事务

        Returns:
            事务ID
        """
        self.transaction_id = str(uuid.uuid4())
        self.participants.clear()

        logger.info(f"开始事务: {self.transaction_id}")
        return self.transaction_id

    async def register_participant(
        self,
        participant_id: str,
        prepare: Callable,
        commit: Callable,
        rollback: Callable,
    ):
        """
        注册参与者

        Args:
            participant_id: 参与者ID
            prepare: Prepare操作
            commit: Commit操作
            rollback: Rollback操作
        """
        self.participants[participant_id] = {
            'prepare': prepare,
            'commit': commit,
            'rollback': rollback,
            'voted': None,
        }

    async def commit(self) -> bool:
        """
        提交事务

        Returns:
            是否成功
        """
        async with self.lock:
            # 阶段1: Prepare
            if not await self._prepare_phase():
                # 任一参与者拒绝，执行Rollback
                await self._rollback_phase()
                return False

            # 阶段2: Commit
            await self._commit_phase()
            return True

    async def _prepare_phase(self) -> bool:
        """
        Prepare阶段

        Returns:
            是否所有参与者都同意
        """
        logger.info(f"Prepare阶段: {self.transaction_id}")

        tasks = []
        for pid, participant in self.participants.items():
            task = self._prepare_participant(pid, participant)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 检查所有参与者是否都同意
        for pid, result in zip(self.participants.keys(), results):
            if isinstance(result, Exception) or not result:
                logger.warning(f"参与者拒绝: {pid}")
                return False

        return True

    async def _prepare_participant(
        self,
        pid: str,
        participant: Dict,
    ) -> bool:
        """
        准备单个参与者

        Args:
            pid: 参与者ID
            participant: 参与者信息

        Returns:
            是否同意
        """
        try:
            result = await asyncio.wait_for(
                participant['prepare'](),
                timeout=self.timeout,
            )
            participant['voted'] = result
            return result

        except asyncio.TimeoutError:
            logger.error(f"Prepare超时: {pid}")
            return False

        except Exception as e:
            logger.error(f"Prepare异常: {pid}, error: {str(e)}")
            return False

    async def _commit_phase(self):
        """
        Commit阶段
        """
        logger.info(f"Commit阶段: {self.transaction_id}")

        tasks = []
        for pid, participant in self.participants.items():
            task = participant['commit']()
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _rollback_phase(self):
        """
        Rollback阶段
        """
        logger.info(f"Rollback阶段: {self.transaction_id}")

        tasks = []
        for pid, participant in self.participants.items():
            # 只Rollback已投票同意的参与者
            if participant['voted']:
                task = participant['rollback']()
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
```

### 41.3 分布式锁服务

```python
# src/distributed/distributed_lock.py
"""
分布式锁服务

基于Redis实现分布式锁
"""

import asyncio
import uuid
from typing import Optional, Callable
from dataclasses import dataclass

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LockResult:
    """锁结果"""
    acquired: bool
    lock_id: str
    ttl: int


class DistributedLock:
    """
    分布式锁

    基于Redis的Redlock算法实现
    """

    def __init__(
        self,
        redis_client,
        lock_name: str,
        ttl: int = 30,
        retry_delay: float = 0.1,
        max_retries: int = 30,
    ):
        """
        初始化分布式锁

        Args:
            redis_client: Redis客户端
            lock_name: 锁名称
            ttl: 锁过期时间(秒)
            retry_delay: 重试延迟
            max_retries: 最大重试次数
        """
        self.redis = redis_client
        self.lock_name = f"lock:{lock_name}"
        self.ttl = ttl * 1000  # 转换为毫秒
        self.retry_delay = retry_delay
        self.max_retries = max_retries

        self.lock_id: Optional[str] = None
        self.acquired = False

    async def acquire(self) -> bool:
        """
        获取锁

        Returns:
            是否成功
        """
        self.lock_id = str(uuid.uuid4())

        for attempt in range(self.max_retries):
            # 使用SET NX EX命令
            result = await self.redis.set(
                self.lock_name,
                self.lock_id,
                nx=True,
                px=self.ttl,
            )

            if result:
                self.acquired = True
                logger.info(f"获取锁成功: {self.lock_name}", extra={
                    'lock_id': self.lock_id,
                    'attempt': attempt + 1,
                })
                return True

            # 重试前等待
            await asyncio.sleep(self.retry_delay)

        logger.warning(f"获取锁失败: {self.lock_name}")
        return False

    async def release(self) -> bool:
        """
        释放锁

        Returns:
            是否成功
        """
        if not self.acquired:
            return False

        # 使用Lua脚本确保只删除自己创建的锁
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        result = await self.redis.eval(
            lua_script,
            1,
            self.lock_name,
            self.lock_id,
        )

        if result:
            self.acquired = False
            logger.info(f"释放锁成功: {self.lock_name}")
            return True

        logger.warning(f"释放锁失败: {self.lock_name}")
        return False

    async def extend(self, additional_ttl: int = 30) -> bool:
        """
        延长锁的过期时间

        Args:
            additional_ttl: 额外的过期时间(秒)

        Returns:
            是否成功
        """
        if not self.acquired:
            return False

        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("pexpire", KEYS[1], ARGV[2])
        else
            return 0
        end
        """

        result = await self.redis.eval(
            lua_script,
            1,
            self.lock_name,
            self.lock_id,
            additional_ttl * 1000,
        )

        return bool(result)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        await self.release()


class Redlock:
    """
    Redlock算法实现

    在多个Redis实例上获取锁以提高可靠性
    """

    def __init__(
        self,
        redis_clients: list,
        lock_name: str,
        ttl: int = 30,
        retry_delay: float = 0.1,
        max_retries: int = 30,
    ):
        """
        初始化Redlock

        Args:
            redis_clients: Redis客户端列表
            lock_name: 锁名称
            ttl: 锁过期时间
            retry_delay: 重试延迟
            max_retries: 最大重试次数
        """
        self.redis_clients = redis_clients
        self.lock_name = f"lock:{lock_name}"
        self.ttl = ttl
        self.retry_delay = retry_delay
        self.max_retries = max_retries

        self.lock_id: Optional[str] = None
        self.acquired_clients: list = []

    async def acquire(self) -> bool:
        """
        获取锁

        Returns:
            是否成功
        """
        self.lock_id = str(uuid.uuid4())
        self.acquired_clients.clear()

        # 计算需要的最少同意数 (N/2 + 1)
        quorum = len(self.redis_clients) // 2 + 1

        start_time = asyncio.get_event_loop().time()

        for attempt in range(self.max_retries):
            acquired_count = 0

            # 并发在所有Redis实例上尝试获取锁
            tasks = []
            for redis in self.redis_clients:
                task = self._try_acquire(redis)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if result and not isinstance(result, Exception):
                    acquired_count += 1
                    self.acquired_clients.append(self.redis_clients[i])

            # 检查是否获得足够多的锁
            elapsed = asyncio.get_event_loop().time() - start_time
            remaining_ttl = self.ttl - elapsed

            if acquired_count >= quorum and remaining_ttl > 0:
                logger.info(f"Redlock获取成功: {self.lock_name}", extra={
                    'acquired': acquired_count,
                    'quorum': quorum,
                })
                return True

            # 没有获得足够多的锁，释放已获取的
            await self._release_all()

            # 检查是否超时
            if elapsed >= self.ttl:
                logger.warning(f"Redlock获取超时: {self.lock_name}")
                return False

            # 等待重试
            await asyncio.sleep(self.retry_delay)

        return False

    async def _try_acquire(self, redis) -> bool:
        """在单个Redis实例上尝试获取锁"""
        try:
            result = await redis.set(
                self.lock_name,
                self.lock_id,
                nx=True,
                px=self.ttl * 1000,
            )
            return bool(result)
        except Exception as e:
            logger.error(f"Redis获取锁失败: {str(e)}")
            return False

    async def release(self) -> bool:
        """
        释放锁

        Returns:
            是否成功
        """
        return await self._release_all()

    async def _release_all(self) -> bool:
        """在所有Redis实例上释放锁"""
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        tasks = []
        for redis in self.acquired_clients:
            task = redis.eval(
                lua_script,
                1,
                self.lock_name,
                self.lock_id,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        self.acquired_clients.clear()

        return all(r for r in results if not isinstance(r, Exception))

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        await self.release()


class AsyncLockContext:
    """
    异步锁上下文管理器

    提供更简单的分布式锁使用方式
    """

    @staticmethod
    async def lock(
        redis_client,
        lock_name: str,
        func: Callable,
        ttl: int = 30,
    ):
        """
        执行带锁的函数

        Args:
            redis_client: Redis客户端
            lock_name: 锁名称
            func: 要执行的函数
            ttl: 锁过期时间

        Returns:
            函数执行结果
        """
        lock = DistributedLock(
            redis_client,
            lock_name,
            ttl=ttl,
        )

        if await lock.acquire():
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                else:
                    return func()
            finally:
                await lock.release()
        else:
            raise Exception(f"获取锁失败: {lock_name}")
```

---

## 第42章 高级机器学习技术

### 42.1 在线学习系统

```python
# src/ml/online_learning.py
"""
在线学习系统

支持模型实时更新和增量学习
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pickle
import asyncio
from collections import deque

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LearningSample:
    """学习样本"""
    features: np.ndarray
    label: float
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


class OnlineLinearRegression:
    """
    在线线性回归

    使用随机梯度下降(SGD)进行增量学习
    """

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        regularization: float = 0.001,
        window_size: int = 10000,
    ):
        """
        初始化在线线性回归

        Args:
            n_features: 特征数量
            learning_rate: 学习率
            regularization: 正则化系数
            window_size: 滑动窗口大小
        """
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.window_size = window_size

        # 模型参数
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # 样本窗口
        self.samples: deque = deque(maxlen=window_size)

        # 统计信息
        self.sample_count = 0
        self.total_loss = 0.0

    def predict(self, features: np.ndarray) -> float:
        """
        预测

        Args:
            features: 特征向量

        Returns:
            预测值
        """
        return np.dot(self.weights, features) + self.bias

    def partial_fit(
        self,
        features: np.ndarray,
        label: float,
        weight: float = 1.0,
    ) -> float:
        """
        增量拟合

        Args:
            features: 特征向量
            label: 真实标签
            weight: 样本权重

        Returns:
            损失值
        """
        # 预测
        prediction = self.predict(features)

        # 计算损失
        error = prediction - label
        loss = 0.5 * error ** 2

        # 计算梯度
        grad_weights = error * features + self.regularization * self.weights
        grad_bias = error

        # 更新参数
        self.weights -= self.learning_rate * grad_weights * weight
        self.bias -= self.learning_rate * grad_bias * weight

        # 添加样本到窗口
        self.samples.append(LearningSample(
            features=features.copy(),
            label=label,
            weight=weight,
        ))

        # 更新统计
        self.sample_count += 1
        self.total_loss += loss

        return loss

    def get_importance(self) -> np.ndarray:
        """
        获取特征重要性

        Returns:
            特征重要性向量
        """
        return np.abs(self.weights)

    def reset(self):
        """重置模型"""
        self.weights = np.zeros(self.n_features)
        self.bias = 0.0
        self.samples.clear()
        self.sample_count = 0
        self.total_loss = 0.0


class OnlineDecisionTree:
    """
    在线决策树

    使用Hoeffding树算法实现增量学习
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples: int = 100,
        delta: float = 1e-7,
        tau: float = 0.05,
    ):
        """
        初始化在线决策树

        Args:
            max_depth: 最大深度
            min_samples: 最小样本数
            delta: Hoeffding界参数
            tau: 分割阈值
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.delta = delta
        self.tau = tau

        self.root = None
        self.n_features = 0

    def partial_fit(
        self,
        features: np.ndarray,
        label: float,
    ):
        """
        增量拟合

        Args:
            features: 特征向量
            label: 标签
        """
        if self.root is None:
            self.n_features = len(features)
            self.root = self._create_node(depth=0)

        self._update_node(self.root, features, label)

    def _create_node(self, depth: int) -> Dict:
        """创建节点"""
        return {
            'depth': depth,
            'is_leaf': True,
            'samples': [],
            'labels': [],
            'split_feature': None,
            'split_value': None,
            'left': None,
            'right': None,
            'prediction': None,
        }

    def _update_node(
        self,
        node: Dict,
        features: np.ndarray,
        label: float,
    ):
        """更新节点"""
        # 添加样本
        node['samples'].append(features.copy())
        node['labels'].append(label)

        # 更新预测值 (均值)
        node['prediction'] = np.mean(node['labels'])

        # 检查是否需要分裂
        if (
            node['is_leaf'] and
            len(node['samples']) >= self.min_samples and
            node['depth'] < self.max_depth
        ):
            self._try_split(node)

        # 如果不是叶子节点，继续向下
        if not node['is_leaf']:
            feature_value = features[node['split_feature']]

            if feature_value <= node['split_value']:
                self._update_node(node['left'], features, label)
            else:
                self._update_node(node['right'], features, label)

    def _try_split(self, node: Dict):
        """尝试分裂节点"""
        n_samples = len(node['samples'])
        best_feature = None
        best_value = None
        best_gain = -np.inf

        # 当前节点的不纯度
        current_impurity = self._calculate_impurity(node['labels'])

        # 对每个特征尝试分裂
        for feature_idx in range(self.n_features):
            feature_values = [s[feature_idx] for s in node['samples']]

            # 计算Hoeffding界
            epsilon = self._hoeffding_bound(
                n_samples,
                current_impurity,
                self.delta,
            )

            # 尝试不同的分裂点
            unique_values = sorted(set(feature_values))

            for i in range(len(unique_values) - 1):
                split_value = (unique_values[i] + unique_values[i + 1]) / 2

                # 计算信息增益
                gain = self._calculate_information_gain(
                    node['labels'],
                    feature_values,
                    split_value,
                )

                # 检查是否是最佳分裂
                if gain > best_gain + epsilon:
                    best_gain = gain
                    best_feature = feature_idx
                    best_value = split_value

        # 检查是否满足分裂条件
        if best_feature is not None and best_gain > self.tau:
            # 执行分裂
            node['is_leaf'] = False
            node['split_feature'] = best_feature
            node['split_value'] = best_value

            # 创建子节点
            node['left'] = self._create_node(node['depth'] + 1)
            node['right'] = self._create_node(node['depth'] + 1)

            # 重新分配样本
            for sample, label in zip(node['samples'], node['labels']):
                if sample[best_feature] <= best_value:
                    node['left']['samples'].append(sample)
                    node['left']['labels'].append(label)
                else:
                    node['right']['samples'].append(sample)
                    node['right']['labels'].append(label)

            # 计算子节点预测值
            node['left']['prediction'] = np.mean(node['left']['labels'])
            node['right']['prediction'] = np.mean(node['right']['labels'])

            # 清空当前节点样本
            node['samples'] = []
            node['labels'] = []

    def _calculate_impurity(self, labels: List[float]) -> float:
        """计算不纯度 (方差)"""
        if len(labels) == 0:
            return 0.0
        return np.var(labels)

    def _calculate_information_gain(
        self,
        labels: List[float],
        feature_values: List[float],
        split_value: float,
    ) -> float:
        """计算信息增益"""
        # 当前不纯度
        current_impurity = self._calculate_impurity(labels)

        # 分裂后的不纯度
        left_labels = [
            l for l, fv in zip(labels, feature_values)
            if fv <= split_value
        ]
        right_labels = [
            l for l, fv in zip(labels, feature_values)
            if fv > split_value
        ]

        n = len(labels)
        n_left = len(left_labels)
        n_right = len(right_labels)

        weighted_impurity = (
            n_left / n * self._calculate_impurity(left_labels) +
            n_right / n * self._calculate_impurity(right_labels)
        )

        return current_impurity - weighted_impurity

    def _hoeffding_bound(
        self,
        n: int,
        impurity: float,
        delta: float,
    ) -> float:
        """计算Hoeffding界"""
        import math
        return math.sqrt(
            (impurity ** 2) * math.log(1 / delta) / (2 * n)
        )

    def predict(self, features: np.ndarray) -> float:
        """预测"""
        node = self.root

        while not node['is_leaf']:
            if features[node['split_feature']] <= node['split_value']:
                node = node['left']
            else:
                node = node['right']

        return node['prediction']


class OnlineEnsemble:
    """
    在线集成学习

    结合多个在线学习模型
    """

    def __init__(
        self,
        models: List[any],
        weights: Optional[List[float]] = None,
    ):
        """
        初始化在线集成

        Args:
            models: 模型列表
            weights: 模型权重
        """
        self.models = models

        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights) / sum(weights)

        self.performance_history = [
            deque(maxlen=1000)
            for _ in models
        ]

    def predict(self, features: np.ndarray) -> float:
        """
        预测

        Args:
            features: 特征向量

        Returns:
            加权预测值
        """
        predictions = []

        for model in self.models:
            pred = model.predict(features)
            predictions.append(pred)

        return np.average(predictions, weights=self.weights)

    def update_weights(self):
        """根据历史表现更新权重"""
        for i, history in enumerate(self.performance_history):
            if len(history) > 0:
                # 使用最近的平均误差作为权重依据
                errors = np.array(list(history))
                # 误差越小，权重越大
                self.weights[i] = 1.0 / (np.mean(errors) + 1e-6)

        # 归一化
        self.weights = self.weights / np.sum(self.weights)
```

### 42.2 模型A/B测试框架

```python
# src/ml/ab_testing.py
"""
模型A/B测试框架
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio

from src.core.logging import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class ABTestMetric:
    """A/B测试指标"""
    name: str
    type: MetricType
    value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ABTestVariant:
    """A/B测试变体"""
    name: str
    traffic_ratio: float  # 流量比例 (0-1)
    model: any
    metrics: Dict[str, ABTestMetric] = field(default_factory=dict)


class ABTestFramework:
    """
    A/B测试框架

    用于模型性能对比和选择
    """

    def __init__(
        self,
        test_name: str,
        variants: List[ABTestVariant],
        significance_level: float = 0.05,
        min_sample_size: int = 1000,
    ):
        """
        初始化A/B测试

        Args:
            test_name: 测试名称
            variants: 测试变体列表
            significance_level: 显著性水平
            min_sample_size: 最小样本量
        """
        self.test_name = test_name
        self.variants = variants
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size

        # 样本计数
        self.sample_counts = {v.name: 0 for v in variants}

        # 指标记录
        self.metric_history = {v.name: {} for v in variants}

        # 流量分配
        self._allocate_traffic()

    def _allocate_traffic(self):
        """分配流量"""
        total_ratio = sum(v.traffic_ratio for v in self.variants)

        if abs(total_ratio - 1.0) > 1e-6:
            # 归一化
            for variant in self.variants:
                variant.traffic_ratio /= total_ratio

        # 创建累积分布
        self.traffic_distribution = []
        cumulative = 0.0

        for variant in self.variants:
            cumulative += variant.traffic_ratio
            self.traffic_distribution.append((cumulative, variant))

    def assign_variant(self, user_id: str) -> ABTestVariant:
        """
        为用户分配测试变体

        Args:
            user_id: 用户ID

        Returns:
            分配的变体
        """
        # 使用一致性哈希确保同一用户总是分配到同一变体
        import hashlib
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        ratio = (hash_value % 10000) / 10000.0

        for cumulative, variant in self.traffic_distribution:
            if ratio <= cumulative:
                return variant

        return self.variants[-1]

    async def record_metric(
        self,
        variant_name: str,
        metric: ABTestMetric,
    ):
        """
        记录指标

        Args:
            variant_name: 变体名称
            metric: 指标
        """
        if variant_name not in self.metric_history:
            return

        if metric.name not in self.metric_history[variant_name]:
            self.metric_history[variant_name][metric.name] = []

        self.metric_history[variant_name][metric.name].append(metric)
        self.sample_counts[variant_name] += 1

    def get_variant_metrics(
        self,
        variant_name: str,
        metric_name: str,
    ) -> List[float]:
        """
        获取变体指标值

        Args:
            variant_name: 变体名称
            metric_name: 指标名称

        Returns:
            指标值列表
        """
        if variant_name not in self.metric_history:
            return []

        if metric_name not in self.metric_history[variant_name]:
            return []

        return [m.value for m in self.metric_history[variant_name][metric_name]]

    def calculate_statistics(
        self,
        metric_name: str,
    ) -> Dict[str, Dict]:
        """
        计算统计指标

        Args:
            metric_name: 指标名称

        Returns:
            各变体的统计信息
        """
        statistics = {}

        for variant in self.variants:
            values = self.get_variant_metrics(variant.name, metric_name)

            if len(values) == 0:
                continue

            statistics[variant.name] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'p25': np.percentile(values, 25),
                'p75': np.percentile(values, 75),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99),
            }

        return statistics

    def perform_t_test(
        self,
        metric_name: str,
        variant_a: str,
        variant_b: str,
    ) -> Dict:
        """
        执行t检验

        Args:
            metric_name: 指标名称
            variant_a: 变体A
            variant_b: 变体B

        Returns:
            检验结果
        """
        from scipy import stats

        values_a = self.get_variant_metrics(variant_a, metric_name)
        values_b = self.get_variant_metrics(variant_b, metric_name)

        if len(values_a) < self.min_sample_size or len(values_b) < self.min_sample_size:
            return {
                'sufficient_data': False,
                'reason': '样本量不足',
            }

        # 执行t检验
        t_stat, p_value = stats.ttest_ind(values_a, values_b)

        # 计算效应大小 (Cohen's d)
        pooled_std = np.sqrt(
            ((len(values_a) - 1) * np.var(values_a) +
             (len(values_b) - 1) * np.var(values_b)) /
            (len(values_a) + len(values_b) - 2)
        )
        cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std

        # 判断显著性
        is_significant = p_value < self.significance_level

        return {
            'sufficient_data': True,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_cohens_d(cohens_d),
            'mean_a': np.mean(values_a),
            'mean_b': np.mean(values_b),
            'relative_improvement': (
                (np.mean(values_a) - np.mean(values_b)) /
                np.mean(values_b) * 100
                if np.mean(values_b) != 0 else 0
            ),
        }

    def _interpret_cohens_d(self, d: float) -> str:
        """解释Cohen's d"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def select_winner(self, metric_name: str) -> Optional[str]:
        """
        选择获胜变体

        Args:
            metric_name: 指标名称 (越大越好)

        Returns:
            获胜变体名称
        """
        statistics = self.calculate_statistics(metric_name)

        if len(statistics) == 0:
            return None

        # 找到均值最大的变体
        best_variant = max(
            statistics.items(),
            key=lambda x: x[1]['mean']
        )

        # 检查是否显著优于其他变体
        for variant_name, stats in statistics.items():
            if variant_name == best_variant[0]:
                continue

            result = self.perform_t_test(
                metric_name,
                best_variant[0],
                variant_name,
            )

            if not result.get('is_significant', False):
                # 不显著，没有明确的获胜者
                return None

        return best_variant[0]

    def generate_report(self) -> Dict:
        """生成测试报告"""
        report = {
            'test_name': self.test_name,
            'timestamp': datetime.now().isoformat(),
            'variants': [],
            'comparisons': [],
            'recommendation': None,
        }

        # 各变体统计
        for variant in self.variants:
            variant_report = {
                'name': variant.name,
                'traffic_ratio': variant.traffic_ratio,
                'sample_count': self.sample_counts[variant.name],
                'metrics': {},
            }

            for metric_name in ['return', 'sharpe_ratio', 'max_drawdown']:
                stats = self.calculate_statistics(metric_name)

                if variant.name in stats:
                    variant_report['metrics'][metric_name] = stats[variant.name]

            report['variants'].append(variant_report)

        # 变体比较
        if len(self.variants) >= 2:
            for metric_name in ['return', 'sharpe_ratio']:
                comparison = self.perform_t_test(
                    metric_name,
                    self.variants[0].name,
                    self.variants[1].name,
                )
                comparison['metric'] = metric_name
                report['comparisons'].append(comparison)

        # 推荐获胜者
        winner = self.select_winner('return')
        if winner:
            report['recommendation'] = {
                'winner': winner,
                'reason': f"{winner} 在收益指标上表现最佳且显著",
            }

        return report
```

### 42.3 特征工程自动化

```python
# src/ml/feature_engineering.py
"""
自动化特征工程
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats

from src.core.logging import get_logger

logger = get_logger(__name__)


class AutoFeatureEngineer:
    """
    自动特征工程器

    自动进行特征选择、转换和生成
    """

    def __init__(
        self,
        selection_method: str = "k_best",
        n_features: int = 50,
        scaling_method: str = "standard",
    ):
        """
        初始化特征工程器

        Args:
            selection_method: 特征选择方法 (k_best, pca, mutual_info)
            n_features: 选择的特征数
            scaling_method: 缩放方法 (standard, minmax, robust)
        """
        self.selection_method = selection_method
        self.n_features = n_features
        self.scaling_method = scaling_method

        self.selector = None
        self.scaler = None
        self.feature_names = None
        self.selected_features = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        """
        拟合特征工程器

        Args:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # 特征选择
        if self.selection_method == "k_best":
            self.selector = SelectKBest(
                f_regression,
                k=self.n_features,
            )
            X_selected = self.selector.fit_transform(X, y)

            # 记录选中的特征
            mask = self.selector.get_support()
            self.selected_features = [
                self.feature_names[i]
                for i, selected in enumerate(mask)
                if selected
            ]

        elif self.selection_method == "mutual_info":
            mi_scores = mutual_info_regression(X, y)
            top_indices = np.argsort(mi_scores)[-self.n_features:]

            self.selector = lambda x: x[:, top_indices]
            X_selected = X[:, top_indices]

            self.selected_features = [
                self.feature_names[i] for i in top_indices
            ]

        elif self.selection_method == "pca":
            self.selector = PCA(n_components=self.n_features)
            X_selected = self.selector.fit_transform(X)

            self.selected_features = [
                f"pc_{i}" for i in range(self.n_features)
            ]

        else:
            X_selected = X
            self.selected_features = self.feature_names

        # 特征缩放
        if self.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None

        if self.scaler:
            self.scaler.fit(X_selected)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换特征

        Args:
            X: 原始特征矩阵

        Returns:
            转换后的特征矩阵
        """
        # 特征选择
        if self.selection_method == "k_best":
            X_selected = self.selector.transform(X)
        elif self.selection_method == "pca":
            X_selected = self.selector.transform(X)
        elif self.selection_method == "mutual_info":
            X_selected = X  # 需要保存索引
        else:
            X_selected = X

        # 特征缩放
        if self.scaler:
            X_selected = self.scaler.transform(X_selected)

        return X_selected

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        拟合并转换

        Args:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称

        Returns:
            转换后的特征矩阵
        """
        self.fit(X, y, feature_names)
        return self.transform(X)


class FeatureGenerator:
    """
    特征生成器

    自动生成新的特征
    """

    def __init__(self):
        """初始化特征生成器"""
        self.generated_features = []

    def generate_polynomial(
        self,
        X: np.ndarray,
        degree: int = 2,
    ) -> np.ndarray:
        """
        生成多项式特征

        Args:
            X: 原始特征
            degree: 多项式次数

        Returns:
            多项式特征
        """
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=True,
            include_bias=False,
        )

        X_poly = poly.fit_transform(X)
        self.generated_features.extend([
            f"poly_{i}" for i in range(X.shape[1], X_poly.shape[1])
        ])

        return X_poly

    def generate_rolling_features(
        self,
        data: pd.DataFrame,
        windows: List[int] = [5, 10, 20],
    ) -> pd.DataFrame:
        """
        生成滚动窗口特征

        Args:
            data: 时间序列数据
            windows: 窗口大小列表

        Returns:
            带滚动特征的数据
        """
        features = data.copy()

        for col in data.columns:
            for window in windows:
                # 滚动均值
                features[f"{col}_mean_{window}"] = data[col].rolling(window).mean()

                # 滚动标准差
                features[f"{col}_std_{window}"] = data[col].rolling(window).std()

                # 滚动最大值
                features[f"{col}_max_{window}"] = data[col].rolling(window).max()

                # 滚动最小值
                features[f"{col}_min_{window}"] = data[col].rolling(window).min()

        return features

    def generate_lag_features(
        self,
        data: pd.DataFrame,
        lags: List[int] = [1, 2, 5, 10],
    ) -> pd.DataFrame:
        """
        生成滞后特征

        Args:
            data: 时间序列数据
            lags: 滞后期列表

        Returns:
            带滞后特征的数据
        """
        features = data.copy()

        for col in data.columns:
            for lag in lags:
                features[f"{col}_lag_{lag}"] = data[col].shift(lag)

        return features

    def generate_difference_features(
        self,
        data: pd.DataFrame,
        periods: List[int] = [1, 5, 10],
    ) -> pd.DataFrame:
        """
        生成差分特征

        Args:
            data: 时间序列数据
            periods: 差分周期列表

        Returns:
            带差分特征的数据
        """
        features = data.copy()

        for col in data.columns:
            for period in periods:
                features[f"{col}_diff_{period}"] = data[col].diff(periods=period)

                # 百分比变化
                features[f"{col}_pct_change_{period}"] = data[col].pct_change(periods=period)

        return features


class FeatureImportanceAnalyzer:
    """
    特征重要性分析器
    """

    @staticmethod
    def analyze_permutation_importance(
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> Dict[str, np.ndarray]:
        """
        分析排列重要性

        Args:
            model: 训练好的模型
            X: 特征矩阵
            y: 目标变量
            n_repeats: 重复次数
            random_state: 随机种子

        Returns:
            特征重要性字典
        """
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        importances = {
            'importances_mean': result.importances_mean,
            'importances_std': result.importances_std,
            'importances': result.importances,
        }

        return importances

    @staticmethod
    def analyze_shap_values(
        model,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        分析SHAP值

        Args:
            model: 训练好的模型
            X: 特征矩阵
            feature_names: 特征名称

        Returns:
            SHAP分析结果
        """
        try:
            import shap

            # 创建解释器
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

            return {
                'values': shap_values.values,
                'base_values': shap_values.base_values,
                'data': shap_values.data,
                'feature_names': feature_names,
            }

        except ImportError:
            logger.warning("SHAP库未安装，无法计算SHAP值")
            return {}

    @staticmethod
    def plot_feature_importance(
        importances: np.ndarray,
        feature_names: List[str],
        top_n: int = 20,
    ):
        """
        绘制特征重要性图

        Args:
            importances: 重要性值
            feature_names: 特征名称
            top_n: 显示前N个
        """
        import matplotlib.pyplot as plt

        # 排序
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(
            range(top_n),
            importances[indices],
            align="center",
        )
        plt.xticks(
            range(top_n),
            [feature_names[i] for i in indices],
            rotation=45,
            ha="right",
        )
        plt.tight_layout()
        plt.show()
```

---

## 第43章 超低延迟高频交易

### 43.1 FPGA加速订单处理

```python
# src/hft/fpga_accelerator.py
"""
FPGA加速器接口

使用FPGA加速订单处理和风险检查
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FPGARiskCheckResult:
    """FPGA风险检查结果"""
    passed: bool
    reason: str = ""
    check_time_ns: int = 0  # 纳秒级检查时间


class FPGAAccelerator:
    """
    FPGA加速器

    通过FPGA实现超低延迟的订单处理
    """

    def __init__(
        self,
        fpga_ip: str = "192.168.1.100",
        fpga_port: int = 5000,
    ):
        """
        初始化FPGA加速器

        Args:
            fpga_ip: FPGA设备IP
            fpga_port: FPGA端口
        """
        self.fpga_ip = fpga_ip
        self.fpga_port = fpga_port
        self.connected = False

    async def connect(self):
        """连接FPGA设备"""
        # 实际实现中，这里会建立专用的网络连接
        self.connected = True
        logger.info(f"FPGA加速器已连接: {self.fpga_ip}")

    async def check_order_risk(
        self,
        order: Dict,
        portfolio: Dict,
    ) -> FPGARiskCheckResult:
        """
        FPGA加速风险检查

        Args:
            order: 订单信息
            portfolio: 组合信息

        Returns:
            风险检查结果
        """
        import time

        start = time.time_ns()

        # 在实际实现中，这里会将数据发送到FPGA
        # FPGA硬件执行风险检查逻辑，返回结果

        # 模拟FPGA处理 (实际延迟 < 1微秒)
        await asyncio.sleep(0.000001)

        end = time.time_ns()
        check_time = end - start

        # 模拟风险检查逻辑
        passed = True
        reason = ""

        # 持仓限制检查
        position = portfolio.get(order['symbol'], {}).get('quantity', 0)
        if order['side'] == 'buy':
            new_position = position + order['quantity']
        else:
            new_position = position - order['quantity']

        if abs(new_position) * order['price'] > portfolio.get('available_capital', 0):
            passed = False
            reason = "资金不足"

        return FPGARiskCheckResult(
            passed=passed,
            reason=reason,
            check_time_ns=check_time,
        )

    async def batch_check_risks(
        self,
        orders: List[Dict],
        portfolios: List[Dict],
    ) -> List[FPGARiskCheckResult]:
        """
        批量风险检查

        Args:
            orders: 订单列表
            portfolios: 组合列表

        Returns:
            检查结果列表
        """
        # FPGA可以并行处理多个检查
        tasks = [
            self.check_order_risk(order, portfolio)
            for order, portfolio in zip(orders, portfolios)
        ]

        return await asyncio.gather(*tasks)


class UltraLowLatencyEngine:
    """
    超低延迟交易引擎

    结合FPGA和软件优化实现微秒级延迟
    """

    def __init__(
        self,
        fpga_accelerator: Optional[FPGAAccelerator] = None,
    ):
        """
        初始化超低延迟引擎

        Args:
            fpga_accelerator: FPGA加速器
        """
        self.fpga = fpga_accelerator
        self.order_book_cache: Dict[str, any] = {}
        self.risk_cache: Dict[str, any] = {}

        # 性能统计
        self.latency_stats = {
            'risk_check': [],
            'order_routing': [],
            'total': [],
        }

    async def process_order(
        self,
        order: Dict,
        portfolio: Dict,
    ) -> Dict:
        """
        处理订单

        Args:
            order: 订单信息
            portfolio: 组合信息

        Returns:
            处理结果
        """
        import time

        start = time.time_ns()

        # 1. FPGA风险检查
        if self.fpga:
            risk_result = await self.fpga.check_order_risk(order, portfolio)
        else:
            # 软件风险检查
            risk_result = await self._software_risk_check(order, portfolio)

        self.latency_stats['risk_check'].append(risk_result.check_time_ns)

        if not risk_result.passed:
            end = time.time_ns()
            self.latency_stats['total'].append(end - start)
            return {
                'success': False,
                'reason': risk_result.reason,
                'latency_ns': end - start,
            }

        # 2. 订单路由
        route_start = time.time_ns()
        routed_order = await self._route_order(order)
        route_time = time.time_ns() - route_start

        self.latency_stats['order_routing'].append(route_time)

        end = time.time_ns()
        total_latency = end - start
        self.latency_stats['total'].append(total_latency)

        return {
            'success': True,
            'order_id': routed_order['order_id'],
            'latency_ns': total_latency,
            'risk_check_ns': risk_result.check_time_ns,
            'routing_ns': route_time,
        }

    async def _software_risk_check(
        self,
        order: Dict,
        portfolio: Dict,
    ) -> FPGARiskCheckResult:
        """软件风险检查"""
        import time
        start = time.time_ns()

        # 快速风险检查
        passed = True
        reason = ""

        # 检查资金
        required = order['quantity'] * order['price']
        if required > portfolio.get('available_capital', 0):
            passed = False
            reason = "资金不足"

        # 检查持仓限制
        position = portfolio.get(order['symbol'], {}).get('quantity', 0)
        if abs(position + order['quantity']) > 1000000:
            passed = False
            reason = "超过持仓限制"

        end = time.time_ns()

        return FPGARiskCheckResult(
            passed=passed,
            reason=reason,
            check_time_ns=end - start,
        )

    async def _route_order(self, order: Dict) -> Dict:
        """路由订单"""
        # 实际实现中，这里会直接连接到交易所网关
        order_id = f"order_{hash(str(order))}"

        return {
            'order_id': order_id,
            'status': 'submitted',
        }

    def get_latency_stats(self) -> Dict:
        """获取延迟统计"""
        stats = {}

        for key, values in self.latency_stats.items():
            if values:
                values_array = np.array(values)
                stats[key] = {
                    'mean_ns': np.mean(values_array),
                    'min_ns': np.min(values_array),
                    'max_ns': np.max(values_array),
                    'p50_ns': np.percentile(values_array, 50),
                    'p99_ns': np.percentile(values_array, 99),
                    'p99_9_ns': np.percentile(values_array, 99.9),
                }

        return stats
```

### 43.2 内核旁路网络

```python
# src/hft/kernel_bypass_networking.py
"""
内核旁路网络

使用DPDK或Solarflare实现零拷贝网络
"""

import asyncio
import socket
import struct
from typing import Callable, Optional
from dataclasses import dataclass
import numpy as np

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NetworkStats:
    """网络统计"""
    packets_received: int = 0
    packets_sent: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    errors: int = 0


class KernelBypassSocket:
    """
    内核旁路Socket

    使用OpenOnload (Solarflare) 或 DPDK实现零拷贝
    """

    def __init__(
        self,
        use_onload: bool = True,
        interface: str = "eth0",
    ):
        """
        初始化内核旁路Socket

        Args:
            use_onload: 是否使用OpenOnload
            interface: 网络接口
        """
        self.use_onload = use_onload
        self.interface = interface
        self.sock = None
        self.stats = NetworkStats()

    async def connect(
        self,
        host: str,
        port: int,
    ):
        """
        连接到服务器

        Args:
            host: 主机地址
            port: 端口
        """
        if self.use_onload:
            # 使用OpenOnload扩展
            try:
                import onload
                self.sock = onload.socket(socket.AF_INET, socket.SOCK_STREAM)
            except ImportError:
                logger.warning("OpenOnload不可用，使用普通socket")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.sock.connect((host, port))
        logger.info(f"内核旁路连接建立: {host}:{port}")

    async def send(self, data: bytes) -> int:
        """
        发送数据

        Args:
            data: 要发送的数据

        Returns:
            发送的字节数
        """
        sent = self.sock.send(data)
        self.stats.packets_sent += 1
        self.stats.bytes_sent += sent
        return sent

    async def recv(self, size: int = 4096) -> bytes:
        """
        接收数据

        Args:
            size: 接收大小

        Returns:
            接收的数据
        """
        data = self.sock.recv(size)
        self.stats.packets_received += 1
        self.stats.bytes_received += len(data)
        return data

    def close(self):
        """关闭连接"""
        if self.sock:
            self.sock.close()


class MarketDataFeed:
    """
    市场数据源

    使用内核旁路技术实现超低延迟行情接收
    """

    def __init__(
        self,
        feed_address: str,
        feed_port: int,
        use_kernel_bypass: bool = True,
    ):
        """
        初始化行情源

        Args:
            feed_address: 行情地址
            feed_port: 行情端口
            use_kernel_bypass: 是否使用内核旁路
        """
        self.feed_address = feed_address
        self.feed_port = feed_port
        self.use_kernel_bypass = use_kernel_bypass

        self.socket: Optional[KernelBypassSocket] = None
        self.handlers: list = []

        self.running = False
        self.stats = NetworkStats()

    async def start(self):
        """启动行情接收"""
        self.socket = KernelBypassSocket(
            use_onload=self.use_kernel_bypass,
        )
        await self.socket.connect(self.feed_address, self.feed_port)

        self.running = True

        # 启动接收循环
        asyncio.create_task(self._receive_loop())

        logger.info("行情源启动")

    async def stop(self):
        """停止行情接收"""
        self.running = False
        if self.socket:
            self.socket.close()

    def register_handler(self, handler: Callable):
        """
        注册行情处理器

        Args:
            handler: 处理函数
        """
        self.handlers.append(handler)

    async def _receive_loop(self):
        """接收循环"""
        while self.running:
            try:
                # 接收行情数据
                data = await self.socket.recv(4096)

                # 解析并分发
                await self._dispatch(data)

            except Exception as e:
                logger.error(f"行情接收错误: {str(e)}")
                self.stats.errors += 1

    async def _dispatch(self, data: bytes):
        """分发行情数据"""
        # 解析行情数据
        # 这里简化实现，实际需要根据协议解析

        for handler in self.handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"行情处理错误: {str(e)}")


class HFTOrderGateway:
    """
    高频交易订单网关

    使用内核旁路技术实现超低延迟订单提交
    """

    def __init__(
        self,
        gateway_address: str,
        gateway_port: int,
        use_kernel_bypass: bool = True,
    ):
        """
        初始化订单网关

        Args:
            gateway_address: 网关地址
            gateway_port: 网关端口
            use_kernel_bypass: 是否使用内核旁路
        """
        self.gateway_address = gateway_address
        self.gateway_port = gateway_port
        self.use_kernel_bypass = use_kernel_bypass

        self.socket: Optional[KernelBypassSocket] = None
        self.pending_orders: Dict[str, any] = {}

        self.stats = {
            'orders_submitted': 0,
            'orders_accepted': 0,
            'orders_rejected': 0,
            'latencies_ns': [],
        }

    async def connect(self):
        """连接订单网关"""
        self.socket = KernelBypassSocket(
            use_onload=self.use_kernel_bypass,
        )
        await self.socket.connect(
            self.gateway_address,
            self.gateway_port,
        )

        logger.info("订单网关已连接")

    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
    ) -> str:
        """
        提交订单

        Args:
            symbol: 股票代码
            side: 买卖方向
            quantity: 数量
            price: 价格

        Returns:
            订单ID
        """
        import time
        start = time.time_ns()

        # 构建订单消息
        order_msg = self._build_order_message(
            symbol, side, quantity, price
        )

        # 发送订单
        await self.socket.send(order_msg)

        order_id = f"{symbol}_{int(time.time_ns())}"
        self.pending_orders[order_id] = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'submit_time_ns': start,
        }

        self.stats['orders_submitted'] += 1

        return order_id

    def _build_order_message(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
    ) -> bytes:
        """
        构建订单消息

        Args:
            symbol: 股票代码
            side: 买卖方向
            quantity: 数量
            price: 价格

        Returns:
            订单消息字节
        """
        # 简化实现，使用二进制协议
        # 实际应根据交易所协议实现

        msg_type = 1  # 新订单
        side_code = 1 if side == 'buy' else 2

        # 打包消息
        # 格式: 类型(1) | 侧(1) | 数量(4) | 价格(8) | 代码长度(1) | 代码
        symbol_bytes = symbol.encode('ascii')
        msg = struct.pack(
            '!BBIdB',
            msg_type,
            side_code,
            quantity,
            int(price * 100),  # 转换为分
            len(symbol_bytes),
        )
        msg += symbol_bytes

        return msg

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()

        if stats['latencies_ns']:
            latencies = np.array(stats['latencies_ns'])
            stats['latency'] = {
                'mean_ns': np.mean(latencies),
                'min_ns': np.min(latencies),
                'max_ns': np.max(latencies),
                'p50_ns': np.percentile(latencies, 50),
                'p99_ns': np.percentile(latencies, 99),
            }

        return stats
```

### 43.3 内存池技术

```python
# src/hft/memory_pool.py
"""
内存池技术

减少内存分配开销
"""

import asyncio
import mmap
import multiprocessing as mp
from typing import Type, TypeVar, Generic, Optional
from dataclasses import dataclass
import ctypes

from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class MemoryBlock:
    """内存块"""
    address: int
    size: int
    in_use: bool = False


class MemoryPool:
    """
    内存池

    预分配大块内存，减少频繁分配/释放的开销
    """

    def __init__(
        self,
        block_size: int = 1024,
        num_blocks: int = 10000,
    ):
        """
        初始化内存池

        Args:
            block_size: 每个块的大小
            num_blocks: 块的数量
        """
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.total_size = block_size * num_blocks

        # 创建共享内存
        self.shared_memory = mp.SharedMemory(
            create=True,
            size=self.total_size,
        )

        # 初始化块列表
        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(MemoryBlock(
                address=self.shared_memory.buf + i * block_size,
                size=block_size,
            ))

        self.lock = asyncio.Lock()

        logger.info(f"内存池初始化: {self.total_size / 1024 / 1024:.2f}MB")

    async def allocate(self) -> Optional[MemoryBlock]:
        """
        分配内存块

        Returns:
            内存块或None
        """
        async with self.lock:
            for block in self.blocks:
                if not block.in_use:
                    block.in_use = True
                    return block

        return None

    async def free(self, block: MemoryBlock):
        """
        释放内存块

        Args:
            block: 要释放的块
        """
        async with self.lock:
            block.in_use = False

    def cleanup(self):
        """清理内存池"""
        self.shared_memory.close()
        self.shared_memory.unlink()


class ObjectPool(Generic[T]):
    """
    对象池

    复用对象减少GC压力
    """

    def __init__(
        self,
        factory: Type[T],
        initial_size: int = 100,
        max_size: int = 1000,
    ):
        """
        初始化对象池

        Args:
            factory: 对象工厂
            initial_size: 初始大小
            max_size: 最大大小
        """
        self.factory = factory
        self.initial_size = initial_size
        self.max_size = max_size

        self.pool = []
        self.in_use = set()
        self.lock = asyncio.Lock()

        # 预创建对象
        for _ in range(initial_size):
            self.pool.append(self.factory())

    async def acquire(self) -> T:
        """
        获取对象

        Returns:
            对象实例
        """
        async with self.lock:
            if self.pool:
                obj = self.pool.pop()
            elif len(self.in_use) < self.max_size:
                obj = self.factory()
            else:
                raise RuntimeError("对象池已满")

            self.in_use.add(id(obj))
            return obj

    async def release(self, obj: T):
        """
        释放对象

        Args:
            obj: 要释放的对象
        """
        async with self.lock:
            if id(obj) in self.in_use:
                self.in_use.remove(id(obj))
                self.pool.append(obj)

    async def __aenter__(self) -> T:
        return await self.acquire()

    async def __aexit__(self, *args):
        # 需要在上下文中保存对象引用
        pass


class ZeroCopyBuffer:
    """
    零拷贝缓冲区

    使用共享内存实现零拷贝数据传输
    """

    def __init__(self, size: int):
        """
        初始化零拷贝缓冲区

        Args:
            size: 缓冲区大小
        """
        self.size = size
        self.shared_memory = mp.SharedMemory(create=True, size=size)
        self.buffer = memoryview(self.shared_memory.buf)

    def write(self, data: bytes, offset: int = 0):
        """
        写入数据

        Args:
            data: 要写入的数据
            offset: 偏移量
        """
        self.buffer[offset:offset+len(data)] = data

    def read(self, size: int, offset: int = 0) -> bytes:
        """
        读取数据

        Args:
            size: 读取大小
            offset: 偏移量

        Returns:
            读取的数据
        """
        return bytes(self.buffer[offset:offset+size])

    def cleanup(self):
        """清理缓冲区"""
        self.buffer.release()
        self.shared_memory.close()
        self.shared_memory.unlink()
```

---

## 第44章 安全攻防实战

### 44.1 Web应用防火墙

```python
# src/security/waf.py
"""
Web应用防火墙实现
"""

import re
import asyncio
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from ipaddress import ip_address, ip_network

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WAFRule:
    """WAF规则"""
    name: str
    pattern: str
    action: str  # block, log, rate_limit
    severity: str  # low, medium, high, critical
    description: str = ""


@dataclass
class SecurityEvent:
    """安全事件"""
    timestamp: datetime
    source_ip: str
    rule: str
    action: str
    details: str
    severity: str


class WebApplicationFirewall:
    """
    Web应用防火墙

    防护常见Web攻击
    """

    def __init__(
        self,
        blocked_ips: Optional[List[str]] = None,
        rate_limits: Optional[Dict[str, int]] = None,
    ):
        """
        初始化WAF

        Args:
            blocked_ips: 封禁IP列表
            rate_limits: 速率限制
        """
        self.blocked_ips = set(blocked_ips or [])
        self.rate_limits = rate_limits or {
            'default': 100,  # 每分钟请求数
        }

        # 请求计数器
        self.request_counts: Dict[str, List[datetime]] = {}

        # 安全事件
        self.security_events: List[SecurityEvent] = []

        # 规则
        self.rules = self._init_rules()

        # IP白名单
        self.whitelisted_ips: set = set()

    def _init_rules(self) -> List[WAFRule]:
        """初始化WAF规则"""
        return [
            # SQL注入
            WAFRule(
                name="SQL_INJECTION",
                pattern=r"(?i)(union\s+select|';\s*drop|or\s+1\s*=\s*1|admin'--)",
                action="block",
                severity="critical",
                description="SQL注入攻击",
            ),

            # XSS
            WAFRule(
                name="XSS_ATTACK",
                pattern=r"<script[^>]*>.*?</script>|javascript:|onerror\s*=",
                action="block",
                severity="high",
                description="跨站脚本攻击",
            ),

            # 路径遍历
            WAFRule(
                name="PATH_TRAVERSAL",
                pattern=r"\.\./|\.\.\\",
                action="block",
                severity="high",
                description="路径遍历攻击",
            ),

            # 命令注入
            WAFRule(
                name="COMMAND_INJECTION",
                pattern=r"[;&|`$()]",
                action="block",
                severity="critical",
                description="命令注入攻击",
            ),

            # 恶意User-Agent
            WAFRule(
                name="MALICIOUS_UA",
                pattern=r"(?i)(sqlmap|nikto|nmap|metasploit|burp)",
                action="block",
                severity="medium",
                description="恶意扫描工具",
            ),
        ]

    async def check_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[str] = None,
        source_ip: str = "0.0.0.0",
    ) -> tuple[bool, Optional[str]]:
        """
        检查请求

        Args:
            method: HTTP方法
            path: 请求路径
            headers: 请求头
            body: 请求体
            source_ip: 源IP

        Returns:
            (是否通过, 拒绝原因)
        """
        # 1. IP白名单检查
        if source_ip in self.whitelisted_ips:
            return True, None

        # 2. IP黑名单检查
        if source_ip in self.blocked_ips:
            await self._log_event(
                source_ip,
                "BLOCKED_IP",
                "block",
                f"IP在黑名单中: {source_ip}",
                "high",
            )
            return False, "IP被封禁"

        # 3. 速率限制检查
        if not await self._check_rate_limit(source_ip):
            await self._log_event(
                source_ip,
                "RATE_LIMIT",
                "block",
                f"超过速率限制",
                "medium",
            )
            return False, "超过速率限制"

        # 4. 规则检查
        all_data = f"{path} {body or ''}"

        for header_name, header_value in headers.items():
            all_data += f" {header_name}: {header_value}"

        for rule in self.rules:
            if re.search(rule.pattern, all_data):
                await self._log_event(
                    source_ip,
                    rule.name,
                    rule.action,
                    f"匹配规则: {rule.description}",
                    rule.severity,
                )

                if rule.action == "block":
                    return False, f"触犯安全规则: {rule.name}"

        return True, None

    async def _check_rate_limit(self, source_ip: str) -> bool:
        """
        检查速率限制

        Args:
            source_ip: 源IP

        Returns:
            是否通过
        """
        now = datetime.now()

        # 初始化计数器
        if source_ip not in self.request_counts:
            self.request_counts[source_ip] = []

        # 清理旧记录 (1分钟前)
        cutoff = now.replace(second=0, microsecond=0)
        self.request_counts[source_ip] = [
            t for t in self.request_counts[source_ip]
            if t > cutoff
        ]

        # 检查限制
        limit = self.rate_limits.get('default', 100)
        if len(self.request_counts[source_ip]) >= limit:
            return False

        # 记录本次请求
        self.request_counts[source_ip].append(now)
        return True

    async def _log_event(
        self,
        source_ip: str,
        rule: str,
        action: str,
        details: str,
        severity: str,
    ):
        """
        记录安全事件

        Args:
            source_ip: 源IP
            rule: 规则名称
            action: 执行动作
            details: 详细信息
            severity: 严重级别
        """
        event = SecurityEvent(
            timestamp=datetime.now(),
            source_ip=source_ip,
            rule=rule,
            action=action,
            details=details,
            severity=severity,
        )

        self.security_events.append(event)

        # 根据严重级别记录日志
        if severity == "critical":
            logger.error(f"[WAF] {details}", extra={
                'ip': source_ip,
                'rule': rule,
            })
        elif severity == "high":
            logger.warning(f"[WAF] {details}", extra={
                'ip': source_ip,
                'rule': rule,
            })
        else:
            logger.info(f"[WAF] {details}", extra={
                'ip': source_ip,
                'rule': rule,
            })

    def block_ip(self, ip: str, duration: int = 3600):
        """
        封禁IP

        Args:
            ip: IP地址
            duration: 封禁时长(秒)
        """
        self.blocked_ips.add(ip)
        logger.warning(f"封禁IP: {ip}, 时长: {duration}秒")

        # 可以使用asyncio.Task在duration后解封
        asyncio.create_task(self._unblock_ip_after(ip, duration))

    async def _unblock_ip_after(self, ip: str, duration: int):
        """延迟解封IP"""
        await asyncio.sleep(duration)
        self.blocked_ips.discard(ip)
        logger.info(f"解封IP: {ip}")

    def add_to_whitelist(self, ip: str):
        """
        添加到白名单

        Args:
            ip: IP地址
        """
        self.whitelisted_ips.add(ip)
        logger.info(f"添加白名单: {ip}")

    def get_security_report(self) -> Dict:
        """获取安全报告"""
        # 按严重级别统计
        severity_counts = {}
        for event in self.security_events:
            severity_counts[event.severity] = \
                severity_counts.get(event.severity, 0) + 1

        # 按规则统计
        rule_counts = {}
        for event in self.security_events:
            rule_counts[event.rule] = \
                rule_counts.get(event.rule, 0) + 1

        # 最近的被封IP
        recent_blocked = [
            event.source_ip
            for event in self.security_events[-100:]
            if event.action == "block"
        ]

        return {
            'total_events': len(self.security_events),
            'severity_distribution': severity_counts,
            'rule_distribution': rule_counts,
            'recent_blocked_ips': recent_blocked,
            'blocked_ips_count': len(self.blocked_ips),
        }
```

### 44.2 加密通信

```python
# src/security/encryption.py
"""
加密通信实现
"""

import asyncio
from typing import Dict, Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import os
import base64

from src.core.logging import get_logger

logger = get_logger(__name__)


class SymmetricEncryption:
    """
    对称加密

    使用Fernet (AES-128-CBC)
    """

    def __init__(self, key: Optional[bytes] = None):
        """
        初始化对称加密

        Args:
            key: 加密密钥 (如果为None则自动生成)
        """
        if key:
            self.key = key
        else:
            self.key = Fernet.generate_key()

        self.cipher = Fernet(self.key)

    def encrypt(self, data: bytes) -> bytes:
        """
        加密数据

        Args:
            data: 明文

        Returns:
            密文
        """
        return self.cipher.encrypt(data)

    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        解密数据

        Args:
            ciphertext: 密文

        Returns:
            明文
        """
        return self.cipher.decrypt(ciphertext)

    def encrypt_string(self, text: str) -> str:
        """
        加密字符串

        Args:
            text: 明文字符串

        Returns:
            Base64编码的密文
        """
        ciphertext = self.encrypt(text.encode())
        return base64.b64encode(ciphertext).decode()

    def decrypt_string(self, ciphertext: str) -> str:
        """
        解密字符串

        Args:
            ciphertext: Base64编码的密文

        Returns:
            明文字符串
        """
        data = base64.b64decode(ciphertext)
        return self.decrypt(data).decode()


class AsymmetricEncryption:
    """
    非对称加密

    使用RSA
    """

    def __init__(
        self,
        private_key: Optional[bytes] = None,
        public_key: Optional[bytes] = None,
    ):
        """
        初始化非对称加密

        Args:
            private_key: 私钥 (PEM格式)
            public_key: 公钥 (PEM格式)
        """
        if private_key:
            self.private_key = serialization.load_pem_private_key(
                private_key,
                password=None,
            )
        else:
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )

        if public_key:
            self.public_key = serialization.load_pem_public_key(public_key)
        else:
            self.public_key = self.private_key.public_key()

    def encrypt(self, data: bytes) -> bytes:
        """
        使用公钥加密

        Args:
            data: 明文

        Returns:
            密文
        """
        ciphertext = self.public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return ciphertext

    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        使用私钥解密

        Args:
            ciphertext: 密文

        Returns:
            明文
        """
        plaintext = self.private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return plaintext

    def sign(self, data: bytes) -> bytes:
        """
        使用私钥签名

        Args:
            data: 要签名的数据

        Returns:
            签名
        """
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return signature

    def verify(self, data: bytes, signature: bytes) -> bool:
        """
        使用公钥验证签名

        Args:
            data: 原始数据
            signature: 签名

        Returns:
            是否验证通过
        """
        try:
            self.public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False

    def export_private_key(self) -> bytes:
        """导出私钥"""
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def export_public_key(self) -> bytes:
        """导出公钥"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )


class SecureChannel:
    """
    安全通信通道

    结合对称和非对称加密
    """

    def __init__(self, asymmetric: AsymmetricEncryption):
        """
        初始化安全通道

        Args:
            asymmetric: 非对称加密实例
        """
        self.asymmetric = asymmetric
        self.session_keys: Dict[str, SymmetricEncryption] = {}

    async def handshake(self, peer_id: str) -> Tuple[bytes, bytes]:
        """
        握手协议

        Args:
            peer_id: 对端ID

        Returns:
            (公钥, 签名)
        """
        # 生成会话密钥
        session_key = Fernet.generate_key()
        symmetric = SymmetricEncryption(session_key)

        # 加密会话密钥
        encrypted_key = self.asymmetric.encrypt(session_key)

        # 签名
        signature = self.asymmetric.sign(session_key)

        # 保存会话密钥
        self.session_keys[peer_id] = symmetric

        logger.info(f"与 {peer_id} 建立安全通道")

        return encrypted_key, signature

    async def receive_handshake(
        self,
        peer_id: str,
        encrypted_key: bytes,
        signature: bytes,
        peer_public_key: bytes,
    ) -> bool:
        """
        接收握手

        Args:
            peer_id: 对端ID
            encrypted_key: 加密的会话密钥
            signature: 签名
            peer_public_key: 对端公钥

        Returns:
            是否成功
        """
        # 解密会话密钥
        session_key = self.asymmetric.decrypt(encrypted_key)

        # 验证签名
        peer_enc = AsymmetricEncryption(public_key=peer_public_key)
        if not peer_enc.verify(session_key, signature):
            logger.warning(f"握手失败: {peer_id} 签名无效")
            return False

        # 保存会话密钥
        self.session_keys[peer_id] = SymmetricEncryption(session_key)

        logger.info(f"接受 {peer_id} 的安全通道")
        return True

    def encrypt_message(self, peer_id: str, message: bytes) -> bytes:
        """
        加密消息

        Args:
            peer_id: 对端ID
            message: 明文消息

        Returns:
            密文
        """
        if peer_id not in self.session_keys:
            raise ValueError(f"未建立与 {peer_id} 的安全通道")

        return self.session_keys[peer_id].encrypt(message)

    def decrypt_message(self, peer_id: str, ciphertext: bytes) -> bytes:
        """
        解密消息

        Args:
            peer_id: 对端ID
            ciphertext: 密文

        Returns:
            明文
        """
        if peer_id not in self.session_keys:
            raise ValueError(f"未建立与 {peer_id} 的安全通道")

        return self.session_keys[peer_id].decrypt(ciphertext)
```

### 44.3 审计日志系统

```python
# src/security/audit_log.py
"""
审计日志系统
"""

import asyncio
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import orjson

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AuditEvent:
    """审计事件"""
    timestamp: datetime
    event_type: str
    user_id: str
    action: str
    resource: str
    result: str  # success, failure
    details: Dict
    ip_address: str = ""
    user_agent: str = ""

    def to_dict(self) -> Dict:
        """转换为字典"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d

    def to_json(self) -> bytes:
        """转换为JSON"""
        return orjson.dumps(self.to_dict())

    def get_hash(self) -> str:
        """获取事件哈希"""
        return hashlib.sha256(self.to_json()).hexdigest()


class AuditLogger:
    """
    审计日志记录器

    记录所有敏感操作
    """

    def __init__(
        self,
        retention_days: int = 90,
        batch_size: int = 100,
    ):
        """
        初始化审计日志器

        Args:
            retention_days: 保留天数
            batch_size: 批处理大小
        """
        self.retention_days = retention_days
        self.batch_size = batch_size

        self.pending_events: List[AuditEvent] = []
        self.lock = asyncio.Lock()

        # 启动刷新任务
        asyncio.create_task(self._flush_loop())

    async def log(
        self,
        event_type: str,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        details: Dict,
        ip_address: str = "",
        user_agent: str = "",
    ):
        """
        记录审计事件

        Args:
            event_type: 事件类型
            user_id: 用户ID
            action: 动作
            resource: 资源
            result: 结果
            details: 详细信息
            ip_address: IP地址
            user_agent: 用户代理
        """
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        async with self.lock:
            self.pending_events.append(event)

        # 检查是否需要立即刷新
        if len(self.pending_events) >= self.batch_size:
            await self._flush()

    async def _flush_loop(self):
        """定期刷新"""
        while True:
            await asyncio.sleep(60)  # 每分钟刷新
            await self._flush()

    async def _flush(self):
        """刷新到存储"""
        async with self.lock:
            if not self.pending_events:
                return

            events = self.pending_events
            self.pending_events = []

        # 写入存储
        await self._write_to_storage(events)

    async def _write_to_storage(self, events: List[AuditEvent]):
        """
        写入存储

        Args:
            events: 事件列表
        """
        # 实际实现中，这里会写入数据库或文件
        for event in events:
            logger.info(f"[AUDIT] {event.event_type} - {event.action}", extra={
                'user_id': event.user_id,
                'resource': event.resource,
                'result': event.result,
                'hash': event.get_hash(),
            })

    async def query(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """
        查询审计日志

        Args:
            start_time: 开始时间
            end_time: 结束时间
            user_id: 用户ID过滤
            event_type: 事件类型过滤
            limit: 限制数量

        Returns:
            事件列表
        """
        # 实际实现中，这里会从数据库查询
        return []

    async def get_user_activity(
        self,
        user_id: str,
        days: int = 30,
    ) -> Dict:
        """
        获取用户活动统计

        Args:
            user_id: 用户ID
            days: 统计天数

        Returns:
            活动统计
        """
        # 实际实现中，这里会从数据库统计
        return {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'most_common_actions': [],
        }
```

---

## 第45章 生产事故管理

### 45.1 故障检测与自愈

```python
# src/operations/incident_management.py
"""
故障检测与自愈系统
"""

import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from src.core.logging import get_logger

logger = get_logger(__name__)


class IncidentSeverity(Enum):
    """事故严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Incident:
    """事故"""
    id: str
    severity: IncidentSeverity
    title: str
    description: str
    service: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    status: str = "open"  # open, acknowledged, resolved


class HealthChecker:
    """
    健康检查器

    定期检查服务健康状态
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        timeout: float = 10.0,
    ):
        """
        初始化健康检查器

        Args:
            check_interval: 检查间隔
            timeout: 超时时间
        """
        self.check_interval = check_interval
        self.timeout = timeout

        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, Dict] = {}

        self.running = False

    def register_check(
        self,
        name: str,
        check_func: Callable,
    ):
        """
        注册健康检查

        Args:
            name: 检查名称
            check_func: 检查函数
        """
        self.checks[name] = check_func
        logger.info(f"注册健康检查: {name}")

    async def start(self):
        """启动健康检查"""
        self.running = True
        asyncio.create_task(self._check_loop())
        logger.info("健康检查器启动")

    async def stop(self):
        """停止健康检查"""
        self.running = False
        logger.info("健康检查器停止")

    async def _check_loop(self):
        """检查循环"""
        while self.running:
            for name, check_func in self.checks.items():
                try:
                    start = asyncio.get_event_loop().time()

                    result = await asyncio.wait_for(
                        check_func(),
                        timeout=self.timeout,
                    )

                    elapsed = (asyncio.get_event_loop().time() - start) * 1000

                    self.results[name] = {
                        'status': 'healthy' if result else 'unhealthy',
                        'latency_ms': elapsed,
                        'timestamp': datetime.now(),
                    }

                except asyncio.TimeoutError:
                    self.results[name] = {
                        'status': 'timeout',
                        'latency_ms': self.timeout * 1000,
                        'timestamp': datetime.now(),
                    }
                    logger.warning(f"健康检查超时: {name}")

                except Exception as e:
                    self.results[name] = {
                        'status': 'error',
                        'error': str(e),
                        'timestamp': datetime.now(),
                    }
                    logger.error(f"健康检查错误: {name}", extra={
                        'error': str(e),
                    })

            await asyncio.sleep(self.check_interval)

    def get_results(self) -> Dict[str, Dict]:
        """获取检查结果"""
        return self.results.copy()

    def is_healthy(self) -> bool:
        """检查整体健康状态"""
        for result in self.results.values():
            if result['status'] != 'healthy':
                return False
        return True


class AutoHealer:
    """
    自动治愈器

    检测到故障时自动执行恢复操作
    """

    def __init__(
        self,
        health_checker: HealthChecker,
    ):
        """
        初始化自动治愈器

        Args:
            health_checker: 健康检查器
        """
        self.health_checker = health_checker
        self.remediation_actions: Dict[str, List[Callable]] = {}

        self.running = False
        self.incident_count = 0

    def register_remediation(
        self,
        check_name: str,
        action: Callable,
    ):
        """
        注册修复动作

        Args:
            check_name: 检查名称
            action: 修复动作
        """
        if check_name not in self.remediation_actions:
            self.remediation_actions[check_name] = []

        self.remediation_actions[check_name].append(action)
        logger.info(f"注册修复动作: {check_name}")

    async def start(self):
        """启动自动治愈"""
        self.running = True
        asyncio.create_task(self._monitor_loop())
        logger.info("自动治愈器启动")

    async def stop(self):
        """停止自动治愈"""
        self.running = False
        logger.info("自动治愈器停止")

    async def _monitor_loop(self):
        """监控循环"""
        prev_results = {}

        while self.running:
            results = self.health_checker.get_results()

            # 检查状态变化
            for name, result in results.items():
                prev_result = prev_results.get(name, {})

                # 检测到故障
                if (
                    result.get('status') != 'healthy' and
                    prev_result.get('status') == 'healthy'
                ):
                    logger.warning(f"检测到故障: {name}", extra={
                        'status': result.get('status'),
                    })

                    # 执行修复动作
                    await self._execute_remediation(name)

            prev_results = results
            await asyncio.sleep(10)

    async def _execute_remediation(self, check_name: str):
        """
        执行修复动作

        Args:
            check_name: 检查名称
        """
        if check_name not in self.remediation_actions:
            return

        self.incident_count += 1

        for action in self.remediation_actions[check_name]:
            try:
                logger.info(f"执行修复动作: {check_name}")
                await asyncio.wait_for(action(), timeout=60)

                # 等待一段时间看是否恢复
                await asyncio.sleep(10)

                result = self.health_checker.results.get(check_name, {})
                if result.get('status') == 'healthy':
                    logger.info(f"修复成功: {check_name}")
                    return

            except Exception as e:
                logger.error(f"修复动作失败: {check_name}", extra={
                    'error': str(e),
                })


class IncidentManager:
    """
    事故管理器

    管理生产事故的生命周期
    """

    def __init__(self):
        """初始化事故管理器"""
        self.incidents: Dict[str, Incident] = {}
        self.notification_handlers: List[Callable] = []

    async def create_incident(
        self,
        severity: IncidentSeverity,
        title: str,
        description: str,
        service: str,
    ) -> Incident:
        """
        创建事故

        Args:
            severity: 严重级别
            title: 标题
            description: 描述
            service: 服务名称

        Returns:
            事故对象
        """
        import uuid

        incident = Incident(
            id=str(uuid.uuid4()),
            severity=severity,
            title=title,
            description=description,
            service=service,
            created_at=datetime.now(),
        )

        self.incidents[incident.id] = incident

        # 发送通知
        await self._notify(incident)

        logger.warning(f"创建事故: {incident.id}", extra={
            'severity': severity.value,
            'title': title,
            'service': service,
        })

        return incident

    async def resolve_incident(self, incident_id: str):
        """
        解决事故

        Args:
            incident_id: 事故ID
        """
        if incident_id not in self.incidents:
            return

        incident = self.incidents[incident_id]
        incident.status = "resolved"
        incident.resolved_at = datetime.now()

        logger.info(f"解决事故: {incident_id}")

    async def _notify(self, incident: Incident):
        """
        发送事故通知

        Args:
            incident: 事故对象
        """
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(incident)
                else:
                    handler(incident)
            except Exception as e:
                logger.error("通知发送失败", extra={'error': str(e)})

    def register_notification_handler(self, handler: Callable):
        """
        注册通知处理器

        Args:
            handler: 处理函数
        """
        self.notification_handlers.append(handler)

    def get_active_incidents(self) -> List[Incident]:
        """获取活动事故列表"""
        return [
            inc for inc in self.incidents.values()
            if inc.status == "open"
        ]
```

### 45.2 灾难恢复

```python
# src/operations/disaster_recovery.py
"""
灾难恢复系统
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import shutil
import os

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BackupConfig:
    """备份配置"""
    name: str
    source_path: str
    destination_path: str
    retention_days: int = 30
    compress: bool = True


class BackupManager:
    """
    备份管理器

    自动执行数据库和文件备份
    """

    def __init__(self):
        """初始化备份管理器"""
        self.backup_configs: List[BackupConfig] = []
        self.running = False

    def register_backup(self, config: BackupConfig):
        """
        注册备份任务

        Args:
            config: 备份配置
        """
        self.backup_configs.append(config)
        logger.info(f"注册备份任务: {config.name}")

    async def start(self):
        """启动备份管理器"""
        self.running = True

        # 每天执行备份
        for config in self.backup_configs:
            asyncio.create_task(self._backup_loop(config))

        logger.info("备份管理器启动")

    async def stop(self):
        """停止备份管理器"""
        self.running = False
        logger.info("备份管理器停止")

    async def _backup_loop(self, config: BackupConfig):
        """备份循环"""
        while self.running:
            # 等待到下一个凌晨3点
            await self._wait_for_schedule(3, 0)

            try:
                await self._perform_backup(config)
            except Exception as e:
                logger.error(f"备份失败: {config.name}", extra={
                    'error': str(e),
                })

    async def _wait_for_schedule(self, hour: int, minute: int):
        """等待到指定时间"""
        now = datetime.now()

        # 计算下一个执行时间
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if now >= next_run:
            # 今天已经过了，等到明天
            from datetime import timedelta
            next_run += timedelta(days=1)

        wait_seconds = (next_run - now).total_seconds()
        await asyncio.sleep(wait_seconds)

    async def _perform_backup(self, config: BackupConfig):
        """
        执行备份

        Args:
            config: 备份配置
        """
        logger.info(f"开始备份: {config.name}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{config.name}_{timestamp}"

        if config.compress:
            backup_path = f"{config.destination_path}/{backup_name}.tar.gz"
            await self._create_compressed_backup(
                config.source_path,
                backup_path,
            )
        else:
            backup_path = f"{config.destination_path}/{backup_name}"
            await self._create_backup(
                config.source_path,
                backup_path,
            )

        # 清理旧备份
        await self._cleanup_old_backups(
            config.destination_path,
            config.name,
            config.retention_days,
        )

        logger.info(f"备份完成: {config.name} -> {backup_path}")

    async def _create_compressed_backup(
        self,
        source: str,
        destination: str,
    ):
        """创建压缩备份"""
        import tarfile

        with tarfile.open(destination, "w:gz") as tar:
            tar.add(source, arcname=os.path.basename(source))

    async def _create_backup(self, source: str, destination: str):
        """创建备份"""
        shutil.copytree(source, destination)

    async def _cleanup_old_backups(
        self,
        destination_path: str,
        name: str,
        retention_days: int,
    ):
        """清理旧备份"""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=retention_days)

        for filename in os.listdir(destination_path):
            if not filename.startswith(name):
                continue

            filepath = os.path.join(destination_path, filename)

            # 获取文件修改时间
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

            if mtime < cutoff:
                try:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                    else:
                        shutil.rmtree(filepath)
                    logger.info(f"删除旧备份: {filename}")
                except Exception as e:
                    logger.error(f"删除备份失败: {filename}", extra={
                        'error': str(e),
                    })


class FailoverManager:
    """
    故障转移管理器

    管理主备切换
    """

    def __init__(
        self,
        primary_endpoint: str,
        secondary_endpoint: str,
    ):
        """
        初始化故障转移管理器

        Args:
            primary_endpoint: 主节点端点
            secondary_endpoint: 备节点端点
        """
        self.primary_endpoint = primary_endpoint
        self.secondary_endpoint = secondary_endpoint

        self.current_primary = "primary"
        self.manual_override = False

    async def check_primary_health(self) -> bool:
        """
        检查主节点健康状态

        Returns:
            是否健康
        """
        # 实际实现中，这里会发送健康检查请求
        return True

    async def initiate_failover(self):
        """启动故障转移"""
        logger.warning("启动故障转移")

        # 1. 检查主节点状态
        primary_healthy = await self.check_primary_health()

        if primary_healthy and not self.manual_override:
            logger.info("主节点健康，无需故障转移")
            return

        # 2. 提升备节点
        self.current_primary = "secondary"
        logger.info("备节点已提升为主节点")

        # 3. 通知客户端
        await self._notify_clients()

        # 4. 尝试恢复主节点
        await self._recover_primary()

    async def _notify_clients(self):
        """通知客户端更新配置"""
        # 实际实现中，这里会通知所有客户端
        pass

    async def _recover_primary(self):
        """尝试恢复主节点"""
        # 实际实现中，这里会尝试修复主节点
        pass

    async def manual_failover(self):
        """手动故障转移"""
        self.manual_override = True
        await self.initiate_failover()
        self.manual_override = False
```

### 45.3 运维自动化

```python
# src/operations/automation.py
"""
运维自动化
"""

import asyncio
import paramiko
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Server:
    """服务器信息"""
    name: str
    host: str
    port: int = 22
    username: str = "root"
    key_path: Optional[str] = None


class RemoteExecutor:
    """
    远程命令执行器

    在多台服务器上执行命令
    """

    def __init__(self, servers: List[Server]):
        """
        初始化远程执行器

        Args:
            servers: 服务器列表
        """
        self.servers = servers
        self.connections: Dict[str, paramiko.SSHClient] = {}

    async def connect_all(self):
        """连接所有服务器"""
        for server in self.servers:
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                if server.key_path:
                    client.connect(
                        server.host,
                        port=server.port,
                        username=server.username,
                        key_filename=server.key_path,
                    )
                else:
                    # 使用密码认证
                    client.connect(
                        server.host,
                        port=server.port,
                        username=server.username,
                        password=os.getenv('SSH_PASSWORD'),
                    )

                self.connections[server.name] = client
                logger.info(f"连接服务器: {server.name}")

            except Exception as e:
                logger.error(f"连接失败: {server.name}", extra={
                    'error': str(e),
                })

    async def execute(
        self,
        server_name: str,
        command: str,
    ) -> str:
        """
        在指定服务器执行命令

        Args:
            server_name: 服务器名称
            command: 要执行的命令

        Returns:
            命令输出
        """
        if server_name not in self.connections:
            raise ValueError(f"未连接到服务器: {server_name}")

        client = self.connections[server_name]

        stdin, stdout, stderr = client.exec_command(command)

        output = stdout.read().decode()
        error = stderr.read().decode()

        if error:
            logger.warning(f"命令错误: {server_name}", extra={
                'command': command,
                'error': error,
            })

        return output

    async def execute_on_all(
        self,
        command: str,
    ) -> Dict[str, str]:
        """
        在所有服务器执行命令

        Args:
            command: 要执行的命令

        Returns:
            服务器名称到输出的映射
        """
        results = {}

        for server_name in self.connections:
            results[server_name] = await self.execute(
                server_name,
                command,
            )

        return results

    def close_all(self):
        """关闭所有连接"""
        for client in self.connections.values():
            client.close()

        self.connections.clear()


class DeploymentOrchestrator:
    """
    部署编排器

    自动化部署流程
    """

    def __init__(
        self,
        executor: RemoteExecutor,
    ):
        """
        初始化部署编排器

        Args:
            executor: 远程执行器
        """
        self.executor = executor
        self.deployments: Dict[str, Dict] = {}

    async def deploy(
        self,
        service: str,
        version: str,
        servers: List[str],
    ):
        """
        部署服务

        Args:
            service: 服务名称
            version: 版本号
            servers: 目标服务器列表
        """
        deployment_id = f"{service}_{version}_{int(datetime.now().timestamp())}"

        logger.info(f"开始部署: {deployment_id}")

        self.deployments[deployment_id] = {
            'service': service,
            'version': version,
            'status': 'deploying',
            'servers': servers,
            'started_at': datetime.now(),
        }

        try:
            # 1. 拉取镜像
            await self._pull_images(service, version, servers)

            # 2. 停止旧服务
            await self._stop_service(service, servers)

            # 3. 启动新服务
            await self._start_service(service, version, servers)

            # 4. 健康检查
            await self._health_check(service, servers)

            self.deployments[deployment_id]['status'] = 'success'

            logger.info(f"部署成功: {deployment_id}")

        except Exception as e:
            self.deployments[deployment_id]['status'] = 'failed'
            self.deployments[deployment_id]['error'] = str(e)

            logger.error(f"部署失败: {deployment_id}", extra={
                'error': str(e),
            })

            # 回滚
            await self._rollback(service, servers)

    async def _pull_images(
        self,
        service: str,
        version: str,
        servers: List[str],
    ):
        """拉取镜像"""
        for server in servers:
            command = f"docker pull registry.example.com/{service}:{version}"
            await self.executor.execute(server, command)

    async def _stop_service(
        self,
        service: str,
        servers: List[str],
    ):
        """停止旧服务"""
        for server in servers:
            command = f"docker stop {service} && docker rm {service}"
            await self.executor.execute(server, command)

    async def _start_service(
        self,
        service: str,
        version: str,
        servers: List[str],
    ):
        """启动新服务"""
        for server in servers:
            command = f"docker run -d --name {service} registry.example.com/{service}:{version}"
            await self.executor.execute(server, command)

    async def _health_check(
        self,
        service: str,
        servers: List[str],
    ):
        """健康检查"""
        for server in servers:
            # 检查容器状态
            command = f"docker ps --filter name={service} --format '{{{{.Status}}}}'"
            output = await self.executor.execute(server, command)

            if "Up" not in output:
                raise Exception(f"服务未启动: {server}")

    async def _rollback(
        self,
        service: str,
        servers: List[str],
    ):
        """回滚"""
        logger.warning(f"回滚服务: {service}")

        # 获取上一个版本
        # 实际实现中需要记录版本历史
        previous_version = "previous"

        await self._stop_service(service, servers)
        await self._start_service(service, previous_version, servers)
```

---

## 总结

本大师级实现细节文档涵盖:

**第41章 - 分布式系统架构深度实现**
- 一致性哈希与数据分片
- 分布式事务管理 (Saga、TCC、2PC)
- 分布式锁服务

**第42章 - 高级机器学习技术**
- 在线学习系统 (增量学习、Hoeffding树)
- 模型A/B测试框架
- 特征工程自动化

**第43章 - 超低延迟高频交易**
- FPGA加速订单处理
- 内核旁路网络 (DPDK、OpenOnload)
- 内存池技术

**第44章 - 安全攻防实战**
- Web应用防火墙 (WAF)
- 加密通信 (对称/非对称)
- 审计日志系统

**第45章 - 生产事故管理**
- 故障检测与自愈
- 灾难恢复系统
- 运维自动化

---

*本文档持续更新中...*

*最后更新: 2026-01-05*
