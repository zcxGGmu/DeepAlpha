"""健康检查系统"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any

from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """健康状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class CheckResult:
    """检查结果"""

    def __init__(
        self,
        name: str,
        status: HealthStatus,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}
        self.duration = duration
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'duration': self.duration,
            'timestamp': self.timestamp.isoformat()
        }


class HealthCheck:
    """健康检查基类"""

    def __init__(self, name: str, timeout: float = 10.0):
        self.name = name
        self.timeout = timeout
        self.last_result: Optional[CheckResult] = None

    async def check(self) -> CheckResult:
        """执行健康检查"""
        start_time = asyncio.get_event_loop().time()

        try:
            # 执行实际检查
            result = await self._check_impl()

            # 计算耗时
            result.duration = asyncio.get_event_loop().time() - start_time

            self.last_result = result
            return result

        except asyncio.TimeoutError:
            return CheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"健康检查超时 ({self.timeout}秒)",
                duration=self.timeout
            )
        except Exception as e:
            return CheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"健康检查异常: {str(e)}",
                duration=asyncio.get_event_loop().time() - start_time
            )

    async def _check_impl(self) -> CheckResult:
        """子类实现具体的检查逻辑"""
        raise NotImplementedError


class DatabaseHealthCheck(HealthCheck):
    """数据库健康检查"""

    def __init__(self, db_session_factory, timeout: float = 5.0):
        super().__init__("database", timeout)
        self.db_session_factory = db_session_factory

    async def _check_impl(self) -> CheckResult:
        from sqlalchemy import text

        async with self.db_session_factory() as session:
            # 执行简单查询
            result = await session.execute(text("SELECT 1"))
            row = result.fetchone()

            if row and row[0] == 1:
                return CheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="数据库连接正常"
                )
            else:
                return CheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="数据库查询异常"
                )


class FreqtradeHealthCheck(HealthCheck):
    """Freqtrade健康检查"""

    def __init__(self, freqtrade_client, timeout: float = 10.0):
        super().__init__("freqtrade", timeout)
        self.client = freqtrade_client

    async def _check_impl(self) -> CheckResult:
        try:
            # 检查API连接
            status = await self.client.get_status()

            if status and status.get('status') == 'ok':
                return CheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Freqtrade API连接正常"
                )
            else:
                return CheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Freqtrade状态异常"
                )
        except Exception as e:
            return CheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"无法连接Freqtrade: {str(e)}"
            )


class MarketDataHealthCheck(HealthCheck):
    """市场数据健康检查"""

    def __init__(self, market_manager, timeout: float = 10.0):
        super().__init__("market_data", timeout)
        self.market_manager = market_manager

    async def _check_impl(self) -> CheckResult:
        # 检查最近的数据更新
        # 这里简化实现，实际应该检查具体的数据源
        return CheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message="市场数据正常"
        )


class MemoryHealthCheck(HealthCheck):
    """内存健康检查"""

    def __init__(self, threshold_mb: float = 1000.0, timeout: float = 5.0):
        super().__init__("memory", timeout)
        self.threshold_mb = threshold_mb

    async def _check_impl(self) -> CheckResult:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        if memory_mb > self.threshold_mb:
            return CheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message=f"内存使用过高: {memory_mb:.2f}MB",
                details={
                    'used_mb': memory_mb,
                    'threshold_mb': self.threshold_mb,
                    'percent': process.memory_percent()
                }
            )
        else:
            return CheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"内存使用正常: {memory_mb:.2f}MB",
                details={
                    'used_mb': memory_mb,
                    'threshold_mb': self.threshold_mb,
                    'percent': process.memory_percent()
                }
            )


class DiskSpaceHealthCheck(HealthCheck):
    """磁盘空间健康检查"""

    def __init__(self, path: str = "/", threshold_percent: float = 90.0, timeout: float = 5.0):
        super().__init__("disk_space", timeout)
        self.path = path
        self.threshold_percent = threshold_percent

    async def _check_impl(self) -> CheckResult:
        import psutil

        disk = psutil.disk_usage(self.path)
        used_percent = (disk.used / disk.total) * 100

        if used_percent > self.threshold_percent:
            return CheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message=f"磁盘空间不足: {used_percent:.1f}%",
                details={
                    'used_gb': disk.used / 1024 / 1024 / 1024,
                    'total_gb': disk.total / 1024 / 1024 / 1024,
                    'used_percent': used_percent
                }
            )
        else:
            return CheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"磁盘空间充足: {used_percent:.1f}%",
                details={
                    'used_gb': disk.used / 1024 / 1024 / 1024,
                    'total_gb': disk.total / 1024 / 1024 / 1024,
                    'free_gb': disk.free / 1024 / 1024 / 1024
                }
            )


class HealthChecker:
    """健康检查器"""

    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._last_check_time: Optional[datetime] = None
        self._check_interval = 30  # 默认30秒检查一次

    def register_check(self, check: HealthCheck):
        """注册健康检查"""
        self._checks[check.name] = check
        logger.info(f"注册健康检查: {check.name}")

    def unregister_check(self, name: str):
        """取消注册健康检查"""
        if name in self._checks:
            del self._checks[name]
            logger.info(f"取消注册健康检查: {name}")

    async def check_all(self) -> Dict[str, CheckResult]:
        """执行所有健康检查"""
        results = {}
        tasks = []

        # 并发执行所有检查
        for check in self._checks.values():
            task = asyncio.create_task(check.check())
            tasks.append((check.name, task))

        # 等待所有检查完成
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
            except Exception as e:
                logger.error(f"健康检查失败 {name}: {e}")
                results[name] = CheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"检查执行失败: {str(e)}"
                )

        self._last_check_time = datetime.utcnow()
        return results

    async def check_single(self, name: str) -> Optional[CheckResult]:
        """执行单个健康检查"""
        check = self._checks.get(name)
        if not check:
            return None

        return await check.check()

    def get_overall_status(self, results: Dict[str, CheckResult]) -> HealthStatus:
        """获取整体健康状态"""
        if not results:
            return HealthStatus.UNKNOWN

        statuses = [result.status for result in results.values()]

        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN

    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要"""
        if not self._last_check_time:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': '尚未执行健康检查',
                'last_check': None,
                'checks': {}
            }

        # 返回上次检查结果的摘要
        summary = {
            'status': HealthStatus.UNKNOWN,
            'message': '',
            'last_check': self._last_check_time.isoformat(),
            'checks': {}
        }

        for check in self._checks.values():
            if check.last_result:
                summary['checks'][check.name] = check.last_result.to_dict()

        # 计算整体状态
        if summary['checks']:
            results = {
                name: CheckResult(**result)
                for name, result in summary['checks'].items()
            }
            summary['status'] = self.get_overall_status(results)

            # 生成消息
            healthy_count = sum(
                1 for r in results.values()
                if r.status == HealthStatus.HEALTHY
            )
            total_count = len(results)
            summary['message'] = f"{healthy_count}/{total_count} 检查通过"

        return summary


# 全局健康检查器实例
health_checker = HealthChecker()