"""指标收集器，对应 Go 版本的 MetricsService"""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class MetricType:
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class Metric:
    """指标基类"""

    def __init__(self, name: str, help_text: str, metric_type: str):
        self.name = name
        self.help_text = help_text
        self.metric_type = metric_type
        self.labels = {}
        self.timestamp = datetime.utcnow()

    def with_labels(self, **labels):
        """添加标签"""
        metric = self.__class__(self.name, self.help_text, self.metric_type)
        metric.labels = labels
        return metric

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'help': self.help_text,
            'type': self.metric_type,
            'labels': self.labels,
            'timestamp': self.timestamp.isoformat()
        }


class Counter(Metric):
    """计数器"""

    def __init__(self, name: str, help_text: str = ""):
        super().__init__(name, help_text, MetricType.COUNTER)
        self.value = 0

    def inc(self, amount: float = 1):
        """增加计数"""
        self.value += amount
        self.timestamp = datetime.utcnow()

    def to_dict(self):
        base = super().to_dict()
        base['value'] = self.value
        return base


class Gauge(Metric):
    """仪表盘"""

    def __init__(self, name: str, help_text: str = ""):
        super().__init__(name, help_text, MetricType.GAUGE)
        self.value = 0

    def set(self, value: float):
        """设置值"""
        self.value = value
        self.timestamp = datetime.utcnow()

    def inc(self, amount: float = 1):
        """增加值"""
        self.value += amount
        self.timestamp = datetime.utcnow()

    def dec(self, amount: float = 1):
        """减少值"""
        self.value -= amount
        self.timestamp = datetime.utcnow()

    def to_dict(self):
        base = super().to_dict()
        base['value'] = self.value
        return base


class Histogram(Metric):
    """直方图"""

    def __init__(self, name: str, help_text: str = "", buckets: Optional[List[float]] = None):
        super().__init__(name, help_text, MetricType.HISTOGRAM)
        self.buckets = buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        self.bucket_counts = {b: 0 for b in self.buckets}
        self.bucket_counts[float('inf')] = 0
        self.count = 0
        self.sum = 0

    def observe(self, value: float):
        """观察值"""
        self.count += 1
        self.sum += value
        self.timestamp = datetime.utcnow()

        # 更新桶计数
        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] += 1
        self.bucket_counts[float('inf')] += 1

    def to_dict(self):
        base = super().to_dict()
        base.update({
            'count': self.count,
            'sum': self.sum,
            'buckets': self.bucket_counts
        })
        return base


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = asyncio.Lock()

        # 内置指标
        self._init_builtin_metrics()

    def _init_builtin_metrics(self):
        """初始化内置指标"""
        # 请求计数
        self.register_counter(
            'http_requests_total',
            'HTTP请求总数'
        ).with_labels(method='GET', endpoint='/api/live/status')

        # 响应时间
        self.register_histogram(
            'http_request_duration_seconds',
            'HTTP请求耗时'
        ).with_labels(method='GET')

        # 决策计数
        self.register_counter(
            'decisions_total',
            '决策总数'
        ).with_labels(action='enter_long')

        # 交易执行
        self.register_counter(
            'trades_total',
            '交易总数'
        ).with_labels(side='long')

        # 持仓数量
        self.register_gauge(
            'positions_count',
            '当前持仓数量'
        )

        # 系统资源
        self.register_gauge(
            'memory_usage_bytes',
            '内存使用量'
        )

        self.register_gauge(
            'cpu_usage_percent',
            'CPU使用率'
        )

    def register_counter(self, name: str, help_text: str = "") -> Counter:
        """注册计数器"""
        return self._register_metric(name, Counter(name, help_text))

    def register_gauge(self, name: str, help_text: str = "") -> Gauge:
        """注册仪表盘"""
        return self._register_metric(name, Gauge(name, help_text))

    def register_histogram(self, name: str, help_text: str = "", buckets: Optional[List[float]] = None) -> Histogram:
        """注册直方图"""
        return self._register_metric(name, Histogram(name, help_text, buckets))

    def _register_metric(self, name: str, metric: Metric) -> Metric:
        """注册指标"""
        async def _register():
            async with self._lock:
                if name not in self._metrics:
                    self._metrics[name] = metric
                return self._metrics[name]

        # 同步调用（首次注册）
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._metrics[name] = metric
        else:
            # 后续注册使用异步
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = asyncio.create_task(_register())
                # 这里不能等待，返回一个等待结果的包装器
                metric._wait_task = task
            else:
                loop.run_until_complete(_register())

        return metric

    def get_metric(self, name: str) -> Optional[Metric]:
        """获取指标"""
        return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有指标"""
        metrics = {}
        for name, metric in self._metrics.items():
            metrics[name] = metric.to_dict()
        return metrics

    async def collect_system_metrics(self):
        """收集系统指标"""
        import psutil

        # CPU使用率
        cpu_metric = self.get_metric('cpu_usage_percent')
        if cpu_metric:
            cpu_metric.set(psutil.cpu_percent())

        # 内存使用
        memory_metric = self.get_metric('memory_usage_bytes')
        if memory_metric:
            memory = psutil.virtual_memory()
            memory_metric.set(memory.used)

        # 磁盘使用
        disk_metric = self.get_metric('disk_usage_bytes')
        if disk_metric:
            disk = psutil.disk_usage('/')
            disk_metric.set(disk.used)

        # 网络IO
        net_metric = self.get_metric('network_io_bytes')
        if net_metric:
            net = psutil.net_io_counters()
            net_metric.set(net.bytes_sent + net.bytes_recv)

    async def start_collection(self, interval: int = 30):
        """启动指标收集"""
        logger.info(f"启动指标收集，间隔: {interval}秒")

        while True:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"收集系统指标失败: {e}")
                await asyncio.sleep(5)  # 错误时短暂等待

    def increment_counter(self, name: str, amount: float = 1, **labels):
        """增加计数器"""
        metric = self.get_metric(name)
        if metric and isinstance(metric, Counter):
            if labels:
                # 带标签的计数器
                label_key = str(labels)
                full_name = f"{name}_{label_key}"
                if full_name not in self._metrics:
                    labeled_metric = Counter(full_name, metric.help_text)
                    labeled_metric.labels = labels
                    self._metrics[full_name] = labeled_metric
                    metric = labeled_metric
            metric.inc(amount)

    def set_gauge(self, name: str, value: float, **labels):
        """设置仪表盘值"""
        metric = self.get_metric(name)
        if metric and isinstance(metric, Gauge):
            if labels:
                label_key = str(labels)
                full_name = f"{name}_{label_key}"
                if full_name not in self._metrics:
                    labeled_metric = Gauge(full_name, metric.help_text)
                    labeled_metric.labels = labels
                    self._metrics[full_name] = labeled_metric
                    metric = labeled_metric
            metric.set(value)

    def observe_histogram(self, name: str, value: float, **labels):
        """观察直方图值"""
        metric = self.get_metric(name)
        if metric and isinstance(metric, Histogram):
            if labels:
                label_key = str(labels)
                full_name = f"{name}_{label_key}"
                if full_name not in self._metrics:
                    labeled_metric = Histogram(full_name, metric.help_text)
                    labeled_metric.labels = labels
                    self._metrics[full_name] = labeled_metric
                    metric = labeled_metric
            metric.observe(value)


# 全局指标收集器实例
metrics = MetricsCollector()