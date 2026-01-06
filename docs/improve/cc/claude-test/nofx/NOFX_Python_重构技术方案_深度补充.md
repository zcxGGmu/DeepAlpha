# NOFX Python 重构技术方案：深度补充（十四-十六章及更多）

> 本文档是前两个文档的进一步补充，包含第十四章至第十六章的详细内容，以及额外的实现细节、最佳实践和完整示例。

---

## 十四、监控与告警系统

### 14.1 监控架构设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         监控与告警系统架构                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     应用层监控                                   │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │   API监控    │  │   业务监控   │  │   决策监控   │          │   │
│  │  │ - 请求量     │  │ - 交易员状态 │  │ - AI调用     │          │   │
│  │  │ - 响应时间   │  │ - 持仓变化   │  │ - 决策质量   │          │   │
│  │  │ - 错误率     │  │ - 盈亏统计   │  │ - Prompt效果 │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                     │
│                                  ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     指标收集层                                   │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │ Prometheus   │  │    StatsD    │  │   Custom     │          │   │
│  │  │  (时序数据)  │  │  (计数器)    │  │  (业务指标)  │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                     │
│                                  ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     存储层                                       │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │ Prometheus   │  │ PostgreSQL   │  │    Redis     │          │   │
│  │  │   TSDB       │  │  (业务数据)  │  │   (缓存)     │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                     │
│                                  ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     告警层                                       │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │ AlertManager │  │   自定义规则 │  │   通知渠道   │          │   │
│  │  │ (告警路由)   │  │  (业务规则)  │  │ (邮件/短信)  │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                     │
│                                  ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     可视化层                                     │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │   Grafana    │  │   Dashboard  │  │   Reports    │          │   │
│  │  │  (仪表盘)    │  │  (自定义)    │  │   (报表)     │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 14.2 监控指标定义

```python
# backend/monitoring/metrics.py
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info,
    Enum,
    CollectorRegistry,
)
from prometheus_client.exposition import generate_latest

# 自定义 Registry
custom_registry = CollectorRegistry()

# ========== API 指标 ==========

# HTTP 请求总数
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=custom_registry,
)

# HTTP 请求延迟
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
    registry=custom_registry,
)

# 当前活跃请求
http_requests_active = Gauge(
    "http_requests_active",
    "Active HTTP requests",
    ["endpoint"],
    registry=custom_registry,
)

# ========== 业务指标 ==========

# 交易员总数
traders_total = Gauge(
    "traders_total",
    "Total number of traders",
    ["market_type", "status"],
    registry=custom_registry,
)

# 决策总数
decisions_total = Counter(
    "decisions_total",
    "Total decisions made",
    ["trader_id", "action"],
    registry=custom_registry,
)

# 决策延迟
decision_duration_seconds = Histogram(
    "decision_duration_seconds",
    "Decision making duration",
    ["trader_id"],
    buckets=(1, 5, 10, 30, 60, 120, 300),
    registry=custom_registry,
)

# AI 调用总数
ai_calls_total = Counter(
    "ai_calls_total",
    "Total AI API calls",
    ["provider", "model", "status"],
    registry=custom_registry,
)

# AI 调用延迟
ai_call_duration_seconds = Histogram(
    "ai_call_duration_seconds",
    "AI API call duration",
    ["provider", "model"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30),
    registry=custom_registry,
)

# AI Token 使用量
ai_tokens_total = Counter(
    "ai_tokens_total",
    "Total AI tokens used",
    ["provider", "model", "type"],  # type: prompt, completion
    registry=custom_registry,
)

# ========== 交易指标 ==========

# 交易总数
trades_total = Counter(
    "trades_total",
    "Total trades executed",
    ["trader_id", "symbol", "side"],
    registry=custom_registry,
)

# 交易金额
trades_value_total = Counter(
    "trades_value_total",
    "Total trading value",
    ["trader_id", "symbol"],
    registry=custom_registry,
)

# 当前持仓价值
positions_value = Gauge(
    "positions_value",
    "Current positions value",
    ["trader_id", "symbol", "side"],
    registry=custom_registry,
)

# 账户权益
account_equity = Gauge(
    "account_equity",
    "Account equity",
    ["trader_id"],
    registry=custom_registry,
)

# 未实现盈亏
unrealized_pnl = Gauge(
    "unrealized_pnl",
    "Unrealized PnL",
    ["trader_id", "symbol"],
    registry=custom_registry,
)

# 已实现盈亏
realized_pnl = Counter(
    "realized_pnl",
    "Realized PnL",
    ["trader_id", "symbol"],
    registry=custom_registry,
)

# 胜率
win_rate = Gauge(
    "win_rate",
    "Win rate",
    ["trader_id"],
    registry=custom_registry,
)

# 最大回撤
max_drawdown = Gauge(
    "max_drawdown",
    "Maximum drawdown",
    ["trader_id"],
    registry=custom_registry,
)

# ========== 市场数据指标 ==========

# 市场数据请求
market_data_requests_total = Counter(
    "market_data_requests_total",
    "Total market data requests",
    ["source", "symbol", "status"],
    registry=custom_registry,
)

# 市场数据延迟
market_data_age_seconds = Gauge(
    "market_data_age_seconds",
    "Age of market data in seconds",
    ["source", "symbol"],
    registry=custom_registry,
)

# 订阅的股票数
subscribed_symbols = Gauge(
    "subscribed_symbols",
    "Number of subscribed symbols",
    ["market_type"],
    registry=custom_registry,
)

# ========== 系统指标 ==========

# 数据库连接池
db_connections_active = Gauge(
    "db_connections_active",
    "Active database connections",
    ["database"],
    registry=custom_registry,
)

db_connections_idle = Gauge(
    "db_connections_idle",
    "Idle database connections",
    ["database"],
    registry=custom_registry,
)

# Redis 连接
redis_connections_active = Gauge(
    "redis_connections_active",
    "Active Redis connections",
    registry=custom_registry,
)

# 任务队列
task_queue_size = Gauge(
    "task_queue_size",
    "Size of task queue",
    ["queue_name"],
    registry=custom_registry,

task_queue_processing = Gauge(
    "task_queue_processing",
    "Number of tasks being processed",
    ["queue_name"],
    registry=custom_registry,
)

# 缓存命中率
cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["cache_name"],
    registry=custom_registry,
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["cache_name"],
    registry=custom_registry,
)

# ========== LLM 指标 ==========

# LLM Prompt 长度
llm_prompt_length = Summary(
    "llm_prompt_length",
    "LLM prompt length in characters",
    ["provider", "model"],
    registry=custom_registry,
)

# LLM 响应长度
llm_response_length = Summary(
    "llm_response_length",
    "LLM response length in characters",
    ["provider", "model"],
    registry=custom_registry,
)

# LLM 解析成功率
llm_parse_success_rate = Gauge(
    "llm_parse_success_rate",
    "LLM response parse success rate",
    ["provider", "model"],
    registry=custom_registry,
)

# ========== 应用信息 ==========

application_info = Info(
    "application",
    "Application information",
    registry=custom_registry,
)


# ========== 中间件 ==========

from fastapi import Request
import time

async def metrics_middleware(request: Request, call_next):
    """Prometheus 监控中间件"""
    # 开始计时
    start_time = time.time()
    endpoint = request.url.path
    method = request.method

    # 增加活跃请求计数
    http_requests_active.labels(endpoint=endpoint).inc()

    try:
        # 处理请求
        response = await call_next(request)
        status = response.status_code

        # 记录请求
        http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=status,
        ).inc()

        return response

    finally:
        # 减少活跃请求计数
        http_requests_active.labels(endpoint=endpoint).dec()

        # 记录延迟
        duration = time.time() - start_time
        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)


# ========== 指标导出端点 ==========

from fastapi import Response
from fastapi.routing import APIRouter

metrics_router = APIRouter()

@metrics_router.get("/metrics")
async def metrics():
    """Prometheus 指标端点"""
    return Response(
        content=generate_latest(custom_registry),
        media_type="text/plain",
    )

@metrics_router.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
    }
```

### 14.3 告警规则定义

```yaml
# deployment/prometheus/alerts.yml

groups:
  - name: api_alerts
    interval: 30s
    rules:
      # API 错误率告警
      - alert: HighAPIErrorRate
        expr: |
          rate(http_requests_total{status=~"5.."}[5m]) /
          rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
          component: api
        annotations:
          summary: "API 错误率过高"
          description: "API 错误率超过 5% (当前值: {{ $value | humanizePercentage }})"

      # API 响应时间告警
      - alert: SlowAPIResponse
        expr: |
          histogram_quantile(0.95, http_request_duration_seconds) > 1
        for: 5m
        labels:
          severity: warning
          component: api
        annotations:
          summary: "API 响应过慢"
          description: "95分位响应时间超过 1秒 (当前值: {{ $value }}s)"

      # API 完全不可用告警
      - alert: APIDown
        expr: |
          rate(http_requests_total[1m]) == 0
        for: 2m
        labels:
          severity: critical
          component: api
        annotations:
          summary: "API 服务不可用"
          description: "API 在过去 2 分钟内没有收到任何请求"

  - name: business_alerts
    interval: 30s
    rules:
      # 决策失败告警
      - alert: HighDecisionFailureRate
        expr: |
          rate(decisions_total{status="failed"}[5m]) /
          rate(decisions_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
          component: decision
        annotations:
          summary: "决策失败率过高"
          description: "交易员 {{ $labels.trader_id }} 决策失败率超过 10%"

      # AI 调用失败告警
      - alert: HighAIFailureRate
        expr: |
          rate(ai_calls_total{status="error"}[5m]) /
          rate(ai_calls_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
          component: ai
        annotations:
          summary: "AI 调用失败率过高"
          description: "{{ $labels.provider }} AI 调用失败率超过 10%"

      # AI 响应时间告警
      - alert: SlowAIResponse
        expr: |
          histogram_quantile(0.95, ai_call_duration_seconds) > 30
        for: 5m
        labels:
          severity: warning
          component: ai
        annotations:
          summary: "AI 响应过慢"
          description: "{{ $labels.provider }} AI 95分位响应时间超过 30秒"

      # 账户回撤告警
      - alert: HighDrawdown
        expr: |
          max_drawdown{trader_id=~".+"} > 0.15
        for: 1m
        labels:
          severity: warning
          component: trading
        annotations:
          summary: "账户回撤过大"
          description: "交易员 {{ $labels.trader_id }} 回撤超过 15% (当前: {{ $value | humanizePercentage }})"

      # 账户异常亏损告警
      - alert: AbnormalLoss
        expr: |
          delta(account_equity[1h]) / account_equity < -0.1
        for: 1m
        labels:
          severity: critical
          component: trading
        annotations:
          summary: "账户异常亏损"
          description: "交易员 {{ $labels.trader_id }} 过去1小时亏损超过 10%"

  - name: system_alerts
    interval: 30s
    rules:
      # 数据库连接池耗尽
      - alert: DatabasePoolExhausted
        expr: |
          db_connections_active / db_connections_idle < 0.1
        for: 5m
        labels:
          severity: warning
          component: database
        annotations:
          summary: "数据库连接池接近耗尽"
          description: "活跃连接占比超过 90%"

      # Redis 连接异常
      - alert: RedisConnectionFailed
        expr: |
          redis_connections_active == 0
        for: 1m
        labels:
          severity: critical
          component: redis
        annotations:
          summary: "Redis 连接失败"
          description: "无法连接到 Redis"

      # 缓存命中率过低
      - alert: LowCacheHitRate
        expr: |
          rate(cache_hits_total[5m]) /
          (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) < 0.5
        for: 10m
        labels:
          severity: warning
          component: cache
        annotations:
          summary: "缓存命中率过低"
          description: "缓存 {{ $labels.cache_name }} 命中率低于 50%"

      # 市场数据延迟告警
      - alert: StaleMarketData
        expr: |
          market_data_age_seconds > 60
        for: 2m
        labels:
          severity: warning
          component: market_data
        annotations:
          summary: "市场数据过期"
          description: "{{ $labels.symbol }} 市场数据超过 60 秒未更新"
```

### 14.4 告警通知系统

```python
# backend/monitoring/notifier.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import httpx
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """告警消息"""
    title: str
    description: str
    severity: AlertSeverity
    labels: dict[str, str] | None = None
    value: float | None = None
    timestamp: float | None = None


class AlertNotifier(ABC):
    """告警通知器抽象基类"""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """发送告警"""
        pass


class EmailNotifier(AlertNotifier):
    """邮件通知器"""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: list[str],
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    async def send(self, alert: Alert) -> bool:
        """发送邮件告警"""
        try:
            # 创建邮件
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            # 构建邮件内容
            text_content = self._build_text_content(alert)
            html_content = self._build_html_content(alert)

            text_part = MIMEText(text_content, "plain", "utf-8")
            html_part = MIMEText(html_content, "html", "utf-8")

            msg.attach(text_part)
            msg.attach(html_part)

            # 发送邮件
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _build_text_content(self, alert: Alert) -> str:
        """构建文本内容"""
        content = f"""
告警: {alert.title}

严重程度: {alert.severity.value.upper()}
描述: {alert.description}
        """.strip()

        if alert.labels:
            content += "\n\n标签:\n"
            for key, value in alert.labels.items():
                content += f"  {key}: {value}\n"

        if alert.value is not None:
            content += f"\n当前值: {alert.value}"

        return content

    def _build_html_content(self, alert: Alert) -> str:
        """构建 HTML 内容"""
        severity_colors = {
            AlertSeverity.INFO: "#3498db",
            AlertSeverity.WARNING: "#f39c12",
            AlertSeverity.ERROR: "#e74c3c",
            AlertSeverity.CRITICAL: "#8e44ad",
        }

        color = severity_colors.get(alert.severity, "#333333")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .alert-box {{
            border-left: 4px solid {color};
            padding: 15px;
            background-color: #f8f9fa;
            margin: 10px 0;
        }}
        .severity {{
            color: {color};
            font-weight: bold;
            text-transform: uppercase;
        }}
        .label {{
            display: inline-block;
            background: #e9ecef;
            padding: 2px 8px;
            margin: 2px;
            border-radius: 3px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="alert-box">
        <h2>{alert.title}</h2>
        <p><span class="severity">严重程度: {alert.severity.value}</span></p>
        <p><strong>描述:</strong> {alert.description}</p>
        """

        if alert.labels:
            html += "<p><strong>标签:</strong></p>"
            for key, value in alert.labels.items():
                html += f'<span class="label">{key}: {value}</span>'

        if alert.value is not None:
            html += f"<p><strong>当前值:</strong> {alert.value}</p>"

        html += """
    </div>
</body>
</html>
        """

        return html


class WebhookNotifier(AlertNotifier):
    """Webhook 通知器（支持钉钉、企业微信等）"""

    def __init__(self, webhook_url: str, notifier_type: str = "generic"):
        self.webhook_url = webhook_url
        self.notifier_type = notifier_type

    async def send(self, alert: Alert) -> bool:
        """发送 Webhook 告警"""
        try:
            payload = self._build_payload(alert)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                response.raise_for_status()

            logger.info(f"Webhook alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

    def _build_payload(self, alert: Alert) -> dict[str, Any]:
        """构建 Webhook 载荷"""

        if self.notifier_type == "dingtalk":
            # 钉钉格式
            return {
                "msgtype": "markdown",
                "markdown": {
                    "title": alert.title,
                    "text": f"### {alert.title}\n\n"
                           f"**严重程度**: {alert.severity.value}\n\n"
                           f"**描述**: {alert.description}\n\n",
                },
            }

        elif self.notifier_type == "wecom":
            # 企业微信格式
            return {
                "msgtype": "text",
                "text": {
                    "content": f"[{alert.severity.value.upper()}] {alert.title}\n"
                              f"{alert.description}",
                },
            }

        else:
            # 通用格式
            return {
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "labels": alert.labels,
                "value": alert.value,
                "timestamp": alert.timestamp,
            }


class SlackNotifier(AlertNotifier):
    """Slack 通知器"""

    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        self.webhook_url = webhook_url
        self.channel = channel

    async def send(self, alert: Alert) -> bool:
        """发送 Slack 告警"""
        try:
            severity_colors = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9800",
                AlertSeverity.ERROR: "#f44336",
                AlertSeverity.CRITICAL: "#9c27b0",
            }

            color = severity_colors.get(alert.severity, "#cccccc")

            payload = {
                "channel": self.channel,
                "username": "DeepAlpha Alerts",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True,
                            },
                        ],
                    }
                ],
            }

            if alert.labels:
                payload["attachments"][0]["fields"].extend([
                    {
                        "title": key,
                        "value": str(value),
                        "short": True,
                    }
                    for key, value in alert.labels.items()
                ])

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                response.raise_for_status()

            logger.info(f"Slack alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.notifiers: list[AlertNotifier] = []
        self.alert_cooldown: dict[str, float] = {}
        self.cooldown_period = 300  # 5分钟冷却

    def register_notifier(self, notifier: AlertNotifier):
        """注册通知器"""
        self.notifiers.append(notifier)

    async def send_alert(self, alert: Alert):
        """发送告警"""
        # 检查冷却时间
        alert_key = f"{alert.title}:{alert.severity.value}"
        last_sent = self.alert_cooldown.get(alert_key, 0)

        import time
        if time.time() - last_sent < self.cooldown_period:
            logger.debug(f"Alert {alert_key} is in cooldown, skipping")
            return

        # 发送告警
        tasks = [notifier.send(alert) for notifier in self.notifiers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 更新冷却时间
        if any(results):
            self.alert_cooldown[alert_key] = time.time()


# 全局告警管理器
alert_manager = AlertManager()
```

### 14.5 Grafana 仪表盘配置

```json
// deployment/grafana/dashboards/trading-dashboard.json
{
  "dashboard": {
    "title": "DeepAlpha 交易系统监控",
    "tags": ["trading", "ai", "stocks"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "API 请求量",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "id": 2,
        "title": "API 响应时间 (P95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds)",
            "legendFormat": "P95 延迟"
          }
        ]
      },
      {
        "id": 3,
        "title": "AI 调用延迟",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, ai_call_duration_seconds)",
            "legendFormat": "{{provider}} P95"
          }
        ]
      },
      {
        "id": 4,
        "title": "账户权益曲线",
        "type": "graph",
        "targets": [
          {
            "expr": "account_equity",
            "legendFormat": "{{trader_id}}"
          }
        ]
      },
      {
        "id": 5,
        "title": "胜率统计",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(win_rate)",
            "legendFormat": "平均胜率"
          }
        ]
      },
      {
        "id": 6,
        "title": "当前持仓价值",
        "type": "table",
        "targets": [
          {
            "expr": "positions_value",
            "legendFormat": "{{trader_id}} {{symbol}}"
          }
        ]
      },
      {
        "id": 7,
        "title": "决策执行统计",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (action) (decisions_total)",
            "legendFormat": "{{action}}"
          }
        ]
      }
    ]
  }
}
```

---

## 十五、性能优化详细方案

### 15.1 性能优化清单

| 类别 | 优化项 | 预期提升 | 优先级 |
|------|--------|----------|--------|
| **数据库** | 连接池配置 | 20-30% | P0 |
| | 索引优化 | 50-70% | P0 |
| | 查询优化 | 30-50% | P0 |
| | 批量操作 | 40-60% | P1 |
| | 读写分离 | 100%+ | P2 |
| **缓存** | Redis 缓存 | 50-80% | P0 |
| | 查询结果缓存 | 60-90% | P0 |
| | 会话缓存 | 40-50% | P1 |
| **API** | 异步处理 | 100%+ | P0 |
| | 响应压缩 | 20-30% | P1 |
| | 连接复用 | 10-20% | P1 |
| **LLM** | 请求批处理 | 30-40% | P0 |
| | 响应缓存 | 50-70% | P1 |
| | Token 优化 | 20-30% | P2 |
| **架构** | 负载均衡 | 线性扩展 | P1 |
| | 服务拆分 | 50%+ | P2 |
| | 消息队列 | 30%+ | P2 |

### 15.2 数据库优化

```python
# backend/optimizations/database.py
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class DatabaseOptimizer:
    """数据库优化器"""

    def __init__(self, session: AsyncSession):
        self.session = session

    @asynccontextmanager
    async def batch_operation(
        self,
        batch_size: int = 1000,
    ) -> AsyncGenerator[Any, None]:
        """
        批量操作上下文管理器

        使用示例:
            async with optimizer.batch_operation(100) as batch:
                for item in items:
                    batch.add(item)
        """
        from sqlalchemy import bulk_saveobjects

        items = []

        class Batch:
            def __init__(self, optimizer, size):
                self.optimizer = optimizer
                self.size = size
                self.items = items

            async def add(self, item):
                self.items.append(item)
                if len(self.items) >= self.size:
                    await self.flush()

            async def flush(self):
                if self.items:
                    await self.optimizer.session.execute(
                        bulk_saveobjects(self.items)
                    )
                    self.items.clear()

        batch = Batch(self, batch_size)
        try:
            yield batch
            await batch.flush()
        finally:
            pass

    async def analyze_query(self, query: str) -> dict[str, Any]:
        """分析查询性能"""
        result = await self.session.execute(
            text(f"EXPLAIN ANALYZE {query}")
        )
        rows = result.fetchall()

        # 解析 EXPLAIN ANALYZE 输出
        total_time = None
        for row in rows:
            if "Execution Time" in str(row):
                total_time = float(str(row).split(":")[1].strip().replace(" ms", ""))
                break

        return {
            "total_time_ms": total_time,
            "plan": [str(row) for row in rows],
        }

    async def create_indexes(self):
        """创建优化索引"""

        # 交易员表索引
        await self.session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_traders_user_market
            ON traders(user_id, market_type)
        """))

        # 决策表索引
        await self.session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_decisions_trader_created
            ON decisions(trader_id, created_at DESC)
        """))

        # 持仓表索引
        await self.session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_positions_open_trader
            ON positions(is_open, trader_id)
        """))

        # 账户快照索引
        await self.session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_trader_time
            ON account_snapshots(trader_id, created_at DESC)
        """))

        # 交易记录索引
        await self.session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_trades_trader_time
            ON trades(trader_id, executed_at DESC)
        """))

        # 部分索引（只索引活跃交易员）
        await self.session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_traders_running
            ON traders(user_id)
            WHERE is_running = true
        """))

        await self.session.commit()

    async def update_statistics(self):
        """更新表统计信息"""
        await self.session.execute(text("ANALYZE traders"))
        await self.session.execute(text("ANALYZE decisions"))
        await self.session.execute(text("ANALYZE positions"))
        await self.session.execute(text("ANALYZE trades"))
        await self.session.execute(text("ANALYZE account_snapshots"))
        await self.session.commit()

    async def vacuum(self):
        """清理和优化表"""
        await self.session.execute(text("VACUUM ANALYZE traders"))
        await self.session.execute(text("VACUUM ANALYZE decisions"))
        await self.session.execute(text("VACUUM ANALYZE positions"))
        await self.session.commit()


# ========== 查询优化示例 ==========

class OptimizedQueries:
    """优化后的查询"""

    @staticmethod
    async def get_trader_performance(
        trader_id: str,
        session: AsyncSession,
    ) -> dict[str, Any]:
        """获取交易员性能统计（优化版本）"""

        # 使用 CTE 和窗口函数一次查询获取所有数据
        query = text("""
            WITH trader_stats AS (
                SELECT
                    t.id as trader_id,
                    t.initial_balance,
                    t.total_equity,
                    t.total_pnl,
                    t.total_trades,
                    t.winning_trades,
                    -- 计算胜率
                    CASE
                        WHEN t.total_trades > 0
                        THEN CAST(t.winning_trades AS FLOAT) / t.total_trades
                        ELSE 0
                    END as win_rate,
                    -- 获取最近交易日期
                    (
                        SELECT MAX(executed_at)
                        FROM trades
                        WHERE trader_id = t.id
                    ) as last_trade_at
                FROM traders t
                WHERE t.id = :trader_id
            ),
            position_stats AS (
                SELECT
                    trader_id,
                    COUNT(*) as position_count,
                    SUM(unrealized_pnl) as total_unrealized_pnl,
                    SUM(CASE WHEN side = 'long' THEN 1 ELSE 0 END) as long_positions,
                    SUM(CASE WHEN side = 'short' THEN 1 ELSE 0 END) as short_positions
                FROM positions
                WHERE trader_id = :trader_id AND is_open = true
                GROUP BY trader_id
            ),
            equity_curve AS (
                SELECT
                    trader_id,
                    array_agg(total_equity ORDER BY created_at) as equity_values,
                    array_agg(created_at ORDER BY created_at) as timestamps
                FROM account_snapshots
                WHERE trader_id = :trader_id
                    AND created_at >= NOW() - INTERVAL '30 days'
                GROUP BY trader_id
            )
            SELECT
                ts.*,
                COALESCE(ps.position_count, 0) as position_count,
                COALESCE(ps.total_unrealized_pnl, 0) as total_unrealized_pnl,
                COALESCE(ps.long_positions, 0) as long_positions,
                COALESCE(ps.short_positions, 0) as short_positions,
                ec.equity_values,
                ec.timestamps
            FROM trader_stats ts
            LEFT JOIN position_stats ps ON ps.trader_id = ts.trader_id
            LEFT JOIN equity_curve ec ON ec.trader_id = ts.trader_id
        """)

        result = await session.execute(query, {"trader_id": trader_id})
        row = result.fetchone()

        if not row:
            return None

        return {
            "trader_id": row.trader_id,
            "initial_balance": float(row.initial_balance),
            "total_equity": float(row.total_equity),
            "total_pnl": float(row.total_pnl),
            "win_rate": float(row.win_rate),
            "position_count": row.position_count,
            "total_unrealized_pnl": float(row.total_unrealized_pnl),
            "equity_curve": {
                "values": row.equity_values,
                "timestamps": row.timestamps,
            },
        }


# ========== 缓存优化 ==========

class CacheOptimizer:
    """缓存优化器"""

    def __init__(self, redis_client):
        self.redis = redis_client

    async def cache_market_data(
        self,
        symbol: str,
        data: dict[str, Any],
        ttl: int = 5,
    ):
        """缓存市场数据"""
        key = f"market:{symbol}"
        await self.redis.setex(
            key,
            ttl,
            json.dumps(data),
        )

    async def get_cached_market_data(
        self,
        symbol: str,
    ) -> dict[str, Any] | None:
        """获取缓存的市场数据"""
        key = f"market:{symbol}"
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None

    async def cache_indicators(
        self,
        symbol: str,
        indicators: dict[str, Any],
        ttl: int = 60,
    ):
        """缓存技术指标"""
        key = f"indicators:{symbol}"
        await self.redis.setex(
            key,
            ttl,
            json.dumps(indicators),
        )

    async def invalidate_symbol_cache(self, symbol: str):
        """使股票相关缓存失效"""
        keys = [
            f"market:{symbol}",
            f"indicators:{symbol}",
            f"klines:{symbol}:*",
        ]

        for key in keys:
            await self.redis.delete(key)

    async def warm_up_cache(self, symbols: list[str]):
        """预热缓存"""
        from backend.services.market_service import MarketService

        market_service = MarketService()

        for symbol in symbols:
            try:
                data = await market_service.get_realtime_data(symbol)
                await self.cache_market_data(symbol, data)

                indicators = await market_service.get_technical_indicators(symbol)
                await self.cache_indicators(symbol, indicators)

            except Exception as e:
                logger.warning(f"Failed to warm up cache for {symbol}: {e}")
```

### 15.3 API 性能优化

```python
# backend/optimizations/api.py
from functools import lru_cache
from typing import Any, Callable
import asyncio
from fastapi import Request, Response
import gzip
import json
import hashlib


class ResponseCompressor:
    """响应压缩器"""

    async def __call__(self, request: Request, call_next):
        """压缩响应"""
        response = await call_next(request)

        # 检查是否应该压缩
        if not self._should_compress(request, response):
            return response

        # 压缩响应体
        body = response.body.decode("utf-8")
        compressed_body = gzip.compress(body.encode("utf-8"))

        # 更新响应头
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Content-Length"] = str(len(compressed_body))
        response.body = compressed_body

        return response

    def _should_compress(self, request: Request, response: Response) -> bool:
        """判断是否应该压缩"""
        # 只压缩 JSON 响应
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return False

        # 检查客户端是否支持 gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding:
            return False

        # 只压缩大于 1KB 的响应
        return len(response.body) > 1024


class ETagMiddleware:
    """ETag 中间件（支持缓存）"""

    async def __call__(self, request: Request, call_next):
        """处理 ETag"""
        response = await call_next(request)

        # 只为 GET 请求生成 ETag
        if request.method != "GET":
            return response

        # 生成 ETag
        body = response.body
        etag = self._generate_etag(body)

        response.headers["ETag"] = etag

        # 检查 If-None-Match
        if_none_match = request.headers.get("if-none-match")
        if if_none_match == etag:
            # 返回 304 Not Modified
            return Response(
                status_code=304,
                headers=response.headers,
            )

        return response

    def _generate_etag(self, body: bytes) -> str:
        """生成 ETag"""
        return hashlib.md5(body).hexdigest()


class QueryOptimizer:
    """查询优化器"""

    @staticmethod
    def optimize_pagination(
        query: str,
        page: int = 1,
        page_size: int = 20,
        max_page_size: int = 100,
    ) -> tuple[str, int, int]:
        """
        优化分页查询

        使用 keyset pagination 替代 OFFSET
        """
        # 限制 page_size
        page_size = min(page_size, max_page_size)

        # 使用 WHERE 替代 OFFSET（更高效）
        if page == 1:
            optimized_query = f"{query} ORDER BY created_at DESC LIMIT {page_size}"
        else:
            # 假设使用 ID 作为游标
            offset = (page - 1) * page_size
            optimized_query = f"{query} WHERE id > {offset} ORDER BY created_at DESC LIMIT {page_size}"

        return optimized_query, page, page_size


class BatchProcessor:
    """批处理器"""

    def __init__(self, batch_size: int = 10, timeout: float = 1.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = asyncio.Queue()
        self.task = None

    async def add(self, item: Any):
        """添加项目到批处理队列"""
        await self.queue.put(item)

    async def start(self, processor: Callable[[list[Any]], Any]):
        """启动批处理"""
        async def process_loop():
            while True:
                batch = []
                try:
                    # 等待第一个项目
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=self.timeout,
                    )
                    batch.append(item)

                    # 收集更多项目
                    while len(batch) < self.batch_size:
                        try:
                            item = await asyncio.wait_for(
                                self.queue.get(),
                                timeout=0.1,
                            )
                            batch.append(item)
                        except asyncio.TimeoutError:
                            break

                    # 处理批次
                    await processor(batch)

                except asyncio.TimeoutError:
                    # 超时后处理当前批次
                    if batch:
                        await processor(batch)

        self.task = asyncio.create_task(process_loop())

    async def stop(self):
        """停止批处理"""
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def flush(self, processor: Callable[[list[Any]], Any]):
        """刷新剩余项目"""
        batch = []
        while not self.queue.empty():
            item = await self.queue.get()
            batch.append(item)

        if batch:
            await processor(batch)
```

### 15.4 LLM 调用优化

```python
# backend/optimizations/llm.py
from functools import wraps
from typing import Any
import asyncio
from collections import defaultdict
import time

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class LLMRateLimiter:
    """LLM API 限流器"""

    def __init__(self, max_requests: int = 10, period: float = 60.0):
        self.max_requests = max_requests
        self.period = period
        self.requests = defaultdict(list)

    async def acquire(self, provider: str) -> bool:
        """获取许可"""
        now = time.time()
        requests = self.requests[provider]

        # 清理过期记录
        self.requests[provider] = [
            req_time for req_time in requests
            if now - req_time < self.period
        ]

        # 检查是否超过限制
        if len(self.requests[provider]) >= self.max_requests:
            wait_time = self.period - (now - requests[0])
            logger.warning(f"Rate limit reached for {provider}, wait {wait_time:.1f}s")
            await asyncio.sleep(wait_time)

        # 记录请求
        self.requests[provider].append(now)
        return True


class LLMPromptCache:
    """LLM Prompt 缓存"""

    def __init__(self, ttl: int = 3600):
        self.cache: dict[str, tuple[Any, float]] = {}
        self.ttl = ttl

    def _generate_key(self, prompt: str, model: str) -> str:
        """生成缓存键"""
        import hashlib
        content = f"{model}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Any | None:
        """获取缓存"""
        key = self._generate_key(prompt, model)

        if key in self.cache:
            value, timestamp = self.cache[key]

            # 检查是否过期
            if time.time() - timestamp < self.ttl:
                logger.debug(f"LLM cache hit: {model}")
                return value
            else:
                del self.cache[key]

        return None

    def set(self, prompt: str, model: str, value: Any):
        """设置缓存"""
        key = self._generate_key(prompt, model)
        self.cache[key] = (value, time.time())

    def invalidate(self, prompt: str | None = None, model: str | None = None):
        """使缓存失效"""
        if prompt is None and model is None:
            self.cache.clear()
        else:
            keys_to_delete = []
            for key in self.cache:
                if (prompt is None or prompt in key) and (model is None or model in key):
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del self.cache[key]


class LLMResponseOptimizer:
    """LLM 响应优化器"""

    @staticmethod
    def compress_prompt(prompt: str) -> str:
        """压缩 Prompt"""
        # 移除多余空格和换行
        import re
        prompt = re.sub(r"\n+", "\n", prompt)
        prompt = re.sub(r" +", " ", prompt)

        # 移除注释（如果不需要保留）
        prompt = re.sub(r"#.*$", "", prompt, flags=re.MULTILINE)

        return prompt.strip()

    @staticmethod
    def truncate_prompt(
        prompt: str,
        max_tokens: int,
        tokenizer: Any,
    ) -> str:
        """截断 Prompt 以适应 token 限制"""
        # 粗略估算：1 token ≈ 4 字符（中文）或 0.75 token ≈ 1 单词（英文）
        # 更准确的方法是使用 tokenizer 计算

        # 简化实现：按字符数估算
        max_chars = max_tokens * 3

        if len(prompt) <= max_chars:
            return prompt

        # 从后往前截断，保留系统指令
        lines = prompt.split("\n")
        result = []
        current_length = 0

        for line in reversed(lines):
            if current_length + len(line) > max_chars:
                break
            result.insert(0, line)
            current_length += len(line) + 1

        return "\n".join(result)

    @staticmethod
    def optimize_response_format(response: str) -> str:
        """优化响应格式"""
        # 移除多余内容
        if "```json" in response:
            # 提取 JSON 代码块
            import re
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return match.group(1)

        return response.strip()


def llm_cache(
    cache: LLMPromptCache,
    model: str,
):
    """LLM 缓存装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(prompt: str, **kwargs):
            # 尝试从缓存获取
            cached = cache.get(prompt, model)
            if cached is not None:
                return cached

            # 调用 LLM
            response = await func(prompt, **kwargs)

            # 缓存响应
            cache.set(prompt, model, response)

            return response

        return wrapper
    return decorator


def llm_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
):
    """LLM 重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import random

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise

                    # 指数退避
                    delay = min(
                        base_delay * (2 ** attempt) + random.uniform(0, 1),
                        max_delay,
                    )

                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )

                    await asyncio.sleep(delay)

            return None

        return wrapper
    return decorator
```

---

## 十六、安全设计详细方案

### 16.1 安全架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          安全防护层次                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       应用层安全                                 │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  • 认证 (JWT)  • 授权 (RBAC)  • 会话管理  • CSRF 防护            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                     │
│                                  ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       数据层安全                                 │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  • 加密存储  • 数据脱敏  • 备份加密  • 审计日志                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                     │
│                                  ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       网络层安全                                 │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  • HTTPS/TLS  • API 限流  • IP 白名单  • WAF 规则                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                     │
│                                  ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       基础设施安全                               │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  • 容器安全  • 密钥管理  • 访问控制  • 漏洞扫描                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 16.2 认证授权系统

```python
# backend/security/auth.py
from datetime import datetime, timedelta
from typing import Any
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt

from backend.config import get_config
from backend.exceptions import AuthenticationError, AuthorizationError
from backend.models.database import User
from backend.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()


# 密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class PasswordManager:
    """密码管理器"""

    @staticmethod
    def hash_password(password: str) -> str:
        """哈希密码"""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, list[str]]:
        """
        验证密码强度

        Returns:
            (是否有效, 错误消息列表)
        """
        errors = []

        if len(password) < config.security.password_min_length:
            errors.append(
                f"密码长度至少 {config.security.password_min_length} 个字符"
            )

        if not any(c.isupper() for c in password):
            errors.append("密码必须包含至少一个大写字母")

        if not any(c.islower() for c in password):
            errors.append("密码必须包含至少一个小写字母")

        if not any(c.isdigit() for c in password):
            errors.append("密码必须包含至少一个数字")

        # 检查常见弱密码
        common_passwords = [
            "password123",
            "12345678",
            "qwerty123",
            "admin123",
        ]
        if password.lower() in common_passwords:
            errors.append("密码过于常见，请使用更强的密码")

        return len(errors) == 0, errors


class TokenManager:
    """Token 管理器"""

    def __init__(self):
        self.secret_key = config.security.jwt_secret
        self.algorithm = config.security.jwt_algorithm
        self.access_token_expire = timedelta(seconds=config.security.jwt_expiration)
        self.refresh_token_expire = timedelta(days=30)

    def create_access_token(
        self,
        user_id: str,
        extra_claims: dict[str, Any] | None = None,
    ) -> str:
        """创建访问令牌"""
        claims = {
            "sub": user_id,
            "type": "access",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.access_token_expire,
            **(extra_claims or {}),
        }

        return jwt.encode(claims, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(
        self,
        user_id: str,
    ) -> str:
        """创建刷新令牌"""
        claims = {
            "sub": user_id,
            "type": "refresh",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.refresh_token_expire,
        }

        return jwt.encode(claims, self.secret_key, algorithm=self.algorithm)

    def decode_token(self, token: str) -> dict[str, Any]:
        """解码令牌"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token 已过期")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"无效的 Token: {str(e)}")

    def verify_access_token(self, token: str) -> dict[str, Any]:
        """验证访问令牌"""
        payload = self.decode_token(token)

        if payload.get("type") != "access":
            raise AuthenticationError("Token 类型错误")

        return payload

    def verify_refresh_token(self, token: str) -> dict[str, Any]:
        """验证刷新令牌"""
        payload = self.decode_token(token)

        if payload.get("type") != "refresh":
            raise AuthenticationError("Token 类型错误")

        return payload


class PermissionChecker:
    """权限检查器"""

    def __init__(self):
        # 定义权限
        self.permissions = {
            # 交易员权限
            "trader:create": ["user"],
            "trader:read": ["user"],
            "trader:update": ["user"],
            "trader:delete": ["user"],
            "trader:control": ["user"],

            # 决策权限
            "decision:read": ["user"],
            "decision:execute": ["user"],

            # 市场数据权限
            "market:read": ["user"],

            # 管理员权限
            "admin:users": ["admin"],
            "admin:system": ["admin"],
            "admin:logs": ["admin"],
        }

    def check_permission(
        self,
        user: User,
        permission: str,
    ) -> bool:
        """检查用户权限"""
        # 超级用户拥有所有权限
        if user.is_superuser:
            return True

        # 获取允许的角色
        allowed_roles = self.permissions.get(permission, [])

        # 检查用户角色
        user_roles = []
        if user.is_active:
            user_roles.append("user")
        if user.is_superuser:
            user_roles.append("admin")

        return any(role in allowed_roles for role in user_roles)

    def require_permission(self, permission: str):
        """权限检查装饰器"""
        from fastapi import Depends, HTTPException, status
        from functools import wraps

        def decorator(func):
            @wraps(func)
            async def wrapper(
                current_user: User = Depends(get_current_user),
                *args,
                **kwargs,
            ):
                if not self.check_permission(current_user, permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"权限不足: {permission}",
                    )

                return await func(current_user, *args, **kwargs)

            return wrapper

        return decorator


# ========== 会话管理 ==========

class SessionManager:
    """会话管理器"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.session_ttl = 3600  # 1小时

    async def create_session(
        self,
        user_id: str,
        token: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """创建会话"""
        import uuid

        session_id = str(uuid.uuid4())

        session_data = {
            "user_id": user_id,
            "token": token,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        # 存储会话
        await self.redis.setex(
            f"session:{session_id}",
            self.session_ttl,
            json.dumps(session_data),
        )

        # 存储用户到会话的映射（用于查询用户的所有会话）
        await self.redis.sadd(f"user_sessions:{user_id}", session_id)

        return session_id

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """获取会话"""
        data = await self.redis.get(f"session:{session_id}")
        if data:
            return json.loads(data)
        return None

    async def update_activity(self, session_id: str):
        """更新会话活动时间"""
        session = await self.get_session(session_id)
        if session:
            session["last_activity"] = datetime.utcnow().isoformat()
            await self.redis.setex(
                f"session:{session_id}",
                self.session_ttl,
                json.dumps(session),
            )

    async def revoke_session(self, session_id: str):
        """撤销会话"""
        session = await self.get_session(session_id)
        if session:
            user_id = session.get("user_id")

            # 删除会话
            await self.redis.delete(f"session:{session_id}")

            # 从用户会话集合中移除
            if user_id:
                await self.redis.srem(f"user_sessions:{user_id}", session_id)

    async def revoke_user_sessions(self, user_id: str):
        """撤销用户的所有会话"""
        session_ids = await self.redis.smembers(f"user_sessions:{user_id}")

        for session_id in session_ids:
            await self.redis.delete(f"session:{session_id}")

        await self.redis.delete(f"user_sessions:{user_id}")

    async def cleanup_expired_sessions(self):
        """清理过期会话"""
        # Redis 的 TTL 会自动清理过期键
        # 这里可以添加额外的清理逻辑
        pass
```

### 16.3 数据加密

```python
# backend/security/encryption.py
from base64 import urlsafe_b64encode, urlsafe_b64decode
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class EncryptionService:
    """加密服务"""

    def __init__(self, master_key: str | None = None):
        """
        初始化加密服务

        Args:
            master_key: 主密钥（从环境变量或配置读取）
        """
        if not master_key:
            master_key = os.environ.get("ENCRYPTION_KEY")

        if not master_key:
            # 生成新密钥（仅用于开发环境）
            logger.warning("No encryption key provided, generating temporary key")
            master_key = self._generate_key()

        # 派生加密密钥
        self.encryption_key = self._derive_key(master_key)
        self.cipher = Fernet(self.encryption_key)

    def _generate_key(self) -> str:
        """生成随机密钥"""
        return urlsafe_b64encode(os.urandom(32)).decode()

    def _derive_key(self, master_key: str) -> bytes:
        """从主密钥派生加密密钥"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"deepalpha_encryption",
            iterations=100000,
            backend=default_backend(),
        )
        return urlsafe_b64encode(kdf.derive(master_key.encode()))

    def encrypt(self, plaintext: str) -> str:
        """加密数据"""
        try:
            encrypted = self.cipher.encrypt(plaintext.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt(self, ciphertext: str) -> str:
        """解密数据"""
        try:
            decrypted = self.cipher.decrypt(ciphertext.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def encrypt_dict(self, data: dict[str, Any]) -> dict[str, str]:
        """加密字典中的敏感字段"""
        encrypted = {}
        sensitive_fields = [
            "password",
            "api_key",
            "secret_key",
            "access_token",
            "private_key",
        ]

        for key, value in data.items():
            if key in sensitive_fields:
                encrypted[key] = self.encrypt(str(value))
            else:
                encrypted[key] = str(value)

        return encrypted

    def decrypt_dict(self, data: dict[str, str]) -> dict[str, Any]:
        """解密字典中的敏感字段"""
        decrypted = {}
        sensitive_fields = [
            "password",
            "api_key",
            "secret_key",
            "access_token",
            "private_key",
        ]

        for key, value in data.items():
            if key in sensitive_fields:
                try:
                    decrypted[key] = self.decrypt(value)
                except Exception as e:
                    logger.warning(f"Failed to decrypt {key}: {e}")
                    decrypted[key] = value
            else:
                decrypted[key] = value

        return decrypted


class FieldEncryption:
    """字段级加密（数据库层面）"""

    @staticmethod
    def encrypt_field(value: str, key: bytes) -> str:
        """加密字段"""
        cipher = Fernet(key)
        encrypted = cipher.encrypt(value.encode())
        return encrypted.decode()

    @staticmethod
    def decrypt_field(encrypted_value: str, key: bytes) -> str:
        """解密字段"""
        cipher = Fernet(key)
        decrypted = cipher.decrypt(encrypted_value.encode())
        return decrypted.decode()


# ========== 数据脱敏 ==========

class DataMasker:
    """数据脱敏器"""

    @staticmethod
    def mask_email(email: str) -> str:
        """脱敏邮箱"""
        if "@" not in email:
            return email

        username, domain = email.split("@", 1)
        username_masked = username[0] + "*" * (len(username) - 2) + username[-1] if len(username) > 2 else "*"

        return f"{username_masked}@{domain}"

    @staticmethod
    def mask_phone(phone: str) -> str:
        """脱敏手机号"""
        if len(phone) < 7:
            return phone

        return phone[:3] + "****" + phone[-4:]

    @staticmethod
    def mask_card(card: str) -> str:
        """脱敏银行卡号"""
        if len(card) < 8:
            return card

        return "*" * (len(card) - 4) + card[-4:]

    @staticmethod
    def mask_api_key(api_key: str) -> str:
        """脱敏 API Key"""
        if len(api_key) < 8:
            return "***"

        return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]


# ========== 安全中间件 ==========

from fastapi import Request

async def security_middleware(request: Request, call_next):
    """安全中间件"""

    # 添加安全响应头
    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"

    # 移除服务器信息
    response.headers["Server"] = "DeepAlpha"

    return response


# ========== 输入验证 ==========

from pydantic import validator, BaseModel
import re


class SecureInput(BaseModel):
    """安全输入模型"""

    @validator("*", pre=True)
    def sanitize_string(cls, v):
        """清理字符串输入"""
        if isinstance(v, str):
            # 移除控制字符
            v = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", v)
            # 移除危险字符
            v = v.replace("\x00", "")
        return v


class SQLInjectionChecker:
    """SQL 注入检查器"""

    DANGEROUS_PATTERNS = [
        r"(\bunion\b.*\bselect\b)",
        r"(\bselect\b.*\bfrom\b.*\bwhere\b.*\bor\b)",
        r"(\bdrop\b.*\btable\b)",
        r"(\binsert\b.*\binto\b)",
        r"(\bupdate\b.*\bset\b)",
        r"(\bdelete\b.*\bfrom\b)",
        r"(--|;|/\*|\*/|xp_|sp_)",
    ]

    @classmethod
    def check(cls, input_string: str) -> bool:
        """检查是否包含 SQL 注入模式"""
        input_lower = input_string.lower()

        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, input_lower, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {input_string[:100]}")
                return True

        return False


class XSSChecker:
    """XSS 检查器"""

    DANGEROUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<style[^>]*>.*?</style>",
        r"<img[^>]*onerror[^>]*>",
    ]

    @classmethod
    def check(cls, input_string: str) -> bool:
        """检查是否包含 XSS 模式"""
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, input_string, re.IGNORECASE | re.DOTALL):
                logger.warning(f"Potential XSS detected: {input_string[:100]}")
                return True

        return False
```

---

## 十七、完整运行示例

### 17.1 完整的交易流程示例

```python
# examples/complete_trading_flow.py
"""
完整的交易流程示例

这个示例展示了从初始化到执行交易的完整流程
"""
import asyncio
from datetime import datetime

from backend.config import get_config
from backend.models.database import async_session_maker
from backend.services.trader_service import TraderService
from backend.services.decision_service import DecisionService
from backend.services.market_service import MarketService
from backend.tasks.decision_task import run_decision_cycle
from backend.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()


async def main():
    """主流程"""

    # 1. 初始化数据库会话
    async with async_session_maker() as db:
        # 2. 创建服务实例
        trader_service = TraderService(db)
        decision_service = DecisionService(db)
        market_service = MarketService()

        # 3. 创建 AI 模型配置
        ai_model = await create_ai_model(db)

        # 4. 创建交易员
        trader = await create_trader(db, trader_service, ai_model.id)

        # 5. 设置股票池
        await setup_stock_pool(db, trader.id)

        # 6. 启动交易员
        logger.info("启动交易员...")
        await trader_service.start_trader(trader.id)

        # 7. 运行决策周期
        logger.info("运行决策周期...")
        await run_decision_cycle(trader.id)

        # 8. 查看结果
        await display_results(db, trader.id)

        # 9. 停止交易员
        logger.info("停止交易员...")
        await trader_service.stop_trader(trader.id)


async def create_ai_model(db):
    """创建 AI 模型配置"""
    from backend.models.database import AIModel
    from backend.utils.crypto import encrypt_api_key

    # 加密 API Key
    encrypted_key = encrypt_api_key(config.llm.deepseek_api_key)

    ai_model = AIModel(
        user_id="default",  # 默认用户
        name="deepseek-main",
        provider="deepseek",
        model_name="deepseek-chat",
        api_key_encrypted=encrypted_key,
        base_url=config.llm.deepseek_base_url,
        max_tokens=config.llm.deepseek_max_tokens,
        temperature=config.llm.deepseek_temperature,
        is_active=True,
    )

    db.add(ai_model)
    await db.commit()
    await db.refresh(ai_model)

    logger.info(f"创建 AI 模型: {ai_model.name}")
    return ai_model


async def create_trader(db, trader_service, ai_model_id):
    """创建交易员"""
    trader_data = {
        "user_id": "default",
        "name": "A股AI交易员1号",
        "description": "专注A股主板市场的AI交易员",
        "ai_model_id": ai_model_id,
        "market_type": "a_stock",
        "initial_balance": 1_000_000,
    }

    trader = await trader_service.create_trader(**trader_data)

    logger.info(f"创建交易员: {trader.name} (ID: {trader.id})")
    logger.info(f"  初始资金: ¥{trader.initial_balance:,.2f}")
    logger.info(f"  市场类型: {trader.market_type}")

    return trader


async def setup_stock_pool(db, trader_id):
    """设置股票池"""
    from backend.models.database import StockPool

    # A股主板蓝筹股
    symbols = {
        "symbols": [
            {"code": "600519.SH", "name": "贵州茅台", "weight": 0.15},
            {"code": "000858.SZ", "name": "五粮液", "weight": 0.10},
            {"code": "600036.SH", "name": "招商银行", "weight": 0.10},
            {"code": "000001.SZ", "name": "平安银行", "weight": 0.08},
            {"code": "601318.SH", "name": "中国平安", "weight": 0.12},
            {"code": "000333.SZ", "name": "美的集团", "weight": 0.10},
            {"code": "600276.SH", "name": "恒瑞医药", "weight": 0.08},
            {"code": "300750.SZ", "name": "宁德时代", "weight": 0.10},
            {"code": "601012.SH", "name": "隆基绿能", "weight": 0.08},
            {"code": "002594.SZ", "name": "比亚迪", "weight": 0.09},
        ],
        "created_at": datetime.now().isoformat(),
    }

    stock_pool = StockPool(
        user_id="default",
        name="A股蓝筹股",
        description="A股主板蓝筹股股票池",
        market_type="a_stock",
        symbols=symbols,
        is_default=True,
    )

    db.add(stock_pool)
    await db.commit()
    await db.refresh(stock_pool)

    # 关联到交易员
    trader = await db.get(Trader, trader_id)
    trader.stock_pool_id = stock_pool.id
    await db.commit()

    logger.info(f"设置股票池: {stock_pool.name} ({len(symbols['symbols'])} 只股票)")


async def display_results(db, trader_id):
    """显示结果"""
    from sqlalchemy import select, func
    from backend.models.database import Trader, Position, Decision

    # 获取交易员信息
    trader = await db.get(Trader, trader_id)

    print("\n" + "=" * 60)
    print("交易员运行结果")
    print("=" * 60)

    # 基本信息
    print(f"\n交易员: {trader.name}")
    print(f"初始资金: ¥{trader.initial_balance:,.2f}")
    print(f"当前权益: ¥{trader.total_equity:,.2f}")
    print(f"可用余额: ¥{trader.current_balance:,.2f}")
    print(f"总盈亏: ¥{trader.total_pnl:,.2f} ({trader.total_pnl_pct:.2%})")

    # 持仓信息
    positions_result = await db.execute(
        select(Position).where(
            Position.trader_id == trader_id,
            Position.is_open == True,
        )
    )
    positions = positions_result.scalars().all()

    if positions:
        print(f"\n当前持仓 ({len(positions)} 个):")
        print("-" * 60)
        for pos in positions:
            print(f"  {pos.symbol:12} {pos.side:4} {pos.quantity:6}股 "
                  f"@ ¥{pos.entry_price:8.2f} | "
                  f"浮盈: ¥{pos.unrealized_pnl:10.2f} ({pos.unrealized_pnl_pct:6.2%})")
    else:
        print("\n当前无持仓")

    # 决策记录
    decisions_result = await db.execute(
        select(Decision)
        .where(Decision.trader_id == trader_id)
        .order_by(Decision.created_at.desc())
        .limit(5)
    )
    decisions = decisions_result.scalars().all()

    if decisions:
        print(f"\n最近决策 ({len(decisions)} 条):")
        print("-" * 60)
        for decision in decisions:
            print(f"  [{decision.created_at.strftime('%H:%M:%S')}] "
                  f"{decision.status:10} | "
                  f"{len(decision.decisions)} 个决策")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
```

### 17.2 前端完整组件示例

```typescript
// frontend/src/components/traders/TraderCard.tsx
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useTraderStore } from '@/stores/traderStore';
import { useWebSocket } from '@/hooks/useWebSocket';
import type { Trader } from '@/types/models';

interface TraderCardProps {
  trader: Trader;
  onControl: (id: string, action: string) => void;
}

export function TraderCard({ trader, onControl }: TraderCardProps) {
  const [isUpdating, setIsUpdating] = useState(false);
  const { updateTrader } = useTraderStore();

  // WebSocket 订阅实时更新
  useWebSocket(`/ws/traders/${trader.id}`, (data) => {
    updateTrader(data.trader);
  });

  const handleControl = async (action: string) => {
    setIsUpdating(true);
    try {
      await onControl(trader.id, action);
    } finally {
      setIsUpdating(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-500';
      case 'paused': return 'bg-yellow-500';
      case 'stopped': return 'bg-gray-500';
      default: return 'bg-gray-500';
    }
  };

  const getPnLColor = (pnl: number) => {
    return pnl >= 0 ? 'text-green-600' : 'text-red-600';
  };

  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{trader.name}</CardTitle>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${getStatusColor(trader.is_running ? 'running' : 'stopped')}`} />
            <Badge variant={trader.is_running ? 'default' : 'secondary'}>
              {trader.is_running ? '运行中' : '已停止'}
            </Badge>
          </div>
        </div>
        {trader.description && (
          <p className="text-sm text-muted-foreground mt-1">
            {trader.description}
          </p>
        )}
      </CardHeader>

      <CardContent className="space-y-4">
        {/* 账户信息 */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-muted-foreground">总权益</span>
            <p className="text-lg font-semibold">
              ¥{trader.total_equity?.toLocaleString() || '0'}
            </p>
          </div>
          <div>
            <span className="text-muted-foreground">可用余额</span>
            <p className="text-lg font-semibold">
              ¥{trader.current_balance?.toLocaleString() || '0'}
            </p>
          </div>
          <div>
            <span className="text-muted-foreground">总盈亏</span>
            <p className={`text-lg font-semibold ${getPnLColor(trader.total_pnl || 0)}`}>
              ¥{trader.total_pnl?.toLocaleString() || '0'}
              <span className="text-sm ml-1">
                ({((trader.total_pnl_pct || 0) * 100).toFixed(2)}%)
              </span>
            </p>
          </div>
          <div>
            <span className="text-muted-foreground">胜率</span>
            <p className="text-lg font-semibold">
              {((trader.win_rate || 0) * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* 性能指标 */}
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="text-center p-2 bg-muted rounded">
            <p className="text-muted-foreground">交易次数</p>
            <p className="font-semibold">{trader.total_trades || 0}</p>
          </div>
          <div className="text-center p-2 bg-muted rounded">
            <p className="text-muted-foreground">盈利次数</p>
            <p className="font-semibold text-green-600">{trader.winning_trades || 0}</p>
          </div>
          <div className="text-center p-2 bg-muted rounded">
            <p className="text-muted-foreground">最大回撤</p>
            <p className="font-semibold text-red-600">
              {((trader.max_drawdown || 0) * 100).toFixed(2)}%
            </p>
          </div>
        </div>

        {/* 控制按钮 */}
        <div className="flex gap-2">
          {trader.is_running ? (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleControl('pause')}
                disabled={isUpdating}
                className="flex-1"
              >
                暂停
              </Button>
              <Button
                variant="destructive"
                size="sm"
                onClick={() => handleControl('stop')}
                disabled={isUpdating}
                className="flex-1"
              >
                停止
              </Button>
            </>
          ) : (
            <Button
              variant="default"
              size="sm"
              onClick={() => handleControl('start')}
              disabled={isUpdating}
              className="w-full"
            >
              {trader.is_paused ? '继续' : '启动'}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
```

### 17.3 权益曲线图表组件

```typescript
// frontend/src/components/charts/EquityCurve.tsx
import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import { format } from 'date-fns';
import type { AccountSnapshot } from '@/types/models';

interface EquityCurveProps {
  snapshots: AccountSnapshot[];
  height?: number;
}

export function EquityCurve({ snapshots, height = 300 }: EquityCurveProps) {
  // 准备图表数据
  const chartData = useMemo(() => {
    return snapshots
      .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
      .map((snapshot) => ({
        timestamp: new Date(snapshot.created_at).getTime(),
        date: format(new Date(snapshot.created_at), 'MM-dd HH:mm'),
        equity: snapshot.total_equity,
        balance: snapshot.available_balance,
        pnl: snapshot.total_pnl,
      }));
  }, [snapshots]);

  // 计算初始权益用于计算百分比
  const initialEquity = chartData[0]?.equity || 1000000;

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="date"
            stroke="#6b7280"
            fontSize={12}
            tickLine={false}
          />
          <YAxis
            stroke="#6b7280"
            fontSize={12}
            tickLine={false}
            tickFormatter={(value) => `¥${(value / 10000).toFixed(1)}万`}
          />

          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
            }}
            labelStyle={{ color: '#374151' }}
            formatter={(value: number, name: string) => {
              if (name === '权益') return [`¥${value.toLocaleString()}`, name];
              if (name === '盈亏%') return [`${((value / initialEquity) * 100).toFixed(2)}%`, name];
              return [value, name];
            }}
          />

          <Legend />
          <Area
            type="monotone"
            dataKey="equity"
            name="权益"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#colorEquity)"
          />
          <Line
            type="monotone"
            dataKey="balance"
            name="可用余额"
            stroke="#10b981"
            strokeWidth={2}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
```

---

**文档版本**: v3.0
**创建日期**: 2026-01-05
**作者**: DeepAlpha Team

> 本文档是 NOFX Python 重构技术方案的第三个补充部分，包含第十四章至第十六章以及额外的实现细节和完整示例。
