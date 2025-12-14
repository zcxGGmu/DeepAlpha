"""配置模型，对应 Go 版本的配置结构"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class AppConfig(BaseModel):
    """应用配置"""
    # 服务器配置
    http_addr: str = Field(default=":9991", description="HTTP服务地址")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="日志级别")

    # 数据目录
    data_root: str = Field(default="/data", description="数据根目录")
    log_dir: str = Field(default="/data/logs", description="日志目录")
    db_path: str = Field(default="/data/db/deepalpha.db", description="数据库路径")

    # 其他配置
    debug: bool = Field(default=False, description="调试模式")
    timezone: str = Field(default="UTC", description="时区")

    class Config:
        env_prefix = "DEEPALPHA_"


class AIModelConfig(BaseModel):
    """AI模型配置"""
    name: str = Field(description="模型名称")
    provider: str = Field(description="提供商")
    model: str = Field(description="模型标识")
    api_key: Optional[str] = Field(None, description="API密钥")
    api_base: Optional[str] = Field(None, description="API基础URL")
    max_tokens: int = Field(default=4096, description="最大令牌数")
    temperature: float = Field(default=0.7, description="温度参数")
    timeout: int = Field(default=30, description="超时时间(秒)")


class AIConfig(BaseModel):
    """AI配置"""
    # 决策间隔
    decision_interval: str = Field(default="5m", description="决策间隔")
    analysis_interval: str = Field(default="1m", description="分析间隔")

    # 模型配置
    models: List[AIModelConfig] = Field(default_factory=list, description="AI模型列表")

    # 决策参数
    confidence_threshold: int = Field(default=70, description="置信度阈值")
    max_decisions_per_hour: int = Field(default=20, description="每小时最大决策数")

    # LLM配置
    llm_retry_count: int = Field(default=3, description="LLM重试次数")
    llm_retry_delay: int = Field(default=1, description="LLM重试延迟(秒)")


class ExchangeConfig(BaseModel):
    """交易所配置"""
    name: str = Field(description="交易所名称")
    api_key: Optional[str] = Field(None, description="API密钥")
    api_secret: Optional[str] = Field(None, description="API密钥")
    sandbox: bool = Field(default=False, description="是否为沙盒环境")
    rate_limit: int = Field(default=1200, description="速率限制(请求/分钟)")


class MarketConfig(BaseModel):
    """市场配置"""
    # 数据源
    exchanges: List[ExchangeConfig] = Field(default_factory=list, description="交易所列表")
    symbols: List[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"], description="交易对列表")

    # 代理配置
    proxy_url: Optional[str] = Field(None, description="代理URL")
    proxy_auth: Optional[str] = Field(None, description="代理认证")

    # 数据配置
    data_retention_days: int = Field(default=30, description="数据保留天数")
    candle_intervals: List[str] = Field(
        default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"],
        description="K线周期"
    )


class FreqtradeConfig(BaseModel):
    """Freqtrade配置"""
    # API配置
    api_url: str = Field(default="http://localhost:8080/api/v1", description="Freqtrade API地址")
    api_key: Optional[str] = Field(None, description="API密钥")
    username: Optional[str] = Field(None, description="用户名")
    password: Optional[str] = Field(None, description="密码")

    # Webhook配置
    webhook_url: str = Field(default="http://localhost:9991/api/live/freqtrade/webhook", description="Webhook地址")
    webhook_secret: Optional[str] = Field(None, description="Webhook密钥")

    # 交易配置
    enabled: bool = Field(default=True, description="是否启用")
    dry_run: bool = Field(default=False, description="是否为模拟模式")
    stake_currency: str = Field(default="USDT", description="基础货币")
    stake_amount: float = Field(default=100, description="默认投资金额")

    # 交易对
    trading_pairs: List[str] = Field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        description="交易对列表"
    )


class RiskConfig(BaseModel):
    """风险配置"""
    # 仓位限制
    max_position_size: float = Field(default=0.02, description="最大仓位比例")
    max_total_position: float = Field(default=0.5, description="总仓位上限")

    # 止损止盈
    stop_loss: float = Field(default=0.02, description="止损比例")
    take_profit: float = Field(default=0.06, description="止盈比例")
    min_profit_ratio: float = Field(default=1.5, description="最小盈亏比")

    # 风险限制
    max_drawdown: float = Field(default=0.10, description="最大回撤")
    max_trades_per_day: int = Field(default=10, description="每日最大交易数")
    max_leverage: int = Field(default=3, description="最大杠杆倍数")


class TradingConfig(BaseModel):
    """交易配置"""
    # 基础配置
    enabled: bool = Field(default=True, description="是否启用交易")
    horizon: str = Field(default="24h", description="持仓时间范围")

    # 风险管理
    risk: RiskConfig = Field(default_factory=RiskConfig, description="风险配置")

    # 策略参数
    min_confidence: int = Field(default=70, description="最小置信度")
    position_sizing_method: str = Field(default="fixed", description="仓位计算方法")
    default_leverage: int = Field(default=1, description="默认杠杆")

    # 交易时间
    trading_hours: Dict[str, Any] = Field(
        default_factory=lambda: {
            "monday": {"start": "00:00", "end": "23:59"},
            "tuesday": {"start": "00:00", "end": "23:59"},
            "wednesday": {"start": "00:00", "end": "23:59"},
            "thursday": {"start": "00:00", "end": "23:59"},
            "friday": {"start": "00:00", "end": "23:59"},
            "saturday": {"start": "00:00", "end": "23:59"},
            "sunday": {"start": "00:00", "end": "23:59"}
        },
        description="交易时间"
    )


class TelegramConfig(BaseModel):
    """Telegram配置"""
    enabled: bool = Field(default=False, description="是否启用")
    bot_token: Optional[str] = Field(None, description="Bot Token")
    chat_id: Optional[str] = Field(None, description="Chat ID")
    parse_mode: str = Field(default="HTML", description="消息格式")

    # 通知类型
    notify_on_entry: bool = Field(default=True, description="开仓通知")
    notify_on_exit: bool = Field(default=True, description="平仓通知")
    notify_on_error: bool = Field(default=True, description="错误通知")
    notify_on_decision: bool = Field(default=False, description="决策通知")


class NotificationConfig(BaseModel):
    """通知配置"""
    # Telegram
    telegram: TelegramConfig = Field(default_factory=TelegramConfig, description="Telegram配置")

    # 邮件
    email_enabled: bool = Field(default=False, description="是否启用邮件")
    email_smtp_host: Optional[str] = Field(None, description="SMTP主机")
    email_smtp_port: int = Field(default=587, description="SMTP端口")
    email_username: Optional[str] = Field(None, description="邮箱用户名")
    email_password: Optional[str] = Field(None, description="邮箱密码")
    email_from: Optional[str] = Field(None, description="发件人")
    email_to: List[str] = Field(default_factory=list, description="收件人列表")

    # 其他
    webhook_enabled: bool = Field(default=False, description="是否启用Webhook")
    webhook_url: Optional[str] = Field(None, description="Webhook地址")


class DatabaseConfig(BaseModel):
    """数据库配置"""
    type: str = Field(default="sqlite", description="数据库类型")
    url: Optional[str] = Field(None, description="数据库URL")

    # SQLite配置
    sqlite_path: str = Field(default="/data/db/deepalpha.db", description="SQLite文件路径")

    # 连接池配置
    pool_size: int = Field(default=10, description="连接池大小")
    max_overflow: int = Field(default=20, description="最大溢出连接数")
    pool_timeout: int = Field(default=30, description="连接超时时间")

    # 其他配置
    echo: bool = Field(default=False, description="是否打印SQL")
    pool_pre_ping: bool = Field(default=True, description="连接前ping")


class MonitoringConfig(BaseModel):
    """监控配置"""
    # 指标
    metrics_enabled: bool = Field(default=True, description="是否启用指标")
    metrics_port: int = Field(default=9090, description="指标端口")

    # 健康检查
    health_check_interval: int = Field(default=30, description="健康检查间隔(秒)")

    # 性能监控
    performance_monitoring: bool = Field(default=True, description="性能监控")
    slow_query_threshold: float = Field(default=1.0, description="慢查询阈值(秒)")

    # 告警
    alert_enabled: bool = Field(default=True, description="是否启用告警")
    alert_webhook: Optional[str] = Field(None, description="告警Webhook")


class DeepAlphaConfig(BaseModel):
    """DeepAlpha完整配置"""
    app: AppConfig = Field(default_factory=AppConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    freqtrade: FreqtradeConfig = Field(default_factory=FreqtradeConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    notification: NotificationConfig = Field(default_factory=NotificationConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # 兼容性字段
    version: str = Field(default="1.0.0", description="配置版本")

    class Config:
        env_nested_delimiter = "__"
        case_sensitive = False

    @validator('app')
    def validate_app_dirs(cls, v):
        """确保目录存在"""
        import os
        os.makedirs(v.data_root, exist_ok=True)
        os.makedirs(v.log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(v.db_path), exist_ok=True)
        return v