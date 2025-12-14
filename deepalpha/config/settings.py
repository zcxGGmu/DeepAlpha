"""配置管理系统"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class AppSettings(BaseModel):
    """应用配置"""
    name: str = "DeepAlpha"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class DatabaseSettings(BaseModel):
    """数据库配置"""
    url: str = "sqlite:///./deepalpha.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

    @validator("url")
    def validate_url(cls, v):
        if not v:
            raise ValueError("Database URL cannot be empty")
        return v


class RedisSettings(BaseModel):
    """Redis配置"""
    url: str = "redis://localhost:6379/0"
    max_connections: int = 10

    @validator("url")
    def validate_url(cls, v):
        if not v:
            raise ValueError("Redis URL cannot be empty")
        return v


class ExchangeSettings(BaseModel):
    """交易所配置"""
    name: str = "binance"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox: bool = True
    test_mode: bool = True

    class Config:
        validate_assignment = True


class LLMProviderSettings(BaseModel):
    """LLM Provider配置"""
    name: str
    api_key: Optional[str] = None
    model: str
    weight: float = Field(default=1.0, ge=0, le=1)
    max_tokens: int = 4000
    temperature: float = Field(default=0.7, ge=0, le=2)


class AISettings(BaseModel):
    """AI配置"""
    providers: List[LLMProviderSettings] = []
    min_confidence: float = Field(default=0.7, ge=0, le=1)
    cache_duration: int = Field(default=300, ge=0)  # 秒
    memory_size: int = Field(default=100, ge=0)

    @validator("providers")
    def validate_providers(cls, v):
        if not v:
            raise ValueError("At least one LLM provider must be configured")

        total_weight = sum(p.weight for p in v)
        if total_weight <= 0:
            raise ValueError("Sum of provider weights must be greater than 0")

        return v


class TradingSettings(BaseModel):
    """交易配置"""
    symbols: List[str] = ["BTC/USDT", "ETH/USDT"]
    timeframes: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]

    # 风险管理
    max_position_size: float = Field(default=0.02, gt=0, le=1)  # 每笔交易最大仓位
    max_total_position: float = Field(default=0.5, gt=0, le=1)  # 总仓位上限
    stop_loss: float = Field(default=0.02, gt=0)  # 止损百分比
    take_profit: float = Field(default=0.06, gt=0)  # 止盈百分比
    trailing_stop: bool = True  # 是否使用追踪止损

    # 交易限制
    max_trades_per_day: int = Field(default=10, ge=0)
    max_drawdown: float = Field(default=0.10, gt=0)  # 最大回撤
    min_profit_ratio: float = Field(default=1.5, gt=0)  # 最小盈亏比

    # 决策间隔
    decision_interval: int = Field(default=60, ge=1)  # 秒
    horizon_hours: int = Field(default=24, ge=1)  # 持仓周期（小时）


class NotificationSettings(BaseModel):
    """通知配置"""
    telegram: Optional[Dict[str, Any]] = None
    email: Optional[Dict[str, Any]] = None
    webhook: Optional[Dict[str, Any]] = None


class APISettings(BaseModel):
    """API配置"""
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    cors_origins: List[str] = ["*"]
    rate_limit: str = "100/minute"

    # 安全配置
    secret_key: str
    access_token_expire_minutes: int = 30

    @validator("cors_origins")
    def validate_cors_origins(cls, v):
        if "*" in v and len(v) > 1:
            raise ValueError("Cannot specify specific origins when using wildcard '*'")
        return v


class Settings(BaseSettings):
    """主配置类"""
    app: AppSettings = AppSettings()
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    exchange: ExchangeSettings = ExchangeSettings()
    ai: AISettings = AISettings()
    trading: TradingSettings = TradingSettings()
    notification: NotificationSettings = NotificationSettings()
    api: APISettings = APISettings()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False
        extra = "allow"

    @classmethod
    def from_file(cls, config_path: str | Path) -> "Settings":
        """从YAML文件加载配置"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def to_file(self, config_path: str | Path) -> None:
        """保存配置到YAML文件"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.dict(exclude_unset=True),
                f,
                default_flow_style=False,
                allow_unicode=True,
                indent=2,
            )

    def get_llm_provider(self, name: str) -> Optional[LLMProviderSettings]:
        """获取指定名称的LLM Provider配置"""
        for provider in self.ai.providers:
            if provider.name == name:
                return provider
        return None

    def normalize_provider_weights(self) -> None:
        """标准化LLM Provider权重，使总和为1"""
        total_weight = sum(p.weight for p in self.ai.providers)
        if total_weight <= 0:
            return

        for provider in self.ai.providers:
            provider.weight = provider.weight / total_weight


# 全局配置实例
settings = Settings()