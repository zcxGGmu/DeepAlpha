"""配置加载器"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from deepalpha.config.settings import Settings


class ConfigLoader:
    """配置加载器，支持多环境配置"""

    def __init__(self, config_dir: str | Path = "config"):
        self.config_dir = Path(config_dir)
        self.default_config = None
        self.env_config = None
        self.final_config = None

    def load(
        self,
        env: Optional[str] = None,
        config_file: Optional[str] = None,
        **overrides,
    ) -> Settings:
        """加载配置"""
        # 1. 加载默认配置
        self._load_default()

        # 2. 加载环境特定配置
        if env:
            self._load_env(env)

        # 3. 加载指定配置文件
        if config_file:
            self._load_file(config_file)

        # 4. 应用环境变量
        self._load_env_vars()

        # 5. 应用覆盖参数
        if overrides:
            self._apply_overrides(overrides)

        # 6. 创建最终配置
        self.final_config = Settings(**self._merge_configs())

        # 7. 标准化LLM Provider权重
        self.final_config.normalize_provider_weights()

        return self.final_config

    def _load_default(self):
        """加载默认配置"""
        default_file = self.config_dir / "default.yaml"
        if default_file.exists():
            with open(default_file, "r", encoding="utf-8") as f:
                self.default_config = yaml.safe_load(f) or {}
        else:
            self.default_config = {}

    def _load_env(self, env: str):
        """加载环境特定配置"""
        env_file = self.config_dir / f"{env}.yaml"
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                self.env_config = yaml.safe_load(f) or {}
        else:
            self.env_config = {}

    def _load_file(self, config_file: str):
        """加载指定配置文件"""
        file_path = Path(config_file)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                self.file_config = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Config file not found: {config_file}")

    def _load_env_vars(self):
        """加载环境变量"""
        self.env_vars = {}

        # 应用配置
        self.env_vars.setdefault("app", {})
        self.env_vars["app"]["name"] = os.getenv("APP_NAME")
        self.env_vars["app"]["version"] = os.getenv("APP_VERSION")
        self.env_vars["app"]["debug"] = self._str_to_bool(os.getenv("DEBUG"))
        self.env_vars["app"]["log_level"] = os.getenv("LOG_LEVEL")

        # 数据库配置
        self.env_vars.setdefault("database", {})
        self.env_vars["database"]["url"] = os.getenv("DATABASE_URL")

        # Redis配置
        self.env_vars.setdefault("redis", {})
        self.env_vars["redis"]["url"] = os.getenv("REDIS_URL")

        # 交易所配置
        self.env_vars.setdefault("exchange", {})
        self.env_vars["exchange"]["api_key"] = os.getenv("BINANCE_API_KEY")
        self.env_vars["exchange"]["api_secret"] = os.getenv("BINANCE_API_SECRET")
        self.env_vars["exchange"]["sandbox"] = self._str_to_bool(
            os.getenv("BINANCE_SANDBOX", "true")
        )

        # LLM Provider配置
        self.env_vars.setdefault("ai", {})
        providers = []

        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            providers.append(
                {
                    "name": "openai",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "model": "gpt-4",
                    "weight": 0.4,
                }
            )

        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            providers.append(
                {
                    "name": "anthropic",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    "model": "claude-3-opus",
                    "weight": 0.3,
                }
            )

        # DeepSeek
        if os.getenv("DEEPSEEK_API_KEY"):
            providers.append(
                {
                    "name": "deepseek",
                    "api_key": os.getenv("DEEPSEEK_API_KEY"),
                    "model": "deepseek-chat",
                    "weight": 0.3,
                }
            )

        if providers:
            self.env_vars["ai"]["providers"] = providers

        # 交易配置
        self.env_vars.setdefault("trading", {})
        self.env_vars["trading"]["max_position_size"] = self._str_to_float(
            os.getenv("MAX_POSITION_SIZE")
        )
        self.env_vars["trading"]["stop_loss"] = self._str_to_float(
            os.getenv("STOP_LOSS")
        )
        self.env_vars["trading"]["take_profit"] = self._str_to_float(
            os.getenv("TAKE_PROFIT")
        )
        self.env_vars["trading"]["max_drawdown"] = self._str_to_float(
            os.getenv("MAX_DRAWDOWN")
        )

        # API配置
        self.env_vars.setdefault("api", {})
        self.env_vars["api"]["host"] = os.getenv("API_HOST", "0.0.0.0")
        self.env_vars["api"]["port"] = int(os.getenv("API_PORT", "8000"))
        self.env_vars["api"]["secret_key"] = os.getenv("SECRET_KEY")

        # 通知配置
        self.env_vars.setdefault("notification", {})
        if os.getenv("TELEGRAM_BOT_TOKEN"):
            self.env_vars["notification"]["telegram"] = {
                "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
                "chat_id": os.getenv("TELEGRAM_CHAT_ID"),
            }

    def _apply_overrides(self, overrides: Dict[str, Any]):
        """应用覆盖参数"""
        self.overrides = overrides

    def _merge_configs(self) -> Dict[str, Any]:
        """合并所有配置"""
        result = {}

        # 合并默认配置
        if self.default_config:
            result.update(self.default_config)

        # 合并环境配置
        if self.env_config:
            result = self._deep_merge(result, self.env_config)

        # 合并文件配置
        if hasattr(self, "file_config") and self.file_config:
            result = self._deep_merge(result, self.file_config)

        # 合并环境变量
        if hasattr(self, "env_vars") and self.env_vars:
            result = self._deep_merge(result, self.env_vars, skip_none=True)

        # 合并覆盖参数
        if hasattr(self, "overrides") and self.overrides:
            result = self._deep_merge(result, {"override": self.overrides})
            # 处理override参数（支持嵌套路径）
            if "override" in result:
                self._apply_nested_overrides(result, result.pop("override"))

        return result

    def _deep_merge(
        self, base: Dict[str, Any], update: Dict[str, Any], skip_none: bool = False
    ) -> Dict[str, Any]:
        """深度合并字典"""
        result = base.copy()

        for key, value in update.items():
            if skip_none and value is None:
                continue

            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value, skip_none)
            else:
                result[key] = value

        return result

    def _apply_nested_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]):
        """应用嵌套覆盖参数（支持点号分隔的路径）"""
        for key, value in overrides.items():
            if "." in key:
                # 嵌套路径，如 "database.url"
                keys = key.split(".")
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                # 直接设置
                config[key] = value

    @staticmethod
    def _str_to_bool(value: Optional[str]) -> Optional[bool]:
        """将字符串转换为布尔值"""
        if value is None:
            return None
        return value.lower() in ("true", "1", "yes", "on")

    @staticmethod
    def _str_to_float(value: Optional[str]) -> Optional[float]:
        """将字符串转换为浮点数"""
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None