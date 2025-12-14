"""配置管理器，对应 Go 版本的配置加载逻辑"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import toml
from pydantic import ValidationError

from deepalpha.config.models import DeepAlphaConfig
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        self._config: Optional[DeepAlphaConfig] = None
        self._config_path = config_path or self._get_default_config_path()
        self._env_prefix = "DEEPALPHA_"

    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        # 1. 环境变量
        config_path = os.getenv("BRALE_CONFIG")
        if config_path and os.path.exists(config_path):
            return config_path

        # 2. 当前目录的configs
        current_dir = Path(__file__).parent.parent.parent
        config_path = current_dir / "configs" / "config.toml"
        if config_path.exists():
            return str(config_path)

        # 3. 数据目录
        data_root = os.getenv("BRALE_DATA_ROOT", "/data")
        config_path = Path(data_root) / "config.toml"
        if config_path.exists():
            return str(config_path)

        # 4. 创建默认配置
        return str(config_path)

    @property
    def config(self) -> DeepAlphaConfig:
        """获取配置（延迟加载）"""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> DeepAlphaConfig:
        """加载配置"""
        try:
            # 加载TOML文件
            config_data = {}
            if os.path.exists(self._config_path):
                logger.info(f"加载配置文件: {self._config_path}")
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    config_data = toml.load(f)
            else:
                logger.warning(f"配置文件不存在，使用默认配置: {self._config_path}")

            # 加载环境变量
            env_config = self._load_env_config()
            self._merge_config(config_data, env_config)

            # 创建配置对象
            config = DeepAlphaConfig(**config_data)

            logger.info("配置加载成功")
            return config

        except ValidationError as e:
            logger.error(f"配置验证失败: {e}")
            raise
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise

    def _load_env_config(self) -> Dict[str, Any]:
        """从环境变量加载配置"""
        env_config = {}

        # 遍历所有环境变量
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                # 移除前缀并转换为小写
                config_key = key[len(self._env_prefix):].lower()

                # 处理嵌套键（用__分隔）
                keys = config_key.split('__')
                self._set_nested_dict(env_config, keys, self._parse_env_value(value))

        return env_config

    def _set_nested_dict(self, d: Dict[str, Any], keys: list, value: Any):
        """设置嵌套字典"""
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值"""
        # 尝试解析为JSON
        if value.startswith('{') or value.startswith('['):
            try:
                import json
                return json.loads(value)
            except:
                pass

        # 尝试解析为布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # 尝试解析为数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except:
            pass

        # 返回字符串
        return value

    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """合并配置"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def reload(self) -> DeepAlphaConfig:
        """重新加载配置"""
        logger.info("重新加载配置")
        self._config = None
        return self.config

    def save(self, config_path: Optional[str] = None):
        """保存配置到文件"""
        path = config_path or self._config_path

        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 转换为字典并保存
        config_dict = self.config.dict()
        with open(path, 'w', encoding='utf-8') as f:
            toml.dump(config_dict, f)

        logger.info(f"配置已保存到: {path}")

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config.dict()

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def update(self, key: str, value: Any):
        """更新配置值"""
        keys = key.split('.')
        config_dict = self.config.dict()

        # 找到目标位置
        target = config_dict
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        # 更新值
        target[keys[-1]] = value

        # 重新加载配置
        self._config = DeepAlphaConfig(**config_dict)

    def validate(self) -> bool:
        """验证配置"""
        try:
            # 触发配置验证
            self.config.dict()
            return True
        except ValidationError as e:
            logger.error(f"配置验证失败: {e}")
            return False


# 全局配置实例
config = ConfigManager()