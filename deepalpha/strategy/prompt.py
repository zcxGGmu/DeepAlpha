"""策略提示词管理"""

from pathlib import Path
from typing import Dict, Optional
import os

from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class StrategyPrompts:
    """策略提示词管理器"""

    def __init__(self, prompts_dir: Optional[str] = None):
        if prompts_dir is None:
            # 默认使用项目根目录下的prompts目录
            self.prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)

        self._prompts: Dict[str, str] = {}
        self._load_prompts()

    def _load_prompts(self):
        """加载所有提示词文件"""
        prompt_files = {
            "default": "default.txt",
            "indicator": "agent_indicator.txt",
            "pattern": "agent_pattern.txt",
            "trend": "agent_trend.txt",
            "guideline": "decision_guideline.txt",
            "summary": "user_summary.txt",
        }

        for key, filename in prompt_files.items():
            file_path = self.prompts_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self._prompts[key] = f.read().strip()
                    logger.debug(f"加载提示词: {key} from {filename}")
                except Exception as e:
                    logger.error(f"加载提示词失败 {filename}: {e}")
                    self._prompts[key] = ""
            else:
                logger.warning(f"提示词文件不存在: {file_path}")
                self._prompts[key] = ""

    def get_prompt(self, key: str) -> str:
        """获取提示词"""
        return self._prompts.get(key, "")

    def format_prompt(self, key: str, **kwargs) -> str:
        """格式化提示词"""
        prompt = self.get_prompt(key)
        try:
            return prompt.format(**kwargs)
        except KeyError as e:
            logger.error(f"格式化提示词失败，缺少参数: {e}")
            return prompt

    def reload(self):
        """重新加载提示词"""
        self._prompts.clear()
        self._load_prompts()
        logger.info("提示词已重新加载")


# 全局提示词管理器实例
strategy_prompts = StrategyPrompts()