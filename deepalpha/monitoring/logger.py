"""日志系统，对应 Go 版本的日志管理"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from deepalpha.config import config


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }

    def format(self, record):
        # 添加颜色
        if hasattr(record, 'no_color') and record.no_color:
            color = ''
        else:
            color = self.COLORS.get(record.levelname, '')

        # 格式化消息
        formatted = super().format(record)
        return f"{color}{formatted}{self.COLORS['RESET']}"


class ContextFilter(logging.Filter):
    """上下文过滤器"""

    def filter(self, record):
        # 添加时间戳
        record.timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # 添加进程和线程信息
        record.process_id = os.getpid()
        record.thread_id = record.thread

        # 添加模块名
        if not hasattr(record, 'module'):
            record.module = record.name

        return True


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""

    def format(self, record):
        import json

        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': os.getpid(),
            'thread_id': record.thread,
        }

        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # 添加额外字段
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno',
                          'pathname', 'filename', 'module', 'lineno',
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    name: Optional[str] = None,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """设置日志系统"""
    # 获取配置
    app_config = config.app
    log_level = level or app_config.log_level.value
    log_dir = Path(log_file or app_config.log_dir)

    # 创建日志目录
    log_dir.mkdir(parents=True, exist_ok=True)

    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # 避免重复添加handler
    if logger.handlers:
        logger.handlers.clear()

    # 添加过滤器
    context_filter = ContextFilter()
    logger.addFilter(context_filter)

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    if json_format:
        console_formatter = JSONFormatter()
    else:
        console_formatter = ColoredFormatter(
            fmt='%(timestamp)s [%(process_id)d] %(module)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件handler - 主日志
    main_log_file = log_dir / "deepalpha.log"
    file_handler = logging.handlers.RotatingFileHandler(
        filename=main_log_file,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter(
        fmt='%(asctime)s [%(process)d] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 文件handler - 错误日志
    error_log_file = log_dir / "deepalpha_error.log"
    error_handler = logging.handlers.RotatingFileHandler(
        filename=error_log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    # 文件handler - LLM日志（如果有）
    llm_log_file = log_dir / "deepalpha_llm.log"
    llm_handler = logging.handlers.RotatingFileHandler(
        filename=llm_log_file,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=5,
        encoding='utf-8'
    )
    llm_handler.setLevel(logging.DEBUG)
    llm_handler.addFilter(lambda record: 'llm' in record.name.lower())
    llm_handler.setFormatter(file_formatter)
    logger.addHandler(llm_handler)

    # 设置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """获取logger实例"""
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """日志适配器，支持额外上下文"""

    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})

    def process(self, msg, kwargs):
        # 将extra合并到kwargs中
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra'].update(self.extra)

        return msg, kwargs


# 定时器装饰器
def timer(method):
    """计时器装饰器"""
    import time
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        logger = get_logger(self.__class__.__module__)
        start_time = time.perf_counter()

        try:
            result = method(self, *args, **kwargs)
            return result
        finally:
            duration = time.perf_counter() - start_time
            logger.debug(
                f"{method.__name__} 执行时间: {duration:.4f}秒",
                extra={
                    'method': method.__name__,
                    'duration': duration,
                    'class': self.__class__.__name__
                }
            )

    return wrapper


# 性能计时器
def timer_scope(name: str):
    """性能计时器上下文管理器"""
    import time
    from contextlib import contextmanager

    @contextmanager
    def _timer():
        logger = get_logger(__name__)
        start_time = time.perf_counter()

        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            logger.info(
                f"[性能] {name}: {duration:.4f}秒",
                extra={
                    'timer_scope': name,
                    'duration': duration,
                    'type': 'performance'
                }
            )

    return _timer()


# 初始化默认日志
_default_logger = None


def get_default_logger() -> logging.Logger:
    """获取默认logger"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging('deepalpha')
    return _default_logger