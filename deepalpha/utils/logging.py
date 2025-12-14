"""日志系统"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    COLORS = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",   # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[35m",  # 紫色
    }
    RESET = "\033[0m"

    def format(self, record):
        # 保存原始颜色
        level_color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{level_color}{record.levelname}{self.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""

    def format(self, record):
        import json

        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加异常信息
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # 添加额外字段
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    format_type: str = "text",
    log_file: Optional[str] = None,
    max_size: str = "100MB",
    backup_count: int = 5,
    enable_rich: bool = True,
) -> None:
    """设置日志系统

    Args:
        level: 日志级别
        format_type: 格式类型，"text" 或 "json"
        log_file: 日志文件路径
        max_size: 日志文件最大大小
        backup_count: 备份文件数量
        enable_rich: 是否启用Rich格式化
    """
    # 获取根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # 清除现有的handlers
    root_logger.handlers.clear()

    # 创建格式化器
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        if enable_rich:
            formatter = None  # RichHandler有自己的格式化
        else:
            formatter = ColoredFormatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

    # 控制台处理器
    if enable_rich:
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 解析文件大小
        size_bytes = _parse_size(max_size)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=size_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter() if format_type == "json" else formatter)
        root_logger.addHandler(file_handler)

    # 配置structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """获取结构化日志器

    Args:
        name: 日志器名称

    Returns:
        结构化日志器实例
    """
    return structlog.get_logger(name)


class LogContext:
    """日志上下文管理器，用于添加临时字段"""

    def __init__(self, logger: structlog.stdlib.BoundLogger, **kwargs):
        self.logger = logger
        self.context = kwargs

    def __enter__(self):
        self.logger = self.logger.bind(**self.context)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def _parse_size(size_str: str) -> int:
    """解析大小字符串为字节数"""
    size_str = size_str.upper()
    if size_str.endswith("KB"):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith("MB"):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith("GB"):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


class TradingLogger:
    """交易专用日志器"""

    def __init__(self):
        self.logger = get_logger("trading")

    def log_decision(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        reason: str,
        agent_name: Optional[str] = None,
    ):
        """记录交易决策"""
        self.logger.info(
            "Trading decision",
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            reason=reason,
            agent_name=agent_name,
        )

    def log_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        status: str = "created",
    ):
        """记录订单"""
        self.logger.info(
            "Order created",
            order_id=order_id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            status=status,
        )

    def log_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        current_price: float,
        pnl: float,
    ):
        """记录持仓"""
        self.logger.info(
            "Position updated",
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            current_price=current_price,
            pnl=pnl,
        )

    def log_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ):
        """记录错误"""
        self.logger.error(
            "Trading error",
            error=str(error),
            error_type=type(error).__name__,
            **(context or {}),
        )


class PerformanceLogger:
    """性能监控日志器"""

    def __init__(self):
        self.logger = get_logger("performance")

    def log_request_time(self, endpoint: str, duration: float, status_code: int):
        """记录API请求时间"""
        self.logger.info(
            "API request",
            endpoint=endpoint,
            duration_ms=duration * 1000,
            status_code=status_code,
        )

    def log_decision_latency(self, symbol: str, duration: float):
        """记录决策延迟"""
        self.logger.info(
            "Decision latency",
            symbol=symbol,
            duration_ms=duration * 1000,
        )

    def log_throughput(self, operation: str, count: int, duration: float):
        """记录吞吐量"""
        self.logger.info(
            "Throughput",
            operation=operation,
            count=count,
            duration_ms=duration * 1000,
            rate=count / duration if duration > 0 else 0,
        )


# 创建全局日志器实例
trading_logger = TradingLogger()
performance_logger = PerformanceLogger()