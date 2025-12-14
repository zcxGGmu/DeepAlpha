"""自定义异常类"""


class DeepAlphaError(Exception):
    """DeepAlpha基础异常类"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ConfigurationError(DeepAlphaError):
    """配置错误"""
    pass


class DatabaseError(DeepAlphaError):
    """数据库错误"""
    pass


class ExchangeError(DeepAlphaError):
    """交易所API错误"""
    pass


class TradingError(DeepAlphaError):
    """交易相关错误"""
    pass


class OrderError(TradingError):
    """订单错误"""
    pass


class PositionError(TradingError):
    """持仓错误"""
    pass


class InsufficientBalanceError(TradingError):
    """余额不足错误"""
    pass


class RiskLimitExceededError(TradingError):
    """风险限制超出错误"""
    pass


class AgentError(DeepAlphaError):
    """智能体错误"""
    pass


class DecisionError(DeepAlphaError):
    """决策错误"""
    pass


class LLMError(DeepAlphaError):
    """LLM调用错误"""
    pass


class DataError(DeepAlphaError):
    """数据错误"""
    pass


class ValidationError(DeepAlphaError):
    """验证错误"""
    pass


class AuthenticationError(DeepAlphaError):
    """认证错误"""
    pass


class PermissionError(DeepAlphaError):
    """权限错误"""
    pass


class RateLimitError(DeepAlphaError):
    """速率限制错误"""
    pass