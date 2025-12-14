"""技术指标计算模块，对应 Go 版本的 indicator 包"""

import math
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import talib

from deepalpha.config.settings import TradingSettings
from deepalpha.market.data import Candle
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class IndicatorValue(NamedTuple):
    """指标值，对应 Go 版本的 IndicatorValue"""
    latest: float
    series: List[float]
    state: Optional[str]
    note: Optional[str]


class IndicatorReport(NamedTuple):
    """指标报告，对应 Go 版本的 Report"""
    symbol: str
    interval: str
    count: int
    values: Dict[str, IndicatorValue]
    warnings: List[str]


class IndicatorSettings(NamedTuple):
    """指标计算设置，对应 Go 版本的 Settings"""
    symbol: str
    interval: str
    ema_fast: int = 7
    ema_mid: int = 25
    ema_slow: int = 99
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70


def compute_all(candles: List[Candle], settings: IndicatorSettings) -> IndicatorReport:
    """计算所有技术指标，对应 Go 版本的 ComputeAll"""
    if not candles:
        return IndicatorReport(
            symbol=settings.symbol,
            interval=settings.interval,
            count=0,
            values={},
            warnings=["No candles provided"]
        )

    # 提取OHLCV数据
    closes = np.array([c.close for c in candles], dtype=np.float64)
    highs = np.array([c.high for c in candles], dtype=np.float64)
    lows = np.array([c.low for c in candles], dtype=np.float64)
    volumes = np.array([c.volume for c in candles], dtype=np.float64)

    values = {}
    warnings = []

    # EMA指标
    ema_fast = _trim_ema_leading_zeros(_sanitize_series(talib.EMA(closes, timeperiod=settings.ema_fast)))
    ema_mid = _trim_ema_leading_zeros(_sanitize_series(talib.EMA(closes, timeperiod=settings.ema_mid)))
    ema_slow = _trim_ema_leading_zeros(_sanitize_series(talib.EMA(closes, timeperiod=settings.ema_slow)))
    last_close = closes[-1]

    values["ema_fast"] = IndicatorValue(
        latest=_last_valid(ema_fast),
        series=ema_fast,
        state=_relative_state(last_close, _last_valid(ema_fast)),
        note=f"EMA{settings.ema_fast} vs price"
    )

    values["ema_mid"] = IndicatorValue(
        latest=_last_valid(ema_mid),
        series=ema_mid,
        state=_relative_state(last_close, _last_valid(ema_mid)),
        note=f"EMA{settings.ema_mid} vs price"
    )

    values["ema_slow"] = IndicatorValue(
        latest=_last_valid(ema_slow),
        series=ema_slow,
        state=_relative_state(last_close, _last_valid(ema_slow)),
        note=f"EMA{settings.ema_slow} vs price"
    )

    # RSI指标
    rsi_series = _sanitize_series(talib.RSI(closes, timeperiod=settings.rsi_period))
    rsi_val = _last_valid(rsi_series)
    rsi_state = "neutral"
    if rsi_val >= settings.rsi_overbought:
        rsi_state = "overbought"
    elif rsi_val <= settings.rsi_oversold:
        rsi_state = "oversold"

    values["rsi"] = IndicatorValue(
        latest=rsi_val,
        series=rsi_series,
        state=rsi_state,
        note=f"period={settings.rsi_period} thresholds={settings.rsi_oversold}/{settings.rsi_overbought}"
    )

    # MACD指标
    macd, signal, hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
    macd_series = _sanitize_series(macd)
    signal_series = _sanitize_series(signal)
    hist_series = _sanitize_series(hist)
    macd_note = f"signal={_last_valid(signal_series):.4f} hist={_last_valid(hist_series):.4f}"
    macd_state = "flat"
    if _last_valid(hist_series) > 0:
        macd_state = "bullish"
    elif _last_valid(hist_series) < 0:
        macd_state = "bearish"

    values["macd"] = IndicatorValue(
        latest=_last_valid(macd_series),
        series=hist_series,
        state=macd_state,
        note=macd_note
    )

    # ROC指标
    roc_series = _sanitize_series(talib.ROC(closes, timeperiod=9))
    roc_val = _last_valid(roc_series)
    values["roc"] = IndicatorValue(
        latest=roc_val,
        series=roc_series,
        state=_polarity_state(roc_val),
        note="period=9"
    )

    # Stochastic指标
    k, d = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    k_series = _sanitize_series(k)
    d_series = _sanitize_series(d)
    values["stoch_k"] = IndicatorValue(
        latest=_last_valid(k_series),
        series=k_series,
        state=_stochastic_state(_last_valid(k_series)),
        note=f"d={_last_valid(d_series):.2f}"
    )

    # Williams %R指标
    williams_r_series = _sanitize_series(talib.WILLR(highs, lows, closes, timeperiod=14))
    values["williams_r"] = IndicatorValue(
        latest=_last_valid(williams_r_series),
        series=williams_r_series,
        state=_stochastic_state(-_last_valid(williams_r_series)),
        note="period=14"
    )

    # ATR指标
    atr_series = _sanitize_series(talib.ATR(highs, lows, closes, timeperiod=14))
    values["atr"] = IndicatorValue(
        latest=_last_valid(atr_series),
        series=atr_series,
        state="volatility",
        note="period=14"
    )

    # OBV指标
    obv_series = _sanitize_series(talib.OBV(closes, volumes))
    values["obv"] = IndicatorValue(
        latest=_last_valid(obv_series),
        series=obv_series,
        state=_polarity_state(roc_val),
        note="volume thrust"
    )

    return IndicatorReport(
        symbol=settings.symbol,
        interval=settings.interval,
        count=len(candles),
        values=values,
        warnings=warnings
    )


def _sanitize_series(series: np.ndarray) -> List[float]:
    """清理序列数据，移除NaN和Inf值"""
    result = []
    for v in series:
        if not math.isnan(v) and not math.isinf(v):
            result.append(_round4(v))
    return result


def _trim_ema_leading_zeros(series: List[float]) -> List[float]:
    """移除EMA前导零值，对应 Go 版本的 trimEMALeadingZeros"""
    start = 0
    while start < len(series) and _almost_zero(series[start]):
        start += 1
    return series[start:]


def _almost_zero(v: float) -> bool:
    """判断是否接近零"""
    return abs(v) <= 1e-9


def _last_valid(series: List[float]) -> float:
    """获取序列中最后一个有效值"""
    for i in range(len(series) - 1, -1, -1):
        if not math.isnan(series[i]) and not math.isinf(series[i]):
            return series[i]
    return 0.0


def _relative_state(price: float, ref: float) -> str:
    """计算价格相对参考值的状态"""
    if ref == 0:
        return "unknown"
    if price > ref * 1.002:
        return "above"
    if price < ref * 0.998:
        return "below"
    return "touch"


def _polarity_state(v: float) -> str:
    """计算极性状态"""
    if v > 0:
        return "positive"
    if v < 0:
        return "negative"
    return "flat"


def _stochastic_state(v: float) -> str:
    """计算随机指标状态"""
    if v >= 80:
        return "overbought"
    if v <= 20:
        return "oversold"
    return "neutral"


def _round4(v: float) -> float:
    """四舍五入到4位小数"""
    return round(v * 10000) / 10000


# 单独计算各个指标的函数，便于单独使用
def calculate_ema(closes: List[float], period: int) -> List[float]:
    """计算EMA"""
    arr = np.array(closes, dtype=np.float64)
    result = talib.EMA(arr, timeperiod=period)
    return _sanitize_series(result)


def calculate_rsi(closes: List[float], period: int = 14) -> List[float]:
    """计算RSI"""
    arr = np.array(closes, dtype=np.float64)
    result = talib.RSI(arr, timeperiod=period)
    return _sanitize_series(result)


def calculate_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """计算MACD，返回(macd, signal, histogram)"""
    arr = np.array(closes, dtype=np.float64)
    macd, signal_line, histogram = talib.MACD(arr, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    return (
        _sanitize_series(macd),
        _sanitize_series(signal_line),
        _sanitize_series(histogram)
    )


def calculate_bollinger_bands(closes: List[float], period: int = 20, std_dev: int = 2) -> tuple:
    """计算布林带，返回(upper, middle, lower)"""
    arr = np.array(closes, dtype=np.float64)
    upper, middle, lower = talib.BBANDS(arr, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
    return (
        _sanitize_series(upper),
        _sanitize_series(middle),
        _sanitize_series(lower)
    )


def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> tuple:
    """计算随机指标，返回(k, d)"""
    high_arr = np.array(highs, dtype=np.float64)
    low_arr = np.array(lows, dtype=np.float64)
    close_arr = np.array(closes, dtype=np.float64)
    k, d = talib.STOCH(high_arr, low_arr, close_arr, fastk_period=k_period, slowk_period=d_period, slowk_matype=0, slowd_period=d_period, slowd_matype=0)
    return _sanitize_series(k), _sanitize_series(d)