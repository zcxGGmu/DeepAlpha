"""策略技术指标模块"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IndicatorSignal:
    """指标信号"""
    name: str
    value: float
    signal: str  # buy, sell, neutral
    strength: float  # 0-1, 信号强度


class IndicatorType(Enum):
    """指标类型"""
    MOMENTUM = "momentum"  # 动量指标
    TREND = "trend"  # 趋势指标
    VOLATILITY = "volatility"  # 波动率指标
    VOLUME = "volume"  # 成交量指标


class StrategyIndicators:
    """策略指标计算器"""

    def __init__(self):
        self.indicators: Dict[str, IndicatorSignal] = {}

    def calculate_all_signals(
        self,
        df: pd.DataFrame,
        config: Optional[Dict] = None,
    ) -> List[IndicatorSignal]:
        """计算所有指标信号"""
        signals = []

        # RSI信号
        rsi_signal = self.analyze_rsi(df)
        if rsi_signal:
            signals.append(rsi_signal)

        # MACD信号
        macd_signal = self.analyze_macd(df)
        if macd_signal:
            signals.append(macd_signal)

        # EMA信号
        ema_signal = self.analyze_ema(df)
        if ema_signal:
            signals.append(ema_signal)

        # 布林带信号
        bb_signal = self.analyze_bollinger_bands(df)
        if bb_signal:
            signals.append(bb_signal)

        # 随机指标信号
        stoch_signal = self.analyze_stochastic(df)
        if stoch_signal:
            signals.append(stoch_signal)

        # 威廉指标信号
        wr_signal = self.analyze_williams_r(df)
        if wr_signal:
            signals.append(wr_signal)

        # ATR信号
        atr_signal = self.analyze_atr(df)
        if atr_signal:
            signals.append(atr_signal)

        # OBV信号
        obv_signal = self.analyze_obv(df)
        if obv_signal:
            signals.append(obv_signal)

        return signals

    def analyze_rsi(self, df: pd.DataFrame, period: int = 14) -> Optional[IndicatorSignal]:
        """分析RSI指标"""
        if len(df) < period:
            return None

        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        # 生成信号
        if current_rsi < 30:
            signal = "buy"
            strength = (30 - current_rsi) / 30
        elif current_rsi > 70:
            signal = "sell"
            strength = (current_rsi - 70) / 30
        else:
            signal = "neutral"
            strength = 0

        return IndicatorSignal(
            name="RSI",
            value=current_rsi,
            signal=signal,
            strength=min(1.0, max(0.0, strength))
        )

    def analyze_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Optional[IndicatorSignal]:
        """分析MACD指标"""
        if len(df) < slow:
            return None

        # 计算MACD
        exp1 = df['close'].ewm(span=fast).mean()
        exp2 = df['close'].ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line

        current_macd = macd.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2] if len(histogram) > 1 else 0

        # 生成信号
        if current_macd > current_signal and current_hist > 0:
            if prev_hist < 0:  # 金叉
                signal = "buy"
                strength = 1.0
            else:
                signal = "buy"
                strength = 0.6
        elif current_macd < current_signal and current_hist < 0:
            if prev_hist > 0:  # 死叉
                signal = "sell"
                strength = 1.0
            else:
                signal = "sell"
                strength = 0.6
        else:
            signal = "neutral"
            strength = 0

        return IndicatorSignal(
            name="MACD",
            value=current_hist,
            signal=signal,
            strength=strength
        )

    def analyze_ema(
        self,
        df: pd.DataFrame,
        short: int = 20,
        long: int = 50,
    ) -> Optional[IndicatorSignal]:
        """分析EMA指标"""
        if len(df) < long:
            return None

        # 计算EMA
        ema_short = df['close'].ewm(span=short).mean()
        ema_long = df['close'].ewm(span=long).mean()

        current_short = ema_short.iloc[-1]
        current_long = ema_long.iloc[-1]
        current_price = df['close'].iloc[-1]

        # 生成信号
        if current_price > current_short > current_long:
            signal = "buy"
            strength = 0.8
        elif current_price < current_short < current_long:
            signal = "sell"
            strength = 0.8
        elif current_short > current_long:
            signal = "buy"
            strength = 0.5
        elif current_short < current_long:
            signal = "sell"
            strength = 0.5
        else:
            signal = "neutral"
            strength = 0

        return IndicatorSignal(
            name="EMA",
            value=current_short - current_long,
            signal=signal,
            strength=strength
        )

    def analyze_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std: float = 2,
    ) -> Optional[IndicatorSignal]:
        """分析布林带指标"""
        if len(df) < period:
            return None

        # 计算布林带
        sma = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        upper_band = sma + (rolling_std * std)
        lower_band = sma - (rolling_std * std)

        current_price = df['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]

        # 生成信号
        if current_price < current_lower:
            signal = "buy"
            strength = 0.9
        elif current_price > current_upper:
            signal = "sell"
            strength = 0.9
        elif current_price < lower_band.iloc[-2]:  # 触及下轨后反弹
            signal = "buy"
            strength = 0.6
        elif current_price > upper_band.iloc[-2]:  # 触及上轨后回落
            signal = "sell"
            strength = 0.6
        else:
            signal = "neutral"
            strength = 0

        # 计算布林带位置 (0-1, 0.5在中轨)
        bb_position = (current_price - current_lower) / (current_upper - current_lower)

        return IndicatorSignal(
            name="BB",
            value=bb_position,
            signal=signal,
            strength=strength
        )

    def analyze_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Optional[IndicatorSignal]:
        """分析随机指标"""
        if len(df) < k_period:
            return None

        # 计算随机指标
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()

        current_k = k_percent.iloc[-1]
        current_d = d_percent.iloc[-1]

        # 生成信号
        if current_k < 20 and current_d < 20:
            signal = "buy"
            strength = (20 - current_k) / 20
        elif current_k > 80 and current_d > 80:
            signal = "sell"
            strength = (current_k - 80) / 20
        elif current_k > current_d and current_k > 50:
            signal = "buy"
            strength = 0.6
        elif current_k < current_d and current_k < 50:
            signal = "sell"
            strength = 0.6
        else:
            signal = "neutral"
            strength = 0

        return IndicatorSignal(
            name="Stoch",
            value=current_k,
            signal=signal,
            strength=min(1.0, max(0.0, strength))
        )

    def analyze_williams_r(
        self,
        df: pd.DataFrame,
        period: int = 14,
    ) -> Optional[IndicatorSignal]:
        """分析威廉指标"""
        if len(df) < period:
            return None

        # 计算威廉指标
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        wr = -100 * ((high_max - df['close']) / (high_max - low_min))

        current_wr = wr.iloc[-1]

        # 生成信号
        if current_wr < -80:
            signal = "buy"
            strength = (-80 - current_wr) / 20
        elif current_wr > -20:
            signal = "sell"
            strength = (current_wr + 20) / 20
        else:
            signal = "neutral"
            strength = 0

        return IndicatorSignal(
            name="WR",
            value=current_wr,
            signal=signal,
            strength=min(1.0, max(0.0, strength))
        )

    def analyze_atr(
        self,
        df: pd.DataFrame,
        period: int = 14,
    ) -> Optional[IndicatorSignal]:
        """分析ATR指标"""
        if len(df) < period:
            return None

        # 计算ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()

        current_atr = atr.iloc[-1]
        current_price = df['close'].iloc[-1]
        atr_percent = (current_atr / current_price) * 100

        # ATR主要用于判断波动率，不直接产生买卖信号
        # 但可以作为其他指标的辅助
        signal = "neutral"
        strength = min(1.0, atr_percent / 5)  # 标准化波动率

        return IndicatorSignal(
            name="ATR",
            value=atr_percent,
            signal=signal,
            strength=strength
        )

    def analyze_obv(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        """分析OBV指标"""
        if len(df) < 2:
            return None

        # 计算OBV
        obv = np.where(df['close'] > df['close'].shift(), df['volume'],
                       np.where(df['close'] < df['close'].shift(), -df['volume'], 0)).cumsum()

        current_obv = obv[-1]
        prev_obv = obv[-10] if len(obv) > 10 else obv[0]

        # 计算OBV趋势
        obv_ma = pd.Series(obv).rolling(window=10).mean()
        current_ma = obv_ma.iloc[-1]

        # 生成信号
        if current_obv > current_ma and current_obv > prev_obv:
            signal = "buy"
            strength = 0.7
        elif current_obv < current_ma and current_obv < prev_obv:
            signal = "sell"
            strength = 0.7
        else:
            signal = "neutral"
            strength = 0

        return IndicatorSignal(
            name="OBV",
            value=current_obv,
            signal=signal,
            strength=strength
        )