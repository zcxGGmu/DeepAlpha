"""K线数据格式化，对应 Go 版本的 candle_format"""

import csv
import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from deepalpha.market.data import Candle
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


def format_candle_csv(candles: List[Candle]) -> str:
    """将K线数据格式化为CSV字符串"""
    if not candles:
        return ""

    # CSV头部
    headers = [
        "OpenTime",
        "CloseTime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Trades"
    ]

    rows = []
    rows.append(",".join(headers))

    # 添加K线数据
    for candle in candles:
        row = [
            str(candle.open_time),
            str(candle.close_time),
            str(candle.open),
            str(candle.high),
            str(candle.low),
            str(candle.close),
            str(candle.volume),
            str(candle.trades)
        ]
        rows.append(",".join(row))

    return "\n".join(rows)


def format_candle_json(candles: List[Candle], symbol: str, interval: str) -> Dict[str, Any]:
    """将K线数据格式化为JSON对象"""
    return {
        "symbol": symbol,
        "interval": interval,
        "count": len(candles),
        "data": [
            {
                "openTime": candle.open_time,
                "closeTime": candle.close_time,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "trades": candle.trades
            }
            for candle in candles
        ]
    }


def format_candle_table(candles: List[Candle], max_rows: Optional[int] = None) -> str:
    """将K线数据格式化为表格字符串"""
    if not candles:
        return "No data"

    if max_rows and len(candles) > max_rows:
        candles = candles[-max_rows:]  # 显示最新的N行

    # 计算列宽
    col_widths = {
        "time": 20,
        "open": 12,
        "high": 12,
        "low": 12,
        "close": 12,
        "volume": 15
    }

    # 构建表格
    lines = []

    # 表头
    header = f"{'Time':^{col_widths['time']}} | {'Open':^{col_widths['open']}} | {'High':^{col_widths['high']}} | {'Low':^{col_widths['low']}} | {'Close':^{col_widths['close']}} | {'Volume':^{col_widths['volume']}}"
    lines.append(header)
    lines.append("-" * len(header))

    # 数据行
    for candle in candles:
        time_str = datetime.fromtimestamp(candle.open_time / 1000).strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"{time_str:^{col_widths['time']}} | "
            f"{candle.open:^{col_widths['open']}.4f} | "
            f"{candle.high:^{col_widths['high']}.4f} | "
            f"{candle.low:^{col_widths['low']}.4f} | "
            f"{candle.close:^{col_widths['close']}.4f} | "
            f"{candle.volume:^{col_widths['volume']}.2f}"
        )
        lines.append(line)

    return "\n".join(lines)


def format_candle_summary(candles: List[Candle]) -> Dict[str, Any]:
    """生成K线数据摘要统计"""
    if not candles:
        return {}

    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]
    high_low_spreads = [c.high - c.low for c in candles]
    open_close_spreads = [abs(c.close - c.open) for c in candles]

    # 计算统计信息
    price_change = closes[-1] - closes[0] if len(closes) > 1 else 0
    price_change_pct = (price_change / closes[0] * 100) if closes[0] != 0 else 0

    return {
        "count": len(candles),
        "first_price": closes[0],
        "last_price": closes[-1],
        "price_change": price_change,
        "price_change_percent": price_change_pct,
        "highest_price": max(closes),
        "lowest_price": min(closes),
        "avg_volume": sum(volumes) / len(volumes),
        "total_volume": sum(volumes),
        "avg_spread": sum(high_low_spreads) / len(high_low_spreads),
        "max_spread": max(high_low_spreads),
        "avg_body": sum(open_close_spreads) / len(open_close_spreads),
        "green_candles": sum(1 for c in candles if c.close > c.open),
        "red_candles": sum(1 for c in candles if c.close < c.open),
        "doji_candles": sum(1 for c in candles if abs(c.close - c.open) < 0.0001)
    }


def parse_candle_from_csv(csv_string: str) -> List[Candle]:
    """从CSV字符串解析K线数据"""
    candles = []
    reader = csv.DictReader(csv_string.strip().split('\n'))

    for row in reader:
        candle = Candle(
            open_time=int(row["OpenTime"]),
            close_time=int(row["CloseTime"]),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
            trades=int(row["Trades"])
        )
        candles.append(candle)

    return candles


def parse_candle_from_binance(klines: List[List[Any]]) -> List[Candle]:
    """从Binance API响应解析K线数据"""
    candles = []
    for kline in klines:
        if len(kline) < 9:
            continue

        candle = Candle(
            open_time=kline[0],
            close_time=kline[6],
            open=float(kline[1]),
            high=float(kline[2]),
            low=float(kline[3]),
            close=float(kline[4]),
            volume=float(kline[5]),
            trades=int(kline[8])
        )
        candles.append(candle)

    return candles


def merge_candles(candles1: List[Candle], candles2: List[Candle]) -> List[Candle]:
    """合并两个K线列表，按时间排序"""
    all_candles = candles1 + candles2
    all_candles.sort(key=lambda x: x.open_time)

    # 去重（相同时间戳的K线）
    seen = set()
    unique_candles = []
    for candle in all_candles:
        if candle.open_time not in seen:
            seen.add(candle.open_time)
            unique_candles.append(candle)

    return unique_candles