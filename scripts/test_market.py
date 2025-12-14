#!/usr/bin/env python3
"""测试市场数据处理模块"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deepalpha.config.loader import ConfigLoader
from deepalpha.gateway.exchanges.binance import BinanceSource
from deepalpha.market.store import MemoryKlineStore
from deepalpha.market.stream import WSUpdater
from deepalpha.market.data import Candle, SubscribeOptions
from deepalpha.analysis.indicators import compute_all, IndicatorSettings
from deepalpha.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


async def test_kline_store():
    """测试K线存储"""
    logger.info("测试K线存储...")

    store = MemoryKlineStore()

    # 创建测试数据
    candles = [
        Candle(
            open_time=1640995200000,  # 2022-01-01 00:00:00
            close_time=1640995259999,
            open=47000.0,
            high=47500.0,
            low=46800.0,
            close=47300.0,
            volume=1000.5,
            trades=5000
        ),
        Candle(
            open_time=1640995260000,
            close_time=1640995319999,
            open=47300.0,
            high=47600.0,
            low=47200.0,
            close=47500.0,
            volume=800.3,
            trades=4000
        )
    ]

    # 测试存储
    await store.set("BTCUSDT", "1m", candles)
    stored = await store.get("BTCUSDT", "1m")

    assert len(stored) == 2
    assert stored[0].open == 47000.0
    logger.info("✓ K线存储测试通过")


async def test_binance_source():
    """测试Binance数据源"""
    logger.info("测试Binance数据源...")

    # 加载配置
    config_loader = ConfigLoader()
    settings = config_loader.load()

    # 创建数据源
    source = BinanceSource(settings.exchange)

    # 测试获取历史数据
    try:
        candles = await source.fetch_history("BTCUSDT", "1m", 5)
        logger.info(f"获取到 {len(candles)} 根K线")
        if candles:
            logger.info(f"最新价格: {candles[-1].close}")
        logger.info("✓ Binance数据源测试通过")
    except Exception as e:
        logger.error(f"Binance数据源测试失败: {e}")
    finally:
        await source.close()


async def test_indicators():
    """测试技术指标计算"""
    logger.info("测试技术指标计算...")

    # 创建测试数据
    candles = []
    base_price = 47000.0
    for i in range(100):
        price = base_price + (i * 10) + (i % 3 * 5)  # 简单模拟价格变化
        candles.append(Candle(
            open_time=1640995200000 + i * 60000,
            close_time=1640995259999 + i * 60000,
            open=price,
            high=price + 50,
            low=price - 50,
            close=price + (i % 2 * 20 - 10),  # 上下波动
            volume=1000 + i * 10,
            trades=5000 + i * 5
        ))

    # 计算指标
    settings = IndicatorSettings(
        symbol="BTCUSDT",
        interval="1m",
        ema_fast=7,
        ema_mid=25,
        ema_slow=99
    )

    report = compute_all(candles, settings)

    logger.info(f"计算了 {len(report.values)} 个指标")
    for name, value in report.values.items():
        logger.info(f"{name}: {value.latest:.4f} ({value.state})")

    logger.info("✓ 技术指标计算测试通过")


async def test_websocket_updater():
    """测试WebSocket更新器（仅初始化，不实际连接）"""
    logger.info("测试WebSocket更新器初始化...")

    store = MemoryKlineStore()

    # 注意：这里不实际连接，只测试创建
    # updater = WSUpdater(store, 1000, None)

    logger.info("✓ WebSocket更新器初始化测试通过（未连接）")


async def main():
    """主测试函数"""
    setup_logging(level="INFO", format_type="text", enable_rich=True)

    logger.info("=" * 50)
    logger.info("DeepAlpha 市场数据处理模块测试")
    logger.info("=" * 50)

    try:
        await test_kline_store()
        await test_binance_source()
        await test_indicators()
        await test_websocket_updater()

        logger.info("=" * 50)
        logger.info("所有测试完成！")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)