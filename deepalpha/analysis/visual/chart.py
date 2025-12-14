"""图表可视化模块"""

import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import asyncio
from pathlib import Path

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from deepalpha.market.models import Candle, TimeFrame
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class ChartRenderer:
    """图表渲染器"""

    def __init__(self, output_dir: str = "data/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.playwright = None

    async def _init_playwright(self):
        """初始化Playwright"""
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright not installed. Install with: pip install playwright")

        if not self.playwright:
            self.playwright = await async_playwright().start()

    async def render_candlestick_chart(
        self,
        symbol: str,
        candles: List[Candle],
        indicators: Optional[Dict[str, List[float]]] = None,
        patterns: Optional[List[Dict]] = None,
        timeframe: TimeFrame = TimeFrame.M1,
        width: int = 1200,
        height: int = 600,
    ) -> str:
        """渲染K线图"""
        await self._init_playwright()

        # 生成HTML内容
        html_content = self._generate_chart_html(
            symbol, candles, indicators, patterns, timeframe, width, height
        )

        # 保存HTML文件
        html_file = self.output_dir / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # 使用Chrome渲染为图片
        browser = await self.playwright.chromium.launch()
        page = await browser.new_page()
        await page.set_viewport_size({"width": width, "height": height})
        await page.goto(f"file://{html_file.absolute()}")

        # 等待图表加载
        await page.wait_for_selector('#chart')

        # 截图
        image_file = self.output_dir / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        await page.screenshot(path=str(image_file), full_page=True)

        await browser.close()

        logger.info(f"图表已生成: {image_file}")
        return str(image_file)

    def _generate_chart_html(
        self,
        symbol: str,
        candles: List[Candle],
        indicators: Optional[Dict[str, List[float]]],
        patterns: Optional[List[Dict]],
        timeframe: TimeFrame,
        width: int,
        height: int,
    ) -> str:
        """生成图表HTML"""

        # 准备数据
        timestamps = [c.timestamp.isoformat() for c in candles]
        ohlc_data = [[c.open, c.high, c.low, c.close] for c in candles]
        volumes = [c.volume for c in candles]

        # 准备指标数据
        indicator_scripts = ""
        if indicators:
            for name, values in indicators.items():
                if len(values) == len(candles):
                    indicator_scripts += f"""
                    var {name}_data = {values};
                    """

        # 准备形态标记
        pattern_markers = ""
        if patterns:
            for pattern in patterns:
                index = pattern.get('index', 0)
                pattern_type = pattern.get('type', 'unknown')
                color = 'red' if 'bearish' in pattern_type.lower() else 'green'
                pattern_markers += f"""
                {{
                    xaxis: {index},
                    yaxis: ohlc_data[{index}][2],  // 最低价
                    text: '{pattern_type}',
                    textposition: 'bottom',
                    showarrow: true,
                    arrowcolor: '{color}',
                    font: {{color: '{color}'}}
                }},
                """

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{symbol} - {timeframe.value} Chart</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #1e1e1e;
            color: #ffffff;
        }}
        #chart {{
            width: {width}px;
            height: {height}px;
        }}
        .title {{
            text-align: center;
            padding: 10px;
            font-size: 18px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="title">{symbol} - {timeframe.value} Chart</div>
    <div id="chart"></div>

    <script>
        var timestamps = {timestamps};
        var ohlc_data = {ohlc_data};
        var volume_data = {volumes};

        {indicator_scripts}

        var trace1 = {{
            x: timestamps,
            open: ohlc_data.map(d => d[0]),
            high: ohlc_data.map(d => d[1]),
            low: ohlc_data.map(d => d[2]),
            close: ohlc_data.map(d => d[3]),
            type: 'candlestick',
            name: 'Price',
            increasing: {{line: {{color: '#00ff00'}}}},
            decreasing: {{line: {{color: '#ff0000'}}}}
        }};

        var trace2 = {{
            x: timestamps,
            y: volume_data,
            type: 'bar',
            name: 'Volume',
            yaxis: 'y2',
            marker: {{
                color: volume_data.map((v, i) =>
                    ohlc_data[i][3] >= ohlc_data[i][0] ? '#00ff0040' : '#ff000040'
                )
            }}
        }};

        var traces = [trace1, trace2];

        // 添加指标线
        """

        # 添加指标
        if indicators:
            for name, values in indicators.items():
                if len(values) == len(candles):
                    html_template += f"""
        traces.push({{
            x: timestamps,
            y: {name}_data,
            type: 'scatter',
            mode: 'lines',
            name: '{name}',
            line: {{
                color: 'rgba(255, 255, 0, 0.8)',
                width: 1
            }}
        }});
                    """

        html_template += f"""
        var layout = {{
            title: false,
            xaxis: {{
                rangeslider: {{
                    visible: false
                }},
                type: 'date',
                gridcolor: '#444',
                tickcolor: '#444'
            }},
            yaxis: {{
                domain: [0.3, 1],
                gridcolor: '#444',
                tickcolor: '#444'
            }},
            yaxis2: {{
                domain: [0, 0.2],
                gridcolor: '#444',
                tickcolor: '#444'
            }},
            paper_bgcolor: '#1e1e1e',
            plot_bgcolor: '#1e1e1e',
            font: {{
                color: '#ffffff'
            }},
            margin: {{t: 20, r: 20, b: 20, l: 20}},
            annotations: [{pattern_markers}]
        }};

        Plotly.newPlot('chart', traces, layout);
    </script>
</body>
</html>
        """

        return html_template

    async def render_indicator_chart(
        self,
        symbol: str,
        timestamps: List[datetime],
        indicators: Dict[str, List[float]],
        width: int = 1200,
        height: int = 400,
    ) -> str:
        """渲染指标图"""
        await self._init_playwright()

        # 生成HTML
        html_content = self._generate_indicator_html(symbol, timestamps, indicators, width, height)

        # 保存HTML
        html_file = self.output_dir / f"{symbol}_indicators_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # 渲染为图片
        browser = await self.playwright.chromium.launch()
        page = await browser.new_page()
        await page.set_viewport_size({"width": width, "height": height})
        await page.goto(f"file://{html_file.absolute()}")
        await page.wait_for_selector('#chart')

        image_file = self.output_dir / f"{symbol}_indicators_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        await page.screenshot(path=str(image_file), full_page=True)

        await browser.close()

        return str(image_file)

    def _generate_indicator_html(
        self,
        symbol: str,
        timestamps: List[datetime],
        indicators: Dict[str, List[float]],
        width: int,
        height: int,
    ) -> str:
        """生成指标图表HTML"""

        timestamp_strs = [t.isoformat() for t in timestamps]

        traces = []
        colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff']

        for i, (name, values) in enumerate(indicators.items()):
            color = colors[i % len(colors)]
            traces.append(f"""
            {{
                x: {timestamp_strs},
                y: {values},
                type: 'scatter',
                mode: 'lines',
                name: '{name}',
                line: {{
                    color: '{color}',
                    width: 2
                }}
            }}
            """)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{symbol} - Indicators</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #1e1e1e;
            color: #ffffff;
        }}
        #chart {{
            width: {width}px;
            height: {height}px;
        }}
        .title {{
            text-align: center;
            padding: 10px;
            font-size: 18px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="title">{symbol} - Technical Indicators</div>
    <div id="chart"></div>

    <script>
        var traces = [{', '.join(traces)}];

        var layout = {{
            title: false,
            xaxis: {{
                type: 'date',
                gridcolor: '#444',
                tickcolor: '#444'
            }},
            yaxis: {{
                gridcolor: '#444',
                tickcolor: '#444'
            }},
            paper_bgcolor: '#1e1e1e',
            plot_bgcolor: '#1e1e1e',
            font: {{
                color: '#ffffff'
            }},
            margin: {{t: 20, r: 20, b: 20, l: 20}},
            legend: {{
                x: 0,
                y: 1,
                bgcolor: 'rgba(0,0,0,0)',
                bordercolor: '#ffffff'
            }}
        }};

        Plotly.newPlot('chart', traces, layout);
    </script>
</body>
</html>
        """

        return html

    async def close(self):
        """关闭资源"""
        if self.playwright:
            await self.playwright.stop()


# 全局图表渲染器实例
chart_renderer = ChartRenderer()


async def generate_chart(
    symbol: str,
    candles: List[Candle],
    indicators: Optional[Dict[str, List[float]]] = None,
    patterns: Optional[List[Dict]] = None,
    timeframe: TimeFrame = TimeFrame.M1,
) -> str:
    """生成图表"""
    return await chart_renderer.render_candlestick_chart(
        symbol, candles, indicators, patterns, timeframe
    )