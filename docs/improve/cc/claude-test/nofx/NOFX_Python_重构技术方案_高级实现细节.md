# NOFX Python重构技术方案 - 高级实现细节篇

> 本文档是《NOFX_Python_重构技术方案_A股港股》系列的第八部分
> 覆盖第31-35章：高级数据库优化、完整前端实现、高级监控告警、性能优化、生产检查清单

---

## 第31章 高级数据库优化

### 31.1 读写分离架构

```python
# src/database/read_write_split.py
"""
数据库读写分离
"""

from typing import Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class DatabaseCluster:
    """
    数据库集群（读写分离）

    架构：
    1. 主库（Master）：处理所有写操作
    2. 从库（Slave）：只处理读操作
    3. 负载均衡：多个从库之间的负载均衡
    """

    def __init__(
        self,
        master_url: str,
        slave_urls: list[str],
        pool_size: int = 20,
        max_overflow: int = 40,
    ):
        """
        初始化数据库集群

        Args:
            master_url: 主库连接URL
            slave_urls: 从库连接URL列表
            pool_size: 连接池大小
            max_overflow: 最大溢出连接数
        """
        # 主库引擎（写）
        self.master_engine = create_async_engine(
            master_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # 连接健康检查
            pool_recycle=3600,  # 连接回收时间
            echo=settings.DEBUG,
        )

        # 从库引擎（读）
        self.slave_engines = []
        for slave_url in slave_urls:
            engine = create_async_engine(
                slave_url,
                pool_size=pool_size * 2,  # 读库连接池可以更大
                max_overflow=max_overflow * 2,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=settings.DEBUG,
            )
            self.slave_engines.append(engine)

        # 当前使用的从库索引（轮询）
        self.current_slave_index = 0

        logger.info("数据库集群初始化", extra={
            'master_url': master_url.split('@')[-1] if '@' in master_url else 'unknown',
            'slave_count': len(slave_urls),
        })

    def get_master_session(self) -> AsyncSession:
        """获取主库会话（写操作）"""
        return sessionmaker(
            bind=self.master_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )()

    def get_slave_session(self) -> AsyncSession:
        """获取从库会话（读操作）- 轮询负载均衡"""
        # 轮询选择从库
        engine = self.slave_engines[self.current_slave_index]
        self.current_slave_index = (self.current_slave_index + 1) % len(self.slave_engines)

        return sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )()

    async def close(self) -> None:
        """关闭所有连接"""
        await self.master_engine.dispose()

        for engine in self.slave_engines:
            await engine.dispose()

        logger.info("数据库集群连接已关闭")


# 使用示例
db_cluster = DatabaseCluster(
    master_url=settings.DATABASE_MASTER_URL,
    slave_urls=settings.DATABASE_SLAVE_URLS,
)


class ReadWriteSession:
    """
    读写会话管理器

    自动路由读写操作到主库和从库
    """

    def __init__(self, cluster: DatabaseCluster):
        """
        初始化会话管理器

        Args:
            cluster: 数据库集群
        """
        self.cluster = cluster
        self.master_session: Optional[AsyncSession] = None
        self.slave_session: Optional[AsyncSession] = None

    async def __aenter__(self):
        """进入上下文"""
        self.slave_session = self.cluster.get_slave_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        if self.slave_session:
            await self.slave_session.close()

    async def get_write_session(self) -> AsyncSession:
        """获取写会话"""
        if not self.master_session:
            self.master_session = self.cluster.get_master_session()
        return self.master_session

    async def commit(self) -> None:
        """提交事务（使用主库）"""
        if self.master_session:
            await self.master_session.commit()

    async def rollback(self) -> None:
        """回滚事务（使用主库）"""
        if self.master_session:
            await self.master_session.rollback()

    async def close(self) -> None:
        """关闭所有会话"""
        if self.master_session:
            await self.master_session.close()
        if self.slave_session:
            await self.slave_session.close()


# FastAPI依赖注入
async def get_read_write_session() -> AsyncSession:
    """获取读写会话（只读用从库）"""
    async with ReadWriteSession(db_cluster) as session:
        yield session


async def get_write_session() -> AsyncSession:
    """获取写会话（写操作用主库）"""
    return db_cluster.get_master_session()
```

### 31.2 分库分表策略

```python
# src/database/sharding.py
"""
分库分表策略
"""

from typing import Dict, List
from datetime import datetime
import hashlib

from src.core.logging import get_logger

logger = get_logger(__name__)


class ShardingStrategy:
    """分片策略基类"""

    def get_shard(self, key: str) -> int:
        """获取分片ID"""
        raise NotImplementedError


class HashSharding(ShardingStrategy):
    """
    哈希分片

    根据键的哈希值分配到不同的分片
    """

    def __init__(self, shard_count: int):
        """
        初始化哈希分片

        Args:
            shard_count: 分片数量
        """
        self.shard_count = shard_count

    def get_shard(self, key: str) -> int:
        """根据键的哈希值获取分片ID"""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % self.shard_count


class DateSharding(ShardingStrategy):
    """
    日期分片

    按日期分配到不同的分片（常用于日志、交易记录等时序数据）
    """

    def __init__(self, shard_type: str = 'month'):
        """
        初始化日期分片

        Args:
            shard_type: 分片类型（day/month/year）
        """
        self.shard_type = shard_type

    def get_shard(self, date: datetime) -> str:
        """根据日期获取分片标识"""
        if self.shard_type == 'day':
            return date.strftime('%Y%m%d')
        elif self.shard_type == 'month':
            return date.strftime('%Y%m')
        elif self.shard_type == 'year':
            return date.strftime('%Y')
        else:
            raise ValueError(f"Unknown shard_type: {self.shard_type}")


class SymbolSharding(ShardingStrategy):
    """
    股票代码分片

    根据股票代码首字母或哈希值分配分片
    """

    def __init__(self):
        """初始化股票代码分片"""
        # 按首字母分片：0-9, A-Z
        self.shards = {}
        for i in range(10):
            self.shards[str(i)] = len(self.shards)

        for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.shards[c] = len(self.shards)

    def get_shard(self, symbol: str) -> int:
        """根据股票代码获取分片ID"""
        first_char = symbol[0].upper()

        if first_char.isdigit() or first_char.isalpha():
            return self.shards.get(first_char, 0)
        else:
            return 0


class ShardedTableManager:
    """
    分表管理器

    管理分表的创建、路由和迁移
    """

    def __init__(self):
        """初始化分表管理器"""
        self.table_registry: Dict[str, List[str]] = {}

    def register_sharded_table(
        self,
        base_table_name: str,
        shard_strategy: ShardingStrategy,
    ) -> None:
        """
        注册分表

        Args:
            base_table_name: 基础表名
            shard_strategy: 分片策略
        """
        self.table_registry[base_table_name] = {
            'strategy': shard_strategy,
            'shards': [],
        }

        logger.info(f"注册分表: {base_table_name}", extra={
            'strategy': shard_strategy.__class__.__name__,
        })

    def get_shard_table_name(
        self,
        base_table_name: str,
        shard_key: str,
    ) -> str:
        """
        获取分表名称

        Args:
            base_table_name: 基础表名
            shard_key: 分片键

        Returns:
            分表名称
        """
        if base_table_name not in self.table_registry:
            return base_table_name

        shard_info = self.table_registry[base_table_name]
        strategy = shard_info['strategy']

        shard_id = strategy.get_shard(shard_key)

        return f"{base_table_name}_shard_{shard_id}"

    async def create_shard_tables(
        self,
        base_table_name: str,
        shard_count: int,
        create_sql: str,
    ) -> None:
        """
        创建分表

        Args:
            base_table_name: 基础表名
            shard_count: 分片数量
            create_sql: 建表SQL模板
        """
        for i in range(shard_count):
            table_name = f"{base_table_name}_shard_{i}"
            sql = create_sql.format(table_name=table_name)

            # 执行建表SQL
            logger.info(f"创建分表: {table_name}")

    async def migrate_data_to_shards(
        self,
        source_table: str,
        target_base_table: str,
        batch_size: int = 1000,
    ) -> None:
        """
        迁移数据到分表

        Args:
            source_table: 源表名
            target_base_table: 目标基础表名
            batch_size: 批次大小
        """
        logger.info(f"开始数据迁移: {source_table} -> {target_base_table}_shard_*")

        # 读取源表数据
        offset = 0

        while True:
            # 分批读取
            rows = await self._read_batch(source_table, offset, batch_size)

            if not rows:
                break

            # 写入分表
            for row in rows:
                shard_key = self._extract_shard_key(row)
                target_table = self.get_shard_table_name(
                    target_base_table,
                    shard_key,
                )

                await self._write_to_shard(target_table, row)

            offset += batch_size
            logger.info(f"已迁移 {offset} 条数据")

        logger.info("数据迁移完成")

    def _extract_shard_key(self, row: Dict) -> str:
        """提取分片键"""
        # 根据实际业务逻辑实现
        return row.get('symbol', '')

    async def _read_batch(
        self,
        table: str,
        offset: int,
        limit: int,
    ) -> List[Dict]:
        """读取批次数据"""
        # 实现数据库读取逻辑
        pass

    async def _write_to_shard(
        self,
        table: str,
        row: Dict,
    ) -> None:
        """写入分表数据"""
        # 实现数据库写入逻辑
        pass
```

### 31.3 数据库索引优化

```sql
-- src/database/sql/optimization_indexes.sql

-- ============================================
-- 复合索引优化
-- ============================================

-- 订单表：按交易员+状态+创建时间查询
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_trader_status_time
ON orders(trader_id, status, created_at DESC);

-- 订单表：按股票+交易所+状态查询
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_symbol_exchange_status
ON orders(symbol, exchange, status)
WHERE status IN ('open', 'pending');

-- 成交表：按订单ID查询
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_order_id
ON trades(order_id);

-- 成交表：按交易员+时间查询
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_trader_time
ON trades(trader_id, timestamp DESC);

-- 持仓表：按交易员+股票查询
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_trader_symbol
ON positions(trader_id, symbol)
WHERE quantity > 0;

-- ============================================
-- 部分索引（减少索引大小）
-- ============================================

-- 只索引活跃交易员
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_traders_active
ON traders(id, name, type)
WHERE is_active = true;

-- 只索引未完成订单
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_pending
ON orders(id, trader_id, symbol, side, created_at)
WHERE status IN ('pending', 'open', 'partially_filled');

-- ============================================
-- 表达式索引
-- ============================================

-- 索引持仓盈亏比例
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_pnl_ratio
ON positions(((unrealized_pnl / NULLIF(market_value, 0)) * 100))
WHERE quantity > 0;

-- 索引订单价格偏离度
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_price_deviation
ON orders((((price - avg_fill_price) / NULLIF(avg_fill_price, 0)) * 100))
WHERE status = 'filled' AND avg_fill_price > 0;

-- ============================================
-- 覆盖索引（包含查询所需的所有列）
-- ============================================

-- 订单列表查询优化
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_list_covering
ON orders(trader_id, created_at DESC)
INCLUDE (symbol, side, type, quantity, price, status, filled_quantity);

-- 成交记录查询优化
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_detail_covering
ON trades(trader_id, timestamp DESC)
INCLUDE (symbol, side, quantity, price, commission);

-- ============================================
-- 唯一索引
-- ============================================

-- 防止重复订单（去重）
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_unique_pending
ON orders(trader_id, symbol, exchange, side, status)
WHERE status IN ('pending', 'open');

-- ============================================
-- 全文搜索索引
-- ============================================

-- 交易员名称搜索
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_traders_name_gin
ON traders USING gin(to_tsvector('simple', name));

-- 股票代码搜索
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_symbol_gin
ON positions USING gin(to_tsvector('simple', symbol));

-- ============================================
-- BRIN索引（适合时序数据）
-- ============================================

-- 市场数据（按时间范围查询）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_timestamp_brin
ON market_data USING brin(timestamp);

-- 成交记录（按时间范围查询）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_timestamp_brin
ON trades USING brin(timestamp);

-- ============================================
-- 自动清理分析
-- ============================================

-- 启用自动清理
ALTER SYSTEM SET autovacuum = ON;
ALTER SYSTEM SET autovacuum_vacuum_scale_factor = 0.1;
ALTER SYSTEM SET autovacuum_analyze_scale_factor = 0.05;

-- 配置特定表的自动清理
ALTER TABLE orders SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05,
    autovacuum_vacuum_threshold = 1000,
    autovacuum_analyze_threshold = 500
);
```

---

## 第32章 完整前端组件实现

### 32.1 主仪表盘组件

```typescript
// frontend/src/pages/Dashboard.tsx
import { useState, useEffect } from 'react';
import { useAppStore } from '../store/useAppStore';
import { useWebSocket } from '../hooks/useWebSocket';
import { TraderCard } from '../components/TraderCard';
import { EquityChart } from '../components/charts/EquityChart';
import { PositionTable } from '../components/PositionTable';
import { OrderBook } from '../components/OrderBook';
import { MarketOverview } from '../components/MarketOverview';
import { useNotifications } from '../hooks/useNotifications';

export function Dashboard() {
  const { traders, selectedTraderId, selectTrader } = useAppStore();
  const { showSuccess, showError } = useNotifications();
  const [isLoading, setIsLoading] = useState(true);

  // WebSocket连接
  useWebSocket(`${import.meta.env.VITE_WS_URL}/ws/market`);

  const selectedTrader = traders.find(t => t.id === selectedTraderId);

  useEffect(() => {
    // 加载数据
    const loadDashboard = async () => {
      try {
        setIsLoading(true);
        // 数据加载逻辑
      } catch (error) {
        showError('加载数据失败');
      } finally {
        setIsLoading(false);
      }
    };

    loadDashboard();
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* 页面标题 */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          DeepAlpha 交易系统
        </h1>
        <p className="text-gray-600 mt-2">
          AI驱动的A股港股智能交易平台
        </p>
      </div>

      {/* 市场概览 */}
      <MarketOverview className="mb-8" />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 左侧：交易员列表 */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">交易员列表</h2>
            <div className="space-y-4">
              {traders.map(trader => (
                <TraderCard
                  key={trader.id}
                  trader={trader}
                  onSelect={() => selectTrader(trader.id)}
                  isSelected={selectedTraderId === trader.id}
                />
              ))}
            </div>
          </div>
        </div>

        {/* 中间：选中交易员详情 */}
        <div className="lg:col-span-2 space-y-6">
          {selectedTrader ? (
            <>
              {/* 权益曲线 */}
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">权益曲线</h2>
                <EquityChart traderId={selectedTrader.id} />
              </div>

              {/* 持仓和订单 */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* 持仓表格 */}
                <div className="bg-white rounded-lg shadow p-6">
                  <h2 className="text-xl font-semibold mb-4">当前持仓</h2>
                  <PositionTable traderId={selectedTrader.id} />
                </div>

                {/* 订单簿 */}
                <div className="bg-white rounded-lg shadow p-6">
                  <h2 className="text-xl font-semibold mb-4">订单簿</h2>
                  <OrderBook traderId={selectedTrader.id} />
                </div>
              </div>
            </>
          ) : (
            <div className="bg-white rounded-lg shadow p-12 text-center">
              <svg
                className="mx-auto h-12 w-12 text-gray-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.707.293H19a2 2 0 012-2z"
                />
              </svg>
              <h3 className="mt-4 text-lg font-medium text-gray-900">
                请选择一个交易员
              </h3>
              <p className="mt-2 text-gray-500">
                点击左侧列表查看详情
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

### 32.2 高级图表组件

```typescript
// frontend/src/components/charts/EquityChart.tsx
import { useEffect, useRef } from 'react';
import { usePositions } from '../../hooks/usePositions';
import * as echarts from 'echarts';
import { EChartsOption } from 'echarts';

interface EquityChartProps {
  traderId: string;
}

export function EquityChart({ traderId }: EquityChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const { positions } = usePositions(traderId);

  useEffect(() => {
    if (!chartRef.current) return;

    // 初始化图表
    chartInstance.current = echarts.init(chartRef.current);

    const option: EChartsOption = {
      title: {
        text: '权益曲线',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const date = params[0].axisValue;
          const value = params[0].data;
          return `
            <div>
              <div>日期: ${date}</div>
              <div>权益: ¥${value.toFixed(2)}</div>
            </div>
          `;
        },
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: [],  // 从API获取
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLabel: {
          formatter: (value: number) => `¥${value.toFixed(0)}`,
        },
      },
      series: [
        {
          name: '权益',
          type: 'line',
          smooth: true,
          symbol: 'none',
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
              { offset: 1, color: 'rgba(59, 130, 246, 0.05)' },
            ]),
          },
          lineStyle: {
            color: '#3b82f6',
            width: 2,
          },
          data: [],  // 从API获取
        },
      ],
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true,
      },
    };

    chartInstance.current.setOption(option);

    // 响应式
    const handleResize = () => {
      chartInstance.current?.resize();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
    };
  }, []);

  // 更新数据
  useEffect(() => {
    if (!chartInstance.current) return;

    // 计算权益曲线
    // 这里应该从API获取历史权益数据
    const updateChart = async () => {
      const response = await fetch(`/api/v1/traders/${traderId}/equity-curve`);
      const data = await response.json();

      chartInstance.current?.setOption({
        xAxis: {
          data: data.dates,
        },
        series: [
          {
            data: data.values,
          },
        ],
      });
    };

    updateChart();
  }, [traderId]);

  return <div ref={chartRef} style={{ width: '100%', height: 400 }} />;
}
```

### 32.3 实时行情组件

```typescript
// frontend/src/components/MarketOverview.tsx
import { useState, useEffect } from 'react';
import { useMarketData } from '../../hooks/useMarketData';
import { ArrowUpIcon, ArrowDownIcon } from '@heroicons/react/24/solid';

interface MarketOverviewProps {
  className?: string;
}

export function MarketOverview({ className = '' }: MarketOverviewProps) {
  // 监控的热门股票
  watchlist = ['000001', '000002', '600000', '600519', '00700'];

  const { getMarketData } = useMarketData(watchlist);

  return (
    <div className={`bg-white rounded-lg shadow p-6 ${className}`}>
      <h2 className="text-xl font-semibold mb-4">市场概览</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        {watchlist.map((symbol) => {
          const data = getMarketData(symbol);

          if (!data) {
            return (
              <div
                key={symbol}
                className="border rounded-lg p-4 bg-gray-50"
              >
                <div className="text-sm text-gray-500 mb-2">{symbol}</div>
                <div className="text-gray-400">加载中...</div>
              </div>
            );
          }

          const isPositive = data.change >= 0;

          return (
            <div
              key={symbol}
              className="border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">
                  {symbol}
                </span>
                <span className={`text-xs px-2 py-1 rounded ${
                  data.exchange === 'XSHE' || data.exchange === 'XSHG'
                    ? 'bg-red-100 text-red-800'
                    : 'bg-green-100 text-green-800'
                }`}>
                  {data.exchange}
                </span>
              </div>

              <div className="text-2xl font-bold text-gray-900 mb-1">
                ¥{data.price.toFixed(2)}
              </div>

              <div className={`flex items-center text-sm ${
                isPositive ? 'text-red-600' : 'text-green-600'
              }`}>
                {isPositive ? (
                  <ArrowUpIcon className="h-4 w-4 mr-1" />
                ) : (
                  <ArrowDownIcon className="h-4 w-4 mr-1" />
                )}
                <span>
                  {isPositive ? '+' : ''}{data.change_percent.toFixed(2)}%
                </span>
              </div>

              <div className="text-xs text-gray-500 mt-2">
                成交量: {(data.volume / 10000).toFixed(2)}万手
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

### 32.4 订单管理组件

```typescript
// frontend/src/components/OrderForm.tsx
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useOrders } from '../../hooks/useOrders';
import { useNotifications } from '../../hooks/useNotifications';

// 验证Schema
const orderSchema = z.object({
  trader_id: z.string().min(1, '请选择交易员'),
  symbol: z.string().min(1, '请输入股票代码'),
  exchange: z.enum(['XSHE', 'XSHG', 'XHKG'], {
    errorMap: () => ({ message: '请选择交易所' }),
  }),
  side: z.enum(['buy', 'sell'], {
    errorMap: () => ({ message: '请选择买卖方向' }),
  }),
  type: z.enum(['market', 'limit'], {
    errorMap: () => ({ message: '请选择订单类型' }),
  }),
  quantity: z.number().int().positive().multipleOf(100, {
    message: '数量必须是100的整数倍',
  }),
  price: z.number().positive().optional(),
});

type OrderFormData = z.infer<typeof orderSchema>;

interface OrderFormProps {
  traderId: string;
  onSubmit?: () => void;
}

export function OrderForm({ traderId, onSubmit }: OrderFormProps) {
  const { submitOrder } = useOrders(traderId);
  const { showSuccess, showError } = useNotifications();

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors, isSubmitting },
  } = useForm<OrderFormData>({
    resolver: zodResolver(orderSchema),
    defaultValues: {
      trader_id: traderId,
      side: 'buy',
      type: 'limit',
    },
  });

  const orderType = watch('type');
  const side = watch('side');

  const handleFormSubmit = async (data: OrderFormData) => {
    try {
      await submitOrder(data as any);
      showSuccess('订单提交成功');
      onSubmit?.();
    } catch (error: any) {
      showError(error.message || '订单提交失败');
    }
  };

  return (
    <form onSubmit={handleSubmit(handleFormSubmit)} className="space-y-4">
      {/* 买卖方向选择 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          买卖方向
        </label>
        <div className="grid grid-cols-2 gap-4">
          <button
            type="button"
            onClick={() => setValue('side', 'buy')}
            className={`py-3 px-4 rounded-lg font-medium transition-colors ${
              side === 'buy'
                ? 'bg-red-500 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            买入
          </button>
          <button
            type="button"
            onClick={() => setValue('side', 'sell')}
            className={`py-3 px-4 rounded-lg font-medium transition-colors ${
              side === 'sell'
                ? 'bg-green-500 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            卖出
          </button>
        </div>
      </div>

      {/* 股票代码 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          股票代码
        </label>
        <input
          type="text"
          {...register('symbol')}
          placeholder="例如: 000001"
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        {errors.symbol && (
          <p className="mt-1 text-sm text-red-600">{errors.symbol.message}</p>
        )}
      </div>

      {/* 交易所选择 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          交易所
        </label>
        <select
          {...register('exchange')}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="XSHE">深交所</option>
          <option value="XSHG">上交所</option>
          <option value="XHKG">港交所</option>
        </select>
        {errors.exchange && (
          <p className="mt-1 text-sm text-red-600">{errors.exchange.message}</p>
        )}
      </div>

      {/* 订单类型 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          订单类型
        </label>
        <div className="grid grid-cols-2 gap-4">
          <button
            type="button"
            onClick={() => setValue('type', 'market')}
            className={`py-2 px-4 rounded-lg font-medium transition-colors ${
              orderType === 'market'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            市价单
          </button>
          <button
            type="button"
            onClick={() => setValue('type', 'limit')}
            className={`py-2 px-4 rounded-lg font-medium transition-colors ${
              orderType === 'limit'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            限价单
          </button>
        </div>
      </div>

      {/* 价格（限价单） */}
      {orderType === 'limit' && (
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            委托价格
          </label>
          <input
            type="number"
            step="0.01"
            {...register('price', { valueAsNumber: true })}
            placeholder="输入委托价格"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          {errors.price && (
            <p className="mt-1 text-sm text-red-600">{errors.price.message}</p>
          )}
        </div>
      )}

      {/* 数量 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          数量（手）
        </label>
        <input
          type="number"
          {...register('quantity', { valueAsNumber: true })}
          placeholder="必须是100的整数倍"
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        {errors.quantity && (
          <p className="mt-1 text-sm text-red-600">{errors.quantity.message}</p>
        )}
      </div>

      {/* 提交按钮 */}
      <div className="flex gap-4">
        <button
          type="submit"
          disabled={isSubmitting}
          className="flex-1 bg-blue-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {isSubmitting ? '提交中...' : '提交订单'}
        </button>
        <button
          type="button"
          onClick={() => reset()}
          className="px-6 py-3 border border-gray-300 rounded-lg font-medium text-gray-700 hover:bg-gray-50 transition-colors"
        >
          重置
        </button>
      </div>
    </form>
  );
}
```

---

## 第33章 高级监控与告警

### 33.1 Prometheus自定义指标

```python
# src/monitoring/metrics.py
"""
自定义Prometheus指标
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, Info
from prometheus_client.exposition import generate_latest
from fastapi import Response
from src.core.logging import get_logger

logger = get_logger(__name__)

# ============================================
# 业务指标
# ============================================

# 交易员数量
traders_total = Gauge(
    'traders_total',
    'Total number of traders',
    ['status']  # active, inactive
)

# 订单处理
orders_total = Counter(
    'orders_total',
    'Total orders processed',
    ['trader_id', 'symbol', 'side', 'status']
)

# 订单处理时间
order_processing_duration = Histogram(
    'order_processing_duration_seconds',
    'Order processing duration',
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# 成交金额
trade_amount_total = Counter(
    'trade_amount_total',
    'Total trade amount',
    ['trader_id', 'symbol']
)

# 持仓价值
position_value = Gauge(
    'position_value',
    'Current position value',
    ['trader_id', 'symbol', 'side']
)

# 未实现盈亏
unrealized_pnl = Gauge(
    'unrealized_pnl',
    'Unrealized P&L',
    ['trader_id', 'symbol']
)

# 权益曲线
equity = Gauge(
    'equity',
    'Current equity',
    ['trader_id']
)

# ============================================
# 性能指标
# ============================================

# API请求
api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['endpoint', 'method']
)

# API请求计数
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

# 数据库查询
db_query_duration = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['operation', 'table']
)

# 缓存命中率
cache_hits = Counter('cache_hits_total', 'Cache hits', ['cache_type'])
cache_misses = Counter('cache_misses_total', 'Cache misses', ['cache_type'])

# LLM调用
llm_request_duration = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration',
    ['provider', 'model']
)

llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['provider', 'model', 'status']
)

# ============================================
# 系统指标
# ============================================

# 数据库连接池
db_pool_size = Gauge(
    'db_pool_size',
    'Database connection pool size',
    ['state']  # checkedin, idle
)

# Redis连接
redis_connections = Gauge(
    'redis_connections_total',
    'Total Redis connections'
)

# Celery任务
celery_task_duration = Histogram(
    'celery_task_duration_seconds',
    'Celery task duration',
    ['task_name', 'status']
)

celery_tasks_total = Counter(
    'celery_tasks_total',
    'Total Celery tasks',
    ['task_name', 'status']
)

# WebSocket连接
websocket_connections = Gauge(
    'websocket_connections_total',
    'Total WebSocket connections',
    ['channel']
)

# ============================================
# 错误指标
# ============================================

# 业务错误
business_errors_total = Counter(
    'business_errors_total',
    'Total business errors',
    ['error_type', 'trader_id']
)

# 系统错误
system_errors_total = Counter(
    'system_errors_total',
    'Total system errors',
    ['error_type', 'component']
)


class MetricsService:
    """指标服务"""

    @staticmethod
    def record_order(order, status: str) -> None:
        """记录订单指标"""
        orders_total.labels(
            trader_id=order.trader_id,
            symbol=order.symbol,
            side=order.side,
            status=status
        ).inc()

        logger.debug("记录订单指标", extra={
            'order_id': order.id,
            'status': status,
        })

    @staticmethod
    def record_trade(trade) -> None:
        """记录成交指标"""
        amount = trade.quantity * trade.price

        trade_amount_total.labels(
            trader_id=trade.trader_id,
            symbol=trade.symbol
        ).inc(amount)

        logger.debug("记录成交指标", extra={
            'trade_id': trade.id,
            'amount': amount,
        })

    @staticmethod
    def update_position_metrics(position) -> None:
        """更新持仓指标"""
        position_value.labels(
            trader_id=position.trader_id,
            symbol=position.symbol,
            side='long' if position.quantity > 0 else 'short'
        ).set(position.market_value)

        unrealized_pnl.labels(
            trader_id=position.trader_id,
            symbol=position.symbol
        ).set(position.unrealized_pnl)

    @staticmethod
    def update_equity(trader_id: str, equity_value: float) -> None:
        """更新权益指标"""
        equity.labels(trader_id=trader_id).set(equity_value)

    @staticmethod
    def record_api_request(endpoint: str, method: str, duration: float, status: int) -> None:
        """记录API请求指标"""
        api_request_duration.labels(
            endpoint=endpoint,
            method=method
        ).observe(duration)

        api_requests_total.labels(
            endpoint=endpoint,
            method=method,
            status=status
        ).inc()

    @staticmethod
    def record_llm_request(provider: str, model: str, duration: float, success: bool) -> None:
        """记录LLM请求指标"""
        llm_request_duration.labels(
            provider=provider,
            model=model
        ).observe(duration)

        llm_requests_total.labels(
            provider=provider,
            model=model,
            status='success' if success else 'error'
        ).inc()

    @staticmethod
    def record_cache_hit(cache_type: str) -> None:
        """记录缓存命中"""
        cache_hits.labels(cache_type=cache_type).inc()

    @staticmethod
    def record_cache_miss(cache_type: str) -> None:
        """记录缓存未命中"""
        cache_misses.labels(cache_type=cache_type).inc()

    @staticmethod
    def record_business_error(error_type: str, trader_id: str = None) -> None:
        """记录业务错误"""
        labels = {'error_type': error_type}
        if trader_id:
            labels['trader_id'] = trader_id

        business_errors_total.labels(**labels).inc()

    @staticmethod
    def record_system_error(error_type: str, component: str) -> None:
        """记录系统错误"""
        system_errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()


# FastAPI端点：暴露指标
@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus指标端点"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

### 33.2 告警规则配置

```yaml
# config/prometheus/rules/deepalpha-alerts.yaml
groups:
  - name: deepalpha-business
    interval: 30s
    rules:
      # ============================================
      # 业务告警
      # ============================================

      - alert: HighOrderRejectionRate
        expr: |
          rate(orders_total{status="rejected"}[5m]) /
          rate(orders_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
          category: business
        annotations:
          summary: "订单拒绝率过高"
          description: "5分钟内订单拒绝率超过10%"

      - alert: TraderLossLimitExceeded
        expr: |
          unrealized_pnl / (
            position_value + abs(unrealized_pnl))
          ) < -0.05
        for: 1m
        labels:
          severity: critical
          category: business
        annotations:
          summary: "交易员亏损超限"
          description: "交易员 {{ $trader_id }} 未实现亏损超过5%"

      - alert: LowWinRate
        expr: |
          (
            sum(rate(trade_amount_total{status="profit"}[1d])) /
            (sum(rate(trade_amount_total[1d])) + 0.01)
          ) < 0.4
        for: 10m
        labels:
          severity: warning
          category: business
        annotations:
          summary: "胜率过低"
          description: "交易员 {{ $trader_id }} 24小时胜率低于40%"

  - name: deepalpha-performance
    interval: 30s
    rules:
      # ============================================
      # 性能告警
      # ============================================

      - alert: HighAPILatency
        expr: |
          histogram_quantile(0.95, api_request_duration_seconds) > 1
        for: 5m
        labels:
          severity: warning
          category: performance
        annotations:
          summary: "API延迟过高"
          description: "95分位API响应延迟超过1秒: {{ $endpoint }}"

      - alert: HighErrorRate
        expr: |
          rate(api_requests_total{status=~"5.."}[5m]) /
          rate(api_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
          category: performance
        annotations:
          summary: "错误率过高"
          description: "5分钟内API错误率超过5%: {{ $endpoint }}"

      - alert: SlowDatabaseQuery
        expr: |
          histogram_quantile(0.95, db_query_duration_seconds) > 0.5
        for: 10m
        labels:
          severity: warning
          category: performance
        annotations:
          summary: "数据库查询缓慢"
          description: "95分位查询时间超过500ms: {{ $table }}"

  - name: deepalpha-system
    interval: 30s
    rules:
      # ============================================
      # 系统告警
      # ============================================

      - alert: DatabaseConnectionPoolExhausted
        expr: |
          db_pool_size{state="checkedin"} / db_pool_size{state="idle"} > 0.9
        for: 5m
        labels:
          severity: critical
          category: system
        annotations:
          summary: "数据库连接池耗尽"
          description: "使用中的连接占比超过90%"

      - alert: HighLLMErrorRate
        expr: |
          rate(llm_requests_total{status="error"}[5m]) /
          rate(llm_requests_total[5m]) > 0.2
        for: 5m
        labels:
          severity: warning
          category: system
        annotations:
          summary: "LLM错误率过高"
          description: "LLM提供商 {{ $provider }} 错误率超过20%"

      - alert: LowCacheHitRate
        expr: |
          cache_hits_total / (cache_hits_total + cache_misses_total) < 0.8
        for: 15m
        labels:
          severity: warning
          category: system
        annotations:
          summary: "缓存命中率过低"
          description: "缓存 {{ $cache_type }} 命中率低于80%"

  - name: deepalpha-security
    interval: 60s
    rules:
      # ============================================
      # 安全告警
      # ============================================

      - alert: TooManyFailedAuthAttempts
        expr: |
          rate(api_requests_total{endpoint="/api/v1/auth/login",status="401"}[5m]) > 10
        for: 5m
        labels:
          severity: warning
          category: security
        annotations:
          summary: "频繁的认证失败"
          description: "检测到来自单个IP的频繁登录失败"

      - alert: AbnormalOrderPattern
        expr: |
          rate(orders_total[1m]) > 10
        for: 2m
        labels:
          severity: critical
          category: security
        annotations:
          summary: "异常的订单模式"
          description: "1分钟内订单数量超过10个"
```

### 33.3 告警通知服务

```python
# src/services/alert_service.py
"""
告警通知服务
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Alert:
    """告警对象"""
    name: str
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'business', 'performance', 'system', 'security'
    summary: str
    description: str
    labels: Dict[str, str]
    timestamp: datetime
    fingerprint: str  # 用于去重


class AlertNotificationService:
    """
    告警通知服务

    支持多种通知渠道：
    1. 邮件
    2. 短信
    3. 钉钉
    4. 微信
    5. Slack
    6. Webhook
    """

    def __init__(self):
        """初始化告警通知服务"""
        # 告警去重（防止重复发送）
        self.recent_alerts: Dict[str, datetime] = {}
        self.dedup_window = timedelta(minutes=5)

        # 通知渠道配置
        self.channels = {
            'email': settings.ALERT_EMAIL_ENABLED,
            'sms': settings.ALERT_SMS_ENABLED,
            'dingtalk': settings.ALERT_DINGTALK_ENABLED,
            'wechat': settings.ALERT_WECHAT_ENABLED,
            'slack': settings.ALERT_SLACK_ENABLED,
            'webhook': settings.ALERT_WEBHOOK_URL,
        }

        logger.info("告警通知服务初始化", extra=self.channels)

    async def send_alert(self, alert: Alert) -> bool:
        """
        发送告警

        Args:
            alert: 告警对象

        Returns:
            是否发送成功
        """
        # 去重检查
        if self._is_duplicate(alert):
            logger.info("告警重复，跳过发送", extra={
                'alert_name': alert.name,
                'fingerprint': alert.fingerprint,
            })
            return False

        # 记录告警
        self.recent_alerts[alert.fingerprint] = datetime.now()

        # 清理过期记录
        self._cleanup_recent_alerts()

        # 根据严重程度选择通知渠道
        channels = self._get_channels_for_severity(alert.severity)

        # 并发发送到各个渠道
        tasks = []
        for channel in channels:
            if self.channels.get(channel):
                tasks.append(self._send_to_channel(channel, alert))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 检查是否有至少一个成功
            success_count = sum(1 for r in results if r is True or not isinstance(r, Exception))

            logger.info("告警发送完成", extra={
                'alert_name': alert.name,
                'channels': channels,
                'success_count': success_count,
                'total_count': len(tasks),
            })

            return success_count > 0

        return False

    def _is_duplicate(self, alert: Alert) -> bool:
        """检查是否为重复告警"""
        if alert.fingerprint in self.recent_alerts:
            last_time = self.recent_alerts[alert.fingerprint]
            if datetime.now() - last_time < self.dedup_window:
                return True
        return False

    def _cleanup_recent_alerts(self) -> None:
        """清理过期的告警记录"""
        cutoff = datetime.now() - self.dedup_window

        expired = [
            fp for fp, time in self.recent_alerts.items()
            if time < cutoff
        ]

        for fp in expired:
            del self.recent_alerts[fp]

    def _get_channels_for_severity(self, severity: str) -> List[str]:
        """根据严重程度获取通知渠道"""
        if severity == 'critical':
            return ['email', 'sms', 'dingtalk', 'slack', 'webhook']
        elif severity == 'warning':
            return ['email', 'dingtalk', 'slack']
        else:
            return ['email']

    async def _send_to_channel(self, channel: str, alert: Alert) -> bool:
        """发送到指定渠道"""
        try:
            if channel == 'email':
                return await self._send_email(alert)
            elif channel == 'sms':
                return await self._send_sms(alert)
            elif channel == 'dingtalk':
                return await self._send_dingtalk(alert)
            elif channel == 'wechat':
                return await self._send_wechat(alert)
            elif channel == 'slack':
                return await self._send_slack(alert)
            elif channel == 'webhook':
                return await self._send_webhook(alert)
            else:
                logger.warning(f"未知的通知渠道: {channel}")
                return False

        except Exception as e:
            logger.error(f"发送告警到 {channel} 失败", extra={
                'channel': channel,
                'error': str(e),
            })
            return False

    async def _send_email(self, alert: Alert) -> bool:
        """发送邮件告警"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = settings.EMAIL_FROM
            msg['To'] = settings.ALERT_EMAIL_TO
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.summary}"

            # 构建邮件内容
            html = self._build_email_html(alert)
            msg.attach(MIMEText(html, 'html'))

            # 发送邮件
            with smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT) as server:
                server.starttls()
                server.login(settings.EMAIL_USER, settings.EMAIL_PASSWORD)
                server.send_message(msg)

            logger.info("邮件告警发送成功", extra={
                'alert_name': alert.name,
            })
            return True

        except Exception as e:
            logger.error("邮件告警发送失败", extra={
                'alert_name': alert.name,
                'error': str(e),
            })
            return False

    async def _send_dingtalk(self, alert: Alert) -> bool:
        """发送钉钉告警"""
        import aiohttp

        url = settings.DINGTALK_WEBHOOK

        # 构建消息
        if alert.severity == 'critical':
            color = '#FF0000'
        elif alert.severity == 'warning':
            color = '#FFA500'
        else:
            color = '#00FF00'

        message = {
            "msgtype": "markdown",
            "markdown": {
                "title": f"[{alert.severity.upper()}] {alert.summary}",
                "text": f"""
### 告警详情

**类别**: {alert.category}
**描述**: {alert.description}
**时间**: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

### 标签
{self._format_labels(alert.labels)}
                """
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=message) as response:
                    if response.status == 200:
                        logger.info("钉钉告警发送成功")
                        return True
                    else:
                        logger.warning(f"钉钉告警失败: {response.status}")
                        return False

        except Exception as e:
            logger.error("钉钉告警发送失败", extra={'error': str(e)})
            return False

    def _build_email_html(self, alert: Alert) -> str:
        """构建邮件HTML"""
        severity_color = {
            'critical': '#dc2626',
            'warning': '#d97706',
            'info': '#2563eb',
        }.get(alert.severity, '#6b7280')

        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
        }}
        .alert-box {{
            border-left: 4px solid {severity_color};
            padding: 20px;
            margin: 20px 0;
            background-color: #f9fafb;
        }}
        .alert-title {{
            font-size: 24px;
            font-weight: bold;
            color: {severity_color};
            margin-bottom: 10px;
        }}
        .alert-info {{
            margin: 5px 0;
        }}
        .label {{
            display: inline-block;
            background-color: #e5e7eb;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin: 2px;
        }}
    </style>
</head>
<body>
    <div class="alert-box">
        <div class="alert-title">[{alert.severity.upper()}] {alert.summary}</div>
        <div class="alert-info"><strong>类别:</strong> {alert.category}</div>
        <div class="alert-info"><strong>描述:</strong> {alert.description}</div>
        <div class="alert-info"><strong>时间:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>

        <div style="margin-top: 20px;">
            <strong>标签:</strong>
            <div style="margin-top: 10px;">
                {self._format_labels_html(alert.labels)}
            </div>
        </div>
    </div>
</body>
</html>
        """

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """格式化标签为文本"""
        return ' '.join([f"{k}={v}" for k, v in labels.items()])

    def _format_labels_html(self, labels: Dict[str, str]) -> str:
        """格式化标签为HTML"""
        return ''.join([f'<span class="label">{k}={v}</span>' for k, v in labels.items()])

    async def _send_slack(self, alert: Alert) -> bool:
        """发送Slack告警"""
        import aiohttp

        url = settings.SLACK_WEBHOOK_URL

        # Slack颜色
        color = {
            'critical': 'danger',
            'warning': 'warning',
            'info': 'good',
        }.get(alert.severity, 'good')

        attachment = {
            "color": color,
            "title": f"[{alert.severity.upper()}] {alert.summary}",
            "text": alert.description,
            "fields": [
                {
                    "title": "Category",
                    "value": alert.category,
                    "short": True
                },
                {
                    "title": "Time",
                    "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    "short": True
                },
            ],
            "footer": self._format_labels(alert.labels),
        }

        message = {
            "attachments": [attachment]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=message) as response:
                    return response.status == 200

        except Exception as e:
            logger.error("Slack告警发送失败", extra={'error': str(e)})
            return False

    async def _send_webhook(self, alert: Alert) -> bool:
        """发送Webhook告警"""
        import aiohttp

        url = settings.ALERT_WEBHOOK_URL

        if not url:
            return False

        payload = alert.__dict__

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    return response.status == 200

        except Exception as e:
            logger.error("Webhook告警发送失败", extra={'error': str(e)})
            return False

    async def _send_sms(self, alert: Alert) -> bool:
        """发送短信告警"""
        # 实现短信发送逻辑
        pass

    async def _send_wechat(self, alert: Alert) -> bool:
        """发送微信告警"""
        # 实现微信发送逻辑
        pass
```

---

## 第34章 性能优化详解

### 34.1 缓存策略实现

```python
# src/cache/cache_manager.py
"""
缓存管理器
"""

import json
import hashlib
from typing import Optional, Any, TypeVar, Callable
from datetime import datetime, timedelta
from functools import wraps

from redis.asyncio import Redis
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CacheManager:
    """
    缓存管理器

    支持多级缓存：
    1. 内存缓存（本地）
    2. Redis缓存（分布式）
    """

    def __init__(
        self,
        redis_url: str = None,
        default_ttl: int = 300,
    ):
        """
        初始化缓存管理器

        Args:
            redis_url: Redis连接URL
            default_ttl: 默认过期时间（秒）
        """
        self.default_ttl = default_ttl
        self.redis_client: Optional[Redis] = None

        # 内存缓存
        self.memory_cache: Dict[str, tuple[Any, datetime]] = {}

        if redis_url:
            self.redis_client = Redis.from_url(
                redis_url,
                encoding='utf-8',
                decode_responses=True,
            )

        logger.info("缓存管理器初始化", extra={
            'default_ttl': default_ttl,
            'redis_enabled': self.redis_client is not None,
        })

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """生成缓存键"""
        # 将参数序列化为字符串
        key_parts = [prefix]

        if args:
            key_parts.extend(str(arg) for arg in args)

        if kwargs:
            sorted_items = sorted(kwargs.items())
            key_parts.extend(f"{k}={v}" for k, v in sorted_items)

        key_string = ':'.join(key_parts)

        # 使用MD5哈希避免键过长
        return hashlib.md5(key_string.encode()).hexdigest()

    async def get(
        self,
        prefix: str,
        *args,
        default: T = None,
        **kwargs
    ) -> Optional[T]:
        """
        获取缓存

        Args:
            prefix: 缓存前缀
            *args: 位置参数
            default: 默认值
            **kwargs: 关键字参数

        Returns:
            缓存值或默认值
        """
        key = self._generate_key(prefix, *args, **kwargs)

        # 先查内存缓存
        if key in self.memory_cache:
            value, expiry = self.memory_cache[key]

            # 检查是否过期
            if datetime.now() < expiry:
                logger.debug("内存缓存命中", extra={'key': key})
                return value
            else:
                # 过期删除
                del self.memory_cache[key]

        # 再查Redis缓存
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)

                if value:
                    # 写入内存缓存
                    parsed_value = json.loads(value)
                    expiry = datetime.now() + timedelta(seconds=self.default_ttl)

                    self.memory_cache[key] = (parsed_value, expiry)

                    logger.debug("Redis缓存命中", extra={'key': key})
                    return parsed_value

            except Exception as e:
                logger.error("Redis缓存读取失败", extra={'error': str(e)})

        return default

    async def set(
        self,
        prefix: str,
        value: Any,
        *args,
        ttl: int = None,
        **kwargs
    ) -> None:
        """
        设置缓存

        Args:
            prefix: 缓存前缀
            value: 缓存值
            *args: 位置参数
            ttl: 过期时间（秒）
            **kwargs: 关键字参数
        """
        key = self._generate_key(prefix, *args, **kwargs)
        ttl = ttl or self.default_ttl

        # 序列化值
        serialized_value = json.dumps(value)

        # 写入内存缓存
        expiry = datetime.now() + timedelta(seconds=ttl)
        self.memory_cache[key] = (value, expiry)

        # 限制内存缓存大小
        if len(self.memory_cache) > 10000:
            # 删除最旧的20%
            sorted_keys = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1][1]
            )
            for k, _ in sorted_keys[:2000]:
                del self.memory_cache[k]

        # 写入Redis缓存
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    key,
                    ttl,
                    serialized_value
                )

                logger.debug("缓存写入成功", extra={
                    'key': key,
                    'ttl': ttl,
                })

            except Exception as e:
                logger.error("Redis缓存写入失败", extra={'error': str(e)})

    async def delete(self, prefix: str, *args, **kwargs) -> None:
        """
        删除缓存

        Args:
            prefix: 缓存前缀
            *args: 位置参数
            **kwargs: 关键字参数
        """
        key = self._generate_key(prefix, *args, **kwargs)

        # 删除内存缓存
        if key in self.memory_cache:
            del self.memory_cache[key]

        # 删除Redis缓存
        if self.redis_client:
            try:
                await self.redis_client.delete(key)

                logger.debug("缓存删除成功", extra={'key': key})

            except Exception as e:
                logger.error("Redis缓存删除失败", extra={'error': str(e)})

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        按模式批量删除缓存

        Args:
            pattern: 匹配模式

        Returns:
            删除的数量
        """
        count = 0

        # 删除匹配的内存缓存
        keys_to_delete = [
            k for k in self.memory_cache.keys()
            if pattern in k
        ]

        for k in keys_to_delete:
            del self.memory_cache[k]
            count += 1

        # 删除匹配的Redis缓存
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(f"*{pattern}*")

                if keys:
                    count += await self.redis_client.delete(*keys)

            except Exception as e:
                logger.error("Redis批量删除失败", extra={'error': str(e)})

        logger.info("批量缓存删除完成", extra={
            'pattern': pattern,
            'count': count,
        })

        return count


# 全局缓存管理器实例
cache_manager = CacheManager(
    redis_url=settings.REDIS_URL,
    default_ttl=settings.CACHE_TTL,
)


# 缓存装饰器
def cached(
    prefix: str,
    ttl: int = None,
    exclude_params: list = None,
):
    """
    缓存装饰器

    Args:
        prefix: 缓存前缀
        ttl: 过期时间（秒）
        exclude_params: 排除的参数名列表
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # 构建缓存键
            cache_args = []
            cache_kwargs = {}

            # 处理位置参数（跳过self）
            if args:
                # 假设第一个参数是self
                cache_args = args[1:]

            # 处理关键字参数
            if exclude_params:
                cache_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in exclude_params
                }
            else:
                cache_kwargs = kwargs

            # 尝试从缓存获取
            cached_value = await cache_manager.get(
                prefix,
                *cache_args,
                **cache_kwargs
            )

            if cached_value is not None:
                return cached_value

            # 调用函数
            result = await func(*args, **kwargs)

            # 写入缓存
            await cache_manager.set(
                prefix,
                result,
                *cache_args,
                ttl=ttl,
                **cache_kwargs
            )

            return result

        return wrapper
    return decorator


# 使用示例
@cached(prefix='trader:info', ttl=600)
async def get_trader_info(trader_id: str) -> Dict:
    """获取交易员信息（带缓存）"""
    # 从数据库查询
    pass


@cached(prefix='market:bar', ttl=60)
async def get_bar_data(
    symbol: str,
    exchange: str,
    start_date: datetime,
    end_date: datetime,
) -> List[Dict]:
    """获取K线数据（带缓存）"""
    # 从数据库查询
    pass
```

### 34.2 数据库查询优化

```python
# src/database/optimizations.py
"""
数据库查询优化
"""

from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import func, select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from src.models.trader import TraderModel
from src.models.order import OrderModel
from src.models.position import PositionModel
from src.models.trade import TradeModel
from src.core.logging import get_logger

logger = get_logger(__name__)


class QueryOptimizer:
    """查询优化器"""

    @staticmethod
    async def get_trader_with_positions(
        session: AsyncSession,
        trader_id: str,
    ) -> Optional[TraderModel]:
        """
        获取交易员及其持仓（一次查询）

        优化点：
        1. 使用joinedload避免N+1查询
        2. 只查询必要的字段
        """
        query = (
            select(TraderModel)
            .options(
                joinedload(TraderModel.positions).selectinload(PositionModel.trades)
            )
            .where(TraderModel.id == trader_id)
        )

        result = await session.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_orders_with_filters(
        session: AsyncSession,
        trader_id: str = None,
        symbol: str = None,
        status: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[OrderModel], int]:
        """
        获取订单列表（带过滤和分页）

        优化点：
        1. 使用索引字段过滤
        2. 使用count简化总数查询
        3. 限制返回字段
        """
        # 构建查询条件
        conditions = []

        if trader_id:
            conditions.append(OrderModel.trader_id == trader_id)

        if symbol:
            conditions.append(OrderModel.symbol == symbol)

        if status:
            conditions.append(OrderModel.status == status)

        if start_date:
            conditions.append(OrderModel.created_at >= start_date)

        if end_date:
            conditions.append(OrderModel.created_at <= end_date)

        # 构建查询
        query = select(OrderModel)

        if conditions:
            query = query.where(*conditions)

        # 先查总数
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await session.execute(count_query)
        total = total_result.scalar()

        # 查询数据（带分页）
        query = query.order_by(OrderModel.created_at.desc())
        query = query.limit(limit).offset(offset)

        result = await session.execute(query)
        orders = result.scalars().all()

        return orders, total

    @staticmethod
    async def batch_update_positions(
        session: AsyncSession,
        updates: List[dict],
    ) -> int:
        """
        批量更新持仓

        优化点：
        1. 使用bulk_update_mappings
        2. 单次事务提交
        """
        if not updates:
            return 0

        # 构建批量更新
        await session.execute(
            update(PositionModel)
            .where(PositionModel.id.in_(u['id'] for u in updates))
            .values(
                current_price=bindparam('current_price'),
                unrealized_pnl=bindparam('unrealized_pnl'),
                market_value=bindparam('market_value'),
                updated_at=datetime.now(),
            )
        )

        return len(updates)

    @staticmethod
    async def get_trader_statistics(
        session: AsyncSession,
        trader_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        获取交易员统计数据

        优化点：
        1. 使用聚合函数在数据库层面计算
        2. 减少数据传输量
        """
        # 计算日期范围
        start_date = datetime.now() - timedelta(days=days)

        # 总成交额
        trade_amount = await session.execute(
            select(
                func.sum(TradeModel.quantity * TradeModel.price)
            )
            .where(TradeModel.trader_id == trader_id)
            .where(TradeModel.timestamp >= start_date)
        )

        # 总成交次数
        trade_count = await session.execute(
            select(func.count())
            .select_from(TradeModel)
            .where(TradeModel.trader_id == trader_id)
            .where(TradeModel.timestamp >= start_date)
        )

        # 盈利交易数
        winning_trades = await session.execute(
            select(func.count())
            .select_from(TradeModel)
            .where(TradeModel.trader_id == trader_id)
            .where(TradeModel.timestamp >= start_date)
            .where(TradeModel.realized_pnl > 0)
        )

        # 总盈亏
        total_pnl = await session.execute(
            select(func.sum(TradeModel.realized_pnl))
            .where(TradeModel.trader_id == trader_id)
            .where(TradeModel.timestamp >= start_date)
        )

        return {
            'trade_amount': trade_amount.scalar() or 0,
            'trade_count': trade_count.scalar() or 0,
            'winning_trades': winning_trades.scalar() or 0,
            'total_pnl': total_pnl.scalar() or 0,
            'win_rate': (
                winning_trades.scalar() / trade_count.scalar()
                if trade_count.scalar() > 0 else 0
            ),
        }
```

### 34.3 连接池优化

```python
# src/database/pool.py
"""
连接池优化
"""

from typing import Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.pool import QueuePool

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class ConnectionPoolOptimizer:
    """连接池优化器"""

    @staticmethod
    def create_optimized_engine(database_url: str) -> AsyncEngine:
        """
        创建优化的数据库引擎

        优化点：
        1. 预ping连接健康检查
        2. 连接回收时间
        3. 连接池大小
        4. 超时设置
        """
        engine = create_async_engine(
            database_url,
            # 连接池配置
            poolclass=QueuePool,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_timeout=30,
            pool_recycle=3600,  # 1小时回收连接

            # 连接健康检查
            pool_pre_ping=True,

            # 连接创建超时
            connect_args={
                "connect_timeout": 10,
            },

            # 查询优化
            echo=settings.DEBUG,
            echo_pool=False,

            # 日志
            logging_name='sqlalchemy.engine',
        )

        logger.info("数据库引擎创建成功", extra={
            'pool_size': settings.DB_POOL_SIZE,
            'max_overflow': settings.DB_MAX_OVERFLOW,
        })

        return engine

    @staticmethod
    async def test_pool_performance(
        engine: AsyncEngine,
        concurrent_tasks: int = 100,
    ) -> Dict[str, float]:
        """
        测试连接池性能

        Args:
            engine: 数据库引擎
            concurrent_tasks: 并发任务数

        Returns:
            性能指标
        """
        import time

        results = {
            'total_time': 0,
            'avg_time': 0,
            'min_time': float('inf'),
            'max_time': 0,
            'success_rate': 0,
        }

        start_time = time.time()
        times = []
        success_count = 0

        async def query_task(task_id: int):
            """查询任务"""
            try:
                task_start = time.time()

                async with AsyncSession(engine) as session:
                    await session.execute(select(1))

                task_time = time.time() - task_start
                times.append(task_time)
                nonlocal success_count
                success_count += 1

            except Exception as e:
                logger.error(f"任务 {task_id} 失败", extra={'error': str(e)})

        # 并发执行
        tasks = [query_task(i) for i in range(concurrent_tasks)]
        await asyncio.gather(*tasks)

        results['total_time'] = time.time() - start_time

        if times:
            results['avg_time'] = sum(times) / len(times)
            results['min_time'] = min(times)
            results['max_time'] = max(times)

        results['success_rate'] = success_count / concurrent_tasks

        logger.info("连接池性能测试完成", extra=results)

        return results
```

---

## 第35章 生产部署检查清单

### 35.1 上线前检查

```markdown
# DeepAlpha 上线前检查清单

## 1. 代码质量

- [ ] 代码审查完成
- [ ] 单元测试覆盖率 >= 80%
- [ ] 集成测试全部通过
- [ ] 性能测试通过
- [ ] 安全扫描无高危漏洞
- [ ] 代码格式化检查通过
- [ ] 类型检查无错误

## 2. 数据库

- [ ] 数据库迁移脚本准备
- [ ] 索引创建完成
- [ ] 数据备份方案确认
- [ ] 读写分离配置完成
- [ ] 连接池参数调优
- [ ] 慢查询优化完成
- [ ] 数据保留策略设置

## 3. 缓存

- [ ] Redis配置完成
- [ ] 缓存预热脚本准备
- [ ] 缓存过期策略设置
- [ ] 缓存穿透防护
- [ ] 缓存雪崩防护
- [ ] 缓存击穿防护

## 4. 监控告警

- [ ] Prometheus配置完成
- [ ] Grafana仪表板创建
- [ ] 告警规则配置
- [ ] 告警通知渠道测试
- [ ] 日志收集配置
- [ ] 链路追踪配置

## 5. 安全

- [ ] JWT密钥配置
- [ ] 数据加密配置
- [ ] CORS配置正确
- [ ] SQL注入防护
- [ ] XSS防护
- [ ] CSRF防护
- [ ] 速率限制配置
- [ ] SSL证书配置

## 6. 性能

- [ ] 负载测试完成
- [ ] 并发测试通过
- [ ] 资源限制配置
- [ ] HPA配置完成
- [ ] CDN配置完成
- [ ] 静态资源压缩

## 7. 部署

- [ ] Docker镜像构建
- [ ] Kubernetes配置文件
- [ ] Ingress配置
- [ ] Service配置
- [ ] ConfigMap/Secret配置
- [ ] 滚动更新策略
- [ ] 备份回滚方案

## 8. 文档

- [ ] API文档更新
- [ ] 部署文档完成
- [ ] 运维手册编写
- [ ] 应急预案制定
- [ ] 故障排查指南

## 9. 测试

- [ ] 功能测试通过
- [ ] 压力测试通过
- [ ] 故障切换测试
- [ ] 数据恢复测试
- [ ] 回滚测试
```

### 35.2 运维手册

```markdown
# DeepAlpha 运维手册

## 1. 日常运维

### 每日检查项

- 服务健康状态
  ```bash
  kubectl get pods -n deepalpha-production
  ```

- 资源使用情况
  ```bash
  kubectl top nodes
  kubectl top pods -n deepalpha-production
  ```

- 日志错误检查
  ```bash
  kubectl logs -n deepalpha-production -l app=deepalpha --since=1h | grep ERROR
  ```

- 数据库连接
  ```bash
  kubectl exec -n deepalpha-production deployment/deepalpha-api -- pg_isready
  ```

### 每周检查项

- 数据库性能分析
  - 慢查询日志
  - 索引使用情况
  - 表空间使用情况

- 安全漏洞扫描
  - 依赖包更新
  - 安全补丁应用

## 2. 故障处理

### API无响应

1. 检查Pod状态
2. 检查日志
3. 检查资源限制
4. 检查数据库连接

### 数据库连接失败

1. 检查数据库Pod状态
2. 检查连接数
3. 检查网络连接
4. 重启应用Pod

### 内存泄漏

1. 重启Pod
2. 调整内存限制
3. 分析内存dump

## 3. 性能优化

### 数据库优化

- 定期VACUUM
- 更新统计信息
- 重建碎片化索引

### 缓存优化

- 调整缓存大小
- 优化缓存策略
- 预热热点数据

### 应用优化

- 增加Worker数量
- 调整超时时间
- 优化查询语句
```

---

## 总结

本高级实现细节文档涵盖：

**第31章 - 高级数据库优化**
- 读写分离架构（主从复制、负载均衡）
- 分库分表策略（哈希分片、日期分片、股票代码分片）
- 高级索引优化（复合索引、部分索引、表达式索引、覆盖索引、BRIN索引）

**第32章 - 完整前端实现**
- 主仪表盘组件
- 高级图表组件
- 实时行情组件
- 订单管理组件

**第33章 - 高级监控告警**
- Prometheus自定义指标（业务、性能、系统、错误）
- 告警规则配置（业务、性能、系统、安全）
- 多渠道告警通知（邮件、短信、钉钉、微信、Slack、Webhook）

**第34章 - 性能优化**
- 多级缓存策略（内存缓存、Redis缓存）
- 缓存装饰器使用
- 数据库查询优化（N+1问题、批量操作、聚合函数）
- 连接池优化配置

**第35章 - 生产部署检查清单**
- 上线前检查清单（代码、数据库、缓存、监控、安全、性能、部署、文档、测试）
- 运维手册（日常检查、故障处理、性能优化）

---

*本文档持续更新中...*

*最后更新: 2026-01-05*
