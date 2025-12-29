# StockAgents: A股/港股AI智能交易系统融合设计方案

**方案版本**: v1.0
**设计日期**: 2025年12月29日
**设计师**: Claude (高级软件架构师)
**基础项目**: NOFX + TradingAgents-CN

---

## 目录

1. [项目概述](#1-项目概述)
2. [核心设计理念](#2-核心设计理念)
3. [系统架构设计](#3-系统架构设计)
4. [模块融合方案](#4-模块融合方案)
5. [A股/港股特殊处理](#5-a股港股特殊处理)
6. [数据流设计](#6-数据流设计)
7. [AI决策引擎](#7-ai决策引擎)
8. [交易执行层](#8-交易执行层)
9. [前端设计](#9-前端设计)
10. [部署方案](#10-部署方案)
11. [开发路线图](#11-开发路线图)
12. [风险评估](#12-风险评估)

---

## 1. 项目概述

### 1.1 项目定位

**StockAgents** 是一个基于多智能体大语言模型的A股/港股智能交易系统，融合 NOFX 的高性能执行引擎和 TradingAgents-CN 的多智能体分析框架，为中国股票市场提供AI驱动的交易决策和自动执行能力。

### 1.2 核心价值主张

| 价值维度 | 说明 | 来源 |
|----------|------|------|
| **智能决策** | 多智能体协作（技术+基本面+新闻+情绪） | TradingAgents-CN |
| **自动执行** | 券商API对接，全自动交易 | NOFX |
| **多市场支持** | A股 + 港股统一框架 | 两者融合 |
| **实时监控** | WebSocket实时行情 + 9:30-15:00交易时段 | NOFX改造 |
| **中文优化** | 完整中文界面、文档、提示词 | TradingAgents-CN |
| **合规风控** | T+1限制、涨跌停检测、ST股限制 | 新增模块 |

### 1.3 项目特色

```
┌─────────────────────────────────────────────────────────────┐
│                    StockAgents 核心特色                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✅ 多智能体分析       技术面 + 基本面 + 新闻面 + 情绪面        │
│  ✅ 自动交易执行       券商API对接，全自动下单                  │
│  ✅ A股/港股统一       一套代码，两个市场                      │
│  ✅ 合规风控          T+1、涨跌停、ST股、最小单位               │
│  ✅ 实时行情          WebSocket + Level-2数据                 │
│  ✅ 中文本地化        完整中文体验                             │
│  ✅ 多LLM支持         DeepSeek、Qwen、文心一言等              │
│  ✅ 历史回测          完整回测系统，验证策略                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心设计理念

### 2.1 设计原则

#### 2.1.1 技术栈选择原则

| 模块 | 技术选择 | 理由 |
|------|----------|------|
| **执行层** | Go | 高性能、并发安全、借鉴NOFX |
| **AI决策层** | Python | LangChain生态成熟、借鉴TradingAgents-CN |
| **前端** | TypeScript + React | 现代化、组件化、借鉴NOFX |
| **数据库** | PostgreSQL + Redis | 生产级、支持复杂查询 |
| **消息队列** | NATS | 轻量级、高性能、Go/Python互通 |
| **缓存** | Redis | 热数据缓存、实时状态共享 |

#### 2.1.2 架构设计原则

1. **微服务架构**: Go执行服务 + Python AI服务独立部署
2. **消息驱动**: 通过NATS消息总线解耦服务
3. **接口优先**: 所有服务通过RESTful API或gRPC通信
4. **配置驱动**: 策略、风控参数通过配置文件管理
5. **可观测性**: 完整的日志、监控、链路追踪

### 2.2 系统分层

```
┌─────────────────────────────────────────────────────────────┐
│                      表现层 (Presentation)                      │
│  React Web UI | CLI | REST API | WebSocket                  │
├─────────────────────────────────────────────────────────────┤
│                      应用层 (Application)                       │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  Python AI服务  │  │   Go 执行服务    │                   │
│  │  - 多智能体分析   │  │  - 订单管理      │                   │
│  │  - LangGraph     │  │  - 券商API      │                   │
│  │  - LLM适配器     │  │  - 风控执行      │                   │
│  └─────────────────┘  └─────────────────┘                   │
├─────────────────────────────────────────────────────────────┤
│                      领域层 (Domain)                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ 智能体   │  │ 决策引擎 │  │ 风控引擎 │  │ 回测引擎 │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
├─────────────────────────────────────────────────────────────┤
│                      基础设施层 (Infrastructure)                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ 数据源   │  │ 缓存     │  │ 消息队列 │  │ 日志     │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 系统架构设计

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              用户层                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Web Dashboard│  │ CLI (Go)     │  │ REST API     │               │
│  │ React + TS   │  │ 交互式命令行  │  │ 开发接口      │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           API网关层 (Go)                               │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ 认证授权      │  │ 限流控制      │  │ 请求路由      │               │
│  │ JWT + 2FA   │  │ Token Bucket │  │ 负载均衡      │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          微服务层                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────┐  ┌─────────────────────────┐ │
│  │      Python AI决策服务              │  │      Go 执行服务         │ │
│  ├─────────────────────────────────────┤  ├─────────────────────────┤ │
│  │  ┌───────────────────────────────┐  │  │  ┌───────────────────┐ │ │
│  │  │ 多智能体分析框架              │  │  │  │ 券商API适配器     │ │ │
│  │  │ - 市场技术分析师              │  │  │  │ - 华泰证券        │ │ │
│  │  │ - 基本面分析师                │  │  │  │ - 国金证券        │ │ │
│  │  │ - 新闻分析师                  │  │  │  │ - 富途证券        │ │ │
│  │  │ - 社交媒体分析师              │  │  │  │ - 东方财富        │ │ │
│  │  └───────────────────────────────┘  │  │  │ - 老虎证券        │ │ │
│  │                                   │  │  └───────────────────┘ │ │
│  │  ┌───────────────────────────────┐  │  │  ┌───────────────────┐ │ │
│  │  │ LangGraph工作流引擎           │  │  │  │ 订单管理系统      │ │ │
│  │  │ - 看涨/看跌研究员辩论         │  │  │  │ - 订单生成        │ │ │
│  │  │ - 风险管理员                  │  │  │  │ - 订单路由        │ │ │
│  │  │ - 投资法官                    │  │  │  │ - 订单状态跟踪    │ │ │
│  │  └───────────────────────────────┘  │  │  │ - 订单确认        │ │ │
│  │                                   │  │  └───────────────────┘ │ │
│  │  ┌───────────────────────────────┐  │  │  ┌───────────────────┐ │ │
│  │  │ LLM适配器统一层               │  │  │  │ 风控执行引擎      │ │ │
│  │  │ - DeepSeek                   │  │  │  │ - T+1限制检查     │ │ │
│  │  │ - Qwen (阿里百炼)             │  │  │  │ - 涨跌停检测      │ │ │
│  │  │ - 文心一言 (千帆)             │  │  │  │ - 最小单位调整    │ │ │
│  │  │ - 自定义OpenAI兼容API        │  │  │  │ - ST股限制        │ │ │
│  │  └───────────────────────────────┘  │  │  │ - 仓位控制        │ │ │
│  └─────────────────────────────────────┘  │  └───────────────────┘ │ │
│                                             └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          数据层                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ PostgreSQL   │  │ Redis        │  │ NATS         │               │
│  │ 持久化存储    │  │ 缓存/状态    │  │ 消息队列     │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ 时序DB       │  │ 对象存储     │  │ Level-2数据   │               │
│  │ (TimescaleDB)│  │ 决策日志     │  │ 实时行情      │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          外部数据源                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Tushare      │  │ AkShare      │  │ 通达信        │               │
│  │ A股专业数据   │  │ 免费A股数据  │  │ 实时行情      │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ 雪球网        │  │ 东方财富      │  │ 财华社        │               │
│  │ 社交媒体      │  │ 新闻公告     │  │ 港股新闻      │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 服务间通信

#### 3.2.1 消息格式 (NATS)

```json
// AI决策消息
{
  "type": "ai_decision",
  "timestamp": "2025-12-29T10:30:00Z",
  "request_id": "uuid-123",
  "payload": {
    "account_id": "acc-001",
    "market": "CN_A",  // CN_A, CN_HK
    "decisions": [
      {
        "symbol": "600519.SH",
        "action": "buy",
        "quantity": 100,
        "price": 1850.0,
        "reasoning": "..."
      }
    ]
  }
}

// 订单状态消息
{
  "type": "order_status",
  "timestamp": "2025-12-29T10:31:00Z",
  "request_id": "uuid-123",
  "payload": {
    "order_id": "ord-456",
    "status": "filled",
    "filled_quantity": 100,
    "filled_price": 1848.5,
    "commission": 92.43
  }
}
```

---

## 4. 模块融合方案

### 4.1 模块映射表

| 功能模块 | NOFX来源 | TradingAgents-CN来源 | 融合后实现 |
|----------|----------|---------------------|-----------|
| **执行引擎** | `trader/auto_trader.go` | - | Go服务：`execution/executor.go` |
| **多智能体** | - | `agents/` | Python服务：`agents/` |
| **决策引擎** | `decision/engine.go` | `graph/trading_graph.py` | Python服务：`decision/` |
| **LLM适配器** | `mcp/` | `llm_adapters/` | Python服务：`llm/` |
| **市场数据** | `market/` | `dataflows/` | Go服务：`market/` |
| **数据源** | - | `dataflows/tushare_utils.py` | Go服务：`datasource/` |
| **风控引擎** | (简单) | (简单) | Go服务：`risk/` |
| **日志系统** | `logger/` | - | Go服务：`logger/` |
| **前端** | `web/` | `web/` | 独立仓库：`frontend/` |
| **API层** | `api/` | - | Go服务：`api/` |

### 4.2 核心模块设计

#### 4.2.1 Python AI决策服务

```python
# stockagents_ai/decision/engine.py
from langgraph.graph import StateGraph, END
from agents.analysts import MarketAnalyst, FundamentalsAnalyst, NewsAnalyst
from agents.researchers import BullResearcher, BearResearcher, InvestmentJudge
from agents.risk_mgr import RiskManager

class StockDecisionEngine:
    """股票决策引擎 - 融合NOFX和TradingAgents-CN"""

    def __init__(self, config: Dict):
        # 从TradingAgents-CN继承：多智能体框架
        self.analysts = {
            "market": MarketAnalyst(),
            "fundamentals": FundamentalsAnalyst(),
            "news": NewsAnalyst(),
        }

        self.researchers = {
            "bull": BullResearcher(),
            "bear": BearResearcher(),
            "judge": InvestmentJudge(),
        }

        self.risk_manager = RiskManager()

        # 从NOFX继承：提示词管理
        self.prompt_manager = PromptManager()

        # 构建LangGraph工作流
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建决策工作流（保留TradingAgents-CN的LangGraph）"""
        workflow = StateGraph(DecisionState)

        # 分析师阶段
        workflow.add_node("market_analysis", self.market_analyst_node)
        workflow.add_node("fundamentals_analysis", self.fundamentals_analyst_node)
        workflow.add_node("news_analysis", self.news_analyst_node)

        # 研究辩论阶段
        workflow.add_node("bull_research", self.bull_research_node)
        workflow.add_node("bear_research", self.bear_research_node)
        workflow.add_node("investment_judge", self.judge_node)

        # 风险管理阶段
        workflow.add_node("risk_assessment", self.risk_assessment_node)

        # 边定义
        workflow.set_entry_point("market_analysis")
        workflow.add_edge("market_analysis", "fundamentals_analysis")
        workflow.add_edge("fundamentals_analysis", "news_analysis")
        workflow.add_edge("news_analysis", "bull_research")
        workflow.add_edge("bull_research", "bear_research")
        workflow.add_edge("bear_research", "investment_judge")
        workflow.add_edge("investment_judge", "risk_assessment")
        workflow.add_edge("risk_assessment", END)

        return workflow.compile()

    async def make_decision(self, context: Dict) -> DecisionResult:
        """生成交易决策"""
        # 1. 运行多智能体分析
        state = await self.graph.ainvoke(context)

        # 2. 融合NOFX的决策验证
        validated_decisions = self._validate_decisions(state["decisions"])

        # 3. 格式化为Go服务可接受的格式
        return DecisionResult(
            request_id=generate_uuid(),
            timestamp=datetime.now().isoformat(),
            decisions=validated_decisions
        )

    def _validate_decisions(self, decisions: List[Dict]) -> List[Dict]:
        """决策验证（融合NOFX的验证逻辑）"""
        validated = []
        for decision in decisions:
            # NOFX风格的风险回报比验证
            if self._check_risk_reward_ratio(decision):
                validated.append(decision)
        return validated
```

#### 4.2.2 Go执行服务

```go
// stockagents_go/execution/executor.go
package execution

import (
    "context"
    "time"
)

type StockExecutor struct {
    broker      Broker  // 券商接口（适配器模式）
    riskMgr     RiskManager
    orderMgr    OrderManager
    natsConn    *nats.Conn
    logger      Logger
}

// Broker 统一券商接口（借鉴NOFX的Trader接口）
type Broker interface {
    // 账户操作
    GetBalance(ctx context.Context) (*Balance, error)
    GetPositions(ctx context.Context) ([]*Position, error)
    GetOrderStatus(ctx context.Context, orderID string) (*Order, error)

    // 订单操作
    PlaceOrder(ctx context.Context, order *Order) (*OrderReceipt, error)
    CancelOrder(ctx context.Context, orderID string) error

    // 查询操作
    GetQuote(ctx context.Context, symbol string) (*Quote, error)
    GetKLine(ctx context.Context, symbol string, period string) ([]*KLine, error)
}

type Order struct {
    Symbol      string
    Side        string // "buy" or "sell"
    OrderType   string // "limit", "market"
    Quantity    int64  // 股数（A股100的倍数）
    Price       float64
    TimeInForce string // "DAY" (A股), "GTC" (港股)
}

// ExecuteDecision 执行AI决策（接收来自Python服务的决策）
func (e *StockExecutor) ExecuteDecision(ctx context.Context, decision *AIDecision) error {
    // 1. 风控检查（A股/港股特殊规则）
    if err := e.riskMgr.ValidateDecision(ctx, decision); err != nil {
        return fmt.Errorf("风控拒绝: %w", err)
    }

    // 2. 根据市场类型调整订单
    order := e.adaptOrderForMarket(decision)

    // 3. 提交订单
    receipt, err := e.broker.PlaceOrder(ctx, order)
    if err != nil {
        return fmt.Errorf("下单失败: %w", err)
    }

    // 4. 发送订单状态消息
    e.publishOrderStatus(receipt)

    // 5. 记录日志
    e.logger.LogOrder(decision, order, receipt)

    return nil
}

// adaptOrderForMarket 根据市场规则调整订单
func (e *StockExecutor) adaptOrderForMarket(decision *AIDecision) *Order {
    order := &Order{
        Symbol:    decision.Symbol,
        Side:      decision.Side,
        OrderType: "limit",
        Price:     decision.Price,
    }

    if isAStock(decision.Symbol) {
        // A股规则：T+1、100股整数倍
        order.Quantity = adjustToLots(decision.Quantity, 100)
        order.TimeInForce = "DAY"

        // 涨跌停检测
        limitPrice := getLimitPrice(decision.Symbol)
        order.Price = clamp(order.Price, limitPrice.Down, limitPrice.Up)

        // ST股票检查
        if isSTStock(decision.Symbol) {
            order.Quantity = min(order.Quantity, getSTMaxLots())
        }

    } else if isHKStock(decision.Symbol) {
        // 港股规则：T+0、100股整数倍、无涨跌停
        order.Quantity = adjustToLots(decision.Quantity, 100)
        order.TimeInForce = "GTC"

        // 汇率转换（港币→人民币）
        if decision.Currency == "HKD" {
            exchangeRate := getExchangeRate("HKD", "CNY")
            order.Price *= exchangeRate
        }
    }

    return order
}

// adjustToLots 调整为最小单位的整数倍
func adjustToLots(quantity float64, lotSize int) int64 {
    lots := int(math.Ceil(quantity / float64(lotSize)))
    return int64(lots) * int64(lotSize)
}
```

#### 4.2.3 风控引擎（A股/港股专用）

```go
// stockagents_go/risk/manager.go
package risk

import (
    "context"
    "time"
)

type RiskManager struct {
    rules map[string]RiskRule
    db    *Database
}

// RiskRule 风控规则接口
type RiskRule interface {
    Validate(ctx context.Context, decision *AIDecision) error
}

// ChineseMarketRules 中国市场规则（T+1、涨跌停等）
type ChineseMarketRules struct{}

func (r *ChineseMarketRules) Validate(ctx context.Context, decision *AIDecision) error {
    symbol := decision.Symbol

    // 1. T+1检查
    if isAStock(symbol) && decision.Side == "sell" {
        if hasBoughtToday(ctx, symbol) {
            return fmt.Errorf("T+1限制：当日买入的股票不可卖出")
        }
    }

    // 2. 涨跌停检查
    if isAStock(symbol) {
        limits := getLimitPrice(symbol)
        if decision.Side == "buy" && decision.Price >= limits.Up {
            return fmt.Errorf("涨停价附近限制买入: %.2f >= %.2f", decision.Price, limits.Up)
        }
        if decision.Side == "sell" && decision.Price <= limits.Down {
            return fmt.Errorf("跌停价附近限制卖出: %.2f <= %.2f", decision.Price, limits.Down)
        }
    }

    // 3. ST股票限制
    if isSTStock(symbol) {
        maxPosition := getSTMaxPosition()
        currentPosition := getCurrentPosition(symbol)
        if currentPosition >= maxPosition {
            return fmt.Errorf("ST股票仓位已达上限: %d股", maxPosition)
        }
    }

    // 4. 最小单位检查
    minUnit := getMinUnit(symbol) // A股=100, 港股=100
    if decision.Quantity%minUnit != 0 {
        return fmt.Errorf("数量必须是%d的整数倍", minUnit)
    }

    return nil
}

// PositionLimitRule 仓位限制规则
type PositionLimitRule struct {
    maxSingleStock float64 // 单只股票最大仓位比例
    maxTotal        float64 // 总仓位上限
}

func (r *PositionLimitRule) Validate(ctx context.Context, decision *AIDecision) error {
    account, _ := getAccountInfo(ctx)

    // 检查单只股票仓位
    currentValue := getCurrentStockValue(decision.Symbol)
    newValue := currentValue + decision.Price*float64(decision.Quantity)
    if newValue/account.TotalValue > r.maxSingleStock {
        return fmt.Errorf("单只股票仓位超限: %.2f%% > %.2f%%",
            newValue/account.TotalValue*100, r.maxSingleStock*100)
    }

    // 检查总仓位
    totalPosition := account.TotalPosition + decision.Price*float64(decision.Quantity)
    if totalPosition/account.TotalValue > r.maxTotal {
        return fmt.Errorf("总仓位超限: %.2f%% > %.2f%%",
            totalPosition/account.TotalValue*100, r.maxTotal*100)
    }

    return nil
}

// DrawdownControlRule 回撤控制规则
type DrawdownControlRule struct {
    maxDrawdown float64
    peakValue   float64
}

func (r *DrawdownControlRule) Validate(ctx context.Context, decision *AIDecision) error {
    account, _ := getAccountInfo(ctx)

    // 更新峰值
    if account.TotalValue > r.peakValue {
        r.peakValue = account.TotalValue
    }

    // 计算回撤
    drawdown := (r.peakValue - account.TotalValue) / r.peakValue

    if drawdown > r.maxDrawdown {
        return fmt.Errorf("回撤超限，暂停交易: %.2f%% > %.2f%%",
            drawdown*100, r.maxDrawdown*100)
    }

    return nil
}
```

---

## 5. A股/港股特殊处理

### 5.1 市场规则引擎

```go
// stockagents_go/market/rules.go
package market

type MarketType int

const (
    MarketCN_A MarketType = iota // A股
    MarketCN_HK                   // 港股
    MarketUS                      // 美股（预留）
)

type MarketRules struct {
    Type        MarketType
    TPlusOne    bool      // T+1交易制度
    LimitUp     float64   // 涨停限制（1=10%, 0.2=20%）
    LimitDown   float64   // 跌停限制
    MinUnit     int       // 最小交易单位（股）
    TradingHours []string // 交易时段
}

var marketRulesMap = map[string]MarketRules{
    // A股主板
    "600000.SH": {Type: MarketCN_A, TPlusOne: true, LimitUp: 0.10, LimitDown: 0.10, MinUnit: 100},
    "000001.SZ": {Type: MarketCN_A, TPlusOne: true, LimitUp: 0.10, LimitDown: 0.10, MinUnit: 100},

    // 科创板/创业板（20%涨跌停）
    "688981.SH": {Type: MarketCN_A, TPlusOne: true, LimitUp: 0.20, LimitDown: 0.20, MinUnit: 100},
    "300001.SZ": {Type: MarketCN_A, TPlusOne: true, LimitUp: 0.20, LimitDown: 0.20, MinUnit: 100},

    // ST股票（5%涨跌停）
    "600000.ST": {Type: MarketCN_A, TPlusOne: true, LimitUp: 0.05, LimitDown: 0.05, MinUnit: 100},

    // 港股
    "00700.HK": {Type: MarketCN_HK, TPlusOne: false, LimitUp: 0, LimitDown: 0, MinUnit: 100},
}

func GetMarketRules(symbol string) MarketRules {
    // 根据股票代码获取市场规则
    if rules, ok := marketRulesMap[symbol]; ok {
        return rules
    }

    // 默认规则推断
    if strings.HasSuffix(symbol, ".SH") || strings.HasSuffix(symbol, ".SZ") {
        return MarketRules{Type: MarketCN_A, TPlusOne: true, LimitUp: 0.10, LimitDown: 0.10, MinUnit: 100}
    } else if strings.HasSuffix(symbol, ".HK") {
        return MarketRules{Type: MarketCN_HK, TPlusOne: false, LimitUp: 0, LimitDown: 0, MinUnit: 100}
    }

    return MarketRules{}
}

// GetLimitPrice 获取涨跌停价格
func GetLimitPrice(symbol string, prevClose float64) (up, down float64) {
    rules := GetMarketRules(symbol)

    if rules.LimitUp > 0 {
        up = prevClose * (1 + rules.LimitUp)
        down = prevClose * (1 - rules.LimitDown)
    } else {
        up = 0    // 无涨跌停
        down = 0
    }

    return
}

// IsTradingTime 检查是否在交易时间
func IsTradingTime(market MarketType) bool {
    now := time.Now()

    if market == MarketCN_A {
        // A股交易时间：9:30-11:30, 13:00-15:00
        hour := now.Hour()
        minute := now.Minute()
        timeValue := hour*60 + minute

        morning := (9*60 + 30 <= timeValue) && (timeValue <= 11*60 + 30)
        afternoon := (13*60 <= timeValue) && (timeValue <= 15*60)

        return morning || afternoon
    } else if market == MarketCN_HK {
        // 港股交易时间：9:30-12:00, 13:00-16:00
        hour := now.Hour()
        minute := now.Minute()
        timeValue := hour*60 + minute

        morning := (9*60 + 30 <= timeValue) && (timeValue <= 12*60)
        afternoon := (13*60 <= timeValue) && (timeValue <= 16*60)

        return morning || afternoon
    }

    return false
}
```

### 5.2 数据源适配

```python
# stockagents_ai/datasource/tushare_adapter.py
import tushare as ts
from typing import List, Dict, Optional

class TushareDataSource:
    """Tushare数据源适配器（借鉴TradingAgents-CN）"""

    def __init__(self, token: str):
        ts.set_token(token)
        self.pro = ts.pro_api()

    def get_stock_list(self) -> List[Dict]:
        """获取A股股票列表"""
        # 基础股票列表
        stocks = self.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,list_date'
        )

        # 过滤条件
        stocks = stocks[
            (stocks['list_date'] >= '20200101') &  # 排除新股
            (~stocks['name'].str.contains('ST'))     # 排除ST股票
        ]

        return stocks.to_dict('records')

    def get_realtime_quote(self, symbols: List[str]) -> Dict:
        """获取实时行情（使用通达信作为备用）"""
        try:
            # Tushare实时行情（需要高级权限）
            df = self.pro.daily(
                ts_code=','.join(symbols),
                trade_date=datetime.now().strftime('%Y%m%d')
            )
            return df.to_dict('records')
        except Exception as e:
            # 降级到通达信数据
            return self._get_tdx_quote(symbols)

    def get_fundamentals(self, symbol: str) -> Dict:
        """获取基本面数据"""
        # 财务指标
        df = self.pro.fina_indicator(
            ts_code=symbol,
            start_date='20200101',
            end_date=datetime.now().strftime('%Y%m%d')
        )

        # 业绩预告
        forecast = self.pro.forecast(
            ts_code=symbol,
            announce_date='20250101'
        )

        return {
            'indicators': df.to_dict('records')[-10:],  # 最近10期
            'forecast': forecast.to_dict('records')
        }

    def get_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """获取新闻（融合东方财富、雪球等数据源）"""
        news = []

        # Tushare新闻（如有权限）
        try:
            df = self.pro.news(
                src_suggest='simplify',
                ts_code=symbol,
                date=datetime.now().strftime('%Y%m%d')
            )
            news.extend(df.to_dict('records'))
        except:
            pass

        # 东方财富新闻（爬取）
        ef_news = self._get_eastmoney_news(symbol, days)
        news.extend(ef_news)

        # 雪球新闻
        xq_news = self._get_xueqiu_news(symbol, days)
        news.extend(xq_news)

        return self._deduplicate_news(news)

# stockagents_ai/datasource/akshare_adapter.py
import akshare as ak

class AkShareDataSource:
    """AkShare数据源适配器（免费A股数据）"""

    def get_stock_list(self) -> List[Dict]:
        """获取A股股票列表"""
        # 沪市
        sh_stocks = ak.stock_info_sh_name_code(indicator="主板A股")
        sh_stocks['exchange'] = 'SH'

        # 深市
        sz_stocks = ak.stock_info_sz_name_code(indicator="A股列表")
        sz_stocks['exchange'] = 'SZ'

        # 合并
        stocks = pd.concat([sh_stocks, sz_stocks])

        # 格式化代码
        stocks['ts_code'] = stocks['code'] + '.' + stocks['exchange']

        return stocks.to_dict('records')

    def get_realtime_quote(self, symbols: List[str]) -> Dict:
        """获取实时行情"""
        quotes = {}
        for symbol in symbols:
            try:
                df = ak.stock_zh_a_spot_em()  # 东方财富实时行情
                stock_data = df[df['代码'] == symbol.replace('.SH', '').replace('.SZ', '')]
                if not stock_data.empty:
                    quotes[symbol] = stock_data.to_dict('records')[0]
            except Exception as e:
                logger.warning(f"获取{symbol}行情失败: {e}")

        return quotes

# stockagents_ai/datasource/hk_stock_adapter.py
class HKStockDataSource:
    """港股数据源适配器"""

    def __init__(self):
        self.akshare = AkShareDataSource()
        # 可选：富途牛牛数据（需要权限）

    def get_stock_list(self) -> List[Dict]:
        """获取港股列表"""
        # 获取港股列表
        stocks = ak.stock_hk_spot_em()  # 东方财富港股实时行情

        # 过滤条件
        stocks = stocks[
            (stocks['成交量'] > 0) &           # 有成交量
            (stocks['最新价'] > 0.1) &           # 价格大于0.1港币
            (~stocks['名称'].str.contains('控股'))  # 排除某些类型
        ]

        # 格式化代码
        stocks['ts_code'] = stocks['代码'].apply(lambda x: f"{x}.HK")

        return stocks.to_dict('records')

    def get_realtime_quote(self, symbols: List[str]) -> Dict:
        """获取港股实时行情"""
        # 方法1：AkShare
        quotes = self.akshare.get_hk_realtime(symbols)

        # 方法2：Yahoo Finance（备用）
        if not quotes:
            quotes = self._get_yahoo_quotes(symbols)

        return quotes

    def get_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """获取港股新闻"""
        news = []

        # 财华社
        news.extend(self._get_xuehua_news(symbol, days))

        # 经济通
        news.extend(self._get_etnet_news(symbol, days))

        # 香港交易所披露易
        news.extend(self._get_hkex_news(symbol, days))

        return self._deduplicate_news(news)
```

---

## 6. 数据流设计

### 6.1 实时行情数据流

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  通达信      │────▶│  Go Market   │────▶│  Redis       │
│  Level-2     │     │  Service     │     │  Cache       │
└──────────────┘     └──────────────┘     └──────────────┘
                             │
                             ▼
                    ┌──────────────┐
                    │  WebSocket   │
                    │  Broadcast   │
                    └──────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │ AI服务   │      │ Go执行   │      │ Web UI   │
    │ (订阅)   │      │ 服务     │      │ (展示)   │
    └──────────┘      └──────────┘      └──────────┘
```

### 6.2 AI决策数据流

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Python AI   │     │  NATS        │     │  Go Executor │
│  Service     │────▶│  Message Bus │────▶│  Service     │
└──────────────┘     └──────────────┘     └──────────────┘
      ▲                                        │
      │                                        ▼
┌──────────────┐                      ┌──────────────┐
│  Tushare/    │                      │  券商API     │
│  AkShare     │                      │  (下单/查询) │
└──────────────┘                      └──────────────┘
```

---

## 7. AI决策引擎

### 7.1 提示词模板管理

```python
# stockagents_ai/prompts/template_manager.py
from typing import Dict, Optional

class PromptTemplateManager:
    """提示词模板管理器（借鉴NOFX）"""

    def __init__(self):
        self.templates = {
            "default": self._load_template("default.txt"),
            "aggressive": self._load_template("aggressive.txt"),
            "conservative": self._load_template("conservative.txt"),
            "cn_market": self._load_template("cn_market.txt"),  # A股专用
            "hk_market": self._load_template("hk_market.txt"),  # 港股专用
        }

    def build_system_prompt(self, template_name: str, context: Dict) -> str:
        """构建系统提示词"""
        template = self.templates.get(template_name, self.templates["default"])

        # 替换动态部分
        prompt = template.format(
            account_equity=context['account']['total_equity'],
            max_position=context['config']['max_position'],
            risk_ratio=context['config']['risk_ratio'],
            market_type=context['market_type'],  # CN_A 或 CN_HK
            trading_date=context['trading_date'],
        )

        # 添加市场特殊规则
        if context['market_type'] == 'CN_A':
            prompt += self._get_cn_a_rules()
        elif context['market_type'] == 'CN_HK':
            prompt += self._get_cn_hk_rules()

        return prompt

    def _get_cn_a_rules(self) -> str:
        """A股交易规则"""
        return """
# A股交易规则（严格遵守）
1. T+1制度：当日买入的股票不可卖出
2. 涨跌停限制：
   - 主板/中小板：±10%
   - 创业板/科创板：±20%
   - ST股票：±5%
3. 最小单位：100股（1手）
4. 交易时间：9:30-11:30, 13:00-15:00
5. ST股票限制：单只ST股票仓位不超过账户的5%
"""

    def _get_cn_hk_rules(self) -> str:
        """港股交易规则"""
        return """
# 港股交易规则（严格遵守）
1. T+0制度：允许日内买卖
2. 无涨跌停限制：需设置动态止损
3. 最小单位：100股
4. 交易时间：9:30-12:00, 13:00-16:00
5. 汇率风险：港币与人民币汇率波动
6. 小心"老千股":小盘港股需检查流动性
"""
```

### 7.2 上下文构建

```python
# stockagents_ai/decision/context_builder.py
class TradingContextBuilder:
    """交易上下文构建器（融合两者优势）"""

    def build_context(self, request: AnalysisRequest) -> Dict:
        """构建完整的交易上下文"""
        context = {
            'market_type': self._detect_market_type(request.symbols),
            'trading_date': datetime.now().strftime('%Y-%m-%d'),
            'account': self._get_account_info(request.account_id),
            'positions': self._get_positions(request.account_id),
            'market_data': self._get_market_data(request.symbols),
            'news': self._get_news(request.symbols),
            'fundamentals': self._get_fundamentals(request.symbols),
            'performance': self._get_historical_performance(request.account_id),
        }

        # A股/港股特殊数据
        if context['market_type'] == 'CN_A':
            context['limit_rules'] = self._get_limit_rules(request.symbols)
            context['st_stocks'] = self._get_st_stocks(request.symbols)
        elif context['market_type'] == 'CN_HK':
            context['exchange_rate'] = self._get_exchange_rate()

        return context

    def _detect_market_type(self, symbols: List[str]) -> str:
        """检测市场类型"""
        if any(s.endswith('.SH') or s.endswith('.SZ') for s in symbols):
            return 'CN_A'
        elif any(s.endswith('.HK') for s in symbols):
            return 'CN_HK'
        return 'UNKNOWN'

    def _get_market_data(self, symbols: List[str]) -> Dict:
        """获取市场数据（融合NOFX和TradingAgents-CN）"""
        market_data = {}

        # 从Go服务获取实时行情（WebSocket）
        real_time = self._fetch_realtime_quotes(symbols)

        # 从Tushare获取技术指标
        for symbol in symbols:
            data = real_time.get(symbol, {})

            # 计算技术指标（TradingAgents-CN风格）
            data['technical_indicators'] = self._calculate_indicators(symbol)

            # K线数据
            data['klines'] = self._get_klines(symbol, period='1d', count=60)

            market_data[symbol] = data

        return market_data
```

---

## 8. 交易执行层

### 8.1 券商API适配器

```go
// stockagents_go/broker/futu_adapter.go
package broker

import (
    "context"
    "github.com/FutunnOpen/futuapi/pb/qotcommon"
    "github.com/FutunnOpen/futuapi/pb/trdcommon"
    "github.com/FutunnOpen/futuapi/pb/trd"
)

// FutuBroker 富途牛牛券商适配器
type FutuBroker struct {
    client *trd.TrdClient
    accID  string
}

func NewFutuBroker(host string, port int, accID, password string) (*FutuBroker, error) {
    client := trd.NewTrdClient(host, int32(port))

    // 解锁交易
    _, err := client.UnlockTrade(accID, password, true)
    if err != nil {
        return nil, fmt.Errorf("解锁失败: %w", err)
    }

    return &FutuBroker{
        client: client,
        accID:  accID,
    }, nil
}

func (b *FutuBroker) GetBalance(ctx context.Context) (*Balance, error) {
    resp, err := b.client.AccInfoQuery(b.accID, trdcommon.FundsField_All)
    if err != nil {
        return nil, err
    }

    funds := resp.GetFundsList()
    if len(funds) == 0 {
        return nil, fmt.Errorf("无资金数据")
    }

    f := funds[0]
    return &Balance{
        TotalValue:   f.GetTotalAssets(),
        Cash:         f.GetCash(),
        Available:    f.GetAvailWithdrawableCash(),
        MarketValue:  f.GetMarketValue(),
        Currency:     "CNY",
    }, nil
}

func (b *FutuBroker) PlaceOrder(ctx context.Context, order *Order) (*OrderReceipt, error) {
    // 转换订单类型
    trdSide := trdcommon.TrdSide_Buy
    if order.Side == "sell" {
        trdSide = trdcommon.TrdSide_Sell
    }

    // 下单
    resp, err := b.client.PlaceOrder(
        b.accID,
        trdSide,
        qotcommon.OrderType_Limit,  // 限价单
        order.Symbol,
        order.Quantity,
        order.Price,
        []trdcommon.ModifyOrderField{},
        trdcommon.TimeInForceType_Day,  // 当日有效
    )

    if err != nil {
        return nil, fmt.Errorf("下单失败: %w", err)
    }

    orderID := resp.GetOrderId()
    return &OrderReceipt{
        OrderID:     fmt.Sprintf("%d", orderID),
        Status:      "pending",
        Symbol:      order.Symbol,
        Side:        order.Side,
        Quantity:    order.Quantity,
        Price:       order.Price,
        SubmittedAt: time.Now(),
    }, nil
}

// EastMoneyBroker 东方财富券商适配器
type EastMoneyBroker struct {
    client *http.Client
    cookies map[string]string
}

func (b *EastMoneyBroker) PlaceOrder(ctx context.Context, order *Order) (*OrderReceipt, error) {
    // 调用东方财富API
    url := "https://trade.eastmoney.com/api/trade/v1/place"

    payload := map[string]interface{}{
        "stock_code": order.Symbol,
        "trade_type": 1, // 买入
        "order_type":  1, // 限价
        "price":       order.Price,
        "amount":      order.Quantity,
    }

    resp, err := b.client.PostForm(url, payload)
    if err != nil {
        return nil, err
    }

    var result struct {
        ErrorCode int    `json:"errorCode"`
        ErrorMsg  string `json:"errorMsg"`
        OrderID   string `json:"orderId"`
    }

    json.NewDecoder(resp.Body).Decode(&result)

    if result.ErrorCode != 0 {
        return nil, fmt.Errorf("下单失败: %s", result.ErrorMsg)
    }

    return &OrderReceipt{
        OrderID:     result.OrderID,
        Status:      "pending",
        Symbol:      order.Symbol,
        Side:        order.Side,
        Quantity:    order.Quantity,
        Price:       order.Price,
        SubmittedAt: time.Now(),
    }, nil
}
```

---

## 9. 前端设计

### 9.1 技术栈

```json
{
  "name": "stockagents-frontend",
  "dependencies": {
    "react": "^18.3.1",
    "typescript": "^5.8.3",
    "vite": "^6.0.7",
    "react-router-dom": "^7.9.5",
    "zustand": "^5.0.2",
    "swr": "^2.2.5",
    "axios": "^1.13.2",
    "tailwindcss": "^3.4.17",
    "recharts": "^2.15.2",
    "framer-motion": "^12.23.24",
    "socket.io-client": "^4.0.0"
  }
}
```

### 9.2 页面结构

```
web/src/
├── pages/
│   ├── Dashboard.tsx          # 主控面板
│   ├── PositionPage.tsx      # 持仓管理
│   ├── OrderPage.tsx         # 订单管理
│   ├── AnalysisPage.tsx      # AI分析
│   ├── BacktestPage.tsx      # 回测系统
│   └── SettingsPage.tsx      # 系统设置
├── components/
│   ├── market/
│   │   ├── RealtimeQuote.tsx  # 实时行情
│   │   ├── KLineChart.tsx     # K线图表
│   │   └── OrderBook.tsx      # 五档盘口
│   ├── trading/
│   │   ├── PositionList.tsx   # 持仓列表
│   │   ├── OrderForm.tsx      # 下单表单
│   │   └── TradeHistory.tsx   # 交易历史
│   └── ai/
│       ├── AgentOpinion.tsx   # 智能体观点
│       ├── DecisionLog.tsx    # 决策日志
│       └── PerformanceChart.tsx  # 绩效图表
├── stores/
│   ├── accountStore.ts       # 账户状态
│   ├── positionStore.ts      # 持仓状态
│   ├── orderStore.ts         # 订单状态
│   └── marketStore.ts        # 行情状态
└── hooks/
    ├── useRealtimeQuote.ts    # 实时行情Hook
    ├── useWebSocket.ts        # WebSocket Hook
    └── useAIDecision.ts       # AI决策Hook
```

---

## 10. 部署方案

### 10.1 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Go执行服务
  execution-service:
    build: ./services/execution
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgres://stockagents:password@postgres:5432/stockagents
      - REDIS_URL=redis://redis:6379
      - NATS_URL=nats://nats:4222
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
      - nats

  # Python AI服务
  ai-service:
    build: ./services/ai
    ports:
      - "9000:9000"
    environment:
      - DATABASE_URL=postgres://stockagents:password@postgres:5432/stockagents
      - REDIS_URL=redis://redis:6379
      - NATS_URL=nats://nats:4222
      - TUSHARE_TOKEN=${TUSHARE_TOKEN}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
    volumes:
      - ./prompts:/app/prompts
    depends_on:
      - postgres
      - redis
      - nats

  # 前端
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - execution-service
      - ai-service

  # PostgreSQL数据库
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=stockagents
      - POSTGRES_USER=stockagents
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d

  # Redis缓存
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  # NATS消息队列
  nats:
    image: nats:latest
    command: "-js"
    ports:
      - "4222:4222"
    volumes:
      - nats_data:/data

  # TimescaleDB时序数据库（用于K线数据）
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    environment:
      - POSTGRES_DB=market_data
      - POSTGRES_USER=timescale
      - POSTGRES_PASSWORD=password
    volumes:
      - timescale_data:/var/lib/postgresql/data

volumes:
  postgres_data:
  redis_data:
  nats_data:
  timescale_data:
```

---

## 11. 开发路线图

### 11.1 阶段划分

#### 阶段1: 核心框架搭建 (4周)

| 任务 | 负责模块 | 输出 |
|------|----------|------|
| 项目初始化 | 基础设施 | Git仓库、CI/CD |
| Go服务框架 | execution/ | 基础服务骨架 |
| Python服务框架 | ai/ | 基础服务骨架 |
| NATS消息总线 | 基础设施 | 服务间通信 |
| 数据库设计 | 数据层 | PostgreSQL Schema |

#### 阶段2: 数据接入 (3周)

| 任务 | 数据源 | 来源 |
|------|--------|------|
| A股行情接入 | Tushare, AkShare, 通达信 | TradingAgents-CN |
| 港股行情接入 | AkShare, Yahoo Finance | 新增 |
| 基本面数据 | Tushare, 东方财富 | TradingAgents-CN |
| 新闻数据 | 东方财富, 雪球, 财华社 | 两者融合 |
| WebSocket推送 | - | NOFX改造 |

#### 阶段3: AI决策引擎 (4周)

| 任务 | 模块 | 来源 |
|------|------|------|
| 多智能体框架 | agents/ | TradingAgents-CN |
| LangGraph工作流 | decision/ | TradingAgents-CN |
| LLM适配器 | llm/ | 两者融合 |
| 提示词模板 | prompts/ | NOFX |
| A股/港股规则 | market/ | 新增 |

#### 阶段4: 交易执行 (4周)

| 任务 | 模块 | 来源 |
|------|------|------|
| 券商API适配 | broker/ | 新增 |
| 风控引擎 | risk/ | NOFX扩展 |
| 订单管理 | execution/ | NOFX |
| T+1限制 | market/rules/ | 新增 |
| 涨跌停检测 | market/rules/ | 新增 |

#### 阶段5: 前端开发 (3周)

| 任务 | 模块 | 来源 |
|------|------|------|
| 主控面板 | pages/Dashboard.tsx | NOFX |
| 实时行情 | components/market/ | NOFX |
| 交易界面 | components/trading/ | 新增 |
| AI分析展示 | components/ai/ | TradingAgents-CN |

#### 阶段6: 测试与优化 (2周)

| 任务 | 说明 |
|------|------|
| 单元测试 | 核心模块测试 |
| 集成测试 | 服务间通信测试 |
| 回测验证 | 历史数据回测 |
| 性能优化 | 延迟优化、内存优化 |

### 11.2 时间表

```
2025年1月 - 2月:  阶段1 + 2（框架 + 数据）
2025年3月:        阶段3（AI决策引擎）
2025年4月:        阶段4（交易执行）
2025年5月:        阶段5（前端）
2025年6月:        阶段6（测试与优化）
```

---

## 12. 风险评估

### 12.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Go/Python通信延迟 | 中 | 使用高性能NATS，优化消息格式 |
| 券商API稳定性 | 高 | 支持多券商，自动切换 |
| LLM输出不可靠 | 中 | 多轮验证、人工审核 |
| 数据源中断 | 中 | 多数据源备份 |

### 12.2 业务风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 交易损失 | 高 | 严格风控、小资金测试 |
| 合规问题 | 高 | 遵守监管、不提供承诺收益 |
| 系统故障 | 中 | 完善监控、快速响应 |

### 12.3 法律风险

| 风险 | 缓解措施 |
|------|----------|
| 投资建议合规性 | 明确免责声明，不构成投资建议 |
| 数据使用合规 | 遵守数据使用协议 |
| 券商API使用 | 获得券商授权 |

---

## 13. 总结

### 13.1 融合价值

```
┌─────────────────────────────────────────────────────────────┐
│                   StockAgents 融合价值                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NOFX 贡献:                                                   │
│  ✅ 高性能Go执行引擎                                          │
│  ✅ 实时WebSocket行情                                         │
│  ✅ 完整的订单管理系统                                        │
│  ✅ 现代化React前端                                           │
│  ✅ 生产级部署方案                                            │
│                                                              │
│  TradingAgents-CN 贡献:                                       │
│  ✅ 多智能体协作框架                                          │
│  ✅ LangGraph工作流引擎                                       │
│  ✅ 中国数据源集成                                            │
│  ✅ 中文本地化完善                                            │
│  ✅ 提示词管理经验                                            │
│                                                              │
│  新增价值:                                                    │
│  ✅ A股/港股统一框架                                          │
│  ✅ T+1、涨跌停规则适配                                       │
│  ✅ 券商API对接                                               │
│  ✅ 回测系统                                                  │
│  ✅ 多LLM支持                                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 13.2 实施建议

1. **分阶段实施**: 先完成核心框架，再逐步添加功能
2. **小资金验证**: 使用小额资金验证策略有效性
3. **人工审核**: 初期AI决策需人工审核后执行
4. **持续优化**: 根据实际运行数据持续优化策略
5. **合规第一**: 遵守监管要求，不触碰法律红线

---

**方案结束**

*本方案基于NOFX和TradingAgents-CN两个项目的深度分析，融合两者优势，设计了一个适合A股/港股市场的AI智能交易系统。建议在充分测试和风险控制的前提下，分阶段实施。*
