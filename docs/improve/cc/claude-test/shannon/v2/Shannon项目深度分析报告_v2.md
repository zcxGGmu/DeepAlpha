# Shannon 项目深度分析报告 (增强版)

## 项目概览

**项目名称**: Shannon - Production AI Agents That Actually Work
**GitHub仓库**: https://github.com/Kocoro-lab/Shannon
**当前版本**: v0.1.0
**发布日期**: 2025-12-25
**许可证**: MIT License
**报告版本**: v2.0 (迭代优化版)

### 项目定位

Shannon 是一个经过实战检验的生产级 AI 智能体基础设施平台，专门解决规模化部署时的核心问题：

| 问题 | Shannon 解决方案 |
|------|-----------------|
| **成本失控** | 硬性 Token 预算控制，三层数学模型自动降级 |
| **非确定性故障** | Temporal 工作流支持时间旅行调试，完整重放能力 |
| **安全风险** | WASI 沙箱隔离 + OPA 策略管控 + 多租户隔离 |
| **可观测性缺失** | Prometheus + Grafana + OpenTelemetry 完整链路追踪 |

---

## 核心架构设计

### 技术栈组成

```
┌─────────────────────────────────────────────────────────────────┐
│                         客户端层                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ REST API │  │OpenAI API│  │Python SDK│  │桌面应用(Tauri)│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Go Gateway (端口 8080)                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  OpenAI 兼容转换器 | 认证中间件 | 速率限制 | 会话管理       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Go Orchestrator (端口 50052)                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Temporal 工作流 | 多智能体编排 | 任务分解 | 预算执行        │  │
│  │ OPA 策略引擎 | 熔断器 | 健康检查 | 降级策略                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                    ↓                              ↓
┌───────────────────────────────┐  ┌─────────────────────────────┐
│   Rust Agent Core (50051)     │  │   Python LLM Service (8000) │
│  ┌─────────────────────────┐  │  │  ┌───────────────────────┐ │
│  │ WASI 沙箱执行引擎       │  │  │  │ 15+ LLM 提供商抽象    │ │
│  │ 工具注册与发现系统     │  │  │  │ MCP 工具集成          │ │
│  │ LRU 缓存系统           │  │  │  │ OpenAPI 工具解析      │ │
│  │ 执行网关(超时/限流)    │  │  │  │ 提示优化              │ │
│  └─────────────────────────┘  │  │  └───────────────────────┘ │
└───────────────────────────────┘  └─────────────────────────────┘
                    ↓                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        数据层                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │PostgreSQL│  │  Redis   │  │  Qdrant  │  │  Temporal    │  │
│  │ 状态存储 │  │ 会话缓存 │  │ 向量检索 │  │ 工作流持久化 │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 三层架构深度解析

#### 1. Go Orchestrator (编排层) - 生产级工作流引擎

**核心职责**:
- 任务路由与复杂度分析
- Temporal 工作流编排
- 多租户隔离与 API Key 作用域
- Token 预算强制执行
- OPA 策略管控
- 熔断器与降级策略

**关键技术实现**:

```go
// OpenAI 兼容转换器实现
// go/orchestrator/cmd/gateway/internal/openai/translator.go

type Translator struct {
    registry *Registry
}

func (t *Translator) Translate(req *ChatCompletionRequest, userID, tenantID string) (*TranslatedRequest, error) {
    // 1. 提取查询消息
    query := t.extractQuery(req.Messages)

    // 2. 生成或派生会话 ID
    sessionID := t.deriveSessionID(req)

    // 3. 构建上下文映射
    ctxMap := t.buildContext(req, modelConfig)

    // 4. 转换为 protobuf 结构
    ctxStruct, _ := structpb.NewStruct(ctxMap)

    // 5. 构建 gRPC 请求
    grpcReq := &orchpb.SubmitTaskRequest{
        Metadata: &commonpb.TaskMetadata{
            UserId:    userID,
            TenantId:  tenantID,
            SessionId: sessionID,
            Labels:    labels,
        },
        Query:   query,
        Context: ctxStruct,
    }

    return &TranslatedRequest{...}, nil
}
```

**配置要点** (`config/shannon.yaml`):

```yaml
service:
  port: 50052
  health_port: 8081
  graceful_timeout: "60s"

session:
  max_history: 1000              # Redis 中每会话最大消息数
  ttl: "720h"                    # 会话过期时间 (30天)
  cache_size: 20000              # 本地缓存最大会话数
  token_budget_per_task: 10000000   # 每任务 Token 预算 (10M)
  token_budget_per_agent: 500000    # 每智能体 Token 预算 (500K)

# 熔断器配置
circuit_breakers:
  redis:
    max_requests: 5
    interval: "30s"
    timeout: "60s"
    max_failures: 5
    on_state_change: true
    enabled: true

# 降级策略
degradation:
  enabled: true
  mode_downgrade:
    minor_degradation_rules:
      complex_to_standard: true      # 轻度降级: 复杂→标准
    moderate_degradation_rules:
      complex_to_standard: true
      standard_to_simple: true       # 中度降级: 复杂→标准→简单
    severe_degradation_rules:
      force_simple_mode: true        # 重度降级: 强制简单模式
```

#### 2. Rust Agent Core (执行层) - 高性能安全执行引擎

**核心职责**:
- WASI 沙箱执行环境
- 工具发现与注册系统
- 执行网关 (超时/速率限制/熔断)
- LRU 缓存管理
- OpenTelemetry 链路追踪
- Prometheus 指标导出

**关键代码实现**:

```rust
// 执行网关实现 - enforcement.rs
// RequestEnforcer 提供统一的请求策略执行

pub struct RequestEnforcer {
    cfg: EnforcementConfig,
    buckets: Arc<Mutex<HashMap<String, TokenBucket>>,    // 速率限制桶
    breakers: Arc<Mutex<HashMap<String, RollingWindow>>, // 熔断器窗口
    redis: Option<RedisLimiter>,                          // 分布式限流
}

impl RequestEnforcer {
    pub async fn enforce<F, Fut, T>(&self, key: &str, est_tokens: usize, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        // 1. Token 限制检查
        if est_tokens > self.cfg.per_request_max_tokens {
            ENFORCEMENT_DROPS.with_label_values(&["token_limit"]).inc();
            return Err(anyhow!("token_limit_exceeded"));
        }

        // 2. 速率限制检查
        if self.rate_check(key).await.is_err() {
            ENFORCEMENT_DROPS.with_label_values(&["rate_limit"]).inc();
            return Err(anyhow!("rate_limit_exceeded"));
        }

        // 3. 熔断器检查
        if !self.cb_allow(key) {
            ENFORCEMENT_DROPS.with_label_values(&["circuit_breaker"]).inc();
            return Err(anyhow!("circuit_breaker_open"));
        }

        // 4. 执行请求
        let start = Instant::now();
        let result = tokio::spawn(f()).await.unwrap();

        // 5. 记录结果
        let ok = result.is_ok();
        self.cb_record(key, ok);

        result
    }
}
```

```rust
// 工具注册系统 - tool_registry.rs
// 提供工具发现、能力元数据管理

pub struct ToolCapability {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
    pub required_permissions: Vec<String>,
    pub estimated_duration_ms: u64,
    pub is_dangerous: bool,
    pub rate_limit: Option<RateLimit>,
    pub cache_ttl_ms: Option<u64>,
}

pub struct ToolRegistry {
    tools: Arc<RwLock<HashMap<String, ToolCapability>>>,
}

impl ToolRegistry {
    pub fn discover(&self, req: ToolDiscoveryRequest) -> ToolDiscoveryResponse {
        // 支持按类别、标签、危险级别过滤
        // 支持语义搜索工具描述
    }
}
```

**性能指标** (来自 README):

| 操作 | P50 延迟 | P99 延迟 | 吞吐量 | 备注 |
|------|----------|----------|--------|------|
| 工具发现 | 0.5ms | 2ms | 20k/s | 基于元数据快速检索 |
| 工具执行(缓存) | 0.1ms | 0.5ms | 50k/s | LRU 缓存命中 |
| 工具执行(无缓存) | 50ms | 200ms | 1k/s | 实际工具调用 |
| 执行网关 | 0.05ms | 0.2ms | 100k/s | 策略检查开销 |
| WASI 执行 | 10ms | 100ms | 500/s | 沙箱 Python 执行 |

#### 3. Python LLM Service (服务层) - 统一 LLM 抽象

**核心职责**:
- 15+ LLM 提供商统一抽象
- MCP (Model Context Protocol) 工具集成
- OpenAPI 规范自动解析 (~70% 覆盖率)
- 提示优化与缓存
- 模型层级选择

**关键代码实现**:

```python
# Python WASI 执行器实现
# python/llm-service/llm_service/tools/builtin/python_wasi_executor.py

class PythonWasiExecutorTool(Tool):
    """
    生产级 Python 执行器，使用 WASI 沙箱

    特性:
    - 完整 Python 3.11.4 标准库支持
    - 内存和 CPU 资源限制
    - 超时保护
    - 安全文件系统隔离
    - 会话持久化 (可选)
    """

    # 类级别解释器缓存
    _interpreter_cache: Optional[bytes] = None
    _sessions: Dict[str, ExecutionSession] = {}
    _max_sessions: int = 100
    _session_timeout: int = 3600  # 1小时

    def __init__(self):
        self.settings = Settings()
        self.interpreter_path = os.getenv(
            "PYTHON_WASI_WASM_PATH",
            "/opt/wasm-interpreters/python-3.11.4.wasm"
        )
        self.agent_core_addr = os.getenv("AGENT_CORE_ADDR", "agent-core:50051")
        super().__init__()
        self._load_interpreter_cache()

    def _get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="python_executor",
            version="2.0.0",
            description="Execute Python code in secure WASI sandbox with full stdlib support",
            category="code",
            timeout_seconds=30,
            memory_limit_mb=256,
            sandboxed=True,
            dangerous=False,  # 因 WASI 隔离而安全
            cost_per_use=0.001,
        )
```

**支持的 LLM 提供商** (15+):

```yaml
# config/models.yaml 部分列表

model_tiers:
  small:   # 目标分配: 50%
    providers:
      - provider: openai
        model: gpt-5-nano-2025-08-07
        priority: 2
      - provider: anthropic
        model: claude-haiku-4-5-20251001
        priority: 1
      - provider: zai
        model: glm-4.5-flash
        priority: 9

  medium:  # 目标分配: 40%
    providers:
      - provider: openai
        model: gpt-5-mini-2025-08-07
        priority: 2
      - provider: anthropic
        model: claude-sonnet-4-5-20250929
        priority: 1
      - provider: zai
        model: glm-4.5-air
        priority: 10

  large:   # 目标分配: 10%
    providers:
      - provider: openai
        model: gpt-5.1
        priority: 1
      - provider: anthropic
        model: claude-opus-4-1-20250805
        priority: 2
      - provider: zai
        model: glm-4.6
        priority: 9
```

**完整提供商列表**:
- OpenAI (GPT-5.1, GPT-5-mini, GPT-5-nano)
- Anthropic (Claude Opus/Sonnet/Haiku 4.5)
- Google (Gemini 2.5 Pro/Flash/Flash-Lite)
- DeepSeek (DeepSeek-V3.2, DeepSeek-R1)
- xAI (Grok-3-mini, Grok-4-fast)
- Z.ai (GLM-4.5-flash, GLM-4.5-air, GLM-4.6)
- Qwen (Qwen3-4b, 8b, omni-30b)
- Mistral (Ministral, Mistral-Small, Mistral-Nemo)
- Meta (Llama-3.2-3b, Llama-4-scout, Llama-3.1-405b)
- Cohere (Command-R)
- Groq (Llama-3.3-70b, Llama-3.1-8b)
- Ollama (本地模型支持)

---

## 核心能力深度分析

### 1. OpenAI 兼容 API - 零代码迁移

**实现原理**:
- Gateway 层实现完整的 OpenAI ChatCompletion API 规范
- 自动转换请求格式到 Shannon 内部 gRPC 协议
- 支持流式响应 (SSE) 和非流式响应

**使用示例**:

```bash
# 零代码替换 OpenAI API
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=sk-shannon-your-key

# 现有代码无需修改
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5-mini-2025-08-07",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**客户端库支持**:
- OpenAI Python SDK ✅
- OpenAI JavaScript SDK ✅
- LangChain ✅
- LlamaIndex ✅

### 2. 实时事件流 - 完整可观测性

**支持的传输协议**:
- **SSE (Server-Sent Events)**: 浏览器原生支持
- **WebSocket**: 双向实时通信
- **HTTP 长轮询**: 兼容模式

**事件类型**:

| 事件类型 | 触发时机 | 数据内容 |
|----------|----------|----------|
| `WORKFLOW_STARTED` | 工作流开始 | workflow_id, query, strategy |
| `WORKFLOW_COMPLETED` | 工作流完成 | result, token_usage, duration |
| `AGENT_STARTED` | 智能体开始 | agent_id, role, input |
| `AGENT_COMPLETED` | 智能体完成 | output, token_count |
| `TOOL_INVOKED` | 工具调用 | tool_name, parameters |
| `TOOL_OBSERVATION` | 工具返回 | result, duration |
| `LLM_PARTIAL` | LLM 流式输出 | delta_content |
| `LLM_OUTPUT` | LLM 完成 | full_content, model |
| `ERROR` | 任何错误 | error_type, message |

**SSE 流式监听示例**:

```bash
curl -N "http://localhost:8080/api/v1/stream/sse?workflow_id=task-dev-123"

# 输出示例:
data: {"type":"WORKFLOW_STARTED","workflow_id":"task-dev-123","query":"分析市场趋势"}

data: {"type":"AGENT_STARTED","agent_id":"agent-1","role":"researcher"}

data: {"type":"TOOL_INVOKED","tool_name":"web_search","parameters":{"query":"2025年市场趋势"}}

data: {"type":"TOOL_OBSERVATION","result":"根据最新数据..."}

data: {"type":"LLM_PARTIAL","delta_content":"基于搜索结果"}

data: {"type":"WORKFLOW_COMPLETED","result":"...", "token_usage":15432}
```

### 3. 研究工作流 - 多智能体协作

**支持的策略** (`config/research_strategies.yaml`):

| 策略 | 并发智能体 | ReAct 迭代 | 模型层级 | 验证 | 间隙填充 | 适用场景 |
|------|-----------|------------|----------|------|----------|----------|
| `quick` | 3 | 2 | small | ✗ | ✗ | 快速问答 |
| `standard` | 4 | 2 | medium | ✓ | 2 间隙, 1 迭代 | 标准研究 |
| `deep` | 5 | 3 | medium | ✓ | 3 间隙, 2 迭代 | 深度分析 |
| `academic` | 6 | 3 | medium | ✓ | 4 间隙, 3 迭代 | 学术研究 |

**工作流流程**:

```
用户查询
   ↓
[查询分解] → 生成子问题
   ↓
[并行 ReAct 智能体] → 每个智能体独立搜索 + 推理
   ↓
[验证阶段] → 交叉验证信息准确性
   ↓
[间隙填充] → 识别信息缺口并补充
   ↓
[最终综合] → 大模型生成综合报告 (带引用)
```

**API 调用示例**:

```bash
curl -X POST http://localhost:8080/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "query": "比较欧盟和美国的可再生能源采用情况",
    "context": {
      "force_research": true,
      "research_strategy": "deep",
      "max_budget_usd": 2.00
    }
  }'
```

**输出示例**:

```
根据综合分析，欧盟和美国在可再生能源采用方面存在以下差异:

[1] 欧盟 2024 年可再生能源占比达到 42%，超过美国 15 个百分点...
    来源: Eurostat 2024 Report, IEA World Energy Outlook 2024

[2] 美国在风能装机容量方面领先...
    来源: US EIA Wind Report 2024, GWEC Global Wind Report

[3] 政策差异方面...
    ...
```

### 4. 会话连续性 - 分层记忆系统

**记忆架构** (`docs/memory-system-architecture.md`):

```
┌─────────────────────────────────────────────────────────────┐
│                    分层记忆系统                              │
├─────────────────────────────────────────────────────────────┤
│  1. 近期记忆 (Recent Memory)                                │
│     - 最近 N 条消息 (来自当前会话)                           │
│     - 存储位置: Redis                                        │
│     - 检索方式: 时间序列                                     │
├─────────────────────────────────────────────────────────────┤
│  2. 语义记忆 (Semantic Memory)                              │
│     - 基于查询相似度的历史上下文                             │
│     - 存储位置: Qdrant 向量数据库                            │
│     - 检索方式: 向量相似度搜索                               │
├─────────────────────────────────────────────────────────────┤
│  3. 压缩摘要 (Compressed Summaries)                         │
│     - 旧对话的压缩表示                                       │
│     - 触发条件: 消息数/Token数超过阈值                        │
│     - 压缩模型: 大层级模型                                   │
├─────────────────────────────────────────────────────────────┤
│  4. 智能体记忆 (Agent Memory)                               │
│     - 智能体执行记录                                         │
│     - Token 使用和模型信息                                   │
│     - 工具执行和性能指标                                     │
├─────────────────────────────────────────────────────────────┤
│  5. Supervisor 记忆 (Supervisor Memory)                     │
│     - 任务分解模式                                           │
│     - 策略性能指标                                           │
│     - 团队组成配置                                           │
│     - 失败模式和缓解策略                                     │
└─────────────────────────────────────────────────────────────┘
```

**记忆检索策略**:

```python
# 分层检索实现伪代码
async def retrieve_hierarchical_memory(query, session_id, top_k=10):
    results = []

    # 1. 获取近期消息 (权重: 0.4)
    recent = await redis.zrange(f"session:{session_id}", 0, top_k//2)
    results.extend(recent)

    # 2. 语义搜索历史 (权重: 0.6)
    embedding = await generate_embedding(query)
    similar = await qdrant.search(
        collection="task_embeddings",
        vector=embedding,
        limit=top_k,
        filter={"session_id": session_id}
    )
    results.extend(similar)

    # 3. 去重和排序
    deduped = deduplicate_by_qa_id(results)
    sorted_results = mmr_rerank(deduped, lambda=0.7)

    return sorted_results[:top_k]
```

**配置选项**:

```yaml
# config/shannon.yaml
vector:
  enabled: true
  host: "qdrant"
  port: 6333
  top_k: 10
  threshold: 0.5
  # MMR (最大边际相关性) 多样性设置
  mmr_enabled: true
  mmr_lambda: 0.7           # 相关性(1.0)与多样性(0.0)的平衡
  mmr_pool_multiplier: 3    # 获取 3x 候选进行重排序

# 文档分块配置
embeddings:
  chunking:
    enabled: true
    max_tokens: 4000        # 目标块大小
    overlap_tokens: 400     # 块之间的重叠 (10%)
    min_chunk_tokens: 200   # 最小块大小
```

### 5. 定时任务 - Cron 工作流

**支持的 Cron 格式**:
- 标准 5 字段 Cron: `分 时 日 月 周`
- 秒级精度 Cron (可选): `秒 分 时 日 月 周`

**API 创建示例**:

```bash
curl -X POST http://localhost:8080/api/v1/schedules \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "name": "每日市场分析",
    "description": "每天早上9点分析市场趋势",
    "cron_expression": "0 9 * * *",
    "task_query": "分析最新的市场趋势和热点新闻",
    "max_budget_per_run_usd": 0.50,
    "enabled": true
  }'
```

**管理 API**:

```bash
# 列出所有定时任务
curl http://localhost:8080/api/v1/schedules

# 更新任务
curl -X PUT http://localhost:8080/api/v1/schedules/{schedule_id} \
  -d '{"enabled": false}'

# 删除任务
curl -X DELETE http://localhost:8080/api/v1/schedules/{schedule_id}

# 查看执行历史
curl http://localhost:8080/api/v1/schedules/{schedule_id}/executions
```

### 6. MCP 集成 - 标准化工具接口

**配置示例**:

```yaml
# config/shannon.yaml
mcp_tools:
  # 测试工具 (使用 Mock 端点)
  test_config_tool:
    enabled: true
    url: "http://localhost:8000/mcp/mock"
    func_name: "echo"
    description: "测试 MCP 工具"
    category: "test"
    cost_per_use: 0.002
    parameters:
      - name: "message"
        type: "string"
        required: true

  # 高德地图 API 示例
  gaode_maps:
    enabled: true
    url: "https://api.gaode.com/mcp"
    func_name: "geocode"
    description: "高德地图地理编码服务"
    category: "location"
    cost_per_use: 0.001
    headers:
      X-API-Key: "${GAODE_API_KEY}"
```

**安全配置**:

```bash
# .env
# 允许的 MCP 工具域名 (逗号分隔)
# 使用 "*" 允许所有域名 (仅开发环境)
MCP_ALLOWED_DOMAINS="api.example.com,api.gaode.com"
```

### 7. OpenAPI 工具 - 自动 API 集成

**配置示例**:

```yaml
# config/shannon.yaml
openapi_tools:
  # Swagger PetStore 示例
  petstore:
    enabled: true
    spec_url: "https://petstore3.swagger.io/api/v3/openapi.json"
    auth_type: "none"
    category: "data"
    base_cost_per_use: 0.001
    rate_limit: 30  # 每分钟请求数
    timeout_seconds: 30
    # 仅包含特定操作
    operations:
      - "getPetById"
      - "addPet"

  # GitHub API 示例 (Bearer Token)
  github:
    enabled: true
    spec_url: "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json"
    auth_type: "bearer"
    auth_config:
      token: "${GITHUB_TOKEN}"
    category: "development"
    operations:
      - "repos/get"
      - "repos/list-for-user"

  # 天气 API 示例 (API Key in Query)
  weather:
    enabled: true
    spec_url: "https://api.openweathermap.org/data/3.0/openapi.json"
    auth_type: "api_key"
    auth_config:
      api_key_name: "appid"
      api_key_location: "query"
      api_key_value: "${OPENWEATHER_API_KEY}"
    category: "data"
```

**安全配置**:

```bash
# .env
# 允许的 OpenAPI 工具域名
OPENAPI_ALLOWED_DOMAINS="petstore3.swagger.io,api.github.com,api.openweathermap.org"
```

---

## 多智能体工作流架构

### 架构层次

```
┌─────────────────────────────────────────────────────────────┐
│              Orchestrator Router (编排路由器)                │
│   查询分解 | 复杂度分析 | 策略选择 | 预算管理                  │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              Strategy Workflows (策略工作流)                 │
│   DAG | React | Research | Exploratory | Scientific          │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              Patterns Library (模式库)                       │
│   执行模式: Parallel | Sequential | Hybrid                  │
│   推理模式: React | Reflection | CoT | Debate | ToT        │
└─────────────────────────────────────────────────────────────┘
```

### 支持的策略工作流

#### 1. DAGWorkflow (有向无环图)
- **用途**: 通用任务分解和多智能体协调
- **执行模式**: 并行、顺序、混合(依赖图)
- **反射模式**: 可选质量改进

#### 2. ReactWorkflow (推理-行动-观察)
- **用途**: 需要工具使用和环境反馈的迭代问题解决
- **特点**: 维护观察窗口进行上下文跟踪

#### 3. ResearchWorkflow (研究工作流)
- **用途**: 信息收集和综合优化
- **简单研究**: 使用 React 模式
- **复杂研究**: 并行智能体 + 反思

#### 4. ExploratoryWorkflow (探索工作流)
- **用途**: 系统性探索未知解决方案空间
- **模式**: Tree-of-Thoughts 探索
- **回退**: 低置信度时使用 Debate 模式
- **最终**: Reflection 质量检查

#### 5. ScientificWorkflow (科学工作流)
- **用途**: 假设测试和系统分析
- **组合**:
  1. Chain-of-Thought → 生成假设
  2. Debate → 测试竞争假设
  3. Tree-of-Thoughts → 探索影响
  4. Reflection → 最终质量综合

### 模式组合示例

```go
// Scientific Workflow 组合实现伪代码

func ScientificWorkflow(ctx workflow.Context, input TaskInput) (TaskResult, error) {
    // 1. Chain-of-Thought: 生成假设
    hypotheses := ChainOfThought(ctx, input.Query, CoTConfig{
        MaxSteps: 5,
        ConfidenceThreshold: 0.7,
    })

    // 2. Debate: 测试假设
    tested := Debate(ctx, hypotheses, DebateConfig{
        Agents: 3,
        Rounds: 2,
    })

    // 3. Tree-of-Thoughts: 探索影响
    implications := TreeOfThoughts(ctx, tested, ToTConfig{
        MaxDepth: 3,
        BranchingFactor: 2,
    })

    // 4. Reflection: 最终综合
    result := Reflection(ctx, implications, ReflectionConfig{
        Criteria: []string{"completeness", "accuracy", "relevance"},
        ConfidenceThreshold: 0.8,
    })

    return result, nil
}
```

### 工作流选择逻辑

```mermaid
graph TD
    A[查询输入] --> B[分解与分析]
    B --> C{复杂度评分}
    C -->|< 0.3| D[Simple Task]
    C -->|>= 0.3| E{认知策略}

    D --> F[ExecuteSimpleTask]

    E -->|exploratory| G[ExploratoryWorkflow]
    E -->|scientific| H[ScientificWorkflow]
    E -->|react| I[ReactWorkflow]
    E -->|research| J[ResearchWorkflow]
    E -->|default| K[DAGWorkflow]

    G --> L[ToT + Debate + Reflection]
    H --> M[CoT + Debate + ToT + Reflection]
    I --> N[Reason-Act-Observe Loop]
    J --> O[React or Parallel + Reflection]
    K --> P[Parallel/Sequential/Hybrid + Reflection]
```

### 配置示例

```yaml
# config/shannon.yaml
cognitive_workflows:
  exploratory:
    max_iterations: 10
    confidence_threshold: 0.85
    branch_factor: 3
    max_concurrent_agents: 5

  react:
    max_iterations: 15
    observation_window: 5

  research:
    research_depth: 5
    sources_per_round: 6
    min_sources: 8
    max_concurrent_agents: 5

  scientific:
    max_hypotheses: 5
    max_iterations: 8
    confidence_threshold: 0.75
    contradiction_threshold: 0.3
```

---

## 模型层级与成本优化

### 三层模型架构

```yaml
# config/models.yaml
model_tiers:
  small:
    # 目标分配: 50% (非强制配额，实际取决于工作负载)
    # 快速、成本优化的模型，用于基础任务
    providers:
      - provider: openai
        model: gpt-5-nano-2025-08-07
        priority: 2
      - provider: anthropic
        model: claude-haiku-4-5-20251001
        priority: 1
      - provider: zai
        model: glm-4.5-flash
        priority: 9

  medium:
    # 目标分配: 40%
    # 标准模型，良好的能力/成本平衡
    providers:
      - provider: openai
        model: gpt-5-mini-2025-08-07
        priority: 2
      - provider: anthropic
        model: claude-sonnet-4-5-20250929
        priority: 1
      - provider: zai
        model: glm-4.5-air
        priority: 10

  large:
    # 目标分配: 10%
    # 重度推理模型，用于复杂任务
    providers:
      - provider: openai
        model: gpt-5.1
        priority: 1
      - provider: anthropic
        model: claude-opus-4-1-20250805
        priority: 2
      - provider: zai
        model: glm-4.6
        priority: 9
```

### 自动模型选择逻辑

```
任务复杂度评分 < 0.3 → small 层级
0.3 ≤ 评分 < 0.5 → medium 层级
评分 ≥ 0.5 → large 层级

# 研究工作流特殊逻辑:
# - 工具智能体: small/medium 层级
# - 最终综合: large 层级 (硬编码)
```

### 成本控制配置

```yaml
# config/models.yaml
cost_controls:
  max_cost_per_request: 2.00      # 单请求最大成本
  max_tokens_per_request: 100000   # 单请求最大 Token 数
  daily_budget_usd: 1000.0         # 每日预算
  alert_threshold_percent: 90      # 告警阈值

# 提供商特定定价 (每 1K Token USD)
pricing:
  models:
    openai:
      gpt-5-nano-2025-08-07:
        input_per_1k: 0.00005
        output_per_1k: 0.00040
      gpt-5-mini-2025-08-07:
        input_per_1k: 0.00025
        output_per_1k: 0.00200
      gpt-5.1:
        input_per_1k: 0.00125
        output_per_1k: 0.01000

    anthropic:
      claude-haiku-4-5-20251001:
        input_per_1k: 0.00010
        output_per_1k: 0.00050
      claude-sonnet-4-5-20250929:
        input_per_1k: 0.00030
        output_per_1k: 0.00150
      claude-opus-4-1-20250805:
        input_per_1k: 0.0150
        output_per_1k: 0.0750

    zai:
      glm-4.5-flash:
        input_per_1k: 0.0
        output_per_1k: 0.0
      glm-4.5-air:
        input_per_1k: 0.0002
        output_per_1k: 0.0011
      glm-4.6:
        input_per_1k: 0.0006
        output_per_1k: 0.0022
```

### 预算执行实现

```go
// Go 编排器中的预算执行中间件
func BudgetMiddleware(ctx workflow.Context, input TaskInput) (TaskResult, error) {
    // 1. 获取会话预算
    session := getSession(input.SessionID)
    remainingBudget := session.TokenBudget

    // 2. 估算所需 Token
    estimatedTokens := estimateTokens(input.Query)

    // 3. 检查预算
    if estimatedTokens > remainingBudget {
        // 自动降级到小模型
        input.ModelTier = "small"
    }

    // 4. 执行任务
    result := executeTask(ctx, input)

    // 5. 更新预算
    session.TokenBudget -= result.ActualTokenUsage

    return result, nil
}
```

---

## 安全与治理

### 1. WASI 沙箱 - 安全代码执行

**安全特性**:
- 无网络访问 (默认)
- 只读文件系统
- 内存限制 (可配置，默认 256MB)
- CPU 时间限制 (可配置，默认 30秒)
- 完整 Python 标准库支持

**Python WASI 执行器实现**:

```python
class PythonWasiExecutorTool(Tool):
    """
    WASI 沙箱 Python 执行器

    安全保证:
    - 使用 CPython 3.11.4 WebAssembly 版本
    - WASI 提供系统调用沙箱
    - 阻止所有文件系统写操作
    - 阻止网络访问
    """

    def execute(self, code: str, timeout: int = 30) -> ToolResult:
        # 1. 代码静态分析
        ast.parse(code)  # 验证语法

        # 2. 通过 gRPC 调用 Rust Agent Core
        stub = agent_pb2_grpc.AgentCoreServiceStub(self.channel)
        request = agent_pb2.ExecuteWasiRequest(
            code=code,
            timeout_ms=timeout * 1000,
            memory_limit_mb=256,
        )

        # 3. 执行并获取结果
        response = stub.ExecuteWasi(request)

        return ToolResult(
            output=response.output,
            error=response.error,
            execution_time_ms=response.duration_ms,
        )
```

**测试安全隔离**:

```bash
# 尝试危险操作
./scripts/submit_task.sh "Execute Python: import os; os.system('rm -rf /')"

# 结果: OSError - system calls blocked by WASI sandbox
```

### 2. OPA 策略管控 - 细粒度访问控制

**策略示例**:

```rego
# config/opa/policies/teams.rego
package shannon.teams

# 允许数据科学团队使用特定模型
allow {
    input.team == "data-science"
    input.model in ["gpt-5-2025-08-07", "claude-sonnet-4-5-20250929"]
}

# 拒绝支持团队执行数据库写操作
deny_tool["database_write"] {
    input.team == "support"
}

# 拒绝未授权团队使用危险工具
deny {
    not input.team in ["data-science", "engineering", "research"]
    input.tool_dangerous == true
}
```

**策略执行流程**:

```
用户请求
   ↓
提取上下文 (user_id, team, tool, model)
   ↓
查询 OPA 策略
   ↓
┌─────────────┐
│  OPA 引擎   │
│  - allow    │
│  - deny     │
│  - default  │
└─────────────┘
   ↓
策略决策 (allow/deny)
   ↓
执行或拒绝请求
```

### 3. 多租户隔离

**隔离机制**:

```go
type TenantContext struct {
    TenantID  string
    UserID    string
    APIKeyID  string
    Labels    map[string]string
}

// 数据库查询自动添加租户过滤
func (r *TaskRepository) GetTasks(ctx context.Context, tenantID string) ([]Task, error) {
    query := `SELECT * FROM task_executions WHERE tenant_id = $1`
    return r.db.Query(ctx, query, tenantID)
}

// Redis 会话隔离
func (s *SessionManager) GetSession(ctx context.Context, tenantID, sessionID string) (*Session, error) {
    key := fmt.Sprintf("tenant:%s:session:%s", tenantID, sessionID)
    return s.redis.Get(ctx, key)
}

// Qdrant 向量存储隔离
func (v *VectorStore) Search(ctx context.Context, tenantID string, query []float32) ([]Result, error) {
    return v.client.Search(collection, search.WithFilter(
        must.NewMatch("tenant_id", tenantID),
    ))
}
```

### 4. API Key 管理

**安全存储**:

```go
// API Key 哈希存储
type APIKey struct {
    ID          string    // UUID
    KeyHash     string    // SHA-256 哈希
    KeyPrefix   string    // 前 8 字符 (用于识别)
    TenantID    string
    UserID      string
    CreatedAt   time.Time
    LastUsedAt  time.Time
}

// 验证 API Key
func (s *APIKeyService) ValidateKey(key string) (*APIKey, error) {
    // 1. 提取前缀
    prefix := extractPrefix(key) // "sk-shannon-xxxx"

    // 2. 查找匹配前缀的密钥
    hashed, err := s.repo.FindByPrefix(prefix)
    if err != nil {
        return nil, ErrInvalidKey
    }

    // 3. 验证哈希
    if !compareHash(key, hashed.KeyHash) {
        return nil, ErrInvalidKey
    }

    // 4. 更新最后使用时间
    hashed.LastUsedAt = time.Now()
    s.repo.Update(hashed)

    return hashed, nil
}
```

---

## 前端与客户端

### 1. 原生桌面应用 (Tauri + Next.js)

**支持平台**:
- macOS: Universal Binary (Intel + Apple Silicon) - `.dmg` / `.app.tar.gz`
- Windows: x64 - `.msi` / `.exe`
- Linux: x64 - `.AppImage` / `.deb`

**技术栈**:
- **前端**: Next.js 14 + React + TypeScript
- **桌面框架**: Tauri 2.0
- **状态管理**: Zustand
- **本地数据库**: Dexie.js (IndexedDB)
- **样式**: Tailwind CSS + shadcn/ui

**构建方式**:

```bash
cd desktop

# 安装依赖
npm install

# 开发模式 (Web UI)
npm run dev
# 访问 http://localhost:3000

# 构建桌面应用
npm run tauri:build

# 输出目录:
# - macOS: src-tauri/target/release/bundle/dmg/
# - Windows: src-tauri/target/release/bundle/msi/
# - Linux: src-tauri/target/release/bundle/deb/
```

**核心功能**:
- ✅ 系统托盘集成和原生通知
- ✅ 离线任务历史 (Dexie.js 本地数据库)
- ✅ 更低内存占用和更好性能
- ✅ GitHub Releases 自动更新
- ✅ 实时事件时间线可视化
- ✅ 多模型提供商切换
- ✅ 深度研究模式

### 2. Python SDK

**安装**:

```bash
pip install shannon-sdk
```

**使用示例**:

```python
from shannon import ShannonClient

# 创建客户端
with ShannonClient(base_url="http://localhost:8080") as client:
    # 提交任务
    handle = client.submit_task(
        "法国的首都是哪里?",
        session_id="demo-session",
        config={
            "max_tokens": 1000,
            "model_tier": "small"
        }
    )

    # 等待完成
    result = client.wait(handle.task_id)

    print(f"结果: {result.result}")
    print(f"Token 使用: {result.token_usage}")
    print(f"耗时: {result.duration_ms}ms")
```

**流式响应**:

```python
async with ShannonClient(base_url="http://localhost:8080") as client:
    # 流式提交
    async for event in client.stream_task(
        "分析市场趋势",
        session_id="market-session"
    ):
        if event.type == "LLM_PARTIAL":
            print(event.delta_content, end="", flush=True)
        elif event.type == "TOOL_INVOKED":
            print(f"\n[工具调用] {event.tool_name}")
        elif event.type == "WORKFLOW_COMPLETED":
            print(f"\n[完成] Token: {event.token_usage}")
```

### 3. CLI 工具

**安装**:

```bash
pip install shannon-sdk
# CLI 自动安装为 shannon 命令
```

**使用示例**:

```bash
# 基础查询
shannon submit "什么是GDP?"

# 指定会话 ID
shannon submit "它与通胀有什么关系?" --session-id econ-101

# 研究模式
shannon submit "比较欧盟和美国的可再生能源采用情况" \
  --research-strategy deep \
  --max-budget-usd 2.00

# 流式输出
shannon submit "分析市场趋势" --stream

# 查看会话历史
shannon history econ-101

# 查看任务详情
shannon get task-abc-123
```

### 4. OpenAI 兼容 API

**使用现有代码**:

```python
from openai import OpenAI

# 只需更改 base URL
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-shannon-your-key"
)

# 所有现有代码无需修改
response = client.chat.completions.create(
    model="gpt-5-mini-2025-08-07",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

**流式响应**:

```python
stream = client.chat.completions.create(
    model="gpt-5-mini-2025-08-07",
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 部署与运维

### 端口映射

| 服务 | 端口 | 协议 | 用途 |
|------|------|------|------|
| Gateway | 8080 | HTTP | REST API、OpenAI 兼容 `/v1` |
| Admin/Events | 8081 | HTTP/WS | SSE/WebSocket 流式传输、健康检查 |
| Orchestrator | 50052 | gRPC | 内部服务通信 |
| Agent Core | 50051 | gRPC | Rust 执行层 |
| LLM Service | 8000 | HTTP | Python LLM 服务 |
| Temporal | 7233 | gRPC | Temporal 工作流引擎 |
| Temporal UI | 8088 | HTTP | 工作流调试界面 |
| PostgreSQL | 5432 | TCP | 状态存储 |
| Redis | 6379 | TCP | 会话缓存 |
| Qdrant | 6333 | HTTP | 向量数据库 |
| Grafana | 3030 | HTTP | 指标仪表板 |
| Prometheus | 9090 | HTTP | 指标采集 |

### 快速安装

```bash
# 一键安装脚本
curl -fsSL https://raw.githubusercontent.com/Kocoro-lab/Shannon/v0.1.0/scripts/install.sh | bash

# 脚本会自动:
# 1. 下载配置文件
# 2. 提示输入 API Keys
# 3. 拉取 Docker 镜像
# 4. 启动所有服务
```

### Docker Compose 部署

```bash
# 1. 克隆仓库
git clone https://github.com/Kocoro-lab/Shannon.git
cd Shannon

# 2. 配置环境变量
cp .env.example .env
nano .env

# 必需配置:
OPENAI_API_KEY=sk-...
# 或
ANTHROPIC_API_KEY=sk-ant-...

# 可选配置:
WEB_SEARCH_PROVIDER=serpapi
SERPAPI_API_KEY=...

# 3. 启动服务
docker compose -f deploy/compose/docker-compose.release.yml up -d

# 4. 检查状态
docker compose -f deploy/compose/docker-compose.release.yml ps

# 5. 查看日志
docker compose -f deploy/compose/docker-compose.release.yml logs -f
```

### 从源码构建

```bash
# 1. 克隆仓库
git clone https://github.com/Kocoro-lab/Shannon.git
cd Shannon

# 2. 环境设置
make setup
# 这会:
# - 创建 .env 文件
# - 生成 protobuf 文件
# - 链接配置文件

# 3. 添加 API Keys
echo "OPENAI_API_KEY=sk-..." >> .env

# 4. 设置 Python WASI (用于代码执行)
./scripts/setup_python_wasi.sh

# 5. 启动所有服务 (本地构建)
make dev

# 6. 运行测试
make smoke  # E2E 烟雾测试
```

### 健康检查

```bash
# Gateway 健康检查
curl http://localhost:8080/health
# 返回: {"status":"healthy","services":{...}}

# Admin 健康检查
curl http://localhost:8081/health

# 检查所有服务
docker compose ps
```

### 日志查看

```bash
# 所有服务日志
docker compose -f deploy/compose/docker-compose.release.yml logs -f

# 特定服务日志
docker compose logs -f orchestrator
docker compose logs -f gateway
docker compose logs -f llm-service
docker compose logs -f agent-core

# 最近 100 行
docker compose logs --tail=100 orchestrator
```

---

## 可观测性

### Prometheus 指标

**Agent Core 指标** (`http://localhost:2113/metrics`):

```
# 工具执行计数
agent_tool_executions_total{tool_name="web_search"} 1234

# 工具执行延迟
agent_tool_execution_duration_seconds{tool_name="web_search",quantile="0.5"} 0.05
agent_tool_execution_duration_seconds{tool_name="web_search",quantile="0.99"} 0.20

# 缓存命中
agent_cache_hits_total 456
agent_cache_misses_total 78

# 内存使用
agent_memory_usage_bytes 134217728

# 活动任务
agent_active_tasks 5
```

**Orchestrator 指标**:

```
# 工作流计数
shannon_workflows_total{strategy="dag",status="completed"} 234
shannon_workflows_total{strategy="research",status="failed"} 2

# Token 使用
shannon_tokens_consumed_total{model_tier="small"} 50000
shannon_tokens_consumed_total{model_tier="medium"} 30000
shannon_tokens_consumed_total{model_tier="large"} 5000

# 预算执行
shannon_budget_enforced_total{reason="token_limit"} 5
shannon_budget_enforced_total{reason="rate_limit"} 12
```

**Gateway 指标**:

```
# HTTP 请求
http_requests_total{method="POST",endpoint="/v1/tasks",status="200"} 456

# 请求延迟
http_request_duration_seconds{endpoint="/v1/tasks",quantile="0.5"} 0.5
http_request_duration_seconds{endpoint="/v1/tasks",quantile="0.99"} 2.3

# OpenAI 兼容请求
openai_compatibility_requests_total 789
openai_compatibility_translations_total 789
```

### OpenTelemetry 链路追踪

**配置**:

```bash
# .env
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
OTEL_SERVICE_NAME=shannon-orchestrator
OTEL_ENABLED=true
```

**追踪示例**:

```mermaid
sequenceDiagram
    participant Client
    participant Gateway
    participant Orchestrator
    participant AgentCore
    participant LLMService

    Client->>Gateway: HTTP POST /v1/tasks
    Gateway->>Orchestrator: gRPC SubmitTask
    Orchestrator->>AgentCore: gRPC ExecuteAgent
    AgentCore->>LLMService: HTTP GenerateCompletion
    LLMService-->>AgentCore: LLM Response
    AgentCore-->>Orchestrator: Agent Result
    Orchestrator-->>Gateway: Task Result
    Gateway-->>Client: HTTP 200 Response
```

**Jaeger UI**: `http://localhost:16686`

### Grafana 仪表板

**预配置仪表板**:

1. **系统概览** (`deploy/compose/grafana/config/provisioning/dashboards/node-exporter-1860.json`)
   - CPU、内存、磁盘使用率
   - 网络流量
   - 系统负载

2. **层级漂移监控** (`tier-drift-monitoring.json`)
   - 实际 vs 目标模型层级分布
   - 成本差异分析
   - 自动降级事件

**访问**: `http://localhost:3030`

**默认凭据**:
- 用户: `admin`
- 密码: `admin` (首次登录后更改)

### 告警配置

**Prometheus 告警规则** (可选):

```yaml
# alerting_rules.yml
groups:
  - name: shannon_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(shannon_workflows_total{status="failed"}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High workflow error rate"

      - alert: BudgetExceeded
        expr: shannon_daily_cost_usd > 100
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Daily budget exceeded"

      - alert: ServiceDown
        expr: up{job=~"shannon.*"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Shannon service is down"
```

---

## 竞品对比分析 (增强版)

### 详细功能对比

| 能力 | Shannon | LangGraph | Dify | AutoGen | CrewAI | LangChain |
|------|---------|-----------|------|---------|--------|-----------|
| **定时任务** | ✓ Cron 工作流 | ✗ | ⚠ 基础 | ✗ | ✗ | ✗ |
| **研究工作流** | ✓ 4种策略 | ⚠ 手动 | ⚠ 手动 | ⚠ 手动 | ⚠ 手动 | ✗ |
| **确定性重放** | ✓ 时间旅行调试 | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Token 预算** | ✓ 硬限制+自动降级 | ✗ | ✗ | ✗ | ✗ | ✗ |
| **安全沙箱** | ✓ WASI 隔离 | ⚠ Docker | ⚠ Docker | ⚠ Docker | ⚠ Docker | ⚠ Docker |
| **OPA 策略** | ✓ 细粒度管控 | ✗ | ⚠ 基础 | ✗ | ✗ | ✗ |
| **生产指标** | ✓ Prometheus+Grafana | ⚠ 需自行搭建 | ⚠ 基础 | ✗ | ✗ | ⚠ LangSmith |
| **原生桌面** | ✓ macOS/Win/Linux | ✗ | ✗ | ✗ | ✗ | ✗ |
| **多语言核心** | ✓ Go/Rust/Python | ⚠ Python | ⚠ Python | ⚠ Python | ⚠ Python | ⚠ Python |
| **会话持久化** | ✓ Redis+PostgreSQL | ⚠ 内存 | ✓ 数据库 | ⚠ 有限 | ✗ | ⚠ 内存 |
| **多智能体** | ✓ DAG/Supervisor/5策略 | ✓ 图结构 | ⚠ 工作流 | ✓ 群聊 | ✓ Crews | ⚠ Chain |
| **向量记忆** | ✓ Qdrant+MMR | ⚠ 需集成 | ✓ 向量库 | ⚠ 需集成 | ⚠ 需集成 | ⚠ 需集成 |
| **OpenAI 兼容** | ✓ 完整 API | ✗ | ⚠ 部分 | ✗ | ✗ | ✗ |
| **MCP 集成** | ✓ 原生支持 | ⚠ 需集成 | ⚠ 需集成 | ⚠ 需集成 | ⚠ 需集成 | ⚠ 需集成 |
| **OpenAPI 工具** | ✓ 70% 覆盖率 | ⚠ 手动 | ⚠ 手动 | ⚠ 手动 | ⚠ 手动 | ⚠ 手动 |
| **LLM 提供商** | 15+ | 10+ | 5+ | 5+ | 5+ | 10+ |
| **代码执行** | ✓ WASI 安全 | ⚠ 限制 | ⚠ 限制 | ⚠ 限制 | ⚠ 限制 | ⚠ 限制 |
| **降级策略** | ✓ 自动多级 | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Human-in-Loop** | ✓ 审批工作流 | ✗ | ⚠ 基础 | ⚠ 基础 | ⚠ 基础 | ✗ |
| **P2P 协调** | ✓ 智能体间通信 | ✗ | ✗ | ⚠ 有限 | ⚠ 有限 | ✗ |

### 架构对比

| 维度 | Shannon | LangGraph | Dify | AutoGen | CrewAI |
|------|---------|-----------|------|---------|--------|
| **架构模式** | 微服务 (Go+Rust+Python) | 单体 (Python) | 微服务 (Python) | 单体 (Python) | 单体 (Python) |
| **编排引擎** | Temporal | LangGraph Runtime | 自研 | 自研 | 自研 |
| **状态存储** | PostgreSQL+Redis | 内存 | 数据库 | 内存 | 内存 |
| **向量存储** | Qdrant | 需集成 | 向量库 | 需集成 | 需集成 |
| **追踪** | OpenTelemetry | LangSmith | 自研 | ✗ | ✗ |
| **部署** | Docker Compose/K8s | 容器 | K8s | 容器 | 容器 |
| **配置方式** | YAML+环境变量 | Python 代码 | UI+YAML | Python 代码 | Python 代码 |

### 成本对比

| 成本类型 | Shannon | LangGraph | Dify | AutoGen | CrewAI |
|----------|---------|-----------|------|---------|--------|
| **软件许可** | MIT (免费) | MIT (免费) | Apache 2.0 (免费) | MIT (免费) | MIT (免费) |
| **基础设施** | 最低 (多层优化) | 中等 | 中等 | 中等 | 中等 |
| **Token 优化** | ✓ 三层自动降级 | ✗ | ✗ | ✗ | ✗ |
| **监控成本** | 内置 (无额外) | LangSmith ($$$) | 内置 | ✗ | ✗ |

### Shannon 独有优势

1. **生产级可靠性**
   - Temporal 工作流引擎 (Uber 开源)
   - 确定性重放调试
   - 完整的故障恢复机制

2. **精细化成本控制**
   - 三层模型自动选择
   - 硬性 Token 预算
   - 实时成本追踪和告警

3. **安全优先**
   - WASI 沙箱 (真·隔离)
   - OPA 策略引擎 (Cloud Native 标准)
   - 多租户隔离

4. **完整可观测性**
   - Prometheus + Grafana (开源标准)
   - OpenTelemetry 链路追踪
   - 预配置仪表板

5. **多语言架构**
   - Go: 编排和并发 (天然优势)
   - Rust: 执行和安全 (内存安全)
   - Python: LLM 集成 (生态丰富)

6. **开箱即用**
   - OpenAI 兼容 API (零迁移)
   - 原生桌面应用
   - Python SDK + CLI

### 何时选择 Shannon

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| 企业级 AI 平台 | **Shannon** | 多租户、策略管控、审计追踪 |
| 成本敏感应用 | **Shannon** | 精细化成本控制、自动降级 |
| 安全关键场景 | **Shannon** | WASI 沙箱、OPA 策略 |
| 研究型应用 | **Shannon** | 多源检索、引用生成 |
| 快速原型 | LangGraph | Python 纯代码、快速迭代 |
| 低代码平台 | Dify | 可视化工作流、非友好 |
| 对话系统 | AutoGen | 群聊模式、角色扮演 |
| 任务编排 | CrewAI | 明确分工、队伍模式 |
| 通用集成 | LangChain | 最大生态、丰富集成 |

---

## 路线图分析

### v0.1 - 生产就绪 (当前) ✅

**已完成功能** (31 项):

- ✅ 核心平台稳定 (Go/Rust/Python)
- ✅ 确定性重放调试
- ✅ OPA 策略执行
- ✅ WebSocket/SSE 流式传输
- ✅ WASI 沙箱
- ✅ 多智能体编排 (5 种策略)
- ✅ 向量记忆 (Qdrant + MMR)
- ✅ 分层记忆管理
- ✅ Token 预算管理
- ✅ MCP 集成
- ✅ OpenAPI 集成 (~70% 覆盖率)
- ✅ 统一 Gateway & SDK
- ✅ 原生桌面应用 (macOS/Win/Linux)
- ✅ OpenAI 兼容 API
- ✅ 定时任务 (Cron)
- ✅ 研究工作流 (4 种策略)
- ✅ 会话连续性
- ✅ 15+ LLM 提供商支持
- ✅ 模型分层自动选择
- ✅ 降级策略 (三级)
- ✅ 熔断器模式
- ✅ 健康检查
- ✅ Prometheus 指标
- ✅ Grafana 仪表板
- ✅ OpenTelemetry 追踪
- 🚧 Docker 镜像发布 (进行中)

### v0.2 - 增强能力 (规划中)

**SDK & UI** (2 项):
- [ ] TypeScript/JavaScript SDK (npm 包)
- [ ] 可选拖拽式 UI (AgentKit-like)

**内置工具扩展** (1 项):
- [ ] 更多定制化工具

**平台增强** (10 项):
- [ ] 高级记忆 (情景摘要、实体/时序知识图谱、混合密集+稀疏检索)
- [ ] 高级学习 (模式识别、基于上下文盗匪的智能体选择)
- [ ] 智能体协作基础 (角色/人设、智能体特定记忆、监督者层级)
- [ ] MMR 多样性重排序 (当前配置就绪，40% 完成)
- [ ] 基于性能的智能体选择 (使用 agent_executions 指标的 ε-贪婪路由)
- [ ] 上下文流式事件 (新增 4 种事件类型)
- [ ] Supervisor 中的预算执行
- [ ] 用例预设 (调试/分析模式的 YAML 预设)
- [ ] 辩论结果持久化 (存储到 Qdrant 用于学习)
- [ ] 共享工作空间功能
- [ ] 智能工具选择 (语义工具结果缓存、智能体经验学习)
- [ ] 原生 RAG 系统 (文档分块、知识库集成、上下文注入)
- [ ] 团队级配额与策略 (每团队预算、模型/工具白名单)

### v0.3 - 企业级与规模化 (远期)

**区块链集成** (1 项):
- [ ] Solana 集成 (去中心化信任、链上证明、区块链审计日志)

**生产可观测性** (1 项):
- [ ] 分布式追踪、自定义 Grafana 仪表板、SLO 监控

**企业功能** (3 项):
- [ ] SSO 集成
- [ ] 多租户隔离 (增强)
- [ ] 审批工作流

**边缘部署** (1 项):
- [ ] WASM 浏览器执行、离线优先能力

**自主智能** (3 项):
- [ ] 自组织智能体群体
- [ ] 批评/反思循环
- [ ] 群聊协调

**跨组织** (2 项):
- [ ] 安全跨租户智能体通信
- [ ] 能力协商协议

**监管合规** (3 项):
- [ ] SOC 2
- [ ] GDPR
- [ ] HIPAA 自动化 + 审计追踪

**AI 安全** (1 项):
- [ ] 宪政 AI、对齐机制、对抗测试

**个性化** (1 项):
- [ ] 从用户成功任务模式学习、用户特定交互微调模型

---

## 技术亮点深度解析

### 1. 分层配置系统

```
环境变量 (.env)
     ↓
YAML 配置 (config/)
     ↓
运行时覆盖 (API 参数)
```

**配置文件优先级**:

1. **环境变量** (.env) - 最高优先级，用于敏感信息
2. **YAML 配置** (config/) - 结构化配置
3. **默认值** - 代码中硬编码

**核心配置文件**:

| 文件 | 用途 | 热重载 |
|------|------|--------|
| `config/models.yaml` | LLM 提供商、定价、层级 | ❌ |
| `config/features.yaml` | 功能开关、工作流设置 | ❌ |
| `config/shannon.yaml` | 编排器配置 | ✅ |
| `config/personas.yaml` | 智能体角色定义 (规划中) | ❌ |
| `config/research_strategies.yaml` | 研究策略配置 | ❌ |
| `config/opa/policies/` | 访问控制规则 | ✅ |

**热重载示例**:

```bash
# 修改 config/shannon.yaml
vim config/shannon.yaml

# 发送 SIGHUP 信号重载配置
docker kill -s HUP $(docker ps -q --filter name=orchestrator)

# 或使用 API
curl -X POST http://localhost:8081/admin/reload-config
```

### 2. 降级策略实现

**降级触发条件**:

```yaml
# config/shannon.yaml
degradation:
  enabled: true
  check_interval: "30s"
  thresholds:
    minor:
      circuit_breakers_open: 1
      critical_dependencies_down: 0
      non_critical_dependencies_down: 1
      error_rate_percentage: 5.0
      latency_threshold_ms: 1000
    moderate:
      circuit_breakers_open: 2
      critical_dependencies_down: 1
      non_critical_dependencies_down: 2
      error_rate_percentage: 15.0
      latency_threshold_ms: 2000
    severe:
      circuit_breakers_open: 3
      critical_dependencies_down: 2
      non_critical_dependencies_down: 3
      error_rate_percentage: 30.0
      latency_threshold_ms: 5000
```

**降级决策逻辑**:

```go
func EvaluateDegradation(metrics *SystemMetrics) DegradationLevel {
    score := 0

    if metrics.CircuitBreakersOpen >= thresholds.Minor.CircuitBreakersOpen {
        score++
    }
    if metrics.ErrorRate >= thresholds.Minor.ErrorRatePercentage {
        score++
    }
    if metrics.LatencyMs >= thresholds.Minor.LatencyThresholdMs {
        score++
    }

    switch score {
    case 1:
        return DegradationMinor
    case 2:
        return DegradationModerate
    case 3:
        return DegradationSevere
    default:
        return DegradationNone
    }
}

func ApplyDegradation(level DegradationLevel, workflow *WorkflowConfig) {
    switch level {
    case DegradationMinor:
        workflow.ComplexToStandard = true
    case DegradationModerate:
        workflow.ComplexToStandard = true
        workflow.StandardToSimple = true
    case DegradationSevere:
        workflow.ForceSimpleMode = true
    }
}
```

### 3. 工具系统架构

**工具类型**:

1. **内置工具** (Built-in):
   - `calculator` - 计算器
   - `web_search` - 网络搜索 (SerpAPI、Google、Bing、Exa)
   - `python_wasi_executor` - Python WASI 沙箱执行
   - `file_ops` - 文件操作
   - `session_file` - 会话文件管理

2. **MCP 工具**:
   - 通过 `config/shannon.yaml` 配置
   - 支持自定义 MCP 端点
   - OAuth2 服务器认证

3. **OpenAPI 工具**:
   - 自动解析 OpenAPI 规范
   - ~70% API 覆盖率
   - 支持 API Key、Bearer、Basic 认证

**工具执行流程**:

```
工具调用请求
   ↓
工具发现 (ToolRegistry)
   ↓
能力检查 (required_permissions, is_dangerous)
   ↓
执行网关 (Enforcement Gateway)
   ├─ 超时检查
   ├─ 速率限制
   ├─ Token 限制
   └─ 熔断器检查
   ↓
缓存检查 (LRU Cache)
   ├─ 命中 → 返回缓存结果
   └─ 未命中 → 继续执行
   ↓
实际执行
   ├─ 内置工具 → 直接调用
   ├─ MCP 工具 → HTTP/gRPC 调用
   └─ OpenAPI 工具 → HTTP 请求
   ↓
结果缓存
   ↓
返回结果
```

### 4. 智能体角色系统 (规划中)

**当前状态**: `config/personas.yaml` 已定义但**未启用**

**当前可用方式**:

```python
# 方式 1: 选择预设角色
context["role"] = "researcher"  # generalist, analysis, research, writer, critic

# 方式 2: 完全自定义系统提示
context["system_prompt"] = "你是一个专业的金融分析师..."
```

**规划中的角色**:

| 角色 | 描述 | 工具集 | Token 预算 | 温度 |
|------|------|--------|-----------|------|
| `generalist` | 通用助手 | 所有工具 | medium | 0.7 |
| `researcher` | 研究专家 | 搜索、读取、分析工具 | high | 0.3 |
| `coder` | 编程专家 | 代码执行、文件操作 | medium | 0.2 |
| `analyst` | 数据分析专家 | Python、数据查询、图表 | high | 0.4 |

**实现计划**:
1. 完成 `personas.yaml` 解析器
2. 实现关键词匹配引擎
3. 实现 LLM 辅助角色选择
4. 实现工具白名单映射
5. 集成到工作流路由器

---

## 开发者体验

### Makefile 命令速查

```bash
# === 环境设置 ===
make setup              # 完整设置 (首次克隆)
make setup-env          # 仅环境设置

# === 开发 ===
make dev                # 启动所有服务
make down               # 停止所有服务
make logs               # 查看日志
make ps                 # 服务状态

# === 代码质量 ===
make fmt                # 格式化代码 (Go/Rust/Python)
make lint               # 运行 linter
make proto              # 生成 protobuf 文件
make proto-local        # 本地生成 (BSR 限速时)

# === 测试 ===
make test               # 运行所有测试
make smoke              # E2E 烟雾测试
make ci                 # CI 检查

# === 重放调试 ===
make replay-export WORKFLOW_ID=xxx OUT=history.json
make replay HISTORY=history.json

# === 覆盖率 ===
make coverage           # 覆盖率报告
make coverage-go        # Go 覆盖率
make coverage-python    # Python 覆盖率

# === 集成测试 ===
make integration-tests          # 所有集成测试
make integration-single         # 单智能体流程测试
make integration-session        # 会话记忆测试
make integration-qdrant         # Qdrant 向量数据库测试
```

### 测试结构

**Go 测试**:
```
go/orchestrator/
├── tests/replay/workflow_replay_test.go
├── cmd/gateway/internal/handlers/task_test.go
└── cmd/gateway/internal/openai/translator_test.go
```

**Rust 测试**:
```
rust/agent-core/tests/
├── test_full_integration.rs
├── tool_calls_sequence.rs
├── stress_test_metrics.rs
└── test_wasm_cache.rs
```

**Python 测试**:
```
python/llm-service/tests/
├── test_tier_selection.py
├── test_decomposition_patterns.py
├── test_vendor_adapters.py
└── test_openapi_tool.py
```

### 项目目录结构

```
Shannon/
├── go/                          # Go 编排器
│   └── orchestrator/
│       ├── cmd/gateway/         # API Gateway
│       │   ├── internal/
│       │   │   ├── openai/      # OpenAI 兼容层
│       │   │   ├── handlers/    # HTTP 处理器
│       │   │   └── middleware/  # 认证、限流、追踪
│       │   └── main.go
│       ├── internal/            # 内部模块
│       │   ├── activities/      # Temporal Activities
│       │   ├── workflows/       # Temporal Workflows
│       │   ├── strategies/      # 策略工作流 (DAG, React, etc.)
│       │   ├── patterns/        # 可组合模式
│       │   └── middleware/      # 预算、降级
│       └── tests/               # Go 测试
├── rust/                        # Rust 执行层
│   └── agent-core/
│       ├── src/                 # Rust 源码
│       │   ├── enforcement.rs   # 执行网关
│       │   ├── tools.rs         # 工具执行器
│       │   ├── tool_registry.rs # 工具注册
│       │   ├── tool_cache.rs    # LRU 缓存
│       │   ├── wasi_sandbox.rs  # WASI 沙箱
│       │   ├── memory.rs        # 内存管理
│       │   ├── config.rs        # 配置
│       │   ├── tracing.rs       # OpenTelemetry
│       │   ├── metrics.rs       # Prometheus
│       │   ├── grpc_server.rs   # gRPC 服务
│       │   └── lib.rs
│       └── tests/               # Rust 测试
├── python/                      # Python LLM 服务
│   └── llm-service/
│       ├── llm_service/
│       │   ├── tools/           # 工具实现
│       │   │   ├── builtin/     # 内置工具
│       │   │   ├── vendor_adapters/  # 供应商适配器
│       │   │   └── base.py      # 工具基类
│       │   ├── providers/       # LLM 提供商
│       │   ├── prompts/         # 提示模板
│       │   └── tests/           # Python 测试
│       └── main.py
├── desktop/                     # 桌面应用 (Tauri + Next.js)
│   ├── src/                     # React 源码
│   ├── src-tauri/               # Tauri 后端
│   └── package.json
├── protos/                      # Protobuf 定义
│   ├── agent/                   # 智能体服务
│   ├── common/                  # 通用类型
│   └── orchestrator/            # 编排器服务
├── config/                      # 配置文件
│   ├── models.yaml
│   ├── features.yaml
│   ├── shannon.yaml
│   ├── personas.yaml
│   ├── research_strategies.yaml
│   └── opa/policies/
├── deploy/                      # 部署配置
│   └── compose/
│       ├── docker-compose.yml
│       └── grafana/
├── migrations/                  # 数据库迁移
├── observability/               # 可观测性配置
├── docs/                        # 文档
├── scripts/                     # 实用脚本
├── tests/                       # 集成测试
│   └── histories/               # 重放历史
├── Makefile
├── .env.example
├── README.md
├── ROADMAP.md
├── CHANGELOG.md
└── CONTRIBUTING.md
```

### 开发工作流

```bash
# 1. 克隆并设置
git clone https://github.com/Kocoro-lab/Shannon.git
cd Shannon
make setup

# 2. 配置环境
echo "OPENAI_API_KEY=sk-..." >> .env

# 3. 启动开发环境
make dev

# 4. 运行测试
make smoke

# 5. 代码格式化
make fmt

# 6. 运行 linter
make lint

# 7. 提交代码
git add .
git commit -m "feat: add new feature"
git push
```

---

## 适用场景分析

### 非常适合的场景

| 场景 | 原因 |
|------|------|
| **企业级 AI 平台** | 多租户隔离、策略管控、审计追踪、SSO 集成 |
| **成本敏感应用** | 精细化成本控制、三层模型自动降级、实时成本追踪 |
| **安全关键场景** | WASI 沙箱隔离、OPA 策略管控、API Key 哈希存储 |
| **研究型应用** | 多源信息检索、引用生成、4 种研究策略 |
| **定时任务** | Cron 工作流、Temporal 持久化、重放调试 |
| **需要可观测性** | 完整的链路追踪、Prometheus 指标、Grafana 仪表板 |
| **多 LLM 提供商** | 15+ 提供商统一抽象、自动故障转移 |
| **OpenAI 迁移** | 完整兼容 API、零代码迁移 |

### 可能不是最佳选择的场景

| 场景 | 替代方案 | 原因 |
|------|----------|------|
| **简单单次调用** | 直接调用 LLM API | Shannon 有额外复杂度 |
| **纯 Python 团队** | LangGraph | 多语言架构增加学习成本 |
| **快速原型开发** | LangChain / LangGraph | Python 纯代码更快迭代 |
| **低代码平台** | Dify | Shannon 需要配置和编程 |
| **对话系统** | AutoGen | 群聊模式更适合对话 |
| **任务编排** | CrewAI | 明确分工、队伍模式 |
| **边缘部署** | (当前无好方案) | Shannon 主要为数据中心设计 |

### 成本估算

**基础设施成本** (月度):

| 组件 | 实例类型 | 数量 | 成本 (USD) |
|------|----------|------|------------|
| Gateway (Go) | 2 vCPU, 4GB RAM | 2 | $40 |
| Orchestrator (Go) | 4 vCPU, 8GB RAM | 2 | $80 |
| Agent Core (Rust) | 2 vCPU, 2GB RAM | 2 | $20 |
| LLM Service (Py) | 4 vCPU, 8GB RAM | 2 | $80 |
| PostgreSQL | 4 vCPU, 8GB RAM | 1 | $40 |
| Redis | 2 vCPU, 4GB RAM | 1 | $20 |
| Qdrant | 4 vCPU, 8GB RAM | 1 | $40 |
| Temporal | 2 vCPU, 4GB RAM | 1 | $20 |
| Grafana | 2 vCPU, 2GB RAM | 1 | $10 |
| **总计** | | | **$370/月** |

**Token 成本优化** (三层架构):

| 使用场景 | 传统方案 | Shannon (三层) | 节省 |
|----------|----------|----------------|------|
| 100万次查询 (全部 large) | $3000 | - | - |
| Shannon 分层 (50/40/10) | - | $1200 | **60%** |
| Shannon 降级后 (80/20/0) | - | $600 | **80%** |

---

## 学习资源

### 官方文档

- [完整文档站点](https://docs.shannon.run) - 官方文档主页
- [架构深度解析](docs/multi-agent-workflow-architecture.md) - 多智能体工作流
- [Agent Core API](docs/agent-core-api.md) - Rust 执行层 API
- [流式 API](docs/streaming-api.md) - SSE 和 WebSocket 流式传输
- [Python 执行指南](docs/python-code-execution.md) - WASI 沙箱指南
- [自定义工具开发](docs/adding-custom-tools.md) - 工具开发教程
- [记忆系统架构](docs/memory-system-architecture.md) - 分层记忆系统
- [OpenAPI 工具](docs/openapi-tools.md) - OpenAPI 集成指南
- [环境配置](docs/environment-configuration.md) - 环境变量配置
- [提供商标识](docs/providers.md) - 15+ LLM 提供商

### 平台指南

- [Ubuntu 快速开始](docs/ubuntu-quickstart.md) - Ubuntu 部署
- [Rocky Linux 快速开始](docs/rocky-linux-quickstart.md) - Rocky Linux 部署
- [Windows 设置](docs/windows-setup-guide-en.md) - Windows 英文指南
- [Windows 中文](docs/windows-setup-guide-cn.md) - Windows 中文指南

### 社区与支持

- [GitHub Issues](https://github.com/Kocoro-lab/Shannon/issues) - 报告 Bug 和提问
- [GitHub Discussions](https://github.com/Kocoro-lab/Shannon/discussions) - 功能讨论
- [GitHub Roadmap](ROADMAP.md) - 路线图
- [X (Twitter)](https://x.com/shannon_agents) - @shannon_agents

### 示例与模板

**YAML 工作流模板** (`config/workflows/examples/`):

- `parallel_dag_example.yaml` - 并行 DAG 示例
- `simple_analysis.yaml` - 简单分析任务
- `market_analysis_playbook.yaml` - 市场分析剧本
- `market_analysis.yaml` - 市场分析工作流
- `research_summary.yaml` - 研究总结
- `complex_dag.yaml` - 复杂 DAG
- `research_summary_enterprise.yaml` - 企业级研究总结

### 代码示例

**Python SDK 示例**:

```python
# 基础查询
from shannon import ShannonClient

client = ShannonClient(base_url="http://localhost:8080")
result = client.submit_and_wait("什么是GDP?")
print(result.result)
```

**OpenAI 兼容示例**:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-shannon-your-key"
)

response = client.chat.completions.create(
    model="gpt-5-mini-2025-08-07",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## 总结与建议

### 项目优势 (6 大核心优势)

1. **生产级成熟度**
   - Temporal 工作流引擎 (Uber 开源，生产验证)
   - 确定性重放调试 (时间旅行)
   - 完整的故障恢复机制
   - 多租户隔离

2. **成本控制**
   - 行业领先的三层模型自动选择
   - 精细化的 Token 预算管理
   - 实时成本追踪和告警
   - 自动降级策略 (可节省 60-80%)

3. **安全优先**
   - WASI 沙箱 (真·隔离，非虚拟化)
   - OPA 策略引擎 (Cloud Native 标准)
   - API Key 哈希存储
   - 多租户隔离

4. **可观测性**
   - Prometheus + Grafana (开源标准)
   - OpenTelemetry 链路追踪
   - 预配置仪表板
   - 完整的审计日志

5. **多语言架构**
   - Go: 编排和并发 (天然优势)
   - Rust: 执行和安全 (内存安全)
   - Python: LLM 集成 (生态丰富)

6. **丰富集成**
   - 15+ LLM 提供商
   - MCP 协议支持
   - OpenAPI 自动集成 (~70% 覆盖率)
   - OpenAI 零代码迁移

### 潜在挑战

1. **学习曲线**: 多语言架构增加学习成本 (需了解 Go/Rust/Python)
2. **部署复杂度**: 相比纯 Python 方案部署更复杂 (需 Docker Compose)
3. **角色系统**: personas.yaml 功能尚未完全实现 (规划中)
4. **文档更新**: 部分配置文件标注为未启用 (需跟进)
5. **社区规模**: 相比 LangChain 社区较小

### 使用建议决策树

```
你需要构建什么？
   ↓
┌─────────────────────────────────────────────────────────────┐
│ 企业级 AI 平台?                                             │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
                         是 → 强烈推荐 Shannon
                              ↓
                         否 → 成本敏感?
                              ↓
                    ┌──────────┴──────────┐
                    ↓                     ↓
                   是                     否
                    ↓                     ↓
          强烈推荐 Shannon          需要代码执行安全?
                                   ↓
                          ┌─────────┴─────────┐
                          ↓                   ↓
                         是                  否
                          ↓                   ↓
              强烈推荐 Shannon      需要多智能体协作?
                                   ↓
                          ┌─────────┴─────────┐
                          ↓                   ↓
                         是                  否
                          ↓                   ↓
              推荐 Shannon (5种策略)   快速原型?
                                       ↓
                              ┌─────────┴─────────┐
                              ↓                   ↓
                             是                  否
                              ↓                   ↓
                  考虑 LangGraph      推荐 Shannon
```

### 入门路径

**1. 快速体验** (5 分钟):
```bash
curl -fsSL https://raw.githubusercontent.com/Kocoro-lab/Shannon/v0.1.0/scripts/install.sh | bash
```

**2. 本地开发** (30 分钟):
```bash
git clone https://github.com/Kocoro-lab/Shannon.git
cd Shannon
make setup
echo "OPENAI_API_KEY=sk-..." >> .env
make dev
make smoke
```

**3. 自定义配置** (1 小时):
- 修改 `config/models.yaml` (添加自定义提供商)
- 修改 `config/shannon.yaml` (调整预算和策略)
- 创建自定义工作流 YAML

**4. 深度定制** (数天):
- Fork 仓库
- 修改 Go/Rust/Python 源码
- 添加自定义工具
- 集成内部系统

**5. 生产部署** (数周):
- 参考官方文档优化配置
- 设置监控和告警
- 配置备份和灾难恢复
- 设置 CI/CD 流水线

---

## 附录: 关键配置示例

### 完整 .env 配置

```bash
# === 必需 LLM API Key (选一个) ===
OPENAI_API_KEY=sk-...
# 或
ANTHROPIC_API_KEY=sk-ant-...

# === 可选工具 API Keys ===

# Web Search (选一个)
WEB_SEARCH_PROVIDER=serpapi  # serpapi | google | bing | exa
SERPAPI_API_KEY=...
# 或
GOOGLE_SEARCH_API_KEY=...
GOOGLE_SEARCH_ENGINE_ID=...

# Web Fetch/Crawl
WEB_FETCH_PROVIDER=firecrawl  # firecrawl | exa | python
FIRECRAWL_API_KEY=...

# === 功能开关 ===
GATEWAY_SKIP_AUTH=1  # 开发模式跳过认证 (生产环境设为 0)

# === 可选配置 ===
# Python WASI 沙箱 (默认: /opt/wasm-interpreters/python-3.11.4.wasm)
PYTHON_WASI_WASM_PATH=/opt/wasm-interpreters/python-3.11.4.wasm

# LLM 服务地址 (默认: http://llm-service:8000)
LLM_SERVICE_ADDR=http://llm-service:8000

# Agent Core 地址 (默认: agent-core:50051)
AGENT_CORE_ADDR=agent-core:50051

# === 审批工作流 (可选) ===
APPROVAL_ENABLED=false
APPROVAL_COMPLEXITY_THRESHOLD=0.7
APPROVAL_DANGEROUS_TOOLS=file_system,code_execution
```

### 研究工作流示例

```bash
# 深度研究模式
curl -X POST http://localhost:8080/api/v1/tasks \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "比较不同云服务商 (AWS, Azure, GCP) 的定价策略",
    "context": {
      "force_research": true,
      "research_strategy": "deep",
      "max_budget_usd": 2.00
    }
  }'

# 学术研究模式
curl -X POST http://localhost:8080/api/v1/tasks \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "分析量子计算在密码学中的最新进展",
    "context": {
      "force_research": true,
      "research_strategy": "academic",
      "max_budget_usd": 5.00
    }
  }'
```

### 时间旅行调试

```bash
# 1. 导出失败的工作流历史
make replay-export WORKFLOW_ID=task-prod-failure-123 OUT=failure.json

# 2. 查看历史文件
cat tests/histories/task-prod-failure-123_20250106_150000.json

# 3. 本地重放调试
make replay HISTORY=tests/histories/task-prod-failure-123_20250106_150000.json

# 4. 重放输出会显示每一步的决策、工具调用和状态变化

# 5. 修复后再次验证
make replay HISTORY=tests/histories/task-prod-failure-123_20250106_150000.json
```

### 定时任务管理

```bash
# 创建每日市场分析任务
curl -X POST http://localhost:8080/api/v1/schedules \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "name": "每日市场分析",
    "description": "每天早上9点分析市场趋势",
    "cron_expression": "0 9 * * *",
    "task_query": "分析最新的市场趋势和热点新闻",
    "max_budget_per_run_usd": 0.50,
    "enabled": true
  }'

# 创建每周报告 (每周一早上8点)
curl -X POST http://localhost:8080/api/v1/schedules \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "name": "每周技术总结",
    "cron_expression": "0 8 * * 1",
    "task_query": "总结上周AI领域的技术进展",
    "max_budget_per_run_usd": 1.00,
    "enabled": true
  }'

# 列出所有定时任务
curl http://localhost:8080/api/v1/schedules \
  -H "X-API-Key: your-api-key"

# 禁用任务
curl -X PUT http://localhost:8080/api/v1/schedules/{schedule_id} \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"enabled": false}'

# 删除任务
curl -X DELETE http://localhost:8080/api/v1/schedules/{schedule_id} \
  -H "X-API-Key: your-api-key"
```

### MCP 工具配置

```yaml
# config/shannon.yaml
mcp_tools:
  # Slack 集成示例
  slack_notifier:
    enabled: true
    url: "https://api.slack.com/mcp"
    func_name: "send_message"
    description: "发送 Slack 消息通知"
    category: "communication"
    cost_per_use: 0.0
    parameters:
      - name: "channel"
        type: "string"
        required: true
        description: "目标频道"
      - name: "message"
        type: "string"
        required: true
        description: "消息内容"
    headers:
      Authorization: "Bearer ${SLACK_BOT_TOKEN}"
```

```bash
# .env
MCP_ALLOWED_DOMAINS="api.slack.com,api.example.com"
SLACK_BOT_TOKEN=xoxb-your-token
```

### OpenAPI 工具配置

```yaml
# config/shannon.yaml
openapi_tools:
  # Stripe API 示例
  stripe:
    enabled: true
    spec_url: "https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.json"
    auth_type: "api_key"
    auth_config:
      api_key_name: "Authorization"
      api_key_location: "header"
      api_key_value: "Bearer ${STRIPE_API_KEY}"
    category: "payment"
    base_cost_per_use: 0.0
    operations:
      - "createCustomer"
      - "createPaymentIntent"
      - "retrieveBalance"
```

```bash
# .env
OPENAPI_ALLOWED_DOMAINS="api.stripe.com"
STRIPE_API_KEY=sk_test_...
```

---

## 结论

Shannon 是一个**生产就绪**的企业级 AI 智能体基础设施平台，在以下方面表现出色:

1. **可靠性**: Temporal 工作流 + 确定性重放
2. **成本控制**: 三层模型自动选择 + 精细预算管理
3. **安全性**: WASI 沙箱 + OPA 策略管控
4. **可观测性**: 完整的监控、追踪、日志体系
5. **多语言架构**: 充分发挥各语言优势
6. **丰富集成**: 15+ LLM 提供商、MCP、OpenAPI

**适用场景**: 企业级 AI 平台、成本敏感应用、安全关键场景、研究型应用

**不适用场景**: 简单单次调用、纯 Python 团队、快速原型

**推荐指数**: ⭐⭐⭐⭐⭐ (5/5) - 对于需要生产级部署的企业

---

**报告生成时间**: 2026-01-06
**分析版本**: Shannon v0.1.0
**报告版本**: v2.0 (迭代优化版)
**报告作者**: Claude (Ralph Wiggum Loop - Iteration 2/5)
