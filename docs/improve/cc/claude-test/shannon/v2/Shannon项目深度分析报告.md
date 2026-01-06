# Shannon 项目深度分析报告

## 项目概览

**项目名称**: Shannon - Production AI Agents That Actually Work
**GitHub仓库**: https://github.com/Kocoro-lab/Shannon
**当前版本**: v0.1.0
**发布日期**: 2025-12-25
**许可证**: MIT License

### 项目定位

Shannon 是一个经过实战检验的生产级 AI 智能体基础设施平台，专门解决规模化部署时的核心问题：
- **成本失控** - 硬性 Token 预算控制，自动模型降级
- **非确定性故障** - Temporal 工作流支持时间旅行调试
- **安全风险** - WASI 沙箱隔离、OPA 策略管控、多租户隔离

---

## 核心架构设计

### 技术栈组成

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│ Orchestrator │────▶│ Agent Core  │
│  (SDK/API)  │     │     (Go)     │     │   (Rust)    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                    │
                    ┌──────┴──────┐      ┌──────┴──────┐
                    │  Temporal   │      │    WASI     │
                    │  Workflows  │      │   Sandbox   │
                    └─────────────┘      └─────────────┘
                           │
                    ┌──────┴──────┐
                    │ LLM Service │
                    │  (Python)   │
                    └─────────────┘
```

### 三层架构详解

#### 1. Orchestrator (Go 编排层)
**职责**: 任务路由、预算执行、会话管理、OPA 策略

**核心功能**:
- Temporal 工作流引擎集成
- 多租户隔离与 API Key 作用域管理
- 熔断器模式 (Circuit Breaker)
- 健康检查与降级策略
- Token 预算管理

**关键配置** (`config/shannon.yaml`):
```yaml
service:
  port: 50052
  health_port: 8081
  graceful_timeout: "60s"

session:
  max_history: 1000
  ttl: "720h"
  token_budget_per_task: 10000000
  token_budget_per_agent: 500000

circuit_breakers:
  redis:
    max_requests: 5
    interval: "30s"
    timeout: "60s"
    max_failures: 5
```

#### 2. Agent Core (Rust 执行层)
**职责**: WASI 沙箱、策略执行、智能体间通信

**核心特性**:
- 现代化 Rust 架构 (2025 最佳实践)
- WASI 安全沙箱执行环境
- 智能工具发现与缓存系统
- OpenTelemetry 链路追踪 + Prometheus 指标
- LRU 缓存机制

**性能指标** (来自 README):
| 操作 | P50 延迟 | P99 延迟 | 吞吐量 |
|------|----------|----------|--------|
| 工具发现 | 0.5ms | 2ms | 20k/s |
| 工具执行(缓存) | 0.1ms | 0.5ms | 50k/s |
| 工具执行(无缓存) | 50ms | 200ms | 1k/s |
| 执行网关 | 0.05ms | 0.2ms | 100k/s |
| WASI 执行 | 10ms | 100ms | 500/s |

#### 3. LLM Service (Python 服务层)
**职责**: LLM 提供商抽象、MCP 工具、提示优化

**支持的 LLM 提供商** (15+):
- OpenAI (GPT-5.1, GPT-5-mini, GPT-5-nano)
- Anthropic (Claude Opus/Sonnet/Haiku 4.5)
- Google (Gemini 2.5 Pro/Flash)
- DeepSeek (DeepSeek-V3.2, DeepSeek-R1)
- xAI (Grok 系列)
- Z.ai (GLM-4.5/4.6)
- 其他: Qwen、Mistral、Meta、Cohere、Ollama

### 数据层架构

```
PostgreSQL  ←→  Redis  ←→  Qdrant (向量数据库)
    ↓             ↓           ↓
  状态存储      会话缓存    语义检索
```

---

## 核心能力分析

### 1. OpenAI 兼容 API

提供即插即用的 OpenAI 替换接口：

```bash
export OPENAI_API_BASE=http://localhost:8080/v1
# 现有 OpenAI 代码无需修改
```

### 2. 实时事件流

支持 SSE (Server-Sent Events) 实时流式传输：

```bash
curl -N "http://localhost:8080/api/v1/stream/sse?workflow_id=task-dev-123"

# 事件类型包括:
# - WORKFLOW_STARTED, WORKFLOW_COMPLETED
# - AGENT_STARTED, AGENT_COMPLETED
# - TOOL_INVOKED, TOOL_OBSERVATION
# - LLM_PARTIAL, LLM_OUTPUT
```

### 3. 研究工作流

多智能体研究模式，自动综合发现并生成引用：

```bash
curl -X POST http://localhost:8080/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "query": "比较欧盟和美国的可再生能源采用情况",
    "context": {
      "force_research": true,
      "research_strategy": "deep"
    }
  }'
```

**研究策略配置** (`config/research_strategies.yaml`):

| 策略 | 并发智能体 | 迭代次数 | 模型层级 | 验证 |
|------|-----------|----------|----------|------|
| quick | 3 | 2 | small | ✗ |
| standard | 4 | 2 | medium | ✓ |
| deep | 5 | 3 | medium | ✓ |
| academic | 6 | 3 | medium | ✓ |

### 4. 会话连续性

多轮对话支持上下文记忆：

```bash
# 第一轮
curl -X POST http://localhost:8080/api/v1/tasks \
  -d '{"query": "什么是GDP?", "session_id": "econ-101"}'

# 第二轮 (记住上下文)
curl -X POST http://localhost:8080/api/v1/tasks \
  -d '{"query": "它与通胀有什么关系?", "session_id": "econ-101"}'
```

### 5. 定时任务

支持 Cron 语法的定期任务执行：

```bash
curl -X POST http://localhost:8080/api/v1/schedules \
  -d '{
    "name": "每日市场分析",
    "cron_expression": "0 9 * * *",
    "task_query": "分析市场趋势",
    "max_budget_per_run_usd": 0.50
  }'
```

### 6. MCP 集成

原生支持 Model Context Protocol，用于自定义工具集成。

---

## 模型层级与成本优化

### 模型分层策略

```yaml
model_tiers:
  small:   # 快速、成本优化 (目标 50%)
    - gpt-5-nano-2025-08-07
    - claude-haiku-4-5-20251001
    - glm-4.5-flash

  medium:  # 标准能力/成本平衡 (目标 40%)
    - gpt-5-mini-2025-08-07
    - claude-sonnet-4-5-20250929
    - gemini-2.5-flash

  large:   # 重度推理 (目标 10%)
    - gpt-5.1
    - claude-opus-4-1-20250805
    - gemini-2.5-pro
```

### 自动模型选择逻辑

```
任务复杂度评分 < 0.3 → small 层级
0.3 ≤ 评分 < 0.5 → medium 层级
评分 ≥ 0.5 → large 层级
```

### 成本控制配置

```yaml
cost_controls:
  max_cost_per_request: 2.00
  max_tokens_per_request: 100000
  daily_budget_usd: 1000.0
  alert_threshold_percent: 90
```

---

## 安全与治理

### 1. WASI 沙箱

Python 代码在 WebAssembly 沙箱中隔离执行：
- 无网络访问
- 只读文件系统
- 资源限制 (内存、超时)

### 2. OPA 策略管控

```rego
# config/opa/policies/teams.rego
package shannon.teams

allow {
    input.team == "data-science"
    input.model in ["gpt-5-2025-08-07", "claude-sonnet-4-5-20250929"]
}

deny_tool["database_write"] {
    input.team == "support"
}
```

### 3. 多租户隔离

- 用户/租户作用域
- API Key 哈希存储
- 每租户内存、预算、策略隔离

---

## 前端与客户端

### 1. 原生桌面应用

**支持平台**:
- macOS (Universal - Intel + Apple Silicon)
- Windows (x64 - MSI/EXE)
- Linux (x64 - AppImage/DEB)

**构建方式**:
```bash
cd desktop
npm install
npm run tauri:build
```

### 2. Python SDK

```bash
pip install shannon-sdk
```

```python
from shannon import ShannonClient

with ShannonClient(base_url="http://localhost:8080") as client:
    handle = client.submit_task("法国的首都是哪里?")
    result = client.wait(handle.task_id)
    print(result.result)
```

### 3. CLI 工具

```bash
shannon submit "分析最新市场趋势"
```

---

## 部署与运维

### 端口映射

| 服务 | 端口 | 用途 |
|------|------|------|
| Gateway | 8080 | REST API、OpenAI 兼容 `/v1` |
| Admin/Events | 8081 | SSE/WebSocket 流式传输、健康检查 |
| Orchestrator | 50052 | gRPC (内部) |
| Temporal UI | 8088 | 工作流调试 |
| Grafana | 3030 | 指标仪表板 |

### 快速安装

```bash
curl -fsSL https://raw.githubusercontent.com/Kocoro-lab/Shannon/v0.1.0/scripts/install.sh | bash
```

### Docker Compose 部署

```bash
# 使用预构建镜像
cp .env.example .env
nano .env  # 添加 API Keys
docker compose -f deploy/compose/docker-compose.release.yml up -d
```

### 从源码构建

```bash
git clone https://github.com/Kocoro-lab/Shannon.git
cd Shannon
make setup
echo "OPENAI_API_KEY=sk-..." >> .env
make dev
```

### 健康检查

```bash
# 检查所有服务
docker compose ps

# Gateway 健康检查
curl http://localhost:8080/health

# Admin 健康检查
curl http://localhost:8081/health
```

---

## 可观测性

### Prometheus 指标

可用端点: `http://localhost:2113/metrics`

- `agent_tool_executions_total` - 工具执行计数
- `agent_tool_execution_duration_seconds` - 执行延迟
- `agent_cache_hits_total` - 缓存命中
- `agent_memory_usage_bytes` - 内存使用
- `agent_active_tasks` - 活动任务数

### OpenTelemetry 链路追踪

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=shannon-orchestrator
```

### Grafana 仪表板

预配置仪表板位于 `deploy/compose/grafana/config/provisioning/dashboards/`

---

## 竞品对比分析

| 能力 | Shannon | LangGraph | Dify | AutoGen | CrewAI |
|------|---------|-----------|------|---------|--------|
| **定时任务** | ✓ Cron 工作流 | ✗ | ⚠ 基础 | ✗ | ✗ |
| **研究工作流** | ✓ 多策略(5种) | ⚠ 手动配置 | ⚠ 手动配置 | ⚠ 手动配置 | ⚠ 手动配置 |
| **确定性重放** | ✓ 时间旅行调试 | ✗ | ✗ | ✗ | ✗ |
| **Token 预算限制** | ✓ 硬限制+自动降级 | ✗ | ✗ | ✗ | ✗ |
| **安全沙箱** | ✓ WASI 隔离 | ✗ | ✗ | ✗ | ✗ |
| **OPA 策略控制** | ✓ 细粒度治理 | ✗ | ✗ | ✗ | ✗ |
| **生产指标** | ✓ Dashboard/Prometheus | ⚠ 需自行搭建 | ⚠ 基础 | ✗ | ✗ |
| **原生桌面应用** | ✓ macOS/iOS | ✗ | ✗ | ✗ | ✗ |
| **多语言核心** | ✓ Go/Rust/Python | ⚠ 仅 Python | ⚠ 仅 Python | ⚠ 仅 Python | ⚠ 仅 Python |
| **会话持久化** | ✓ Redis 后端 | ⚠ 内存 | ✓ 数据库 | ⚠ 有限 | ✗ |
| **多智能体编排** | ✓ DAG/Supervisor/Strategies | ✓ 图结构 | ⚠ 工作流 | ✓ 群聊 | ✓ 队伍 |

**核心差异化优势**:

1. **生产级可靠性**: Temporal 工作流 + 确定性重放
2. **成本控制**: 精细化的 Token 预算管理和模型分层
3. **安全优先**: WASI 沙箱 + OPA 策略管控
4. **多语言架构**: Go(编排) + Rust(执行) + Python(LLM)
5. **完整可观测性**: Prometheus + Grafana + OpenTelemetry

---

## 路线图分析

### v0.1 - 生产就绪 (当前) ✅

已完成功能:
- ✅ 核心平台稳定
- ✅ 确定性重放调试
- ✅ OPA 策略执行
- ✅ WebSocket/SSE 流式传输
- ✅ WASI 沙箱
- ✅ 多智能体编排
- ✅ 向量记忆 (Qdrant)
- ✅ 分层记忆管理
- ✅ Token 预算管理
- ✅ MCP 集成
- ✅ OpenAPI 集成
- ✅ 统一 Gateway & SDK
- 🚧 Docker 镜像发布

### v0.2 - 增强能力 (规划中)

**SDK & UI**:
- TypeScript/JavaScript SDK
- 可选拖拽式 UI (AgentKit-like)

**内置工具扩展**:
- 更多定制化工具

**平台增强**:
- 高级记忆 (情景摘要、知识图谱)
- 高级学习 (模式识别、智能体选择)
- 智能体协作基础
- MMR 多样性重排序
- RAG 系统
- 团队级配额与策略

### v0.3 - 企业级与规模化 (远期)

- Solana 集成 (去中心化信任、链上证明)
- 生产可观测性增强
- 企业功能 (SSO、多租户隔离、审批工作流)
- 边缘部署 (WASM 浏览器执行)
- 自主智能 (自组织群体、反思循环)
- 跨组织联邦
- 监管合规 (SOC 2、GDPR、HIPAA)
- AI 安全框架

---

## 技术亮点

### 1. 分层配置系统

```
环境变量 (.env) → YAML 配置 (config/) → 运行时覆盖
```

**核心配置文件**:
- `config/models.yaml` - LLM 提供商、定价、层级配置
- `config/features.yaml` - 功能开关、工作流设置
- `config/shannon.yaml` - 编排器配置
- `config/personas.yaml` - 智能体角色定义 (规划中)
- `config/research_strategies.yaml` - 研究策略配置
- `config/opa/policies/` - 访问控制规则

### 2. 降级策略

```yaml
degradation:
  mode_downgrade:
    minor_degradation_rules:
      complex_to_standard: true
    moderate_degradation_rules:
      complex_to_standard: true
      standard_to_simple: true
    severe_degradation_rules:
      force_simple_mode: true
```

### 3. 工具系统架构

**内置工具** (来自 Rust README):
- `calculator` - 计算器
- `web_search` - 网络搜索 (支持 SerpAPI、Google、Bing、Exa)
- `python_wasi_executor` - Python WASI 沙箱执行
- `file_ops` - 文件操作
- `session_file` - 会话文件管理

**MCP 工具**:
- 通过 `config/shannon.yaml` 中的 `mcp_tools` 配置
- 支持自定义 MCP 端点

**OpenAPI 工具**:
- 通过 `openapi_tools` 配置
- 支持 OpenAPI 规范自动解析
- ~70% API 覆盖率

### 4. 智能体角色系统 (规划中)

虽然 `config/personas.yaml` 已定义，但当前**未启用**。

**当前可用方式**:
```python
context["role"] = "researcher"  # 选择角色: generalist, analysis, research, writer, critic
context["system_prompt"] = "自定义系统提示词"
```

**规划中的角色**:
- `generalist` - 通用助手
- `researcher` - 研究专家
- `coder` - 编程专家
- `analyst` - 数据分析专家

---

## 开发者体验

### Makefile 命令速查

```bash
# 环境设置
make setup              # 完整设置 (首次克隆)
make setup-env          # 仅环境设置

# 开发
make dev                # 启动所有服务
make down               # 停止所有服务
make logs               # 查看日志
make ps                 # 服务状态

# 代码质量
make fmt                # 格式化代码
make lint               # 运行 linter
make proto              # 生成 protobuf 文件
make proto-local        # 本地生成 (BSR 限速时)

# 测试
make test               # 运行所有测试
make smoke              # E2E 烟雾测试
make ci                 # CI 检查

# 重放调试
make replay-export WORKFLOW_ID=xxx OUT=history.json
make replay HISTORY=history.json

# 覆盖率
make coverage           # 覆盖率报告
make coverage-go        # Go 覆盖率
make coverage-python    # Python 覆盖率
```

### 测试结构

**Go 测试**:
```
go/orchestrator/tests/replay/workflow_replay_test.go
go/orchestrator/cmd/gateway/internal/handlers/task_test.go
```

**Rust 测试**:
```
rust/agent-core/tests/test_full_integration.rs
rust/agent-core/tests/tool_calls_sequence.rs
```

**Python 测试**:
```
python/llm-service/tests/test_tier_selection.py
python/llm-service/tests/test_decomposition_patterns.py
python/llm-service/tests/test_vendor_adapters.py
```

### 项目目录结构

```
Shannon/
├── go/                          # Go 编排器
│   └── orchestrator/
│       ├── cmd/gateway/         # API Gateway
│       ├── internal/            # 内部模块
│       └── tests/               # Go 测试
├── rust/                        # Rust 执行层
│   └── agent-core/
│       ├── src/                 # Rust 源码
│       └── tests/               # Rust 测试
├── python/                      # Python LLM 服务
│   └── llm-service/
│       ├── llm_service/
│       │   ├── tools/           # 工具实现
│       │   └── tests/           # Python 测试
│       └── main.py
├── desktop/                     # 桌面应用 (Tauri + Next.js)
├── protos/                      # Protobuf 定义
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
└── docs/                        # 文档
```

---

## 适用场景分析

### 非常适合的场景

1. **企业级 AI 平台**: 需要多租户隔离、策略管控、审计追踪
2. **成本敏感应用**: 需要精细化成本控制和模型分层
3. **安全关键场景**: 需要代码沙箱隔离、策略管控
4. **研究型应用**: 需要多源信息检索、引用生成
5. **定时任务**: 需要定期执行的自动化工作流
6. **需要可观测性**: 需要完整的链路追踪和指标监控

### 可能不是最佳选择的场景

1. **简单单次调用**: Shannon 相比直接调用 LLM API 有额外复杂度
2. **纯 Python 团队**: 如果团队只有 Python 经验，多语言架构可能增加维护成本
3. **边缘部署**: 当前主要设计为数据中心部署
4. **极简原型**: LangGraph 等纯 Python 框架可能更适合快速原型

---

## 学习资源

### 官方文档

- [完整文档站点](https://docs.shannon.run)
- [架构深度解析](docs/multi-agent-workflow-architecture.md)
- [Agent Core API](docs/agent-core-api.md)
- [流式 API](docs/streaming-api.md)
- [Python 执行指南](docs/python-code-execution.md)
- [自定义工具开发](docs/adding-custom-tools.md)

### 平台指南

- [Ubuntu 快速开始](docs/ubuntu-quickstart.md)
- [Rocky Linux 快速开始](docs/rocky-linux-quickstart.md)
- [Windows 设置](docs/windows-setup-guide-en.md)
- [Windows 中文](docs/windows-setup-guide-cn.md)

### 社区

- GitHub Issues: 报告 Bug 和提问
- GitHub Discussions: 功能讨论
- X (Twitter): @shannon_agents

---

## 总结与建议

### 项目优势

1. **生产级成熟度**: 从架构设计到运维工具都面向生产环境
2. **成本控制**: 行业领先的精细化成本管理
3. **安全优先**: 多层安全防护 (WASI、OPA、多租户隔离)
4. **可观测性**: 完整的监控、追踪、日志体系
5. **多语言架构**: 充分发挥各语言优势
6. **丰富集成**: 15+ LLM 提供商、MCP、OpenAPI

### 潜在挑战

1. **学习曲线**: 多语言架构增加学习成本
2. **部署复杂度**: 相比纯 Python 方案部署更复杂
3. **角色系统**: personas.yaml 功能尚未完全实现
4. **文档更新**: 部分配置文件(如 personas.yaml)标注为未启用

### 使用建议

**如果你正在构建**:
- 企业内部 AI 平台 → **强烈推荐 Shannon**
- 需要精细成本控制的大规模应用 → **强烈推荐 Shannon**
- 需要代码执行安全隔离的场景 → **强烈推荐 Shannon**
- 快速原型或个人项目 → 考虑 LangGraph 或直接调用 LLM API
- 纯 Python 技术栈团队 → 评估团队多语言能力后再决定

### 入门路径

1. **快速体验**: 使用一键安装脚本
2. **本地开发**: Docker Compose 部署
3. **自定义**: 修改 config/ 配置文件
4. **深度定制**: Fork 后修改 Go/Rust/Python 源码
5. **生产部署**: 参考官方文档进行部署优化

---

## 附录: 关键配置示例

### 完整 .env 配置

```bash
# 必需 LLM API Key (选一个)
OPENAI_API_KEY=sk-...
# 或
ANTHROPIC_API_KEY=sk-ant-...

# 可选工具 API Keys
WEB_SEARCH_PROVIDER=serpapi
SERPAPI_API_KEY=...

WEB_FETCH_PROVIDER=firecrawl
FIRECRAWL_API_KEY=...

# 功能开关
GATEWAY_SKIP_AUTH=1  # 开发模式跳过认证
```

### 研究工作流示例

```bash
curl -X POST http://localhost:8080/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "query": "比较不同云服务商的定价策略",
    "context": {
      "force_research": true,
      "research_strategy": "deep"
    }
  }'
```

### 时间旅行调试

```bash
# 导出失败的工作流历史
make replay-export WORKFLOW_ID=task-prod-failure-123 OUT=failure.json

# 本地重放调试
make replay HISTORY=failure.json
```

---

**报告生成时间**: 2026-01-06
**分析的版本**: Shannon v0.1.0
**报告作者**: Claude (Ralph Wiggum Loop - Iteration 1/5)
