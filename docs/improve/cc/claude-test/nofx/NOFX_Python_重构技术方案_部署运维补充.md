# NOFX Python重构技术方案 - 部署运维补充篇

> 本文档是《NOFX_Python_重构技术方案_A股港股》系列的第六部分
> 覆盖第21-25章：CI/CD、容器化部署、数据迁移、前端高级模式、生产运维

---

## 第21章 CI/CD流水线配置

### 21.1 GitHub Actions完整配置

#### 21.1.1 主流水线配置

```yaml
# .github/workflows/ci-cd-pipeline.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop, 'feature/**']
  pull_request:
    branches: [main, develop]
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      environment:
        description: '部署环境'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '20'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ========================================
  # 代码质量检查
  # ========================================
  code-quality:
    name: Code Quality Check
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - name: Checkout代码
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 完整历史用于SonarQube

      - name: 设置Python环境
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: 安装依赖
        run: |
          python -m pip install --upgrade pip
          pip install ruff black isort mypy pylint bandit
          pip install -r requirements-dev.txt

      - name: Ruff代码检查
        run: |
          ruff check . --output-format=github
        continue-on-error: false

      - name: Black格式检查
        run: |
          black --check --diff .
        continue-on-error: false

      - name: isort导入排序检查
        run: |
          isort --check-only --diff .
        continue-on-error: false

      - name: MyPy类型检查
        run: |
          mypy src/ --config-file pyproject.toml
        continue-on-error: true

      - name: Pylint代码评分
        run: |
          pylint src/ --rcfile .pylintrc --fail-under=8.0
        continue-on-error: true

      - name: Bandit安全扫描
        run: |
          bandit -r src/ -f json -o bandit-report.json
        continue-on-error: true

      - name: 上传Bandit报告
        uses: actions/upload-artifact@v4
        with:
          name: bandit-security-report
          path: bandit-report.json
          retention-days: 30

  # ========================================
  # 单元测试
  # ========================================
  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: code-quality
    timeout-minutes: 20

    strategy:
      matrix:
        python-version: ['3.11', '3.12']
      fail-fast: false

    steps:
      - name: Checkout代码
        uses: actions/checkout@v4

      - name: 设置Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: 安装依赖
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: 运行单元测试（带覆盖率）
        run: |
          pytest tests/unit/ \
            --cov=src \
            --cov-report=xml \
            --cov-report=html \
            --cov-report=term-missing \
            --junitxml=test-results.xml \
            -v \
            --tb=short
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379/0

      - name: 上传覆盖率到Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          flags: unit-tests
          name: codecov-${{ matrix.python-version }}
          fail_ci_if_error: false

      - name: 上传测试报告
        uses: actions/upload-artifact@v4
        with:
          name: test-results-py${{ matrix.python-version }}
          path: |
            test-results.xml
            htmlcov/
            .coverage
          retention-days: 30

      - name: 覆盖率检查
        run: |
          coverage report --fail-under=80

  # ========================================
  # 集成测试
  # ========================================
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: unit-tests
    timeout-minutes: 30

    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - name: Checkout代码
        uses: actions/checkout@v4

      - name: 设置Python环境
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: 安装依赖
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: 数据库迁移
        run: |
          alembic upgrade head
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db

      - name: 运行集成测试
        run: |
          pytest tests/integration/ \
            --cov=src \
            --cov-report=xml \
            --cov-append \
            --junitxml=integration-test-results.xml \
            -v \
            --tb=short
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379/0

      - name: 上传集成测试报告
        uses: actions/upload-artifact@v4
        with:
          name: integration-test-results
          path: integration-test-results.xml
          retention-days: 30

  # ========================================
  # 端到端测试
  # ========================================
  e2e-tests:
    name: E2E Tests
    runs-on: ubuntu-latest
    needs: integration-tests
    timeout-minutes: 45

    steps:
      - name: Checkout代码
        uses: actions/checkout@v4

      - name: 设置Python环境
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: 设置Node.js环境
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: 安装Python依赖
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: 安装前端依赖
        run: |
          cd frontend
          npm ci

      - name: 构建前端
        run: |
          cd frontend
          npm run build

      - name: 启动测试服务器
        run: |
          uvicorn src.main:app --host 0.0.0.0 --port 8000 &
          sleep 10
        env:
          DATABASE_URL: sqlite:///./test.db
          REDIS_URL: redis://localhost:6379/0

      - name: 安装Playwright
        run: |
          npm init -y
          npm install -D @playwright/test
          npx playwright install --with-deps

      - name: 运行E2E测试
        run: |
          npx playwright test
        working-directory: tests/e2e

      - name: 上传Playwright报告
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: playwright-report
          path: tests/e2e/playwright-report/
          retention-days: 30

  # ========================================
  # 性能测试
  # ========================================
  performance-tests:
    name: Performance Tests
    runs-on: ubuntu-latest
    needs: unit-tests
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    timeout-minutes: 60

    steps:
      - name: Checkout代码
        uses: actions/checkout@v4

      - name: 设置Python环境
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: 安装依赖
        run: |
          pip install -r requirements.txt
          pip install locust pytest-benchmark

      - name: 运行负载测试
        run: |
          locust -f tests/performance/locustfile.py \
            --headless \
            --users 100 \
            --spawn-rate 10 \
            --run-time 5m \
            --host http://localhost:8000 \
            --html performance-report.html \
            --csv performance
        continue-on-error: true

      - name: 运行基准测试
        run: |
          pytest tests/performance/benchmarks.py \
            --benchmark-only \
            --benchmark-json=benchmark-results.json
        continue-on-error: true

      - name: 上传性能测试报告
        uses: actions/upload-artifact@v4
        with:
          name: performance-reports
          path: |
            performance-report.html
            benchmark-results.json
          retention-days: 30

  # ========================================
  # 安全扫描
  # ========================================
  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: code-quality
    timeout-minutes: 15

    steps:
      - name: Checkout代码
        uses: actions/checkout@v4

      - name: 运行Trivy漏洞扫描
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: 依赖安全检查
        run: |
          pip install safety
          safety check --json > safety-report.json || true
        continue-on-error: true

      - name: 代码漏洞扫描
        uses: github/codeql-action/analyze@v3
        with:
          languages: python, javascript
          category: "/language:python"

      - name: 上传安全报告
        uses: actions/upload-artifact@v4
        with:
          name: security-reports
          path: |
            trivy-results.sarif
            safety-report.json
          retention-days: 90

  # ========================================
  # 构建Docker镜像
  # ========================================
  build-image:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests, security-scan]
    if: github.event_name != 'pull_request'
    timeout-minutes: 30
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}

    steps:
      - name: Checkout代码
        uses: actions/checkout@v4

      - name: 设置Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: 登录到Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: 提取镜像元数据
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}

      - name: 构建并推送镜像
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            BUILD_DATE=${{ github.event.head_commit.timestamp }}
            VCS_REF=${{ github.sha }}
            VERSION=${{ steps.meta.outputs.version }}

  # ========================================
  # 部署到Staging
  # ========================================
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [build-image, e2e-tests]
    if: github.ref == 'refs/heads/develop'
    environment:
      name: staging
      url: https://staging.deepalpha.example.com

    steps:
      - name: Checkout代码
        uses: actions/checkout@v4

      - name: 设置kubectl
        uses: azure/setup-kubectl@v4
        with:
          version: 'latest'

      - name: 配置Kubeconfig
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > ~/.kube/config

      - name: 部署到Kubernetes
        run: |
          kubectl set image deployment/deepalpha-api \
            deepalpha-api=${{ needs.build-image.outputs.image-tag }} \
            -n deepalpha-staging

          kubectl rollout status deployment/deepalpha-api \
            -n deepalpha-staging \
            --timeout=5m

      - name: 运行数据库迁移
        run: |
          kubectl exec -n deepalpha-staging \
            deployment/deepalpha-api \
            -- alembic upgrade head

      - name: 健康检查
        run: |
          kubectl wait --for=condition=ready pod \
            -l app=deepalpha-api \
            -n deepalpha-staging \
            --timeout=60s

          curl -f https://staging.deepalpha.example.com/health || exit 1

      - name: 通知Slack
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "🚀 部署成功",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*DeepAlpha已部署到Staging环境*\n• 分支: `${{ github.ref }}`\n• 提交: `${{ github.sha }}`\n• 作者: `${{ github.actor }}`\n• 镜像: `${{ needs.build-image.outputs.image-tag }}`"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}

  # ========================================
  # 部署到Production
  # ========================================
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [build-image, e2e-tests]
    if: github.event_name == 'release'
    environment:
      name: production
      url: https://deepalpha.example.com

    steps:
      - name: Checkout代码
        uses: actions/checkout@v4

      - name: 设置kubectl
        uses: azure/setup-kubectl@v4

      - name: 配置Kubeconfig
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBE_CONFIG_PRODUCTION }}" | base64 -d > ~/.kube/config

      - name: 创建Git标签
        run: |
          git tag -a v${{ github.event.release.tag_name }} -m "Release v${{ github.event.release.tag_name }}"
          git push origin v${{ github.event.release.tag_name }}

      - name: 蓝绿部署 - 切换流量
        run: |
          # 部署到Green环境
          helm upgrade --install deepalpha-green ./helm/deepalpha \
            --namespace deepalpha-production \
            --set image.tag=${{ github.event.release.tag_name }} \
            --set environment=production \
            --values helm/deepalpha/values-production.yaml \
            --wait \
            --timeout 10m

          # 健康检查
          kubectl wait --for=condition=ready pod \
            -l app=deepalpha,environment=green \
            -n deepalpha-production \
            --timeout=120s

          # 金丝雀发布：10%流量到Green
          kubectl patch service deepalpha-api \
            -n deepalpha-production \
            -p '{"spec":{"selector":{"version":"green"}}}' \
            --type=merge

          sleep 60  # 观察期

          # 100%流量到Green
          kubectl patch service deepalpha-api \
            -n deepalpha-production \
            -p '{"spec":{"selector":{"version":"green"}}}' \
            --type=merge

          # 清理Blue环境
          helm uninstall deepalpha-blue -n deepalpha-production || true

      - name: 运行数据库迁移
        run: |
          kubectl exec -n deepalpha-production \
            deployment/deepalpha-api \
            -- alembic upgrade head

      - name: 验证部署
        run: |
          # API健康检查
          curl -f https://deepalpha.example.com/health || exit 1

          # 关键端点检查
          curl -f https://deepalpha.example.com/api/v1/traders || exit 1

          # 监控指标检查
          curl -f https://deepalpha.example.com/metrics || exit 1

      - name: 创建部署回滚任务
        if: failure()
        run: |
          echo "部署失败，触发回滚流程"
          # 自动回滚到上一个稳定版本
          helm rollback deepalpha-green -n deepalpha-production

      - name: 通知生产部署
        if: success()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "🎉 生产环境部署成功",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*DeepAlpha v${{ github.event.release.tag_name }}已部署到生产环境*\n• 发布: <${{ github.event.release.html_url }}|${{ github.event.release.name }}>\n• 提交: `${{ github.sha }}`\n• 作者: `${{ github.actor }}`"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_PRODUCTION }}

  # ========================================
  # 生成测试报告
  # ========================================
  test-report:
    name: Generate Test Report
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests, e2e-tests]
    if: always()

    steps:
      - name: 下载所有测试报告
        uses: actions/download-artifact@v4

      - name: 发布测试报告
        uses: mikepenz/action-junit-report@v4
        with:
          report_paths: '**/test-results.xml'
          check_name: 测试结果汇总
          detailed_summary: true
          include_passed: true

  # ========================================
  # 发布GitHub Release
  # ========================================
  release:
    name: Create Release
    runs-on: ubuntu-latest
    needs: [build-image]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    permissions:
      contents: write

    steps:
      - name: Checkout代码
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: 生成变更日志
        id: changelog
        uses: conventional-changelog/conventional-changelog-action@v5
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          output-file: 'CHANGELOG.md'

      - name: 创建Release
        uses: softprops/action-gh-release@v1
        with:
          body_path: CHANGELOG.md
          draft: false
          prerelease: false
          generate_release_notes: true
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

#### 21.1.2 Dockerfile多阶段构建

```dockerfile
# Dockerfile
# 多阶段构建，优化镜像大小和安全性

# ============================================
# Stage 1: Base Builder
# ============================================
FROM python:3.11-slim as base-builder

# 设置工作目录
WORKDIR /build

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt requirements-dev.txt ./

# ============================================
# Stage 2: Dependencies Builder
# ============================================
FROM base-builder as dependencies-builder

# 创建虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级pip和安装构建工具
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 安装生产依赖（分离编译和运行时依赖）
RUN pip install --no-cache-dir --no-deps -r requirements.txt

# ============================================
# Stage 3: Test Runner
# ============================================
FROM dependencies-builder as test-runner

# 安装测试依赖
RUN pip install --no-cache-dir -r requirements-dev.txt

# 复制源代码
COPY . .

# 运行测试
RUN pytest tests/unit/ --cov=src --cov-report=term || echo "Tests completed with status: $?"

# ============================================
# Stage 4: Production Image
# ============================================
FROM python:3.11-slim as production

# 获取构建参数
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.0.0

# 添加标签
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.title="DeepAlpha Trading System" \
      org.opencontainers.image.description="AI-powered trading system for A-shares and HK stocks" \
      org.opencontainers.image.vendor="DeepAlpha" \
      org.opencontainers.image.authors="DeepAlpha Team" \
      org.opencontainers.image.licenses="MIT"

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# 从依赖阶段复制虚拟环境
COPY --from=dependencies-builder /opt/venv /opt/venv

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    APP_HOME="/app"

# 创建应用目录
WORKDIR $APP_HOME

# 复制应用代码
COPY --chown=appuser:appuser . .

# 复制Alembic迁移文件
COPY --chown=appuser:appuser alembic/ alembic/
COPY --chown=appuser:appuser alembic.ini ./

# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/tmp && \
    chown -R appuser:appuser /app/logs /app/data /app/tmp

# 切换到非root用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### 21.1.3 Docker Compose开发环境

```yaml
# docker-compose.yml
version: '3.8'

services:
  # ========================================
  # 主应用
  # ========================================
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    image: deepalpha-api:latest
    container_name: deepalpha-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://deepalpha:deepalpha_pass@postgres:5432/deepalpha
      - REDIS_URL=redis://redis:6379/0
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
      - SENTRY_DSN=${SENTRY_DSN}
    volumes:
      - ./src:/app/src:ro
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    networks:
      - deepalpha-network

  # ========================================
  # PostgreSQL数据库
  # ========================================
  postgres:
    image: postgres:16-alpine
    container_name: deepalpha-postgres
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=deepalpha
      - POSTGRES_PASSWORD=deepalpha_pass
      - POSTGRES_DB=deepalpha
      - PGDATA=/var/lib/postgresql/data/pgdata
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U deepalpha -d deepalpha"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - deepalpha-network

  # ========================================
  # Redis缓存
  # ========================================
  redis:
    image: redis:7-alpine
    container_name: deepalpha-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --requirepass redis_pass
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - deepalpha-network

  # ========================================
  # Prometheus监控
  # ========================================
  prometheus:
    image: prom/prometheus:latest
    container_name: deepalpha-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./config/prometheus/rules:/etc/prometheus/rules:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    networks:
      - deepalpha-network

  # ========================================
  # Grafana可视化
  # ========================================
  grafana:
    image: grafana/grafana:latest
    container_name: deepalpha-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_INSTALL_PLUGINS=redis-datasource
      - GF_SERVER_ROOT_URL=http://localhost:3000
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    networks:
      - deepalpha-network

  # ========================================
  # Jaeger链路追踪
  # ========================================
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: deepalpha-jaeger
    restart: unless-stopped
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"
      - "14268:14268"
      - "14250:14250"
      - "9411:9411"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - deepalpha-network

  # ========================================
  # 前端开发服务器
  # ========================================
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    container_name: deepalpha-frontend
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app:delegated
      - /app/node_modules
    environment:
      - NODE_ENV=development
      - VITE_API_URL=http://localhost:8000
      - VITE_WS_URL=ws://localhost:8000
    command: npm run dev
    networks:
      - deepalpha-network

  # ========================================
  # Nginx反向代理
  # ========================================
  nginx:
    image: nginx:alpine
    container_name: deepalpha-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./config/nginx/ssl:/etc/nginx/ssl:ro
      - ./frontend/dist:/usr/share/nginx/html:ro
    depends_on:
      - api
      - frontend
    networks:
      - deepalpha-network

  # ========================================
  # Worker后台任务
  # ========================================
  worker:
    build:
      context: .
      dockerfile: Dockerfile
    image: deepalpha-worker:latest
    container_name: deepalpha-worker
    restart: unless-stopped
    command: celery -A src.tasks.worker worker --loglevel=info --concurrency=4
    environment:
      - DATABASE_URL=postgresql://deepalpha:deepalpha_pass@postgres:5432/deepalpha
      - REDIS_URL=redis://:redis_pass@redis:6379/0
      - CELERY_BROKER_URL=redis://:redis_pass@redis:6379/0
      - CELERY_RESULT_BACKEND=redis://:redis_pass@redis:6379/0
    volumes:
      - ./src:/app/src:ro
      - ./logs:/app/logs
    depends_on:
      - redis
      - postgres
    networks:
      - deepalpha-network

  # ========================================
  # Celery Beat定时任务
  # ========================================
  celery-beat:
    build:
      context: .
      dockerfile: Dockerfile
    image: deepalpha-beat:latest
    container_name: deepalpha-beat
    restart: unless-stopped
    command: celery -A src.tasks.beat beat --loglevel=info --scheduler redbeat.RedBeatScheduler
    environment:
      - REDIS_URL=redis://:redis_pass@redis:6379/0
      - CELERY_BROKER_URL=redis://:redis_pass@redis:6379/0
    volumes:
      - ./src:/app/src:ro
    depends_on:
      - redis
    networks:
      - deepalpha-network

  # ========================================
  # Flower任务监控
  # ========================================
  flower:
    build:
      context: .
      dockerfile: Dockerfile
    image: deepalpha-flower:latest
    container_name: deepalpha-flower
    restart: unless-stopped
    ports:
      - "5555:5555"
    command: celery -A src.tasks.worker flower --port=5555
    environment:
      - CELERY_BROKER_URL=redis://:redis_pass@redis:6379/0
      - CELERY_RESULT_BACKEND=redis://:redis_pass@redis:6379/0
      - FLOWER_BASIC_AUTH=${FLOWER_USER}:${FLOWER_PASSWORD}
    depends_on:
      - redis
      - worker
    networks:
      - deepalpha-network

networks:
  deepalpha-network:
    driver: bridge

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
```

---

## 第22章 Kubernetes生产部署

### 22.1 Namespace与资源配额

```yaml
# k8s/00-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: deepalpha-production
  labels:
    name: deepalpha-production
    environment: production

---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: deepalpha-quota
  namespace: deepalpha-production
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "10"
    services.loadbalancers: "2"
    services.nodeports: "0"

---
apiVersion: v1
kind: LimitRange
metadata:
  name: deepalpha-limits
  namespace: deepalpha-production
spec:
  limits:
  - default:
      cpu: 500m
      memory: 512Mi
    defaultRequest:
      cpu: 100m
      memory: 128Mi
    type: Container
```

### 22.2 ConfigMap配置管理

```yaml
# k8s/01-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: deepalpha-config
  namespace: deepalpha-production
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  DATABASE_POOL_SIZE: "20"
  REDIS_MAX_CONNECTIONS: "50"

  # API配置
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  API_WORKERS: "4"
  API_TIMEOUT: "30"

  # 市场数据配置
  MARKET_DATA_PROVIDERS: "tushare,akshare"
  TUSHARE_TOKEN: "${TUSHARE_TOKEN}"
  AKSHARE_TIMEOUT: "10"

  # LLM配置
  LLM_PROVIDER: "deepseek"
  LLM_MODEL: "deepseek-chat"
  LLM_TEMPERATURE: "0.7"
  LLM_MAX_TOKENS: "2000"
  LLM_TIMEOUT: "30"

  # 交易配置
  TRADING_ENABLED: "true"
  DRY_RUN: "false"
  MAX_POSITION_SIZE: "100000"
  MAX_DAILY_LOSS: "50000"

  # 监控配置
  SENTRY_DSN: "${SENTRY_DSN}"
  PROMETHEUS_PORT: "9090"
  JAEGER_HOST: "jaeger"
  JAEGER_PORT: "6831"
```

### 22.3 Secret密钥管理

```yaml
# k8s/02-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: deepalpha-secrets
  namespace: deepalpha-production
type: Opaque
stringData:
  # 数据库凭证
  DATABASE_URL: "postgresql://deepalpha:${DB_PASSWORD}@postgres:5432/deepalpha"

  # Redis凭证
  REDIS_URL: "redis://:${REDIS_PASSWORD}@redis:6379/0"

  # JWT密钥
  JWT_SECRET_KEY: "${JWT_SECRET_KEY}"
  JWT_ALGORITHM: "HS256"

  # API加密
  API_ENCRYPTION_KEY: "${API_ENCRYPTION_KEY}"

  # 第三方服务密钥
  TUSHARE_TOKEN: "${TUSHARE_TOKEN}"
  DEEPSEEK_API_KEY: "${DEEPSEEK_API_KEY}"
  QWEN_API_KEY: "${QWEN_API_KEY}"

  # Broker API凭证
  BROKER_API_KEY: "${BROKER_API_KEY}"
  BROKER_API_SECRET: "${BROKER_API_SECRET}"

  # 通知服务
  SLACK_WEBHOOK_URL: "${SLACK_WEBHOOK_URL}"
  DINGTALK_WEBHOOK: "${DINGTALK_WEBHOOK}"
  EMAIL_SMTP_PASSWORD: "${EMAIL_SMTP_PASSWORD}"

---
# 使用Sealed Secrets或External Secrets Operator管理敏感数据
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: deepalpha-sealed-secrets
  namespace: deepalpha-production
spec:
  encryptedData:
    DATABASE_PASSWORD: AgBy3i4OJSWK+PiTySY...
    JWT_SECRET_KEY: AgBy3i4OJSWK+PiTySY...
  template:
    metadata:
      name: deepalpha-secrets
      namespace: deepalpha-production
    type: Opaque
```

### 22.4 Deployment部署配置

```yaml
# k8s/03-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepalpha-api
  namespace: deepalpha-production
  labels:
    app: deepalpha
    component: api
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: deepalpha
      component: api
  template:
    metadata:
      labels:
        app: deepalpha
        component: api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: deepalpha-sa

      # 安全上下文
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000

      # 初始化容器
      initContainers:
      - name: wait-for-postgres
        image: busybox:1.36
        command: ['sh', '-c', 'until nc -z postgres 5432; do echo waiting for postgres; sleep 2; done;']

      - name: wait-for-redis
        image: busybox:1.36
        command: ['sh', '-c', 'until nc -z redis 6379; do echo waiting for redis; sleep 2; done;']

      - name: run-migrations
        image: ghcr.io/your-org/deepalpha:latest
        command: ['alembic', 'upgrade', 'head']
        envFrom:
        - secretRef:
            name: deepalpha-secrets

      # 主容器
      containers:
      - name: api
        image: ghcr.io/your-org/deepalpha:{{ .Values.image.tag }}
        imagePullPolicy: Always

        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP

        env:
        - name: ENVIRONMENT
          value: "production"

        envFrom:
        - configMapRef:
            name: deepalpha-config
        - secretRef:
            name: deepalpha-secrets

        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 2000m
            memory: 2Gi

        # 探针配置
        livenessProbe:
          httpGet:
            path: /health/live
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /health/ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        startupProbe:
          httpGet:
            path: /health/startup
            port: http
          initialDelaySeconds: 0
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 30

        # 生命周期钩子
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]

      # 优雅终止
      terminationGracePeriodSeconds: 30

      # 亲和性规则
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - deepalpha
              topologyKey: kubernetes.io/hostname

---
# Worker部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepalpha-worker
  namespace: deepalpha-production
  labels:
    app: deepalpha
    component: worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: deepalpha
      component: worker
  template:
    metadata:
      labels:
        app: deepalpha
        component: worker
    spec:
      containers:
      - name: worker
        image: ghcr.io/your-org/deepalpha:latest
        command: ["celery", "-A", "src.tasks.worker", "worker", "--loglevel=info"]

        envFrom:
        - configMapRef:
            name: deepalpha-config
        - secretRef:
            name: deepalpha-secrets

        resources:
          requests:
            cpu: 1000m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi

        livenessProbe:
          exec:
            command:
            - celery
            - -A
            - src.tasks.worker
            - inspect
            - ping
          initialDelaySeconds: 30
          periodSeconds: 60
```

### 22.5 Service服务配置

```yaml
# k8s/04-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: deepalpha-api
  namespace: deepalpha-production
  labels:
    app: deepalpha
    component: api
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "http"
    service.beta.kubernetes.io/aws-load-balancer-connection-idle-timeout: "60"
spec:
  type: ClusterIP
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800
  ports:
  - name: http
    port: 80
    targetPort: http
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: metrics
    protocol: TCP
  selector:
    app: deepalpha
    component: api

---
# Headless Service用于StatefulSet
apiVersion: v1
kind: Service
metadata:
  name: deepalpha-api-headless
  namespace: deepalpha-production
spec:
  type: ClusterIP
  clusterIP: None
  ports:
  - port: 8000
    targetPort: http
    name: http
  selector:
    app: deepalpha
    component: api
```

### 22.6 Ingress路由配置

```yaml
# k8s/05-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: deepalpha-ingress
  namespace: deepalpha-production
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"

    # 速率限制
    nginx.ingress.kubernetes.io/limit-rps: "100"
    nginx.ingress.kubernetes.io/limit-burst-multiplier: "2"

    # 超时设置
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"

    # WebSocket支持
    nginx.ingress.kubernetes.io/proxy-http-version: "1.1"
    nginx.ingress.kubernetes.io/upgrade: "$http_upgrade"
    nginx.ingress.kubernetes.io/connection: "upgrade"

    # CORS
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://deepalpha.example.com"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"

    # 安全头
    nginx.ingress.kubernetes.io/configuration-snippet: |
      add_header X-Frame-Options "SAMEORIGIN" always;
      add_header X-Content-Type-Options "nosniff" always;
      add_header X-XSS-Protection "1; mode=block" always;
      add_header Referrer-Policy "strict-origin-when-cross-origin" always;
spec:
  tls:
  - hosts:
    - deepalpha.example.com
    - api.deepalpha.example.com
    secretName: deepalpha-tls

  rules:
  - host: deepalpha.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: deepalpha-frontend
            port:
              number: 80

  - host: api.deepalpha.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: deepalpha-api
            port:
              number: 80
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: deepalpha-api
            port:
              number: 80
      - path: /metrics
        pathType: Prefix
        backend:
          service:
            name: deepalpha-api
            port:
              number: 9090
```

### 22.7 HPA自动扩缩容

```yaml
# k8s/06-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: deepalpha-api-hpa
  namespace: deepalpha-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: deepalpha-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 4
        periodSeconds: 30
      selectPolicy: Max

---
# KEDA事件驱动扩缩容
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: deepalpha-worker-scaler
  namespace: deepalpha-production
spec:
  scaleTargetRef:
    name: deepalpha-worker
  minReplicaCount: 2
  maxReplicaCount: 10
  triggers:
  - type: redis
    metadata:
      address: redis:6379
      listName: celery
      listLength: "5"
      enableTLS: "false"
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: celery_queue_length
      threshold: "100"
      query: celery_queue_length
```

### 22.8 StatefulSet有状态服务

```yaml
# k8s/07-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: deepalpha-trader
  namespace: deepalpha-production
spec:
  serviceName: deepalpha-trader-headless
  replicas: 3
  selector:
    matchLabels:
      app: deepalpha
      component: trader
  template:
    metadata:
      labels:
        app: deepalpha
        component: trader
    spec:
      containers:
      - name: trader
        image: ghcr.io/your-org/deepalpha:latest
        command: ["python", "-m", "src.trader.main"]

        ports:
        - containerPort: 8001
          name: trader

        envFrom:
        - configMapRef:
            name: deepalpha-config
        - secretRef:
            name: deepalpha-secrets

        volumeMounts:
        - name: trader-data
          mountPath: /app/data
        - name: trader-logs
          mountPath: /app/logs

        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi

  volumeClaimTemplates:
  - metadata:
      name: trader-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 10Gi
  - metadata:
      name: trader-logs
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: standard
      resources:
        requests:
          storage: 5Gi
```

### 22.9 PodDisruptionBudget中断预算

```yaml
# k8s/08-pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: deepalpha-api-pdb
  namespace: deepalpha-production
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: deepalpha
      component: api

---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: deepalpha-worker-pdb
  namespace: deepalpha-production
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: deepalpha
      component: worker
```

### 22.10 NetworkPolicy网络策略

```yaml
# k8s/09-networkpolicy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deepalpha-network-policy
  namespace: deepalpha-production
spec:
  podSelector:
    matchLabels:
      app: deepalpha
  policyTypes:
  - Ingress
  - Egress
  ingress:
  # 允许来自Ingress的流量
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000

  # 允许来自监控系统的流量
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090

  egress:
  # 允许DNS查询
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: UDP
      port: 53

  # 允许访问数据库
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432

  # 允许访问Redis
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379

  # 允许访问外部API
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443
```

---

## 第23章 数据迁移与备份策略

### 23.1 NOFX Go到Python数据迁移

```python
# scripts/migrate_nofx_data.py
"""
NOFX Go数据迁移到Python系统
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json

import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.models.trader import TraderModel
from src.models.position import PositionModel
from src.models.order import OrderModel
from src.models.trade import TradeModel
from src.core.config import settings

logger = logging.getLogger(__name__)


class NOFXDataMigrator:
    """NOFX数据迁移器"""

    def __init__(self, nofx_db_url: str, target_db_url: str):
        self.nofx_db_url = nofx_db_url
        self.target_db_url = target_db_url
        self.nofx_conn = None
        self.target_engine = None
        self.target_session_factory = None

        # 统计信息
        self.stats = {
            'traders': {'migrated': 0, 'failed': 0, 'skipped': 0},
            'positions': {'migrated': 0, 'failed': 0, 'skipped': 0},
            'orders': {'migrated': 0, 'failed': 0, 'skipped': 0},
            'trades': {'migrated': 0, 'failed': 0, 'skipped': 0},
        }

    async def connect(self):
        """连接数据库"""
        # 连接NOFX数据库（PostgreSQL）
        self.nofx_conn = await asyncpg.connect(self.nofx_db_url)

        # 创建目标数据库连接
        self.target_engine = create_async_engine(
            self.target_db_url,
            pool_size=20,
            max_overflow=40,
        )
        self.target_session_factory = sessionmaker(
            self.target_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info("数据库连接建立成功")

    async def close(self):
        """关闭连接"""
        if self.nofx_conn:
            await self.nofx_conn.close()
        if self.target_engine:
            await self.target_engine.dispose()
        logger.info("数据库连接已关闭")

    async def migrate_all(self):
        """执行完整迁移"""
        try:
            await self.connect()

            # 按依赖顺序迁移
            await self.migrate_traders()
            await self.migrate_positions()
            await self.migrate_orders()
            await self.migrate_trades()

            self.print_summary()

        finally:
            await self.close()

    async def migrate_traders(self):
        """迁移交易员数据"""
        logger.info("开始迁移交易员数据...")

        # 从NOFX读取交易员
        rows = await self.nofx_conn.fetch("""
            SELECT
                id,
                name,
                type,
                initial_capital,
                is_active,
                created_at,
                updated_at,
                config,
                metadata
            FROM traders
            ORDER BY created_at
        """)

        async with self.target_session_factory() as session:
            for row in rows:
                try:
                    # 检查是否已存在
                    existing = await session.get(TraderModel, row['id'])
                    if existing:
                        self.stats['traders']['skipped'] += 1
                        continue

                    # 数据映射与转换
                    trader_data = {
                        'id': row['id'],
                        'name': row['name'],
                        'type': self._convert_trader_type(row['type']),
                        'initial_capital': row['initial_capital'],
                        'is_active': row['is_active'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'config': self._convert_config(row['config']),
                        'metadata': self._convert_metadata(row['metadata']),
                    }

                    trader = TraderModel(**trader_data)
                    session.add(trader)
                    await session.flush()

                    self.stats['traders']['migrated'] += 1
                    logger.info(f"迁移交易员: {trader.name}")

                except Exception as e:
                    logger.error(f"迁移交易员失败 {row['id']}: {e}")
                    self.stats['traders']['failed'] += 1
                    session.rollback()

            await session.commit()

        logger.info(f"交易员迁移完成: {self.stats['traders']}")

    async def migrate_positions(self):
        """迁移持仓数据"""
        logger.info("开始迁移持仓数据...")

        rows = await self.nofx_conn.fetch("""
            SELECT
                id,
                trader_id,
                symbol,
                exchange,
                quantity,
                entry_price,
                current_price,
                market_value,
                unrealized_pnl,
                created_at,
                updated_at
            FROM positions
            WHERE quantity > 0  # 只迁移当前持仓
            ORDER BY trader_id, symbol
        """)

        async with self.target_session_factory() as session:
            for row in rows:
                try:
                    # 验证交易员存在
                    trader = await session.get(TraderModel, row['trader_id'])
                    if not trader:
                        logger.warning(f"交易员不存在，跳过持仓: {row['trader_id']}")
                        self.stats['positions']['skipped'] += 1
                        continue

                    position_data = {
                        'id': row['id'],
                        'trader_id': row['trader_id'],
                        'symbol': self._convert_symbol(row['symbol']),
                        'exchange': self._convert_exchange(row['exchange']),
                        'quantity': row['quantity'],
                        'entry_price': row['entry_price'],
                        'current_price': row['current_price'],
                        'market_value': row['market_value'],
                        'unrealized_pnl': row['unrealized_pnl'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                    }

                    position = PositionModel(**position_data)
                    session.add(position)
                    await session.flush()

                    self.stats['positions']['migrated'] += 1

                except Exception as e:
                    logger.error(f"迁移持仓失败 {row['id']}: {e}")
                    self.stats['positions']['failed'] += 1
                    session.rollback()

            await session.commit()

        logger.info(f"持仓迁移完成: {self.stats['positions']}")

    async def migrate_orders(self):
        """迁移订单数据"""
        logger.info("开始迁移订单数据...")

        # 分批迁移以提高性能
        batch_size = 1000
        offset = 0

        while True:
            rows = await self.nofx_conn.fetch("""
                SELECT
                    id,
                    trader_id,
                    symbol,
                    exchange,
                    side,
                    order_type,
                    quantity,
                    price,
                    status,
                    filled_quantity,
                    avg_fill_price,
                    created_at,
                    updated_at,
                    filled_at,
                    cancelled_at,
                    rejected_reason
                FROM orders
                ORDER BY created_at
                LIMIT $1 OFFSET $2
            """, batch_size, offset)

            if not rows:
                break

            async with self.target_session_factory() as session:
                for row in rows:
                    try:
                        order_data = {
                            'id': row['id'],
                            'trader_id': row['trader_id'],
                            'symbol': self._convert_symbol(row['symbol']),
                            'exchange': self._convert_exchange(row['exchange']),
                            'side': row['side'],
                            'order_type': self._convert_order_type(row['order_type']),
                            'quantity': row['quantity'],
                            'price': row['price'],
                            'status': self._convert_order_status(row['status']),
                            'filled_quantity': row['filled_quantity'],
                            'avg_fill_price': row['avg_fill_price'],
                            'created_at': row['created_at'],
                            'updated_at': row['updated_at'],
                            'filled_at': row['filled_at'],
                            'cancelled_at': row['cancelled_at'],
                            'rejected_reason': row['rejected_reason'],
                        }

                        order = OrderModel(**order_data)
                        session.add(order)
                        await session.flush()

                        self.stats['orders']['migrated'] += 1

                    except Exception as e:
                        logger.error(f"迁移订单失败 {row['id']}: {e}")
                        self.stats['orders']['failed'] += 1
                        session.rollback()

                await session.commit()

            offset += batch_size
            logger.info(f"已迁移 {offset + len(rows)} 条订单")

        logger.info(f"订单迁移完成: {self.stats['orders']}")

    async def migrate_trades(self):
        """迁移成交记录"""
        logger.info("开始迁移成交记录...")

        batch_size = 1000
        offset = 0

        while True:
            rows = await self.nofx_conn.fetch("""
                SELECT
                    id,
                    order_id,
                    trader_id,
                    symbol,
                    exchange,
                    side,
                    quantity,
                    price,
                    commission,
                    timestamp,
                    external_trade_id
                FROM trades
                ORDER BY timestamp
                LIMIT $1 OFFSET $2
            """, batch_size, offset)

            if not rows:
                break

            async with self.target_session_factory() as session:
                for row in rows:
                    try:
                        trade_data = {
                            'id': row['id'],
                            'order_id': row['order_id'],
                            'trader_id': row['trader_id'],
                            'symbol': self._convert_symbol(row['symbol']),
                            'exchange': self._convert_exchange(row['exchange']),
                            'side': row['side'],
                            'quantity': row['quantity'],
                            'price': row['price'],
                            'commission': row['commission'],
                            'timestamp': row['timestamp'],
                            'external_trade_id': row['external_trade_id'],
                        }

                        trade = TradeModel(**trade_data)
                        session.add(trade)
                        await session.flush()

                        self.stats['trades']['migrated'] += 1

                    except Exception as e:
                        logger.error(f"迁移成交失败 {row['id']}: {e}")
                        self.stats['trades']['failed'] += 1
                        session.rollback()

                await session.commit()

            offset += batch_size
            logger.info(f"已迁移 {offset + len(rows)} 条成交记录")

        logger.info(f"成交记录迁移完成: {self.stats['trades']}")

    @staticmethod
    def _convert_trader_type(nofx_type: str) -> str:
        """转换交易员类型"""
        type_mapping = {
            'manual': 'discretionary',
            'ai': 'ai',
            'hybrid': 'hybrid',
        }
        return type_mapping.get(nofx_type, 'discretionary')

    @staticmethod
    def _convert_symbol(nofx_symbol: str) -> str:
        """转换交易代码格式"""
        # NOFX使用BTC/USD，新系统使用BTCUSD
        return nofx_symbol.replace('/', '').upper()

    @staticmethod
    def _convert_exchange(nofx_exchange: str) -> str:
        """转换交易所代码"""
        exchange_mapping = {
            'binance': 'BN',
            'okx': 'OKX',
            'bybit': 'BYBIT',
        }
        return exchange_mapping.get(nofx_exchange.lower(), 'XSHE')  # 默认深交所

    @staticmethod
    def _convert_order_type(nofx_type: str) -> str:
        """转换订单类型"""
        return nofx_type.lower()  # MARKET, LIMIT

    @staticmethod
    def _convert_order_status(nofx_status: str) -> str:
        """转换订单状态"""
        status_mapping = {
            'PENDING': 'pending',
            'OPEN': 'open',
            'FILLED': 'filled',
            'PARTIALLY_FILLED': 'partially_filled',
            'CANCELLED': 'cancelled',
            'REJECTED': 'rejected',
            'EXPIRED': 'expired',
        }
        return status_mapping.get(nofx_status, 'pending')

    @staticmethod
    def _convert_config(nofx_config: Dict) -> Dict:
        """转换配置格式"""
        if isinstance(nofx_config, str):
            nofx_config = json.loads(nofx_config)

        # 转换配置字段
        return {
            'risk_limit': nofx_config.get('risk_limit', 0.02),
            'max_position': nofx_config.get('max_position', 100000),
            'strategy_params': nofx_config.get('strategy', {}),
        }

    @staticmethod
    def _convert_metadata(nofx_metadata: Dict) -> Dict:
        """转换元数据格式"""
        if isinstance(nofx_metadata, str):
            nofx_metadata = json.loads(nofx_metadata)
        return nofx_metadata or {}

    def print_summary(self):
        """打印迁移摘要"""
        print("\n" + "="*50)
        print("数据迁移摘要")
        print("="*50)

        for entity, stats in self.stats.items():
            print(f"\n{entity.upper()}:")
            print(f"  迁移成功: {stats['migrated']}")
            print(f"  跳过:     {stats['skipped']}")
            print(f"  失败:     {stats['failed']}")


async def main():
    """主函数"""
    migrator = NOFXDataMigrator(
        nofx_db_url="postgresql://user:pass@localhost:5432/nofx",
        target_db_url=str(settings.DATABASE_URL),
    )

    await migrator.migrate_all()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
```

### 23.2 增量数据同步

```python
# scripts/incremental_sync.py
"""
增量数据同步脚本
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.models.trader import TraderModel
from src.models.order import OrderModel
from src.models.trade import TradeModel

logger = logging.getLogger(__name__)


class IncrementalSyncer:
    """增量同步器"""

    def __init__(self, source_db_url: str, target_db_url: str):
        self.source_db_url = source_db_url
        self.target_db_url = target_db_url
        self.last_sync_time = self._get_last_sync_time()

    @staticmethod
    def _get_last_sync_time() -> datetime:
        """获取上次同步时间"""
        # 从文件或数据库读取
        # 这里简化为10分钟前
        return datetime.now() - timedelta(minutes=10)

    async def sync_orders(self):
        """同步增量订单"""
        logger.info(f"同步 {self.last_sync_time} 之后的订单...")

        async with get_session() as source_session:
            # 查询增量数据
            query = select(OrderModel).where(
                OrderModel.updated_at > self.last_sync_time
            ).order_by(OrderModel.updated_at)

            result = await source_session.execute(query)
            orders = result.scalars().all()

            logger.info(f"找到 {len(orders)} 条增量订单")

            # 写入目标数据库
            async with get_session() as target_session:
                for order in orders:
                    # 检查是否已存在
                    existing = await target_session.get(OrderModel, order.id)
                    if existing:
                        # 更新
                        for key, value in order.__dict__.items():
                            if not key.startswith('_'):
                                setattr(existing, key, value)
                    else:
                        # 插入
                        target_session.add(order)

                await target_session.commit()

        # 更新同步时间
        self._update_last_sync_time()

    async def sync_trades(self):
        """同步增量成交"""
        logger.info(f"同步 {self.last_sync_time} 之后的成交...")

        async with get_session() as source_session:
            query = select(TradeModel).where(
                TradeModel.timestamp > self.last_sync_time
            ).order_by(TradeModel.timestamp)

            result = await source_session.execute(query)
            trades = result.scalars().all()

            logger.info(f"找到 {len(trades)} 条增量成交")

            async with get_session() as target_session:
                for trade in trades:
                    existing = await target_session.get(TradeModel, trade.id)
                    if existing:
                        for key, value in trade.__dict__.items():
                            if not key.startswith('_'):
                                setattr(existing, key, value)
                    else:
                        target_session.add(trade)

                await target_session.commit()

        self._update_last_sync_time()

    def _update_last_sync_time(self):
        """更新同步时间"""
        self.last_sync_time = datetime.now()
        # 持久化到文件或数据库

    async def run(self):
        """执行同步"""
        await self.sync_orders()
        await self.sync_trades()
        logger.info("增量同步完成")


async def main():
    """定时任务"""
    syncer = IncrementalSyncer(
        source_db_url="postgresql://...",
        target_db_url="postgresql://...",
    )

    while True:
        try:
            await syncer.run()
        except Exception as e:
            logger.error(f"同步失败: {e}")

        # 每10分钟同步一次
        await asyncio.sleep(600)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

### 23.3 备份恢复脚本

```bash
#!/bin/bash
# scripts/backup.sh

set -e

# 配置
BACKUP_DIR="/data/backups/deepalpha"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATABASE_HOST="postgres"
DATABASE_NAME="deepalpha"
DATABASE_USER="deepalpha"
S3_BUCKET="s3://deepalpha-backups"

# 创建备份目录
mkdir -p "$BACKUP_DIR/$TIMESTAMP"

echo "========================================="
echo "DeepAlpha备份脚本"
echo "时间: $(date)"
echo "========================================="

# 1. 数据库备份
echo "备份数据库..."
pg_dump -h "$DATABASE_HOST" \
        -U "$DATABASE_USER" \
        -d "$DATABASE_NAME" \
        -F c \
        -f "$BACKUP_DIR/$TIMESTAMP/database.dump"

# 2. Redis备份
echo "备份Redis..."
redis-cli --rdb "$BACKUP_DIR/$TIMESTAMP/redis.rdb"

# 3. 数据目录备份
echo "备份数据目录..."
tar -czf "$BACKUP_DIR/$TIMESTAMP/data.tar.gz" /app/data

# 4. 配置文件备份
echo "备份配置文件..."
tar -czf "$BACKUP_DIR/$TIMESTAMP/config.tar.gz" /etc/deepalpha

# 5. 生成备份清单
echo "生成备份清单..."
cat > "$BACKUP_DIR/$TIMESTAMP/manifest.txt" << EOF
备份时间: $(date)
数据库文件: database.dump
Redis文件: redis.rdb
数据文件: data.tar.gz
配置文件: config.tar.gz
EOF

# 6. 计算校验和
echo "计算校验和..."
sha256sum "$BACKUP_DIR/$TIMESTAMP"/* > "$BACKUP_DIR/$TIMESTAMP/sha256sums.txt"

# 7. 上传到S3
echo "上传到S3..."
aws s3 sync "$BACKUP_DIR/$TIMESTAMP" "$S3_BUCKET/$TIMESTAMP/"

# 8. 清理旧备份
echo "清理旧备份..."
find "$BACKUP_DIR" -maxdepth 1 -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;

# 9. 发送通知
echo "发送备份通知..."
curl -X POST "$SLACK_WEBHOOK" \
  -H 'Content-Type: application/json' \
  -d "{\"text\": \"✅ DeepAlpha备份完成: $TIMESTAMP\"}"

echo "备份完成!"
```

```bash
#!/bin/bash
# scripts/restore.sh

set -e

if [ -z "$1" ]; then
  echo "用法: $0 <备份时间戳>"
  echo "示例: $0 20240101_000000"
  exit 1
fi

BACKUP_ID="$1"
BACKUP_DIR="/data/backups/deepalpha/$BACKUP_ID"
S3_BUCKET="s3://deepalpha-backups"

echo "========================================="
echo "DeepAlpha恢复脚本"
echo "备份ID: $BACKUP_ID"
echo "========================================="

# 1. 从S3下载
if [ ! -d "$BACKUP_DIR" ]; then
  echo "从S3下载备份..."
  aws s3 sync "$S3_BUCKET/$BACKUP_ID/" "$BACKUP_DIR/"
fi

# 2. 验证校验和
echo "验证校验和..."
cd "$BACKUP_DIR"
sha256sum -c sha256sums.txt
if [ $? -ne 0 ]; then
  echo "校验和验证失败!"
  exit 1
fi

# 3. 停止服务
echo "停止服务..."
kubectl scale deployment deepalpha-api --replicas=0 -n deepalpha-production

# 4. 恢复数据库
echo "恢复数据库..."
pg_restore -h postgres -U deepalpha -d deepalpha --clean --if-exists "$BACKUP_DIR/database.dump"

# 5. 恢复Redis
echo "恢复Redis..."
redis-cli --rdb "$BACKUP_DIR/redis.rdb"

# 6. 恢复数据目录
echo "恢复数据目录..."
tar -xzf "$BACKUP_DIR/data.tar.gz" -C /

# 7. 恢复配置文件
echo "恢复配置文件..."
tar -xzf "$BACKUP_DIR/config.tar.gz" -C /

# 8. 启动服务
echo "启动服务..."
kubectl scale deployment deepalpha-api --replicas=3 -n deepalpha-production

# 9. 等待服务就绪
echo "等待服务就绪..."
kubectl wait --for=condition=ready pod -l app=deepalpha -n deepalpha-production --timeout=300s

# 10. 验证
echo "验证服务..."
curl -f http://api.deepalpha.example.com/health || exit 1

echo "恢复完成!"
```

---

## 第24章 前端高级模式与状态管理

### 24.1 React状态管理架构

```typescript
// frontend/src/store/useAppStore.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { subscribeWithSelector } from 'zustand/middleware';

// ============================================
// 类型定义
// ============================================
interface Trader {
  id: string;
  name: string;
  type: 'discretionary' | 'ai' | 'hybrid';
  equity: number;
  pnl: number;
  isActive: boolean;
}

interface Position {
  id: string;
  traderId: string;
  symbol: string;
  exchange: string;
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
}

interface Order {
  id: string;
  traderId: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  quantity: number;
  price?: number;
  status: 'pending' | 'open' | 'filled' | 'cancelled';
  createdAt: string;
}

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
}

// ============================================
// Slice定义
// ============================================

// Auth Slice
interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
}

interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'trader' | 'viewer';
}

// Trader Slice
interface TraderState {
  traders: Trader[];
  selectedTraderId: string | null;
  isLoading: boolean;
  error: string | null;
  fetchTraders: () => Promise<void>;
  selectTrader: (id: string) => void;
  createTrader: (data: Partial<Trader>) => Promise<void>;
  updateTrader: (id: string, data: Partial<Trader>) => Promise<void>;
  deleteTrader: (id: string) => Promise<void>;
}

// Position Slice
interface PositionState {
  positions: Position[];
  filterByTrader: string | null;
  fetchPositions: () => Promise<void>;
  updatePosition: (position: Position) => void;
}

// Order Slice
interface OrderState {
  orders: Order[];
  pendingOrders: Order[];
  fetchOrders: (traderId?: string) => Promise<void>;
  submitOrder: (order: Omit<Order, 'id' | 'status' | 'createdAt'>) => Promise<void>;
  cancelOrder: (orderId: string) => Promise<void>;
}

// Market Data Slice
interface MarketDataState {
  data: Map<string, MarketData>;
  watchlist: string[];
  subscribe: (symbols: string[]) => void;
  unsubscribe: (symbols: string[]) => void;
  updatePrice: (symbol: string, data: Partial<MarketData>) => void;
}

// UI Slice
interface UIState {
  sidebarOpen: boolean;
  theme: 'light' | 'dark';
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id'>) => void;
  removeNotification: (id: string) => void;
  toggleSidebar: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
}

interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  duration?: number;
}

// ============================================
// Store创建
// ============================================
interface AppState extends AuthState, TraderState, PositionState, OrderState, MarketDataState, UIState {}

export const useAppStore = create<AppState>()(
  devtools(
    persist(
      subscribeWithSelector(
        immer((set, get) => ({
          // ============================================
          // Auth State & Actions
          // ============================================
          user: null,
          token: null,
          isAuthenticated: false,

          login: async (email: string, password: string) => {
            try {
              const response = await fetch('/api/v1/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
              });

              if (!response.ok) throw new Error('Login failed');

              const data = await response.json();

              set((state) => {
                state.user = data.user;
                state.token = data.token;
                state.isAuthenticated = true;
              });
            } catch (error) {
              set((state) => {
                state.error = (error as Error).message;
              });
              throw error;
            }
          },

          logout: () => {
            set((state) => {
              state.user = null;
              state.token = null;
              state.isAuthenticated = false;
            });
          },

          refreshToken: async () => {
            const { token } = get();
            if (!token) return;

            try {
              const response = await fetch('/api/v1/auth/refresh', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                  'Authorization': `Bearer ${token}`,
                },
              });

              if (!response.ok) throw new Error('Token refresh failed');

              const data = await response.json();
              set((state) => {
                state.token = data.token;
              });
            } catch (error) {
              get().logout();
              throw error;
            }
          },

          // ============================================
          // Trader State & Actions
          // ============================================
          traders: [],
          selectedTraderId: null,
          isLoading: false,
          error: null,

          fetchTraders: async () => {
            set((state) => {
              state.isLoading = true;
              state.error = null;
            });

            try {
              const { token } = get();
              const response = await fetch('/api/v1/traders', {
                headers: {
                  'Authorization': `Bearer ${token}`,
                },
              });

              if (!response.ok) throw new Error('Failed to fetch traders');

              const data = await response.json();

              set((state) => {
                state.traders = data;
                state.isLoading = false;
              });
            } catch (error) {
              set((state) => {
                state.error = (error as Error).message;
                state.isLoading = false;
              });
            }
          },

          selectTrader: (id: string) => {
            set((state) => {
              state.selectedTraderId = id;
            });
          },

          createTrader: async (data: Partial<Trader>) => {
            const { token } = get();
            const response = await fetch('/api/v1/traders', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
              },
              body: JSON.stringify(data),
            });

            if (!response.ok) throw new Error('Failed to create trader');

            const newTrader = await response.json();

            set((state) => {
              state.traders.push(newTrader);
            });
          },

          updateTrader: async (id: string, data: Partial<Trader>) => {
            const { token } = get();
            const response = await fetch(`/api/v1/traders/${id}`, {
              method: 'PATCH',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
              },
              body: JSON.stringify(data),
            });

            if (!response.ok) throw new Error('Failed to update trader');

            const updatedTrader = await response.json();

            set((state) => {
              const index = state.traders.findIndex((t) => t.id === id);
              if (index !== -1) {
                state.traders[index] = updatedTrader;
              }
            });
          },

          deleteTrader: async (id: string) => {
            const { token } = get();
            const response = await fetch(`/api/v1/traders/${id}`, {
              method: 'DELETE',
              headers: {
                'Authorization': `Bearer ${token}`,
              },
            });

            if (!response.ok) throw new Error('Failed to delete trader');

            set((state) => {
              state.traders = state.traders.filter((t) => t.id !== id);
              if (state.selectedTraderId === id) {
                state.selectedTraderId = null;
              }
            });
          },

          // ============================================
          // Position State & Actions
          // ============================================
          positions: [],
          filterByTrader: null,

          fetchPositions: async () => {
            const { token, filterByTrader } = get();
            const url = filterByTrader
              ? `/api/v1/positions?trader_id=${filterByTrader}`
              : '/api/v1/positions';

            const response = await fetch(url, {
              headers: {
                'Authorization': `Bearer ${token}`,
              },
            });

            if (!response.ok) throw new Error('Failed to fetch positions');

            const data = await response.json();

            set((state) => {
              state.positions = data;
            });
          },

          updatePosition: (position: Position) => {
            set((state) => {
              const index = state.positions.findIndex((p) => p.id === position.id);
              if (index !== -1) {
                state.positions[index] = position;
              } else {
                state.positions.push(position);
              }
            });
          },

          // ============================================
          // Order State & Actions
          // ============================================
          orders: [],
          pendingOrders: [],

          fetchOrders: async (traderId?: string) => {
            const { token } = get();
            const url = traderId
              ? `/api/v1/orders?trader_id=${traderId}`
              : '/api/v1/orders';

            const response = await fetch(url, {
              headers: {
                'Authorization': `Bearer ${token}`,
              },
            });

            if (!response.ok) throw new Error('Failed to fetch orders');

            const data = await response.json();

            set((state) => {
              state.orders = data;
              state.pendingOrders = data.filter((o: Order) => o.status === 'pending' || o.status === 'open');
            });
          },

          submitOrder: async (order: Omit<Order, 'id' | 'status' | 'createdAt'>) => {
            const { token } = get();
            const response = await fetch('/api/v1/orders', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
              },
              body: JSON.stringify(order),
            });

            if (!response.ok) throw new Error('Failed to submit order');

            const newOrder = await response.json();

            set((state) => {
              state.orders.push(newOrder);
              if (newOrder.status === 'pending' || newOrder.status === 'open') {
                state.pendingOrders.push(newOrder);
              }
            });
          },

          cancelOrder: async (orderId: string) => {
            const { token } = get();
            const response = await fetch(`/api/v1/orders/${orderId}/cancel`, {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${token}`,
              },
            });

            if (!response.ok) throw new Error('Failed to cancel order');

            set((state) => {
              const order = state.orders.find((o) => o.id === orderId);
              if (order) {
                order.status = 'cancelled';
              }
              state.pendingOrders = state.pendingOrders.filter((o) => o.id !== orderId);
            });
          },

          // ============================================
          // Market Data State & Actions
          // ============================================
          data: new Map(),
          watchlist: [],

          subscribe: (symbols: string[]) => {
            // WebSocket订阅逻辑
            const ws = new WebSocket(`${import.meta.env.VITE_WS_URL}/ws/market`);

            ws.onopen = () => {
              ws.send(JSON.stringify({
                action: 'subscribe',
                symbols,
              }));
            };

            ws.onmessage = (event) => {
              const data = JSON.parse(event.data);
              set((state) => {
                state.data.set(data.symbol, data);
              });
            };

            set((state) => {
              state.watchlist = [...new Set([...state.watchlist, ...symbols])];
            });
          },

          unsubscribe: (symbols: string[]) => {
            // WebSocket取消订阅逻辑
            set((state) => {
              state.watchlist = state.watchlist.filter((s) => !symbols.includes(s));
              symbols.forEach((symbol) => {
                state.data.delete(symbol);
              });
            });
          },

          updatePrice: (symbol: string, data: Partial<MarketData>) => {
            set((state) => {
              const existing = state.data.get(symbol);
              state.data.set(symbol, { ...existing, ...data } as MarketData);
            });
          },

          // ============================================
          // UI State & Actions
          // ============================================
          sidebarOpen: true,
          theme: 'light',
          notifications: [],

          addNotification: (notification: Omit<Notification, 'id'>) => {
            const id = crypto.randomUUID();
            set((state) => {
              state.notifications.push({ ...notification, id });
            });

            // 自动移除通知
            if (notification.duration !== 0) {
              setTimeout(() => {
                get().removeNotification(id);
              }, notification.duration || 5000);
            }
          },

          removeNotification: (id: string) => {
            set((state) => {
              state.notifications = state.notifications.filter((n) => n.id !== id);
            });
          },

          toggleSidebar: () => {
            set((state) => {
              state.sidebarOpen = !state.sidebarOpen;
            });
          },

          setTheme: (theme: 'light' | 'dark') => {
            set((state) => {
              state.theme = theme;
            });
            document.documentElement.setAttribute('data-theme', theme);
          },
        }))
      ),
      {
        name: 'deepalpha-storage',
        partialize: (state) => ({
          theme: state.theme,
          sidebarOpen: state.sidebarOpen,
          watchlist: state.watchlist,
        }),
      }
    ),
    { name: 'DeepAlphaStore' }
  )
);

// ============================================
// Selectors
// ============================================
export const selectTraders = (state: AppState) => state.traders;
export const selectActiveTraders = (state: AppState) => state.traders.filter((t) => t.isActive);
export const selectSelectedTrader = (state: AppState) =>
  state.traders.find((t) => t.id === state.selectedTraderId) || null;
export const selectPositionsByTrader = (traderId: string) => (state: AppState) =>
  state.positions.filter((p) => p.traderId === traderId);
export const selectOrdersByTrader = (traderId: string) => (state: AppState) =>
  state.orders.filter((o) => o.traderId === traderId);
export const selectMarketData = (symbol: string) => (state: AppState) => state.data.get(symbol);
```

### 24.2 自定义Hooks

```typescript
// frontend/src/hooks/useWebSocket.ts
import { useEffect, useRef, useCallback } from 'react';
import { useAppStore } from '../store/useAppStore';

interface WebSocketMessage {
  type: 'price' | 'order' | 'position' | 'trade' | 'notification';
  data: unknown;
}

export function useWebSocket(url: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const addNotification = useAppStore((state) => state.addNotification);
  const updatePrice = useAppStore((state) => state.updatePrice);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      reconnectAttempts.current = 0;

      // 发送认证消息
      const token = localStorage.getItem('deepalpha-storage');
      if (token) {
        const parsed = JSON.parse(token);
        ws.send(JSON.stringify({
          type: 'auth',
          token: parsed.state.token,
        }));
      }
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);

        switch (message.type) {
          case 'price':
            updatePrice((message.data as MarketData).symbol, message.data as MarketData);
            break;

          case 'order':
            // 处理订单更新
            break;

          case 'notification':
            addNotification(message.data as Omit<Notification, 'id'>);
            break;

          default:
            console.warn('Unknown message type:', message.type);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      addNotification({
        type: 'error',
        message: 'WebSocket连接错误',
      });
    };

    ws.onclose = () => {
      console.log('WebSocket closed');

      // 自动重连
      if (reconnectAttempts.current < maxReconnectAttempts) {
        reconnectAttempts.current++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);

        reconnectTimeoutRef.current = setTimeout(() => {
          console.log(`Reconnecting... Attempt ${reconnectAttempts.current}`);
          connect();
        }, delay);
      } else {
        addNotification({
          type: 'error',
          message: 'WebSocket连接失败，请刷新页面',
          duration: 0,
        });
      }
    };
  }, [url, updatePrice, addNotification]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const send = useCallback((message: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }, []);

  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return { send, disconnect };
}

// ============================================
// useTraders Hook
// ============================================
export function useTraders() {
  const {
    traders,
    selectedTraderId,
    isLoading,
    error,
    fetchTraders,
    selectTrader,
    createTrader,
    updateTrader,
    deleteTrader,
  } = useAppStore();

  useEffect(() => {
    fetchTraders();
  }, [fetchTraders]);

  return {
    traders,
    selectedTrader: traders.find((t) => t.id === selectedTraderId) || null,
    selectedTraderId,
    isLoading,
    error,
    selectTrader,
    createTrader,
    updateTrader,
    deleteTrader,
    refreshTraders: fetchTraders,
  };
}

// ============================================
// usePositions Hook
// ============================================
export function usePositions(traderId?: string) {
  const { positions, fetchPositions, filterByTrader } = useAppStore();

  useEffect(() => {
    if (traderId && filterByTrader !== traderId) {
      // 更新过滤条件并重新获取
    }
    fetchPositions();
  }, [traderId, fetchPositions, filterByTrader]);

  const filteredPositions = traderId
    ? positions.filter((p) => p.traderId === traderId)
    : positions;

  return {
    positions: filteredPositions,
    refreshPositions: fetchPositions,
  };
}

// ============================================
// useOrders Hook
// ============================================
export function useOrders(traderId?: string) {
  const {
    orders,
    pendingOrders,
    fetchOrders,
    submitOrder,
    cancelOrder,
  } = useAppStore();

  useEffect(() => {
    fetchOrders(traderId);
  }, [traderId, fetchOrders]);

  const filteredOrders = traderId
    ? orders.filter((o) => o.traderId === traderId)
    : orders;

  return {
    orders: filteredOrders,
    pendingOrders: traderId
      ? pendingOrders.filter((o) => o.traderId === traderId)
      : pendingOrders,
    refreshOrders: () => fetchOrders(traderId),
    submitOrder,
    cancelOrder,
  };
}

// ============================================
// useMarketData Hook
// ============================================
export function useMarketData(symbols: string[]) {
  const { data, subscribe, unsubscribe } = useAppStore();

  useEffect(() => {
    subscribe(symbols);

    return () => {
      unsubscribe(symbols);
    };
  }, [symbols.join(','), subscribe, unsubscribe]);

  const getMarketData = (symbol: string) => data.get(symbol);

  return {
    getMarketData,
    allData: data,
  };
}

// ============================================
// useNotifications Hook
// ============================================
export function useNotifications() {
  const { notifications, addNotification, removeNotification } = useAppStore();

  const showSuccess = useCallback((message: string, duration?: number) => {
    addNotification({ type: 'success', message, duration });
  }, [addNotification]);

  const showError = useCallback((message: string, duration?: number) => {
    addNotification({ type: 'error', message, duration });
  }, [addNotification]);

  const showWarning = useCallback((message: string, duration?: number) => {
    addNotification({ type: 'warning', message, duration });
  }, [addNotification]);

  const showInfo = useCallback((message: string, duration?: number) => {
    addNotification({ type: 'info', message, duration });
  }, [addNotification]);

  return {
    notifications,
    removeNotification,
    showSuccess,
    showError,
    showWarning,
    showInfo,
  };
}

// ============================================
// useDebounce Hook
// ============================================
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

// ============================================
// useLocalStorage Hook
// ============================================
export function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error loading ${key} from localStorage:`, error);
      return initialValue;
    }
  });

  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(`Error saving ${key} to localStorage:`, error);
    }
  }, [key, storedValue]);

  return [storedValue, setValue] as const;
}
```

### 24.3 高级组件模式

```typescript
// frontend/src/components/common/AsyncBoundary.tsx
import { ComponentType, Suspense, lazy } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { PulseLoader } from 'react-spinners';

interface AsyncBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  errorFallback?: React.ReactNode;
}

export function AsyncBoundary({
  children,
  fallback = <LoadingFallback />,
  errorFallback = <ErrorFallback />,
}: AsyncBoundaryProps) {
  return (
    <Suspense fallback={fallback}>
      <ErrorBoundary FallbackComponent={ErrorFallback}>
        {children}
      </ErrorBoundary>
    </Suspense>
  );
}

function LoadingFallback() {
  return (
    <div className="flex items-center justify-center h-64">
      <PulseLoader color="#3b82f6" size={15} />
    </div>
  );
}

function ErrorFallback({ error, resetErrorBoundary }: { error: Error; resetErrorBoundary: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-64 text-center">
      <div className="text-red-500 text-6xl mb-4">⚠️</div>
      <h3 className="text-lg font-semibold mb-2">出错了</h3>
      <p className="text-gray-600 mb-4">{error.message}</p>
      <button
        onClick={resetErrorBoundary}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        重试
      </button>
    </div>
  );
}

// ============================================
// 懒加载HOC
// ============================================
export function withLazyLoading<P extends object>(
  component: ComponentType<P>,
  loadingComponent?: React.ReactNode
) {
  return lazy(() => {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          default: component,
        });
      }, 300);
    });
  });
}

// ============================================
// VirtualList组件
// ============================================
import { useVirtualizer } from '@tanstack/react-virtual';

interface VirtualListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  itemHeight: number;
  height: number;
  overscan?: number;
}

export function VirtualList<T>({
  items,
  renderItem,
  itemHeight,
  height,
  overscan = 5,
}: VirtualListProps<T>) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => itemHeight,
    overscan,
  });

  return (
    <div ref={parentRef} style={{ height, overflow: 'auto' }}>
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((virtualItem) => (
          <div
            key={virtualItem.key}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              transform: `translateY(${virtualItem.start}px)`,
            }}
          >
            {renderItem(items[virtualItem.index], virtualItem.index)}
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================
// InfiniteScroll组件
// ============================================
interface InfiniteScrollProps<T> {
  fetchMore: (page: number) => Promise<T[]>;
  renderItem: (item: T, index: number) => React.ReactNode;
  initialPage?: number;
  pageSize?: number;
}

export function InfiniteScroll<T>({
  fetchMore,
  renderItem,
  initialPage = 1,
  pageSize = 20,
}: InfiniteScrollProps<T>) {
  const [items, setItems] = useState<T[]>([]);
  const [page, setPage] = useState(initialPage);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);

  const loadMore = useCallback(async () => {
    if (loading || !hasMore) return;

    setLoading(true);
    try {
      const newItems = await fetchMore(page);
      setItems((prev) => [...prev, ...newItems]);
      setPage((prev) => prev + 1);

      if (newItems.length < pageSize) {
        setHasMore(false);
      }
    } catch (error) {
      console.error('Failed to load more items:', error);
    } finally {
      setLoading(false);
    }
  }, [fetchMore, page, loading, hasMore, pageSize]);

  useEffect(() => {
    loadMore();
  }, []);

  const observerTarget = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !loading) {
          loadMore();
        }
      },
      { threshold: 1.0 }
    );

    if (observerTarget.current) {
      observer.observe(observerTarget.current);
    }

    return () => observer.disconnect();
  }, [loadMore, hasMore, loading]);

  return (
    <div>
      {items.map((item, index) => renderItem(item, index))}
      <div ref={observerTarget} className="h-4" />
      {loading && <LoadingFallback />}
    </div>
  );
}
```

### 24.4 性能优化组件

```typescript
// frontend/src/components/common/MemoizedComponents.tsx
import { memo, useMemo, useCallback } from 'react';

// ============================================
// Memoized Trader Card
// ============================================
interface TraderCardProps {
  trader: Trader;
  onSelect: (id: string) => void;
  isSelected: boolean;
}

export const TraderCard = memo<TraderCardProps>(({ trader, onSelect, isSelected }) => {
  const handleClick = useCallback(() => {
    onSelect(trader.id);
  }, [trader.id, onSelect]);

  const pnlColor = useMemo(() => {
    if (trader.pnl > 0) return 'text-green-500';
    if (trader.pnl < 0) return 'text-red-500';
    return 'text-gray-500';
  }, [trader.pnl]);

  const pnlPercent = useMemo(() => {
    return ((trader.pnl / trader.equity) * 100).toFixed(2);
  }, [trader.pnl, trader.equity]);

  return (
    <div
      onClick={handleClick}
      className={`
        p-4 rounded-lg cursor-pointer transition-all
        ${isSelected ? 'bg-blue-500 text-white' : 'bg-white hover:bg-gray-50'}
      `}
    >
      <h3 className="font-semibold">{trader.name}</h3>
      <p className={`text-sm ${pnlColor}`}>
        ¥{trader.pnl.toLocaleString()} ({pnlPercent}%)
      </p>
      <p className="text-xs text-gray-500">
        权益: ¥{trader.equity.toLocaleString()}
      </p>
    </div>
  );
}, (prevProps, nextProps) => {
  return (
    prevProps.trader.id === nextProps.trader.id &&
    prevProps.trader.equity === nextProps.trader.equity &&
    prevProps.trader.pnl === nextProps.trader.pnl &&
    prevProps.isSelected === nextProps.isSelected
  );
});

TraderCard.displayName = 'TraderCard';

// ============================================
// Optimized Position Table
// ============================================
import { useTable, useSortBy, usePagination } from 'react-table';
import { useMemo } from 'react';

interface PositionTableProps {
  positions: Position[];
}

export function PositionTable({ positions }: PositionTableProps) {
  const data = useMemo(() => positions, [positions]);

  const columns = useMemo(() => [
    {
      Header: '股票代码',
      accessor: 'symbol',
    },
    {
      Header: '交易所',
      accessor: 'exchange',
    },
    {
      Header: '数量',
      accessor: 'quantity',
      Cell: ({ value }: { value: number }) => value.toLocaleString(),
    },
    {
      Header: '持仓价',
      accessor: 'entryPrice',
      Cell: ({ value }: { value: number }) => `¥${value.toFixed(2)}`,
    },
    {
      Header: '现价',
      accessor: 'currentPrice',
      Cell: ({ value }: { value: number }) => `¥${value.toFixed(2)}`,
    },
    {
      Header: '未实现盈亏',
      accessor: 'unrealizedPnl',
      Cell: ({ value }: { value: number }) => (
        <span className={value >= 0 ? 'text-green-500' : 'text-red-500'}>
          ¥{value.toFixed(2)}
        </span>
      ),
    },
  ], []);

  const tableInstance = useTable(
    { columns, data },
    useSortBy,
    usePagination
  );

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    page,
    prepareRow,
  } = tableInstance;

  return (
    <div className="overflow-x-auto">
      <table {...getTableProps()} className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          {headerGroups.map((headerGroup) => (
            <tr {...headerGroup.getHeaderGroupProps()}>
              {headerGroup.headers.map((column) => (
                <th
                  {...column.getHeaderProps(column.getSortByToggleProps())}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {column.render('Header')}
                  <span>
                    {column.isSorted
                      ? column.isSortedDesc
                        ? ' 🔽'
                        : ' 🔼'
                      : ''}
                  </span>
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody {...getTableBodyProps()} className="bg-white divide-y divide-gray-200">
          {page.map((row) => {
            prepareRow(row);
            return (
              <tr {...row.getRowProps()}>
                {row.cells.map((cell) => (
                  <td
                    {...cell.getCellProps()}
                    className="px-6 py-4 whitespace-nowrap text-sm text-gray-900"
                  >
                    {cell.render('Cell')}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ============================================
// Debounced Search Input
// ============================================
import { useDebouncedCallback } from 'use-debounce';

interface SearchInputProps {
  onSearch: (query: string) => void;
  placeholder?: string;
  debounceMs?: number;
}

export function SearchInput({
  onSearch,
  placeholder = '搜索...',
  debounceMs = 300,
}: SearchInputProps) {
  const [query, setQuery] = useState('');

  const debouncedSearch = useDebouncedCallback(
    (value: string) => onSearch(value),
    debounceMs
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    debouncedSearch(value);
  };

  return (
    <div className="relative">
      <input
        type="text"
        value={query}
        onChange={handleChange}
        placeholder={placeholder}
        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      />
      <svg
        className="absolute right-3 top-2.5 h-5 w-5 text-gray-400"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
        />
      </svg>
    </div>
  );
}
```

---

## 第25章 生产运维手册

### 25.1 日常运维检查清单

#### 每日检查项

```bash
#!/bin/bash
# scripts/daily_health_check.sh

echo "========================================="
echo "DeepAlpha 日常健康检查"
echo "日期: $(date)"
echo "========================================="

# 1. 服务状态检查
echo -e "\n[1] 检查服务状态..."
kubectl get pods -n deepalpha-production

# 2. 检查Pod健康状态
echo -e "\n[2] 检查Pod健康状态..."
kubectl get pods -n deepalpha-production -o json | \
  jq -r '.items[] | select(.status.containerStatuses[].ready != true) | "\(.metadata.name): \(.status.containerStatuses[].state)"'

# 3. 检查资源使用情况
echo -e "\n[3] 检查资源使用情况..."
kubectl top nodes
kubectl top pods -n deepalpha-production

# 4. 检查磁盘空间
echo -e "\n[4] 检查磁盘空间..."
df -h | grep -E '(Filesystem|/dev/)'

# 5. 检查数据库连接
echo -e "\n[5] 检查数据库连接..."
kubectl exec -n deepalpha-production deployment/deepalpha-api -- \
  pg_isready -h postgres -U deepalpha

# 6. 检查Redis连接
echo -e "\n[6] 检查Redis连接..."
kubectl exec -n deepalpha-production deployment/deepalpha-api -- \
  redis-cli -h redis ping

# 7. 检查API健康端点
echo -e "\n[7] 检查API健康端点..."
curl -f http://api.deepalpha.example.com/health || echo "API健康检查失败"

# 8. 检查日志错误
echo -e "\n[8] 检查最近1小时错误日志..."
kubectl logs -n deepalpha-production -l app=deepalpha --since=1h | grep -i error | tail -20

# 9. 检查Celery任务队列
echo -e "\n[9] 检查Celery任务队列..."
curl -s http://flower.deepalpha.example.com/api/workers | jq -r '.[] | "\(.name): \(.status)"'

# 10. 检查Prometheus告警
echo -e "\n[10] 检查活动告警..."
curl -s 'http://prometheus.deepalpha.example.com/api/v1/alerts' | \
  jq -r '.data.alerts[] | select(.state=="firing") | "\(.labels.alertname): \(.annotations.summary)"'

echo -e "\n========================================="
echo "检查完成"
echo "========================================="
```

#### 每周检查项

```bash
#!/bin/bash
# scripts/weekly_check.sh

echo "========================================="
echo "DeepAlpha 每周检查"
echo "日期: $(date)"
echo "========================================="

# 1. 数据库性能分析
echo -e "\n[1] 数据库慢查询分析..."
kubectl exec -n deepalpha-production postgres-0 -- \
  psql -U deepalpha -d deepalpha -c "
  SELECT query, calls, total_time, mean_time
  FROM pg_stat_statements
  ORDER BY mean_time DESC
  LIMIT 10;
"

# 2. 检查索引使用情况
echo -e "\n[2] 检查未使用的索引..."
kubectl exec -n deepalpha-production postgres-0 -- \
  psql -U deepalpha -d deepalpha -c "
  SELECT schemaname, tablename, indexname
  FROM pg_stat_user_indexes
  WHERE idx_scan = 0
  AND indisunique = false;
"

# 3. 分析表膨胀情况
echo -e "\n[3] 分析表膨胀..."
kubectl exec -n deepalpha-production postgres-0 -- \
  psql -U deepalpha -d deepalpha -c "
  SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS index_size
  FROM pg_tables
  WHERE schemaname = 'public'
  ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
  LIMIT 10;
"

# 4. 检查SSL证书有效期
echo -e "\n[4] 检查SSL证书..."
openssl s_client -connect api.deepalpha.example.com:443 -servername api.deepalpha.example.com </dev/null 2>/dev/null | \
  openssl x509 -noout -dates

# 5. 检查依赖更新
echo -e "\n[5] 检查Python依赖更新..."
pip list --outdated

# 6. 检查安全漏洞
echo -e "\n[6] 检查安全漏洞..."
safety check --json || true

echo -e "\n========================================="
echo "每周检查完成"
echo "========================================="
```

### 25.2 故障排查流程

#### API响应缓慢

```bash
#!/bin/bash
# scripts/troubleshoot/slow_api.sh

echo "诊断API响应缓慢问题..."

# 1. 检查Pod资源限制
echo "[1] 检查Pod资源使用..."
kubectl top pods -n deepalpha-production -l app=deepalpha-api

# 2. 检查数据库慢查询
echo "[2] 检查数据库慢查询..."
kubectl exec -n deepalpha-production postgres-0 -- \
  psql -U deepalpha -d deepalpha -c "
  SELECT pid, now() - query_start as duration, query
  FROM pg_stat_activity
  WHERE state = 'active'
  ORDER BY duration DESC;
"

# 3. 检查数据库连接池
echo "[3] 检查数据库连接池..."
kubectl exec -n deepalpha-production deployment/deepalpha-api -- \
  python -c "
import asyncpg
import asyncio

async def check_connections():
    conn = await asyncpg.connect('postgresql://deepalpha:password@postgres:5432/deepalpha')
    result = await conn.fetchval('SELECT count(*) FROM pg_stat_activity WHERE datname = $1', 'deepalpha')
    print(f'当前数据库连接数: {result}')
    await conn.close()

asyncio.run(check_connections())
"

# 4. 检查Redis性能
echo "[4] 检查Redis性能..."
kubectl exec -n deepalpha-production redis-0 -- redis-cli INFO stats | grep -E '(instantaneous_ops_per_sec|used_memory)'

# 5. 检查网络延迟
echo "[5] 检查Pod间网络延迟..."
kubectl exec -n deepalpha-production deployment/deepalpha-api -- \
  ping -c 10 postgres.deepalpha-production.svc.cluster.local

# 6. 分析应用日志
echo "[6] 检查慢请求日志..."
kubectl logs -n deepalpha-production -l app=deepalpha-api --tail=1000 | \
  grep -i "slow request" | tail -20

# 7. 检查CPU Throttling
echo "[7] 检查CPU限速..."
kubectl get pods -n deepalpha-production -l app=deepalpha-api -o json | \
  jq -r '.items[] | "\(.metadata.name): \(.status.containerStatuses[].state.terminated.reason)"'
```

#### 数据库连接池耗尽

```python
# scripts/troubleshoot/db_pool.py
import asyncio
import asyncpg
from typing import List

async def diagnose_db_pool():
    """诊断数据库连接池问题"""

    conn = await asyncpg.connect(
        'postgresql://deepalpha:password@localhost:5432/deepalpha'
    )

    try:
        # 1. 检查当前连接数
        result = await conn.fetchval("""
            SELECT count(*)
            FROM pg_stat_activity
            WHERE datname = 'deepalpha'
        """)
        print(f"当前连接数: {result}")

        # 2. 检查连接状态分布
        rows = await conn.fetch("""
            SELECT state, count(*)
            FROM pg_stat_activity
            WHERE datname = 'deepalpha'
            GROUP BY state
        """)
        print("\n连接状态分布:")
        for row in rows:
            print(f"  {row['state']}: {row['count']}")

        # 3. 检查长时间运行的查询
        rows = await conn.fetch("""
            SELECT
                pid,
                now() - query_start as duration,
                state,
                query
            FROM pg_stat_activity
            WHERE datname = 'deepalpha'
            AND state = 'active'
            AND now() - query_start > interval '1 minute'
        """)
        if rows:
            print("\n长时间运行的查询 (>1分钟):")
            for row in rows:
                print(f"  PID: {row['pid']}, 时长: {row['duration']}")
                print(f"  查询: {row['query'][:100]}...")

        # 4. 检查空闲连接
        result = await conn.fetchval("""
            SELECT count(*)
            FROM pg_stat_activity
            WHERE datname = 'deepalpha'
            AND state = 'idle'
            AND now() - query_start > interval '5 minutes'
        """)
        print(f"\n长时间空闲连接 (>5分钟): {result}")

        # 5. 建议清理
        if result > 0:
            print("\n建议: 终止长时间空闲连接")
            print("执行: SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE ...")

    finally:
        await conn.close()

if __name__ == '__main__':
    asyncio.run(diagnose_db_pool())
```

### 25.3 应急响应程序

#### 紧急回滚

```bash
#!/bin/bash
# scripts/emergency/rollback.sh

set -e

VERSION_TO_ROLLBACK=$1
NAMESPACE=${2:-deepalpha-production}

if [ -z "$VERSION_TO_ROLLBACK" ]; then
  echo "用法: $0 <版本> [命名空间]"
  echo "示例: $0 v1.2.3 deepalpha-production"
  exit 1
fi

echo "========================================="
echo "紧急回滚到版本: $VERSION_TO_ROLLBACK"
echo "命名空间: $NAMESPACE"
echo "时间: $(date)"
echo "========================================="

# 1. 通知Slack
curl -X POST "$SLACK_WEBHOOK" \
  -H 'Content-Type: application/json' \
  -d "{\"text\": \"🚨 开始紧急回滚到 $VERSION_TO_ROLLBACK\"}"

# 2. 记录当前版本
CURRENT_VERSION=$(kubectl get deployment deepalpha-api -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')
echo "当前版本: $CURRENT_VERSION"

# 3. 执行回滚
echo "执行回滚..."
kubectl set image deployment/deepalpha-api \
  deepalpha-api=ghcr.io/your-org/deepalpha:$VERSION_TO_ROLLBACK \
  -n $NAMESPACE

# 4. 等待回滚完成
echo "等待回滚完成..."
kubectl rollout status deployment/deepalpha-api -n $NAMESPACE --timeout=5m

# 5. 验证健康状态
echo "验证健康状态..."
sleep 30

HEALTH_CHECK_URL="http://api.deepalpha.example.com/health"
MAX_ATTEMPTS=10
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
  if curl -f $HEALTH_CHECK_URL; then
    echo "健康检查通过"
    break
  fi

  ATTEMPT=$((ATTEMPT + 1))
  echo "健康检查失败，重试 ($ATTEMPT/$MAX_ATTEMPTS)..."
  sleep 10
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
  echo "健康检查失败，回滚可能存在问题!"
  exit 1
fi

# 6. 通知回滚完成
curl -X POST "$SLACK_WEBHOOK" \
  -H 'Content-Type: application/json' \
  -d "{\"text\": \"✅ 回滚到 $VERSION_TO_ROLLBACK 完成\"}"

echo "========================================="
echo "回滚成功完成"
echo "========================================="
```

#### 数据库故障恢复

```bash
#!/bin/bash
# scripts/emergency/db_recovery.sh

echo "========================================="
echo "数据库故障恢复程序"
echo "时间: $(date)"
echo "========================================="

# 1. 检查主库状态
echo "[1] 检查主库状态..."
kubectl get pod -n deepalpha-production -l role=postgres,position=primary

# 2. 检查从库状态
echo "[2] 检查从库状态..."
kubectl get pod -n deepalpha-production -l role=postgres,position=replica

# 3. 提升从库为主库
echo "[3] 提升从库为主库..."
kubectl exec -n deepalpha-production postgres-1 -- \
  pg_ctl promote -D /var/lib/postgresql/data

# 4. 更新Service指向新主库
echo "[4] 更新Service..."
kubectl patch svc postgres -n deepalpha-production -p '{"spec":{"selector":{"position":"primary"}}}'

# 5. 验证连接
echo "[5] 验证数据库连接..."
kubectl exec -n deepalpha-production deployment/deepalpha-api -- \
  pg_isready -h postgres -U deepalpha

echo "========================================="
echo "故障转移完成"
echo "========================================="
```

### 25.4 性能调优指南

#### 数据库调优

```sql
-- postgresql-tuning.sql

-- 1. 配置参数调整
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '32MB';
ALTER SYSTEM SET min_wal_size = '1GB';
ALTER SYSTEM SET max_wal_size = '4GB';

-- 重新加载配置
SELECT pg_reload_conf();

-- 2. 创建关键索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_trader_status
ON orders(trader_id, status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_created_at
ON orders(created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_symbol_timestamp
ON trades(symbol, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_trader_symbol
ON positions(trader_id, symbol);

-- 3. 分析表统计信息
ANALYZE orders;
ANALYZE trades;
ANALYZE positions;
ANALYZE traders;

-- 4. 查看配置
SELECT name, setting, unit, context
FROM pg_settings
WHERE name IN (
  'shared_buffers',
  'effective_cache_size',
  'work_mem',
  'maintenance_work_mem'
);
```

#### 应用调优

```python
# config/production.py
from pydantic_settings import BaseSettings

class ProductionSettings(BaseSettings):
    # 数据库连接池优化
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600

    # Redis连接优化
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5

    # API优化
    API_WORKERS: int = 4
    API_MAX_REQUEST_SIZE: int = 10 * 1024 * 1024  # 10MB
    API_TIMEOUT: int = 30

    # 缓存优化
    CACHE_TTL: int = 300  # 5分钟
    CACHE_MAX_SIZE: int = 10000

    # LLM调用优化
    LLM_TIMEOUT: int = 30
    LLM_MAX_RETRIES: int = 3
    LLM_RATE_LIMIT: int = 100  # 每分钟

    class Config:
        env_file = ".env.production"

settings = ProductionSettings()
```

---

## 总结

本补充文档涵盖DeepAlpha交易系统的生产部署和运维的详细内容：

**第21章 - CI/CD流水线**
- GitHub Actions完整工作流
- 代码质量检查、测试、安全扫描
- Docker多阶段构建
- 自动化部署流程

**第22章 - Kubernetes部署**
- 完整的K8s资源配置
- 滚动更新、蓝绿部署、金丝雀发布
- HPA自动扩缩容
- 网络策略和安全配置

**第23章 - 数据迁移与备份**
- NOFX Go到Python的数据迁移
- 增量数据同步
- 完整的备份恢复脚本

**第24章 - 前端高级模式**
- Zustand状态管理
- 自定义Hooks
- 性能优化组件
- 虚拟列表、无限滚动

**第25章 - 生产运维**
- 日常检查清单
- 故障排查流程
- 应急响应程序
- 性能调优指南

---

*本文档持续更新中...*

*最后更新: 2026-01-05*

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"activeForm": "Creating CI/CD pipeline configuration documentation", "content": "Create Chapter 21: CI/CD Pipeline Configuration", "status": "completed"}, {"activeForm": "Creating Docker & Kubernetes deployment documentation", "content": "Create Chapter 22: Docker & Kubernetes Deployment", "status": "completed"}, {"activeForm": "Creating data migration & backup documentation", "content": "Create Chapter 23: Data Migration & Backup Strategies", "status": "completed"}, {"activeForm": "Creating advanced frontend patterns documentation", "content": "Create Chapter 24: Advanced Frontend Patterns & State Management", "status": "in_progress"}, {"activeForm": "Creating production runbooks documentation", "content": "Create Chapter 25: Production Runbooks & Operational Procedures", "status": "pending"}]