# DeepAlpha 部署架构

## 生产环境部署架构

```mermaid
graph TB
    subgraph "Load Balancer Layer"
        LB[Nginx/HAProxy<br/>SSL Termination]
    end

    subgraph "Application Layer - Kubernetes Cluster"
        subgraph "DeepAlpha Pods"
            DA1[DeepAlpha API Pod 1]
            DA2[DeepAlpha API Pod 2]
            DA3[DeepAlpha API Pod 3]
        end

        subgraph "Executor Pods"
            EX1[Executor Pod 1<br/>Rust Engine]
            EX2[Executor Pod 2<br/>Rust Engine]
        end

        subgraph "WebSocket Pods"
            WS1[WebSocket Pod 1<br/>10K Connections]
            WS2[WebSocket Pod 2<br/>10K Connections]
        end

        subgraph "Freqtrade Pods"
            FT1[Freqtrade Pod 1]
            FT2[Freqtrade Pod 2]
        end
    end

    subgraph "Data Layer"
        subgraph "PostgreSQL Cluster"
            PG1[Primary DB]
            PG2[Replica 1]
            PG3[Replica 2]
        end

        subgraph "Redis Cluster"
            RD1[Redis Master]
            RD2[Redis Slave 1]
            RD3[Redis Slave 2]
        end

        subgraph "InfluxDB Cluster"
            ID1[Primary InfluxDB]
            ID2[Secondary InfluxDB]
        end

        subgraph "Object Storage"
            S3[S3/MinIO<br/>Backups & Files]
        end
    end

    subgraph "Monitoring Stack"
        PROM[Prometheus]
        GRAF[Grafana]
        JAE[Jaeger Tracing]
        ELK[ELK Stack]
    end

    subgraph "External Services"
        BIN[Binance API]
        NEW[News APIs]
        LLM[LLM Providers]
        TELE[Telegram Bot]
    end

    %% Connections
    LB --> DA1
    LB --> DA2
    LB --> DA3
    LB --> WS1
    LB --> WS2

    DA1 --> PG1
    DA2 --> PG1
    DA3 --> PG1
    DA1 --> RD1
    DA2 --> RD1
    DA3 --> RD1
    DA1 --> ID1
    DA2 --> ID1
    DA3 --> ID1

    EX1 --> PG1
    EX1 --> RD1
    EX2 --> PG1
    EX2 --> RD1

    WS1 --> RD1
    WS2 --> RD1

    FT1 --> PG1
    FT2 --> PG1

    PG1 --> PG2
    PG1 --> PG3
    RD1 --> RD2
    RD1 --> RD3
    ID1 --> ID2

    S3 --> DA1
    S3 --> DA2
    S3 --> DA3

    PROM --> DA1
    PROM --> DA2
    PROM --> DA3
    PROM --> EX1
    PROM --> EX2
    GRAF --> PROM
    JAE --> DA1
    JAE --> DA2
    JAE --> DA3
    ELK --> DA1
    ELK --> DA2
    ELK --> DA3

    DA1 --> BIN
    DA1 --> NEW
    DA1 --> LLM
    DA1 --> TELE

    %% Styling
    classDef k8s fill:#326ce5,stroke:#fff,color:#fff
    classDef data fill:#3c763d,stroke:#fff,color:#fff
    classDef monitor fill:#e6522c,stroke:#fff,color:#fff
    classDef external fill:#f0ad4e,stroke:#000,color:#000

    class DA1,DA2,DA3,EX1,EX2,WS1,WS2,FT1,FT2 k8s
    class PG1,PG2,PG3,RD1,RD2,RD3,ID1,ID2,S3 data
    class PROM,GRAF,JAE,ELK monitor
    class BIN,NEW,LLM,TELE external
```

## 开发环境架构

```mermaid
graph TB
    subgraph "Developer Machine"
        DEV[Development Environment]
        IDE[IDE/VSCode]
        Docker[Docker Desktop]
    end

    subgraph "Local Containers"
        DA[DeepAlpha Container]
        PG[PostgreSQL Container]
        RD[Redis Container]
        FT[Freqtrade Container]
        WS[WebSocket Container]
    end

    subgraph "Test Services"
        TEST_Bin[Binance Testnet]
        TEST_LLM[LLM Test APIs]
    end

    %% Connections
    IDE --> DEV
    DEV --> Docker
    Docker --> DA
    Docker --> PG
    Docker --> RD
    Docker --> FT
    Docker --> WS

    DA --> PG
    DA --> RD
    DA --> TEST_Bin
    DA --> TEST_LLM
    FT --> DA

    classDef local fill:#5cb85c,stroke:#fff,color:#fff
    classDev test fill:#5bc0de,stroke:#fff,color:#fff

    class DEV,IDE,Docker,DA,PG,RD,FT,WS local
    class TEST_Bin,TEST_LLM test
```

## 容器化配置

### Docker Compose 生产配置

```yaml
version: '3.8'

services:
  # DeepAlpha API服务
  deepalpha-api:
    image: deepalpha:latest
    replicas: 3
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres-primary:5432/deepalpha
      - REDIS_URL=redis://redis-master:6379
      - INFLUXDB_URL=http://influxdb-primary:8086
    ports:
      - "8000:8000"
    depends_on:
      - postgres-primary
      - redis-master
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  # WebSocket服务
  websocket:
    image: deepalpha:latest
    command: python -m deepalpha.transport.websocket
    replicas: 2
    ports:
      - "8001:8001"
    environment:
      - REDIS_URL=redis://redis-master:6379
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G

  # 执行引擎
  executor:
    image: deepalpha:latest
    command: python -m deepalpha.executor.rust_engine
    replicas: 2
    privileged: true  # For Rust performance
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres-primary:5432/deepalpha
      - REDIS_URL=redis://redis-master:6379
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

  # Freqtrade
  freqtrade:
    image: freqtradeorg/freqtrade:stable
    volumes:
      - ./freqtrade/user_data:/freqtrade/user_data
      - ./configs:/freqtrade/configs
    command: trade --config /freqtrade/configs/freqtrade_config.json
    depends_on:
      - deepalpha-api

  # PostgreSQL集群
  postgres-primary:
    image: postgres:15
    environment:
      - POSTGRES_DB=deepalpha
      - POSTGRES_USER=deepalpha
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  postgres-replica:
    image: postgres:15
    environment:
      - POSTGRES_MASTER_SERVICE=postgres-primary
      - POSTGRES_USER=deepalpha
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_replica_data:/var/lib/postgresql/data
    depends_on:
      - postgres-primary

  # Redis集群
  redis-master:
    image: redis:7-alpine
    command: redis-server --appendonly yes --replica-read-only no
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G

  redis-slave:
    image: redis:7-alpine
    command: redis-server --slaveof redis-master 6379
    depends_on:
      - redis-master

  # InfluxDB
  influxdb:
    image: influxdb:2.0
    environment:
      - INFLUXDB_DB=deepalpha
      - INFLUXDB_ADMIN_USER=admin
      - INFLUXDB_ADMIN_PASSWORD=${INFLUX_PASSWORD}
    volumes:
      - influxdb_data:/var/lib/influxdb2
    ports:
      - "8086:8086"

  # 监控服务
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards

  # Jaeger链路追踪
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"

  # Nginx负载均衡
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - deepalpha-api
      - websocket

volumes:
  postgres_data:
  postgres_replica_data:
  redis_data:
  influxdb_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
```

## Kubernetes部署清单

### Namespace和RBAC

```yaml
---
apiVersion: v1
kind: Namespace
metadata:
  name: deepalpha

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: deepalpha
  namespace: deepalpha

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: deepalpha
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: deepalpha
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: deepalpha
subjects:
- kind: ServiceAccount
  name: deepalpha
  namespace: deepalpha
```

### DeepAlpha应用部署

```yaml
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepalpha-api
  namespace: deepalpha
  labels:
    app: deepalpha-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deepalpha-api
  template:
    metadata:
      labels:
        app: deepalpha-api
    spec:
      serviceAccountName: deepalpha
      containers:
      - name: deepalpha
        image: deepalpha:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: deepalpha-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: deepalpha-api-service
  namespace: deepalpha
spec:
  selector:
    app: deepalpha-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 监控配置

### Prometheus配置

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'deepalpha-api'
    static_configs:
      - targets: ['deepalpha-api-service:80']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'deepalpha-executor'
    static_configs:
      - targets: ['deepalpha-executor-service:80']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### Grafana仪表板

```json
{
  "dashboard": {
    "id": null,
    "title": "DeepAlpha System Overview",
    "tags": ["deepalpha", "trading"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{service}}"
          }
        ]
      },
      {
        "id": 2,
        "title": "Order Execution Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(order_latency_seconds_bucket[5m])) by (le))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "WebSocket Connections",
        "type": "singlestat",
        "targets": [
          {
            "expr": "websocket_connections_current",
            "legendFormat": "Active Connections"
          }
        ]
      },
      {
        "id": 4,
        "title": "Portfolio PnL",
        "type": "graph",
        "targets": [
          {
            "expr": "portfolio_pnl_total",
            "legendFormat": "Total PnL"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

## 部署流程

### 1. 基础设施准备

```bash
# 创建Kubernetes集群（AWS EKS示例）
eksctl create cluster \
  --name deepalpha \
  --version 1.27 \
  --region us-west-2 \
  --nodegroup-name deepalpha-nodes \
  --node-type m5.2xlarge \
  --nodes 3

# 安装必要组件
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack
```

### 2. 应用部署

```bash
# 创建命名空间和RBAC
kubectl apply -f k8s/namespace.yaml

# 部署数据库
kubectl apply -f k8s/postgres/
kubectl apply -f k8s/redis/

# 部署DeepAlpha应用
kubectl apply -f k8s/deepalpha/

# 配置Ingress
kubectl apply -f k8s/ingress/
```

### 3. 监控部署

```bash
# 部署Prometheus和Grafana
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# 导入DeepAlpha仪表板
kubectl create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards/ \
  --namespace=monitoring
```

## 性能调优

### 1. Rust性能模块优化

- **CPU亲和性**: 绑定Rust进程到特定CPU核心
- **内存大页**: 使用huge pages减少TLB miss
- **NUMA优化**: 跨NUMA节点的内存分配策略
- **编译优化**: `target-cpu=native` 和 `lto=fat`

### 2. 数据库优化

```sql
-- PostgreSQL配置优化
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- 创建分区表
CREATE TABLE trades (
    id BIGSERIAL,
    symbol VARCHAR(20),
    price DECIMAL(18,8),
    quantity DECIMAL(18,8),
    timestamp TIMESTAMP,
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- 创建月度分区
CREATE TABLE trades_2024_01 PARTITION OF trades
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### 3. 网络优化

- **TCP优化**: 调整缓冲区大小和拥塞控制算法
- **DPDK**: 用户态网络栈减少延迟
- **SR-IOV**: 网卡虚拟化优化
- **eBPF**: 内核级网络处理优化

## 安全配置

### 1. 网络安全

```yaml
# NetworkPolicy示例
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deepalpha-network-policy
  namespace: deepalpha
spec:
  podSelector:
    matchLabels:
      app: deepalpha-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
    - protocol: TCP
      port: 443   # External APIs
```

### 2. Pod安全策略

```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: deepalpha-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## 灾难恢复

### 1. 数据备份策略

```bash
#!/bin/bash
# 每日备份脚本
DATE=$(date +%Y%m%d)

# PostgreSQL备份
pg_dump -h postgres-primary -U deepalpha deepalpha > /backup/postgres_${DATE}.sql

# InfluxDB备份
influxd backup -portable /backup/influxdb_${DATE}

# 上传到S3
aws s3 cp /backup/ s3://deepalpha-backups/${DATE}/ --recursive

# 清理本地文件
find /backup -type f -mtime +7 -delete
```

### 2. 高可用配置

- **多可用区部署**: 跨AZ的Pod分布
- **自动故障转移**: Pod健康检查和自动重启
- **数据同步**: PostgreSQL流复制，Redis集群模式
- **负载均衡**: 多实例负载分发

这个部署架构确保了DeepAlpha系统的高可用性、可扩展性和高性能，能够满足生产环境的严格要求。