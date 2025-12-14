# 安全政策

## 安全报告流程

如果您发现安全漏洞，请遵循以下流程：

1. **请勿** 在公开的 issue 中报告安全漏洞
2. 请发送邮件至：security@brale.example.com
3. 邮件主题应包含：`[Security] 漏洞报告`
4. 请提供详细的信息，包括：
   - 漏洞的类型
   - 漏洞的位置
   - 漏洞的影响
   - 复现步骤
   - 您的建议修复方案（可选）

### 响应时间

- 我们会在收到报告后的 48 小时内回复
- 严重漏洞将在 7 天内修复并发布更新
- 中等漏洞将在 14 天内修复并发布更新
- 低等漏洞将在 30 天内修复并发布更新

### 安全措施

DeepAlpha 采用了以下安全措施：

#### 1. 数据保护
- API 密钥通过环境变量存储，不提交到版本控制
- 支持代理配置，避免直接连接交易所
- 数据库连接使用加密

#### 2. 访问控制
- REST API 支持 JWT 认证
- WebSocket 连接使用 token 验证
- 配置文件包含敏感信息访问控制

#### 3. 交易安全
- 严格的仓位控制（默认每笔交易不超过总资产的 2%）
- 强制止损机制
- 多层风险检查

#### 4. 依赖安全
- 定期更新依赖包
- 使用安全扫描工具（bandit、safety）
- 依赖版本锁定

## 安全最佳实践

### 部署建议

1. **环境变量管理**
   ```bash
   # 使用环境变量存储敏感信息
   export BINANCE_API_KEY="your_api_key"
   export BINANCE_SECRET_KEY="your_secret_key"
   export OPENAI_API_KEY="your_openai_key"
   ```

2. **网络配置**
   ```bash
   # 使用代理（推荐）
   export HTTP_PROXY="http://proxy.example.com:8080"
   export HTTPS_PROXY="http://proxy.example.com:8080"
   ```

3. **Docker 部署**
   ```yaml
   # docker-compose.yml 安全配置
   version: '3.8'
   services:
     deepalpha:
       user: "1000:1000"  # 非root用户
       read_only: true     # 只读文件系统
       no-new-privileges: true
   ```

4. **防火墙配置**
   ```bash
   # 只开放必要的端口
   ufw allow 9991/tcp  # API端口
   ufw allow 8080/tcp   # Freqtrade端口
   ufw deny 3306/tcp    # 数据库端口（仅内部）
   ```

### 运行时安全

1. **使用沙箱模式**
   - 首次使用时启用 dry-run 模式
   - 在测试环境中验证所有配置

2. **监控日志**
   ```bash
   # 监控异常登录尝试
   tail -f running_log/logs/deepalpha.log | grep "auth"

   # 监控异常交易
   tail -f running_log/logs/deepalpha.log | grep "trade"
   ```

3. **定期更新**
   ```bash
   # 更新依赖
   pip install --upgrade -r requirements.txt

   # 安全检查
   make security
   ```

## 已知安全注意事项

### 1. API 密钥管理
- ❌ 不要将 API 密钥硬编码在代码中
- ✅ 使用环境变量或安全的密钥管理服务
- ✅ 定期轮换 API 密钥

### 2. 网络连接
- ❌ 不要在公共网络中运行无加密的服务
- ✅ 使用 HTTPS/WSS 连接
- ✅ 考虑使用 VPN 或代理

### 3. 数据存储
- ❌ 不要在日志中记录敏感信息
- ✅ 加密存储敏感配置
- ✅ 定期备份重要的配置文件

### 4. 权限控制
- ❌ 不要使用 root 权限运行
- ✅ 使用最小权限原则
- ✅ 限制数据库访问权限

## 漏洞分类

### 严重等级定义

#### 🔴 严重 (Critical)
- 远程代码执行
- 权限提升
- 敏感数据泄露（API 密钥等）

#### 🟠 高危 (High)
- 拒绝服务攻击
- 重要功能绕过
- 非授权交易

#### 🟡 中危 (Medium)
- 跨站脚本 (XSS)
- SQL 注入
- 配置信息泄露

#### 🟢 低危 (Low)
- 信息泄露
- 功能限制绕过
- 性能问题

## 安全更新

我们将：

1. 及时响应安全报告
2. 在适当时机发布安全更新
3. 在发布更新前测试修复方案
4. 在更新中感谢贡献者的发现

## 联系方式

- **安全报告邮箱**: security@brale.example.com
- **通用问题**: 使用 GitHub Issues
- **PGP 密钥**: （如需加密通信，请联系获取）

## 致谢

感谢所有帮助改进 DeepAlpha 安全性的研究人员和贡献者。您的努力使我们的产品更加安全可靠。