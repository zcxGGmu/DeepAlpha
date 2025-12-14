#!/bin/bash
# DeepAlpha快速部署脚本
# 对应 Go 版本的 quickstart.sh

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统要求
check_requirements() {
    log_info "检查系统要求..."

    # 检查Python版本
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 未安装"
        exit 1
    fi

    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    log_info "Python版本: $python_version"

    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        log_error "需要Python 3.8或更高版本"
        exit 1
    fi

    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_warn "Docker未安装，将跳过Docker相关部署"
        SKIP_DOCKER=true
    else
        log_info "Docker版本: $(docker --version)"
    fi

    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_warn "Docker Compose未安装，将跳过Docker部署"
        SKIP_DOCKER=true
    fi
}

# 安装依赖
install_dependencies() {
    log_info "安装Python依赖..."

    # 创建虚拟环境（可选）
    if [ "$USE_VENV" = "true" ]; then
        if [ ! -d "venv" ]; then
            log_info "创建Python虚拟环境..."
            python3 -m venv venv
        fi
        source venv/bin/activate
    fi

    # 升级pip
    pip install --upgrade pip

    # 安装依赖
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi

    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
    fi

    # 安装toml（配置解析需要）
    pip install toml
}

# 创建目录结构
create_directories() {
    log_info "创建目录结构..."

    # 创建数据目录
    mkdir -p data/logs
    mkdir -p data/db
    mkdir -p data/configs

    # 创建Freqtrade目录
    mkdir -p freqtrade/user_data/strategies
    mkdir -p freqtrade/user_data/configs

    # 复制配置文件
    if [ ! -f "configs/config.toml" ]; then
        log_info "创建默认配置文件..."
        cp configs/config.toml configs/config.toml.default
    fi

    # 设置权限
    chmod -R 755 data
    chmod -R 755 freqtrade
}

# 配置环境变量
setup_environment() {
    log_info "配置环境变量..."

    # 创建.env文件
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# DeepAlpha环境配置

# 基础配置
DEEPALPHA_APP__LOG_LEVEL=info
DEEPALPHA_APP__DEBUG=false

# 数据库配置
DEEPALPHA_DATABASE__TYPE=sqlite
DEEPALPHA_DATABASE__SQLITE_PATH=/data/db/deepalpha.db

# Freqtrade配置
FREQTRADE_API_URL=http://localhost:8080/api/v1
FREQTRADE_USERDATA_ROOT=./freqtrade/user_data

# AI配置（可选）
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 代理配置（可选）
# HTTP_PROXY=http://proxy.example.com:8080
# HTTPS_PROXY=http://proxy.example.com:8080
EOF
        log_info "创建了.env文件，请根据需要修改配置"
    fi
}

# Docker部署
deploy_docker() {
    if [ "$SKIP_DOCKER" = "true" ]; then
        log_warn "跳过Docker部署"
        return
    fi

    log_info "使用Docker部署..."

    # 构建镜像
    log_info "构建DeepAlpha镜像..."
    docker build -t deepalpha:latest .

    # 启动服务
    log_info "启动服务..."
    docker-compose up -d

    log_info "Docker部署完成"
    log_info "访问地址："
    log_info "  - Web界面: http://localhost:9991/admin"
    log_info "  - API文档: http://localhost:9991/api/docs"
    log_info "  - Freqtrade: http://localhost:8080"
}

# 本地部署
deploy_local() {
    log_info "本地部署..."

    # 检查配置
    python3 -c "
from deepalpha.config import config
print('配置加载成功')
print(f'数据目录: {config.app.data_root}')
print(f'日志目录: {config.app.log_dir}')
"

    log_info "本地部署完成"
    log_info "启动命令："
    log_info "  python run.py"
    log_info "  或使用: make run"
}

# 运行测试
run_tests() {
    log_info "运行测试..."

    if [ -f "run_tests.py" ]; then
        python run_tests.py --type quick
    else
        python -m pytest tests/ -v
    fi
}

# 主函数
main() {
    echo "========================================"
    echo "DeepAlpha 快速部署脚本"
    echo "========================================"

    # 解析参数
    SKIP_DOCKER=false
    USE_VENV=false
    SKIP_TESTS=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --venv)
                USE_VENV=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -h|--help)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --skip-docker  跳过Docker部署"
                echo "  --venv         使用Python虚拟环境"
                echo "  --skip-tests   跳过测试"
                echo "  -h, --help     显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                exit 1
                ;;
        esac
    done

    # 执行步骤
    check_requirements
    install_dependencies
    create_directories
    setup_environment

    # 运行测试（可选）
    if [ "$SKIP_TESTS" != "true" ]; then
        read -p "是否运行测试？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_tests
        fi
    fi

    # 选择部署方式
    echo ""
    echo "请选择部署方式："
    echo "1) Docker部署（推荐）"
    echo "2) 本地部署"
    echo ""
    read -p "请输入选择 (1/2): " -n 1 -r
    echo

    case $REPLY in
        1)
            deploy_docker
            ;;
        2)
            deploy_local
            ;;
        *)
            log_error "无效选择"
            exit 1
            ;;
    esac

    echo ""
    echo "========================================"
    echo "部署完成！"
    echo "========================================"
}

# 错误处理
trap 'log_error "部署失败，请检查错误信息"; exit 1' ERR

# 执行主函数
main "$@"