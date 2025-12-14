#!/bin/bash
# Freqtrade Webhook测试脚本
# 对应Go版本的 freqtrade_webhook_test.sh

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
WEBHOOK_URL="${WEBHOOK_URL:-http://localhost:9991/api/live/freqtrade/webhook}"
BASE_URL="${BASE_URL:-http://localhost:9991}"

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

# 测试数据
test_order_fill='{
    "type": "order_fill",
    "timestamp": "2024-01-01T00:00:00Z",
    "exchange": "binance",
    "pair": "BTC/USDT",
    "order_id": "123456",
    "filled": 0.001,
    "price": 45000.0
}'

test_order_cancel='{
    "type": "order_canceled",
    "timestamp": "2024-01-01T00:00:00Z",
    "exchange": "binance",
    "pair": "BTC/USDT",
    "order_id": "123456",
    "reason": "User cancelled"
}'

test_position_close='{
    "type": "position_closed",
    "timestamp": "2024-01-01T00:00:00Z",
    "pair": "BTC/USDT",
    "pnl": 150.0,
    "close_price": 45150.0
}'

# 发送Webhook请求
send_webhook() {
    local test_name="$1"
    local data="$2"

    log_info "测试 $test_name ..."

    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$data" \
        "$WEBHOOK_URL")

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)

    if [ "$http_code" = "200" ]; then
        log_info "$test_name 测试通过 ✓"
        echo "响应: $body"
    else
        log_error "$test_name 测试失败 ✗"
        echo "HTTP状态码: $http_code"
        echo "响应: $body"
        return 1
    fi

    echo "---"
}

# 检查服务是否运行
check_service() {
    log_info "检查服务状态..."

    response=$(curl -s -w "%{http_code}" "$BASE_URL/api/live/status" || echo "000")

    if [ "$response" = "200" ]; then
        log_info "服务运行正常 ✓"
    else
        log_error "服务未运行或无法访问 ✗"
        log_error "请确保DeepAlpha服务已启动"
        exit 1
    fi
}

# 测试健康检查
test_health() {
    log_info "测试健康检查端点..."

    response=$(curl -s "$BASE_URL/api/live/status")

    if echo "$response" | grep -q "status"; then
        log_info "健康检查通过 ✓"
    else
        log_error "健康检查失败 ✗"
    fi
}

# 批量测试
run_all_tests() {
    log_info "开始执行所有Webhook测试..."

    # 检查服务
    check_service
    test_health

    # Webhook测试
    send_webhook "订单成交" "$test_order_fill"
    send_webhook "订单取消" "$test_order_cancel"
    send_webhook "持仓关闭" "$test_position_close"

    # 无效请求测试
    log_info "测试无效请求..."
    invalid_response=$(curl -s -w "%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "{}" \
        "$WEBHOOK_URL")

    if [ "$invalid_response" = "400" ]; then
        log_info "无效请求正确拒绝 ✓"
    else
        log_warn "无效请求处理异常"
    fi

    log_info "所有测试完成!"
}

# 实时监控模式
monitor_mode() {
    log_info "启动Webhook监控模式..."
    log_info "等待Webhook事件..."

    while true; do
        # 检查最近的决策
        response=$(curl -s "$BASE_URL/api/live/decisions?limit=5" 2>/dev/null || echo "")

        if [ -n "$response" ]; then
            echo "最新决策: $(echo "$response" | head -c 200)..."
        fi

        sleep 5
    done
}

# 主函数
main() {
    echo "========================================"
    echo "Freqtrade Webhook 测试工具"
    echo "========================================"
    echo "Webhook URL: $WEBHOOK_URL"
    echo "Base URL: $BASE_URL"
    echo ""

    case "${1:-test}" in
        "test")
            run_all_tests
            ;;
        "monitor")
            monitor_mode
            ;;
        "check")
            check_service
            ;;
        "health")
            test_health
            ;;
        *)
            echo "用法: $0 [test|monitor|check|health]"
            echo ""
            echo "选项:"
            echo "  test    - 运行所有测试（默认）"
            echo "  monitor - 实时监控模式"
            echo "  check   - 检查服务状态"
            echo "  health  - 健康检查"
            exit 1
            ;;
    esac
}

# 错误处理
trap 'log_error "测试失败"; exit 1' ERR

# 执行主函数
main "$@"