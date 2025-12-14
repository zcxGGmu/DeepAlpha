#!/usr/bin/env python3
"""DeepAlpha运行脚本"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepalpha.config import config
from deepalpha.monitoring import setup_logging, metrics, health_checker
from deepalpha.transport.http.routers import (
    decisions,
    positions,
    freqtrade,
    monitoring,
    websocket,
)
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("DeepAlpha 启动中...")

    # 设置日志
    setup_logging('deepalpha')

    # 注册健康检查
    from deepalpha.monitoring.health import (
        DatabaseHealthCheck,
        MemoryHealthCheck,
        DiskSpaceHealthCheck
    )
    health_checker.register_check(MemoryHealthCheck())
    health_checker.register_check(DiskSpaceHealthCheck())

    # 启动指标收集
    if config.monitoring.metrics_enabled:
        logger.info("启动指标收集...")
        metrics_task = asyncio.create_task(
            metrics.start_collection(config.monitoring.health_check_interval)
        )
    else:
        metrics_task = None

    # 启动服务
    try:
        yield
    finally:
        # 关闭时执行
        logger.info("DeepAlpha 关闭中...")
        if metrics_task:
            metrics_task.cancel()
            try:
                await metrics_task
            except asyncio.CancelledError:
                pass
        logger.info("DeepAlpha 已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="DeepAlpha",
    description="DeepAlpha量化交易系统",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# 添加路由
api_prefix = "/api/live"
app.include_router(decisions.router, prefix=api_prefix, tags=["决策"])
app.include_router(positions.router, prefix=api_prefix, tags=["持仓"])
app.include_router(freqtrade.router, prefix=api_prefix, tags=["Freqtrade"])
app.include_router(monitoring.router, prefix=api_prefix, tags=["监控"])
app.include_router(websocket.router, tags=["WebSocket"])


# 根路径
@app.get("/", include_in_schema=False)
async def root():
    """根路径"""
    return {
        "name": "DeepAlpha",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs",
        "admin": "/admin"
    }


# 健康检查
@app.get("/api/live/status")
async def health_status():
    """健康检查"""
    try:
        results = await health_checker.check_all()
        overall_status = health_checker.get_overall_status(results)

        return {
            "status": overall_status.value,
            "timestamp": results[next(iter(results))].timestamp.isoformat() if results else None,
            "checks": {name: result.to_dict() for name, result in results.items()}
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "message": str(e)
        }


# 管理界面
@app.get("/admin", include_in_schema=False)
async def admin_ui():
    """管理界面"""
    try:
        with open("deepalpha/transport/web/static/index.html", "r") as f:
            from fastapi.responses import HTMLResponse
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return {"error": "管理界面未找到"}


def main():
    """主函数"""
    # 加载配置
    logger.info(f"DeepAlpha 版本: {config.version}")
    logger.info(f"配置文件: {config._config_path}")
    logger.info(f"数据目录: {config.app.data_root}")
    logger.info(f"日志目录: {config.app.log_dir}")

    # 创建必要的目录
    os.makedirs(config.app.data_root, exist_ok=True)
    os.makedirs(config.app.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.app.db_path), exist_ok=True)

    # 配置服务器
    host = config.app.http_addr.replace(':', '').replace('0.0.0.0', '0.0.0.0')
    port = int(config.app.http_addr.split(':')[-1]) if ':' in config.app.http_addr else 9991

    logger.info(f"启动HTTP服务器: {host}:{port}")

    # 启动服务器
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=config.app.log_level.value,
        reload=config.app.debug
    )


if __name__ == "__main__":
    main()