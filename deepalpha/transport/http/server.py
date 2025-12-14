"""HTTP服务器，对应 Go 版本的 server.go"""

import logging
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StaticFiles
from fastapi.staticfiles import StaticFiles
import uvicorn

from deepalpha.transport.http.routers import (
    decisions,
    positions,
    freqtrade,
    monitoring,
    websocket,
)
from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)


class HTTPServer:
    """HTTP服务器，对应 Go 版本的 Server"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9991,
        debug: bool = False,
        cors_origins: Optional[list] = None,
    ):
        self.host = host
        self.port = port
        self.debug = debug

        # 创建 FastAPI 应用
        self.app = self._create_app(cors_origins or ["*"])

        # 配置路由
        self._setup_routes()

        # 服务器实例
        self._server: Optional[uvicorn.Server] = None

    def _create_app(self, cors_origins: list) -> FastAPI:
        """创建FastAPI应用"""
        app = FastAPI(
            title="DeepAlpha API",
            description="DeepAlpha交易系统API",
            version="1.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc",
        )

        # 配置CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 请求日志中间件
        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = logger.timer()

            # 记录请求
            logger.info(
                "HTTP请求",
                method=request.method,
                url=str(request.url),
                client=request.client.host if request.client else None
            )

            # 处理请求
            response = await call_next(request)

            # 记录响应
            duration = logger.timer() - start_time
            logger.info(
                "HTTP响应",
                status_code=response.status_code,
                duration_ms=duration * 1000
            )

            return response

        # 健康检查端点
        @app.get("/healthz")
        async def health_check():
            """健康检查"""
            return {"status": "ok", "service": "deepalpha-http"}

        # 根路径
        @app.get("/", response_class=HTMLResponse)
        async def root():
            """根路径返回简单的HTML"""
            return """
            <html>
                <head>
                    <title>DeepAlpha</title>
                </head>
                <body>
                    <h1>DeepAlpha交易系统</h1>
                    <p><a href="/api/docs">API文档</a></p>
                    <p><a href="/admin">管理界面</a></p>
                </body>
            </html>
            """

        return app

    def _setup_routes(self):
        """设置路由"""
        # API路由
        api_prefix = "/api/live"

        self.app.include_router(
            decisions.router,
            prefix=api_prefix,
            tags=["决策"],
        )

        self.app.include_router(
            positions.router,
            prefix=api_prefix,
            tags=["持仓"],
        )

        self.app.include_router(
            freqtrade.router,
            prefix=api_prefix,
            tags=["Freqtrade"],
        )

        self.app.include_router(
            monitoring.router,
            prefix=api_prefix,
            tags=["监控"],
        )

        # WebSocket路由
        self.app.include_router(
            websocket.router,
            tags=["WebSocket"],
        )

        # 管理界面路由
        @app.get("/admin", response_class=HTMLResponse)
        async def admin_ui():
            """管理界面"""
            try:
                with open("deepalpha/transport/web/static/index.html", "r") as f:
                    return f.read()
            except FileNotFoundError:
                return "<h1>管理界面未找到</h1>", 404

        # 静态文件路由（如果存在）
        try:
            self.app.mount(
                "/static",
                StaticFiles(directory="deepalpha/transport/web/static"),
                name="static"
            )
        except RuntimeError:
            # 静态文件目录不存在
            logger.warning("静态文件目录不存在，跳过挂载")

    async def start(self):
        """启动服务器"""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="debug" if self.debug else "info",
            access_log=True,
        )

        self._server = uvicorn.Server(config)

        logger.info(f"启动HTTP服务器", host=self.host, port=self.port)
        await self._server.serve()

    def stop(self):
        """停止服务器"""
        if self._server:
            logger.info("停止HTTP服务器")
            self._server.should_exit = True

    def run(self):
        """运行服务器（同步）"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="debug" if self.debug else "info",
        )


# 创建全局应用实例
app = HTTPServer().app

# 导出
__all__ = ["HTTPServer", "app"]