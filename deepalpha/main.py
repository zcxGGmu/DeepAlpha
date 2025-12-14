#!/usr/bin/env python3
"""DeepAlpha 主入口"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import typer
from typing import Optional

from deepalpha.config.loader import ConfigLoader
from deepalpha.config.settings import settings
from deepalpha.utils.logging import setup_logging, get_logger

app = typer.Typer(
    name="deepalpha",
    help="DeepAlpha - AI驱动的多智能体量化交易系统",
    no_args_is_help=True,
)

logger = get_logger(__name__)


@app.command()
def start(
    env: Optional[str] = typer.Option(None, "--env", "-e", help="运行环境"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    debug: bool = typer.Option(False, "--debug", "-d", help="调试模式"),
):
    """启动DeepAlpha交易系统"""
    # 加载配置
    config_loader = ConfigLoader()
    global settings
    settings = config_loader.load(env=env, config_file=config)

    # 设置日志
    log_level = "DEBUG" if debug or settings.app.debug else settings.app.log_level
    setup_logging(
        level=log_level,
        format_type="json" if not settings.app.debug else "text",
        log_file="logs/deepalpha.log",
        enable_rich=settings.app.debug,
    )

    logger.info("启动 DeepAlpha 系统", version=settings.app.version)
    logger.info(f"配置加载完成 - 环境: {env or 'default'}")

    # TODO: 实现主循环
    logger.info("系统启动中...")
    logger.info("按 Ctrl+C 停止系统")

    try:
        # 模拟运行
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("正在停止系统...")
        logger.info("系统已停止")


@app.command()
def version():
    """显示版本信息"""
    from deepalpha import __version__
    typer.echo(f"DeepAlpha version {__version__}")


@app.command()
def config_check(
    env: Optional[str] = typer.Option(None, "--env", "-e", help="运行环境"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
):
    """检查配置"""
    try:
        config_loader = ConfigLoader()
        settings = config_loader.load(env=env, config_file=config)

        typer.echo("✅ 配置检查通过")
        typer.echo(f"应用名称: {settings.app.name}")
        typer.echo(f"版本: {settings.app.version}")
        typer.echo(f"数据库: {settings.database.url}")
        typer.echo(f"交易所: {settings.exchange.name}")
        typer.echo(f"LLM Providers: {len(settings.ai.providers)}")

        for provider in settings.ai.providers:
            typer.echo(f"  - {provider.name}: {provider.model} (权重: {provider.weight})")

    except Exception as e:
        typer.echo(f"❌ 配置检查失败: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def migrate(
    command: str = typer.Argument(..., help="迁移命令 (init/upgrade/downgrade/revision)"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="迁移信息"),
    revision: Optional[str] = typer.Option(None, "--revision", "-r", help="版本"),
):
    """数据库迁移"""
    from scripts.migrate import (
        migrate_init,
        migrate_upgrade,
        migrate_downgrade,
        migrate_revision,
    )

    if command == "init":
        migrate_init()
    elif command == "upgrade":
        migrate_upgrade(revision or "head")
    elif command == "downgrade":
        if not revision:
            typer.echo("❌ 降级命令需要指定版本", err=True)
            raise typer.Exit(1)
        migrate_downgrade(revision)
    elif command == "revision":
        if not message:
            typer.echo("❌ 创建迁移需要指定信息", err=True)
            raise typer.Exit(1)
        migrate_revision(message)
    else:
        typer.echo(f"❌ 未知命令: {command}", err=True)
        raise typer.Exit(1)


def main():
    """主函数"""
    app()


if __name__ == "__main__":
    main()