#!/usr/bin/env python3
"""数据库迁移脚本"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import subprocess
from typing import Optional

from deepalpha.config.settings import settings
from deepalpha.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def run_alembic_command(command: str, message: Optional[str] = None) -> None:
    """运行alembic命令"""
    cmd = ["alembic", command]

    if message and command == "revision":
        cmd.extend(["-m", message])

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"成功执行命令: {' '.join(cmd)}")
        if result.stdout:
            logger.info(f"输出: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"执行命令失败: {' '.join(cmd)}")
        logger.error(f"错误: {e.stderr}")
        raise


def migrate_init():
    """初始化迁移环境"""
    logger.info("初始化迁移环境...")
    run_alembic_command("stamp", "head")
    logger.info("迁移环境初始化完成")


def migrate_upgrade(revision: str = "head"):
    """升级数据库"""
    logger.info(f"升级数据库到版本: {revision}")
    run_alembic_command("upgrade", revision)
    logger.info("数据库升级完成")


def migrate_downgrade(revision: str):
    """降级数据库"""
    logger.info(f"降级数据库到版本: {revision}")
    run_alembic_command("downgrade", revision)
    logger.info("数据库降级完成")


def migrate_revision(message: str, autogenerate: bool = True):
    """创建新的迁移文件"""
    logger.info(f"创建迁移文件: {message}")

    cmd = ["alembic", "revision", "-m", message]
    if autogenerate:
        cmd.append("--autogenerate")

    subprocess.run(cmd, check=True)
    logger.info("迁移文件创建完成")


def migrate_current():
    """查看当前版本"""
    run_alembic_command("current")


def migrate_history():
    """查看迁移历史"""
    run_alembic_command("history")


def migrate_branches():
    """查看分支"""
    run_alembic_command("branches")


def migrate_heads():
    """查看头部版本"""
    run_alembic_command("heads")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据库迁移工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 初始化命令
    init_parser = subparsers.add_parser("init", help="初始化迁移环境")

    # 升级命令
    upgrade_parser = subparsers.add_parser("upgrade", help="升级数据库")
    upgrade_parser.add_argument(
        "--revision", default="head", help="目标版本 (默认: head)"
    )

    # 降级命令
    downgrade_parser = subparsers.add_parser("downgrade", help="降级数据库")
    downgrade_parser.add_argument(
        "--revision", required=True, help="目标版本"
    )

    # 创建迁移命令
    revision_parser = subparsers.add_parser("revision", help="创建迁移文件")
    revision_parser.add_argument(
        "--message", "-m", required=True, help="迁移信息"
    )
    revision_parser.add_argument(
        "--no-autogenerate", action="store_true", help="不自动生成"
    )

    # 查看命令
    subparsers.add_parser("current", help="查看当前版本")
    subparsers.add_parser("history", help="查看迁移历史")
    subparsers.add_parser("branches", help="查看分支")
    subparsers.add_parser("heads", help="查看头部版本")

    # 解析参数
    args = parser.parse_args()

    # 设置日志
    setup_logging(level="INFO")

    # 执行命令
    if args.command == "init":
        migrate_init()
    elif args.command == "upgrade":
        migrate_upgrade(args.revision)
    elif args.command == "downgrade":
        migrate_downgrade(args.revision)
    elif args.command == "revision":
        migrate_revision(args.message, not args.no_autogenerate)
    elif args.command == "current":
        migrate_current()
    elif args.command == "history":
        migrate_history()
    elif args.command == "branches":
        migrate_branches()
    elif args.command == "heads":
        migrate_heads()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()