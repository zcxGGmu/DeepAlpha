"""Alembic环境配置"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from deepalpha.db.models import Base
from deepalpha.config import config

# Alembic配置对象
alembic_config = context.config

# 设置数据库URL
alembic_config.set_main_option(
    "sqlalchemy.url",
    config.database.get_url()
)

# 解释配置文件的日志记录
if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

# 添加模型的MetaData对象以支持'autogenerate'
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """在'离线'模式下运行迁移

    这将上下文配置为仅使用URL而不是引擎，
    尽管在这里也接受引擎。通过跳过
    DDL事件处理程序，只需要对迁移脚本进行SQL转储。

    """
    url = alembic_config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在'在线'模式下运行迁移

    这种情况下，我们创建一个引擎并
    将连接与上下文关联。

    """
    connectable = engine_from_config(
        alembic_config.get_section(alembic_config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()