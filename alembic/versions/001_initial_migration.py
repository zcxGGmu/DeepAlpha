"""Initial migration

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import json

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create klines table
    op.create_table('klines',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('symbol', sa.String(length=32), nullable=False),
        sa.Column('interval', sa.String(length=16), nullable=False),
        sa.Column('open_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('close_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('open', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('high', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('low', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('close', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('volume', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('quote_volume', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('trades', sa.Integer(), nullable=True),
        sa.Column('taker_buy_volume', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('taker_buy_quote_volume', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_klines_symbol_interval', 'klines', ['symbol', 'interval'], unique=False)
    op.create_index('idx_klines_symbol_time', 'klines', ['symbol', 'open_time'], unique=False)
    op.create_index('idx_klines_time', 'klines', ['open_time'], unique=False)

    # Create decisions table
    op.create_table('decisions',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('symbol', sa.String(length=32), nullable=False),
        sa.Column('decision_type', sa.Enum('ENTER_LONG', 'ENTER_SHORT', 'EXIT_LONG', 'EXIT_SHORT', 'HOLD', name='decisiontype'), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('reasoning', sa.Text(), nullable=True),
        sa.Column('signals', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('executed', sa.Boolean(), nullable=True, default=False),
        sa.Column('execution_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('suggested_size', sa.Float(), nullable=True),
        sa.Column('stop_loss', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('take_profit', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_decisions_symbol', 'decisions', ['symbol'], unique=False)
    op.create_index('idx_decisions_created', 'decisions', ['created_at'], unique=False)
    op.create_index('idx_decisions_executed', 'decisions', ['executed'], unique=False)
    op.create_index('idx_decisions_symbol_type', 'decisions', ['symbol', 'decision_type'], unique=False)

    # Create orders table
    op.create_table('orders',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('client_order_id', sa.String(length=64), nullable=True),
        sa.Column('symbol', sa.String(length=32), nullable=False),
        sa.Column('type', sa.Enum('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT', 'TAKE_PROFIT', 'STOP_LOSS', name='ordertype'), nullable=False),
        sa.Column('side', sa.Enum('BUY', 'SELL', name='orderside'), nullable=False),
        sa.Column('amount', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('price', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('stop_price', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'OPEN', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED', name='orderstatus'), nullable=False),
        sa.Column('filled', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('remaining', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('average_price', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('fee', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('fee_currency', sa.String(length=16), nullable=True),
        sa.Column('trades', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('decision_id', sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(['decision_id'], ['decisions.id'], ),
        sa.UniqueConstraint('client_order_id'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_orders_symbol', 'orders', ['symbol'], unique=False)
    op.create_index('idx_orders_status', 'orders', ['status'], unique=False)
    op.create_index('idx_orders_created', 'orders', ['created_at'], unique=False)
    op.create_index('idx_orders_symbol_status', 'orders', ['symbol', 'status'], unique=False)

    # Create positions table
    op.create_table('positions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('symbol', sa.String(length=32), nullable=False),
        sa.Column('side', sa.Enum('BUY', 'SELL', name='orderside'), nullable=False),
        sa.Column('size', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('entry_price', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('current_price', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('unrealized_pnl', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('realized_pnl', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.UniqueConstraint('symbol'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_positions_symbol', 'positions', ['symbol'], unique=False)
    op.create_index('idx_positions_side', 'positions', ['side'], unique=False)

    # Create balances table
    op.create_table('balances',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('asset', sa.String(length=16), nullable=False),
        sa.Column('free', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('used', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('total', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.UniqueConstraint('asset'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_balances_asset', 'balances', ['asset'], unique=False)

    # Create agent_results table
    op.create_table('agent_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('agent_name', sa.String(length=64), nullable=False),
        sa.Column('symbol', sa.String(length=32), nullable=False),
        sa.Column('signal_type', sa.Enum('BUY', 'SELL', 'HOLD', name='signaltype'), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('price', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('timeframe', sa.String(length=16), nullable=True),
        sa.Column('analysis', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('processing_time', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_agent_results_symbol', 'agent_results', ['symbol'], unique=False)
    op.create_index('idx_agent_results_agent', 'agent_results', ['agent_name'], unique=False)
    op.create_index('idx_agent_results_created', 'agent_results', ['created_at'], unique=False)

    # Create trades table
    op.create_table('trades',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('order_id', sa.String(length=64), nullable=False),
        sa.Column('symbol', sa.String(length=32), nullable=False),
        sa.Column('side', sa.Enum('BUY', 'SELL', name='orderside'), nullable=False),
        sa.Column('amount', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('price', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('fee', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('fee_currency', sa.String(length=16), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_trades_order', 'trades', ['order_id'], unique=False)
    op.create_index('idx_trades_symbol', 'trades', ['symbol'], unique=False)
    op.create_index('idx_trades_timestamp', 'trades', ['timestamp'], unique=False)


def downgrade() -> None:
    op.drop_table('trades')
    op.drop_table('agent_results')
    op.drop_table('balances')
    op.drop_table('positions')
    op.drop_table('orders')
    op.drop_table('decisions')
    op.drop_table('klines')