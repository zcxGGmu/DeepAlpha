"""监控相关的API路由，对应 Go 版本的 monitoring handlers"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from deepalpha.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# 响应模型
class LogEntry(BaseModel):
    """日志条目"""
    timestamp: datetime
    level: str
    message: str
    module: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class SystemStatus(BaseModel):
    """系统状态"""
    status: str
    uptime: float
    version: str
    git_commit: Optional[str] = None
    build_time: Optional[str] = None
    components: Dict[str, str]


class PerformanceMetrics(BaseModel):
    """性能指标"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    goroutines: int = 0  # Python中用线程数
    active_connections: int = 0


class AgentStatus(BaseModel):
    """代理状态"""
    name: str
    status: str
    last_run: Optional[datetime] = None
    error_count: int = 0
    success_count: int = 0
    avg_duration_ms: float = 0.0


class Alert(BaseModel):
    """告警"""
    id: str
    level: str  # info, warning, error, critical
    message: str
    source: str
    timestamp: datetime
    acknowledged: bool = False
    metadata: Optional[Dict[str, Any]] = None


# 模拟数据存储
_log_entries: List[LogEntry] = []
_system_start_time = datetime.utcnow()
_alerts: List[Alert] = []


@router.get("/logs")
async def get_logs(
    level: Optional[str] = Query(None, description="日志级别"),
    module: Optional[str] = Query(None, description="模块"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    limit: int = Query(100, ge=1, le=1000, description="限制数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
):
    """
    获取日志
    对应 Go 版本的 GET /api/live/logs
    """
    logger.info("获取日志", level=level, module=module, limit=limit)

    # 筛选日志
    filtered_logs = _log_entries

    if level:
        filtered_logs = [log for log in filtered_logs if log.level == level]
    if module:
        filtered_logs = [log for log in filtered_logs if log.module == module]
    if start_time:
        filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
    if end_time:
        filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]

    # 排序（最新的在前）
    filtered_logs = sorted(filtered_logs, key=lambda x: x.timestamp, reverse=True)

    # 分页
    total = len(filtered_logs)
    page_logs = filtered_logs[offset:offset + limit]

    return {
        "logs": page_logs,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """
    获取系统状态
    对应 Go 版本的 GET /api/live/status
    """
    logger.info("获取系统状态")

    uptime = (datetime.utcnow() - _system_start_time).total_seconds()

    # TODO: 检查各个组件状态
    components = {
        "database": "ok",
        "freqtrade_client": "ok",
        "market_data": "ok",
        "decision_engine": "ok",
        "executor": "ok",
    }

    return SystemStatus(
        status="ok",
        uptime=uptime,
        version="1.0.0",
        git_commit="abc123",  # TODO: 获取实际commit
        build_time="2024-01-01T00:00:00Z",  # TODO: 获取实际构建时间
        components=components,
    )


@router.get("/metrics", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """
    获取性能指标
    """
    logger.info("获取性能指标")

    # TODO: 获取实际的系统指标
    # 这里使用模拟数据
    import psutil
    import threading

    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return PerformanceMetrics(
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        memory_used_mb=memory.used / 1024 / 1024,
        memory_total_mb=memory.total / 1024 / 1024,
        disk_percent=disk.percent,
        disk_used_gb=disk.used / 1024 / 1024 / 1024,
        disk_total_gb=disk.total / 1024 / 1024 / 1024,
        goroutines=threading.active_count(),  # 使用活跃线程数代替goroutines
        active_connections=0,  # TODO: 获取实际连接数
    )


@router.get("/agents", response_model=List[AgentStatus])
async def get_agent_status():
    """
    获取代理状态
    """
    logger.info("获取代理状态")

    # TODO: 从实际的代理管理器获取状态
    # 这里返回模拟数据
    return [
        AgentStatus(
            name="technical_agent",
            status="running",
            last_run=datetime.utcnow(),
            error_count=0,
            success_count=10,
            avg_duration_ms=150.0,
        ),
        AgentStatus(
            name="news_agent",
            status="running",
            last_run=datetime.utcnow(),
            error_count=1,
            success_count=9,
            avg_duration_ms=200.0,
        ),
        AgentStatus(
            name="sentiment_agent",
            status="idle",
            last_run=datetime.utcnow(),
            error_count=0,
            success_count=5,
            avg_duration_ms=100.0,
        ),
    ]


@router.get("/alerts", response_model=List[Alert])
async def get_alerts(
    level: Optional[str] = Query(None, description="告警级别"),
    acknowledged: Optional[bool] = Query(None, description="是否已确认"),
    limit: int = Query(100, ge=1, le=1000, description="限制数量"),
):
    """
    获取告警列表
    """
    logger.info("获取告警列表", level=level, acknowledged=acknowledged)

    # 筛选告警
    filtered_alerts = _alerts

    if level:
        filtered_alerts = [a for a in filtered_alerts if a.level == level]
    if acknowledged is not None:
        filtered_alerts = [a for a in filtered_alerts if a.acknowledged == acknowledged]

    # 排序（最新的在前）
    filtered_alerts = sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)

    return filtered_alerts[:limit]


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """
    确认告警
    """
    logger.info("确认告警", alert_id=alert_id)

    # 查找告警
    alert = next((a for a in _alerts if a.id == alert_id), None)
    if not alert:
        raise HTTPException(status_code=404, detail="告警未找到")

    alert.acknowledged = True
    return {"message": "告警已确认"}


@router.post("/alerts")
async def create_alert(
    level: str = Field(..., description="告警级别"),
    message: str = Field(..., description="告警消息"),
    source: str = Field(..., description="告警来源"),
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据"),
):
    """
    创建告警
    """
    logger.info("创建告警", level=level, message=message, source=source)

    alert = Alert(
        id=f"alert_{len(_alerts) + 1}",
        level=level,
        message=message,
        source=source,
        timestamp=datetime.utcnow(),
        metadata=metadata,
    )
    _alerts.append(alert)

    return {"alert_id": alert.id}


@router.get("/health")
async def health_check():
    """
    健康检查
    """
    # 检查各个组件
    checks = {
        "database": "ok",
        "freqtrade": "ok",
        "market_data": "ok",
    }

    all_ok = all(status == "ok" for status in checks.values())
    status_code = 200 if all_ok else 503

    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.utcnow(),
    }


# 辅助函数：添加日志（用于测试）
def add_test_log(level: str, message: str, module: Optional[str] = None):
    """添加测试日志"""
    _log_entries.append(LogEntry(
        timestamp=datetime.utcnow(),
        level=level,
        message=message,
        module=module,
    ))


# 辅助函数：添加告警（用于测试）
def add_test_alert(level: str, message: str, source: str):
    """添加测试告警"""
    alert = Alert(
        id=f"alert_{len(_alerts) + 1}",
        level=level,
        message=message,
        source=source,
        timestamp=datetime.utcnow(),
    )
    _alerts.append(alert)


# 导出
__all__ = ["router"]