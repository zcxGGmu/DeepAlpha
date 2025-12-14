"""依赖注入框架
对应Go版本的 Wire 依赖注入系统
"""

from typing import TypeVar, Type, Dict, Any, Callable, Optional, List
from abc import ABC, abstractmethod
from functools import wraps
import inspect

T = TypeVar('T')


class DIContainer:
    """依赖注入容器"""

    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._instances: Dict[Type, Any] = {}
        self._building: set = set()  # 防止循环依赖

    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> 'DIContainer':
        """注册单例服务"""
        self._services[interface] = implementation
        return self

    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> 'DIContainer':
        """注册工厂函数"""
        self._factories[interface] = factory
        return self

    def register_instance(self, interface: Type[T], instance: T) -> 'DIContainer':
        """注册实例"""
        self._instances[interface] = instance
        return self

    def get(self, interface: Type[T]) -> T:
        """获取服务实例"""
        # 检查循环依赖
        if interface in self._building:
            raise ValueError(f"检测到循环依赖: {interface}")

        # 已有实例直接返回
        if interface in self._instances:
            return self._instances[interface]

        # 检查是否是单例
        if interface in self._singletons:
            return self._singletons[interface]

        self._building.add(interface)

        try:
            # 优先使用注册的实例
            if interface in self._instances:
                instance = self._instances[interface]
            # 使用工厂函数
            elif interface in self._factories:
                factory = self._factories[interface]
                # 检查工厂函数的参数依赖
                instance = self._create_with_injection(factory)
            # 使用注册的实现类
            elif interface in self._services:
                implementation = self._services[interface]
                instance = self._create_with_injection(implementation)
            else:
                # 尝试直接实例化
                instance = self._create_with_injection(interface)

            # 标记为单例
            if hasattr(instance, '_singleton') and instance._singleton:
                self._singletons[interface] = instance
            else:
                self._instances[interface] = instance

            return instance

        finally:
            self._building.discard(interface)

    def _create_with_injection(self, cls: Type[T]) -> T:
        """通过依赖注入创建实例"""
        # 获取构造函数签名
        sig = inspect.signature(cls.__init__)

        # 准备参数
        kwargs = {}
        for name, param in sig.parameters.items():
            if name == 'self':
                continue

            # 获取参数类型注解
            param_type = param.annotation

            # 如果有类型注解且不是Any，尝试从容器获取
            if param_type != inspect.Parameter.empty and param_type != Any:
                if param_type in self._services or param_type in self._factories or param_type in self._instances:
                    kwargs[name] = self.get(param_type)
            # 如果有默认值，使用默认值
            elif param.default != inspect.Parameter.empty:
                kwargs[name] = param.default
            # 其他情况抛出异常
            else:
                raise ValueError(f"无法解析依赖: {cls.__name__}.{name}")

        # 创建实例
        return cls(**kwargs)

    def create_scope(self) -> 'DIScope':
        """创建作用域"""
        return DIScope(self)

    def clear(self):
        """清除所有实例（单例除外）"""
        self._instances.clear()
        self._building.clear()


class DIScope:
    """依赖注入作用域"""

    def __init__(self, container: DIContainer):
        self._container = container
        self._scoped_instances: Dict[Type, Any] = {}

    def get(self, interface: Type[T]) -> T:
        """获取作用域内的实例"""
        if interface in self._scoped_instances:
            return self._scoped_instances[interface]

        instance = self._container.get(interface)
        self._scoped_instances[interface] = instance
        return instance

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 清除作用域内的实例
        self._scoped_instances.clear()


# 全局容器
_global_container = DIContainer()


def inject(interface: Optional[Type] = None):
    """依赖注入装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 调用时注入依赖
            return _global_container.get(interface or func.__annotations__.get('return'))
        return wrapper
    return decorator


def singleton(cls: Type[T]) -> Type[T]:
    """单例装饰器"""
    cls._singleton = True
    return cls


def component(interface: Optional[Type] = None):
    """组件装饰器"""
    def decorator(cls: Type[T]) -> Type[T]:
        impl_type = interface or cls
        _global_container.register_singleton(impl_type, cls)
        return cls
    return decorator


def factory(interface: Type[T]):
    """工厂装饰器"""
    def decorator(func: Callable[[], T]) -> Callable[[], T]:
        _global_container.register_factory(interface, func)
        return func
    return decorator


def get_service(interface: Type[T]) -> T:
    """获取服务实例"""
    return _global_container.get(interface)


def register_singleton(interface: Type[T], implementation: Type[T]):
    """注册单例服务"""
    _global_container.register_singleton(interface, implementation)


def register_factory(interface: Type[T], factory_func: Callable[[], T]):
    """注册工厂函数"""
    _global_container.register_factory(interface, factory_func)


def register_instance(interface: Type[T], instance: T):
    """注册实例"""
    _global_container.register_instance(interface, instance)


# 构建器类（对应Go版本的builder.go）
class ApplicationBuilder:
    """应用构建器"""

    def __init__(self):
        self._container = DIContainer()
        self._config = {}

    def add_singleton(self, interface: Type[T], implementation: Type[T]) -> 'ApplicationBuilder':
        """添加单例服务"""
        self._container.register_singleton(interface, implementation)
        return self

    def add_factory(self, interface: Type[T], factory: Callable[[], T]) -> 'ApplicationBuilder':
        """添加工厂函数"""
        self._container.register_factory(interface, factory)
        return self

    def add_instance(self, interface: Type[T], instance: T) -> 'ApplicationBuilder':
        """添加实例"""
        self._container.register_instance(interface, instance)
        return self

    def use_config(self, config: Dict[str, Any]) -> 'ApplicationBuilder':
        """使用配置"""
        self._config.update(config)
        return self

    def build(self) -> DIContainer:
        """构建容器"""
        # 将配置作为实例注册
        register_instance(dict, self._config)
        return self._container


# 生命周期管理
class ServiceLifecycle(ABC):
    """服务生命周期接口"""

    @abstractmethod
    async def initialize(self) -> None:
        """初始化"""
        pass

    @abstractmethod
    async def dispose(self) -> None:
        """销毁"""
        pass


class LifecycleManager:
    """生命周期管理器"""

    def __init__(self, container: DIContainer):
        self._container = container
        self._services: List[ServiceLifecycle] = []

    async def initialize_all(self) -> None:
        """初始化所有服务"""
        for interface, instance in self._container._instances.items():
            if isinstance(instance, ServiceLifecycle):
                await instance.initialize()
                self._services.append(instance)

    async def dispose_all(self) -> None:
        """销毁所有服务"""
        # 按相反顺序销毁
        for service in reversed(self._services):
            await service.dispose()

        self._services.clear()
        self._container.clear()