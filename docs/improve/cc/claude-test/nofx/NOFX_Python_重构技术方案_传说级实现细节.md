# NOFX Python 重构技术方案 - 传说级实现细节

## 文档说明

本文档是NOFX交易系统从Go重构到Python的**传说级**实现细节指南，涵盖了最前沿的AI/ML技术和金融工程实现。

**适用场景：**
- A股（中国内地股票市场）
- 港股（香港股票市场）
- 加密货币市场

**技术难度：** ⭐⭐⭐⭐⭐（传说级）

---

## 第51章 神经架构搜索（NAS）

### 51.1 概述

神经架构搜索（Neural Architecture Search, NAS）自动搜索最优神经网络架构，用于交易预测和策略生成。

**核心特性：**
- 自动架构搜索
- 多目标优化
- 分布式搜索
- 架构压缩
- 可迁移学习

### 51.2 搜索空间定义

```python
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import logging
import json

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """层类型"""
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    ATTENTION = "attention"
    DENSE = "dense"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    MAX_POOL = "max_pool"
    AVG_POOL = "avg_pool"
    SKIP_CONNECTION = "skip_connection"


class ActivationType(Enum):
    """激活函数类型"""
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    MISH = "mish"
    SELU = "selu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"


@dataclass
class LayerConfig:
    """层配置"""
    layer_type: LayerType
    activation: Optional[ActivationType] = None
    units: Optional[int] = None
    filters: Optional[int] = None
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    padding: Optional[str] = None
    dropout_rate: Optional[float] = None
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    num_heads: Optional[int] = None  # for attention/transformer
    d_model: Optional[int] = None  # for transformer
    ff_dim: Optional[int] = None  # feedforward dimension
    return_sequences: bool = True


@dataclass
class ArchitectureConfig:
    """架构配置"""
    layers: List[LayerConfig]
    input_shape: Tuple[int, ...]
    output_dim: int
    optimizer: str = "adam"
    learning_rate: float = 0.001
    loss_function: str = "mse"
    metrics: List[str] = field(default_factory=lambda: ["mae", "mse"])
    regularization: float = 0.0
    architecture_id: str = ""


class SearchSpace:
    """
    搜索空间
    """

    def __init__(self, max_layers: int = 20):
        """
        初始化搜索空间

        Args:
            max_layers: 最大层数
        """
        self.max_layers = max_layers
        self.valid_layer_types = [
            LayerType.CONV1D,
            LayerType.LSTM,
            LayerType.GRU,
            LayerType.TRANSFORMER,
            LayerType.ATTENTION,
            LayerType.DENSE,
        ]
        self.valid_activations = [
            ActivationType.RELU,
            ActivationType.GELU,
            ActivationType.SWISH,
            ActivationType.MISH,
        ]
        self.valid_kernel_sizes = [1, 3, 5, 7]
        self.valid_filters = [32, 64, 128, 256, 512]
        self.valid_units = [32, 64, 128, 256, 512]
        self.valid_num_heads = [2, 4, 8, 16]
        self.valid_d_models = [64, 128, 256, 512]

    def sample_layer_config(self, layer_idx: int) -> LayerConfig:
        """
        采样层配置

        Args:
            layer_idx: 层索引

        Returns:
            层配置
        """
        import random

        # 随机选择层类型
        layer_type = random.choice(self.valid_layer_types)

        # 随机选择激活函数
        activation = random.choice(self.valid_activations)

        config = LayerConfig(
            layer_type=layer_type,
            activation=activation
        )

        # 根据层类型设置特定参数
        if layer_type in [LayerType.CONV1D]:
            config.filters = random.choice(self.valid_filters)
            config.kernel_size = random.choice(self.valid_kernel_sizes)
            config.stride = random.choice([1, 2])
            config.padding = "same" if config.stride == 1 else "valid"

        elif layer_type in [LayerType.LSTM, LayerType.GRU]:
            config.units = random.choice(self.valid_units)

        elif layer_type in [LayerType.TRANSFORMER, LayerType.ATTENTION]:
            config.num_heads = random.choice(self.valid_num_heads)
            config.d_model = random.choice(self.valid_d_models)
            config.ff_dim = config.d_model * 4

        elif layer_type == LayerType.DENSE:
            config.units = random.choice(self.valid_units)

        # 随机添加dropout
        if random.random() < 0.3:
            config.dropout_rate = random.uniform(0.1, 0.5)

        # 随机添加归一化
        if random.random() < 0.5:
            config.use_batch_norm = random.choice([True, False])
            config.use_layer_norm = random.choice([True, False])

        return config

    def sample_architecture(self, input_shape: Tuple[int, ...], output_dim: int) -> ArchitectureConfig:
        """
        采样架构

        Args:
            input_shape: 输入形状
            output_dim: 输出维度

        Returns:
            架构配置
        """
        num_layers = np.random.randint(5, self.max_layers + 1)

        layers = []
        for i in range(num_layers):
            layer_config = self.sample_layer_config(i)
            layers.append(layer_config)

        # 添加输出层
        output_layer = LayerConfig(
            layer_type=LayerType.DENSE,
            activation=ActivationType.SIGMOID,
            units=output_dim,
            return_sequences=False
        )
        layers.append(output_layer)

        architecture_id = f"arch_{hash(str(layers)) % 1000000:06d}"

        return ArchitectureConfig(
            layers=layers,
            input_shape=input_shape,
            output_dim=output_dim,
            architecture_id=architecture_id
        )

    def mutate_architecture(
        self,
        architecture: ArchitectureConfig,
        mutation_rate: float = 0.2
    ) -> ArchitectureConfig:
        """
        变异架构

        Args:
            architecture: 原始架构
            mutation_rate: 变异率

        Returns:
            变异后的架构
        """
        import random
        import copy

        new_layers = copy.deepcopy(architecture.layers)
        num_mutations = int(len(new_layers) * mutation_rate)

        for _ in range(num_mutations):
            layer_idx = random.randint(0, len(new_layers) - 2)  # 不包括输出层

            # 变异操作
            mutation_type = random.choice([
                "change_activation",
                "change_units",
                "add_dropout",
                "remove_dropout",
                "add_layer",
                "remove_layer"
            ])

            if mutation_type == "change_activation":
                new_layers[layer_idx].activation = random.choice(self.valid_activations)

            elif mutation_type == "change_units":
                if new_layers[layer_idx].units:
                    new_layers[layer_idx].units = random.choice(self.valid_units)

            elif mutation_type == "add_dropout":
                if not new_layers[layer_idx].dropout_rate:
                    new_layers[layer_idx].dropout_rate = random.uniform(0.1, 0.5)

            elif mutation_type == "remove_dropout":
                new_layers[layer_idx].dropout_rate = None

            elif mutation_type == "add_layer" and len(new_layers) < self.max_layers:
                new_layer = self.sample_layer_config(layer_idx)
                new_layers.insert(layer_idx, new_layer)

            elif mutation_type == "remove_layer" and len(new_layers) > 3:
                new_layers.pop(layer_idx)

        new_architecture_id = f"arch_{hash(str(new_layers)) % 1000000:06d}"

        return ArchitectureConfig(
            layers=new_layers,
            input_shape=architecture.input_shape,
            output_dim=architecture.output_dim,
            optimizer=architecture.optimizer,
            learning_rate=architecture.learning_rate,
            loss_function=architecture.loss_function,
            metrics=architecture.metrics,
            regularization=architecture.regularization,
            architecture_id=new_architecture_id
        )
```

### 51.3 架构构建器

```python
class ArchitectureBuilder:
    """
    架构构建器
    """

    def __init__(self):
        """初始化架构构建器"""
        self.activation_map = {
            ActivationType.RELU: nn.ReLU(),
            ActivationType.GELU: nn.GELU(),
            ActivationType.SWISH: nn.SiLU(),  # SiLU is Swish
            ActivationType.MISH: nn.Mish(),
            ActivationType.SELU: nn.SELU(),
            ActivationType.TANH: nn.Tanh(),
            ActivationType.SIGMOID: nn.Sigmoid(),
            ActivationType.LEAKY_RELU: nn.LeakyReLU(0.01),
        }

    def build_architecture(self, config: ArchitectureConfig) -> nn.Module:
        """
        构建架构

        Args:
            config: 架构配置

        Returns:
            PyTorch模型
        """
        layers = []

        for layer_config in config.layers:
            layer = self._build_layer(layer_config)
            layers.append(layer)

        model = nn.Sequential(*layers)

        return model

    def _build_layer(self, config: LayerConfig) -> nn.Module:
        """
        构建单个层

        Args:
            config: 层配置

        Returns:
            PyTorch层
        """
        if config.layer_type == LayerType.CONV1D:
            layers = []
            layers.append(nn.Conv1d(
                in_channels=self._infer_in_channels(config),
                out_channels=config.filters or 64,
                kernel_size=config.kernel_size or 3,
                stride=config.stride or 1,
                padding=config.padding or 0
            ))

            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(config.filters or 64))

            if config.activation:
                layers.append(self.activation_map[config.activation])

            if config.dropout_rate:
                layers.append(nn.Dropout(config.dropout_rate))

            return nn.Sequential(*layers)

        elif config.layer_type == LayerType.LSTM:
            return nn.LSTM(
                input_size=self._infer_input_size(config),
                hidden_size=config.units or 128,
                batch_first=True,
                dropout=config.dropout_rate if config.dropout_rate else 0.0
            )

        elif config.layer_type == LayerType.GRU:
            return nn.GRU(
                input_size=self._infer_input_size(config),
                hidden_size=config.units or 128,
                batch_first=True,
                dropout=config.dropout_rate if config.dropout_rate else 0.0
            )

        elif config.layer_type == LayerType.TRANSFORMER:
            # 简化的Transformer层
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model or 256,
                nhead=config.num_heads or 8,
                dim_feedforward=config.ff_dim or 1024,
                dropout=config.dropout_rate or 0.1,
                activation='gelu',
                batch_first=True
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=1)

        elif config.layer_type == LayerType.ATTENTION:
            return nn.MultiheadAttention(
                embed_dim=config.d_model or 256,
                num_heads=config.num_heads or 8,
                dropout=config.dropout_rate or 0.1,
                batch_first=True
            )

        elif config.layer_type == LayerType.DENSE:
            layers = []
            layers.append(nn.Linear(
                self._infer_input_size(config),
                config.units or 128
            ))

            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(config.units or 128))

            if config.activation:
                layers.append(self.activation_map[config.activation])

            if config.dropout_rate:
                layers.append(nn.Dropout(config.dropout_rate))

            return nn.Sequential(*layers)

        else:
            raise ValueError(f"Unknown layer type: {config.layer_type}")

    def _infer_in_channels(self, config: LayerConfig) -> int:
        """推断输入通道数（简化）"""
        return 64  # 默认值

    def _infer_input_size(self, config: LayerConfig) -> int:
        """推断输入大小（简化）"""
        return 128  # 默认值


class TradingModel(nn.Module):
    """
    交易模型基类
    """

    def __init__(self, architecture: nn.Module, config: ArchitectureConfig):
        """
        初始化

        Args:
            architecture: 网络架构
            config: 配置
        """
        super().__init__()
        self.architecture = architecture
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            输出张量
        """
        return self.architecture(x)

    def get_num_parameters(self) -> int:
        """
        获取参数数量

        Returns:
            参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_flops(self) -> int:
        """
        估算FLOPs

        Returns:
            FLOPs数量
        """
        # 简化实现
        return self.get_num_parameters() * 2
```

### 51.4 NAS优化器

```python
from typing import Dict, List, Any, Optional, Callable
import optuna
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import multiprocessing as mp


class NASOptimizer:
    """
    神经架构搜索优化器
    """

    def __init__(
        self,
        search_space: SearchSpace,
        builder: ArchitectureBuilder,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = -1
    ):
        """
        初始化

        Args:
            search_space: 搜索空间
            builder: 架构构建器
            n_trials: 试验次数
            timeout: 超时时间（秒）
            n_jobs: 并行作业数
        """
        self.search_space = search_space
        self.builder = builder
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()

        self.study = None
        self.best_architecture = None
        self.search_history: List[Dict[str, Any]] = []

    def optimize(
        self,
        train_func: Callable,
        train_data: Any,
        val_data: Any,
        directions: List[str] = ["maximize", "minimize"]
    ) -> ArchitectureConfig:
        """
        优化架构

        Args:
            train_func: 训练函数
            train_data: 训练数据
            val_data: 验证数据
            directions: 优化方向

        Returns:
            最优架构
        """
        # 创建多目标优化研究
        self.study = optuna.create_study(
            directions=directions,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10,
                multivariate=True
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )

        # 定义目标函数
        def objective(trial: optuna.Trial) -> float:
            # 采样架构
            architecture = self.search_space.sample_architecture(
                input_shape=train_data[0].shape[1:],
                output_dim=1
            )

            # 设置超参数
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
            weight_decay = trial.suggest_float("weight_decay", 0, 0.1)

            # 构建模型
            model = self.builder.build_architecture(architecture)

            # 训练和评估
            result = train_func(
                model=model,
                train_data=train_data,
                val_data=val_data,
                learning_rate=learning_rate,
                batch_size=batch_size,
                weight_decay=weight_decay
            )

            # 记录历史
            self.search_history.append({
                "trial": trial.number,
                "architecture_id": architecture.architecture_id,
                "val_loss": result["val_loss"],
                "val_accuracy": result["val_accuracy"],
                "num_parameters": result["num_parameters"],
                "training_time": result["training_time"]
            })

            # 返回优化目标
            return result["val_accuracy"]

        # 并行优化
        if self.n_jobs > 1:
            self._parallel_optimize(objective)
        else:
            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout
            )

        # 获取最优架构
        best_trial = self.study.best_trials[0]
        self.best_architecture = self._reconstruct_architecture(best_trial)

        logger.info(f"Best validation accuracy: {best_trial.value:.4f}")
        logger.info(f"Best architecture ID: {self.best_architecture.architecture_id}")

        return self.best_architecture

    def _parallel_optimize(self, objective: Callable):
        """
        并行优化

        Args:
            objective: 目标函数
        """
        # 使用进程池并行执行
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []

            for i in range(self.n_trials):
                future = executor.submit(
                    self._run_single_trial,
                    objective,
                    i
                )
                futures.append(future)

            # 收集结果
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Trial failed: {e}")

    def _run_single_trial(self, objective: Callable, trial_idx: int):
        """
        运行单个试验

        Args:
            objective: 目标函数
            trial_idx: 试验索引
        """
        trial = self.study.ask()

        try:
            value = objective(trial)
            self.study.tell(trial, value)
        except Exception as e:
            self.study.tell(trial, float("-inf"))
            logger.error(f"Trial {trial_idx} failed: {e}")

    def _reconstruct_architecture(self, trial: optuna.Trial) -> ArchitectureConfig:
        """
        重建架构

        Args:
            trial: 试验对象

        Returns:
            架构配置
        """
        # 从试验中重建架构
        # ...（简化实现）
        return self.search_space.sample_architecture(
            input_shape=(100, 10),
            output_dim=1
        )

    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """
        获取帕累托前沿

        Returns:
            帕累托前沿架构列表
        """
        best_trials = self.study.best_trials

        pareto_front = []
        for trial in best_trials:
            pareto_front.append({
                "values": trial.values,
                "params": trial.params,
                "number": trial.number
            })

        return pareto_front

    def visualize_search_history(self) -> Dict[str, Any]:
        """
        可视化搜索历史

        Returns:
            可视化数据
        """
        import matplotlib.pyplot as plt
        import io
        import base64

        # 准确率历史
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 验证准确率
        axes[0, 0].plot(
            [h["trial"] for h in self.search_history],
            [h["val_accuracy"] for h in self.search_history]
        )
        axes[0, 0].set_title("Validation Accuracy")
        axes[0, 0].set_xlabel("Trial")
        axes[0, 0].set_ylabel("Accuracy")

        # 参数数量
        axes[0, 1].plot(
            [h["trial"] for h in self.search_history],
            [h["num_parameters"] for h in self.search_history]
        )
        axes[0, 1].set_title("Number of Parameters")
        axes[0, 1].set_xlabel("Trial")
        axes[0, 1].set_ylabel("Parameters")

        # 训练时间
        axes[1, 0].plot(
            [h["trial"] for h in self.search_history],
            [h["training_time"] for h in self.search_history]
        )
        axes[1, 0].set_title("Training Time")
        axes[1, 0].set_xlabel("Trial")
        axes[1, 0].set_ylabel("Time (s)")

        # 帕累托前沿
        if self.study:
            best_trials = self.study.best_trials
            accuracies = [t.values[0] for t in best_trials]
            losses = [t.values[1] for t in best_trials]
            axes[1, 1].scatter(accuracies, losses, alpha=0.6)
            axes[1, 1].set_title("Pareto Front")
            axes[1, 1].set_xlabel("Accuracy")
            axes[1, 1].set_ylabel("Loss")

        plt.tight_layout()

        # 转换为base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()

        plt.close()

        return {
            "image": img_base64,
            "summary": {
                "total_trials": len(self.search_history),
                "best_accuracy": max(h["val_accuracy"] for h in self.search_history),
                "avg_parameters": np.mean([h["num_parameters"] for h in self.search_history])
            }
        }


class ENASOptimizer(NASOptimizer):
    """
    Efficient Neural Architecture Search优化器
    """

    def __init__(
        self,
        search_space: SearchSpace,
        builder: ArchitectureBuilder,
        n_trials: int = 100,
        share_weights: bool = True
    ):
        """
        初始化

        Args:
            search_space: 搜索空间
            builder: 架构构建器
            n_trials: 试验次数
            share_weights: 是否共享权重
        """
        super().__init__(search_space, builder, n_trials)
        self.share_weights = share_weights
        self.shared_weights: Dict[str, nn.Module] = {}

    def build_supernet(self) -> nn.Module:
        """
        构建超网络

        Returns:
            超网络模型
        """
        # 构建包含所有可能架构的超网络
        # ...（简化实现）
        supernet_layers = nn.ModuleList()

        for layer_type in self.search_space.valid_layer_types:
            # 添加每种类型的层
            if layer_type == LayerType.CONV1D:
                supernet_layers.append(nn.Conv1d(64, 64, 3))
            elif layer_type == LayerType.LSTM:
                supernet_layers.append(nn.LSTM(64, 128, batch_first=True))
            # ... 其他层类型

        return nn.Sequential(*supernet_layers)
```

---

## 第52章 多智能体强化学习

### 52.1 概述

多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）用于协调多个交易智能体协同工作。

**核心特性：**
- 独立学习
- 集中训练分散执行（CTDE）
- 智能体通信
- 协作与竞争
- 信用分配

### 52.2 智能体架构

```python
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import logging

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """智能体类型"""
    MARKET_MAKER = "market_maker"
    TREND_FOLLOWER = "trend_follower"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGEUR = "arbitrageur"
    HEDGER = "hedger"
    SENTIMENT_ANALYST = "sentiment_analyst"
    PORTFOLIO_MANAGER = "portfolio_manager"
    RISK_MANAGER = "risk_manager"


class ActionType(Enum):
    """动作类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_POSITION = "close_position"
    ADJUST_POSITION = "adjust_position"


@dataclass
class AgentObservation:
    """智能体观测"""
    agent_id: str
    timestep: int
    market_features: np.ndarray
    portfolio_state: Dict[str, Any]
    communication_messages: List[Dict[str, Any]]
    opponent_positions: Optional[Dict[str, float]] = None
    global_state: Optional[Dict[str, Any]] = None


@dataclass
class AgentAction:
    """智能体动作"""
    agent_id: str
    action_type: ActionType
    symbol: str
    quantity: float
    price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentReward:
    """智能体奖励"""
    agent_id: str
    reward: float
    reward_components: Dict[str, float]
    timestep: int


class TradingAgent(nn.Module):
    """
    交易智能体基类
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4
    ):
        """
        初始化

        Args:
            agent_id: 智能体ID
            agent_type: 智能体类型
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            learning_rate: 学习率
        """
        super().__init__()

        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 策略网络（Actor）
        self.actor = self._build_actor_network()

        # 价值网络（Critic）
        self.critic = self._build_critic_network()

        # 优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=learning_rate
        )

        # 经验回放
        self.replay_buffer = deque(maxlen=100000)

        # 训练统计
        self.training_stats = {
            "total_episodes": 0,
            "total_steps": 0,
            "total_rewards": [],
            "losses": []
        }

    def _build_actor_network(self) -> nn.Module:
        """
        构建Actor网络

        Returns:
            Actor网络
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Softmax(dim=-1)
        )

    def _build_critic_network(self) -> nn.Module:
        """
        构建Critic网络

        Returns:
            Critic网络
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        epsilon: float = 0.1
    ) -> int:
        """
        选择动作

        Args:
            state: 状态
            explore: 是否探索
            epsilon: 探索率

        Returns:
            动作索引
        """
        if explore and np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.actor(state_tensor)
            action = action_probs.argmax(dim=-1).item()

        return action

    def compute_advantage(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> np.ndarray:
        """
        计算优势函数（GAE）

        Args:
            rewards: 奖励列表
            values: 价值列表
            next_value: 下一个价值
            gamma: 折扣因子
            gae_lambda: GAE参数

        Returns:
            优势数组
        """
        advantages = []
        gae = 0

        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)

        return np.array(advantages)

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        gamma: float = 0.99
    ) -> Dict[str, float]:
        """
        更新网络

        Args:
            states: 状态
            actions: 动作
            rewards: 奖励
            next_states: 下一个状态
            dones: 是否结束
            gamma: 折扣因子

        Returns:
            损失字典
        """
        # 转换为张量
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)

        # 计算TD目标
        with torch.no_grad():
            next_values = self.critic(next_states_tensor).squeeze()
            targets = rewards_tensor + gamma * next_values * (1 - dones_tensor)

        # 更新Critic
        values = self.critic(states_tensor).squeeze()
        critic_loss = nn.MSELoss()(values, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 更新Actor
        action_probs = self.actor(states_tensor)
        log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze())

        # 计算优势
        advantages = (targets - values).detach()

        actor_loss = -(log_probs * advantages).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 记录统计
        self.training_stats["losses"].append({
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()
        })

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "mean_value": values.mean().item()
        }

    def save(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径
        """
        torch.save({
            'agent_id': self.agent_id,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)

    def load(self, path: str):
        """
        加载模型

        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
```

### 52.3 多智能体环境

```python
class MultiAgentTradingEnvironment:
    """
    多智能体交易环境
    """

    def __init__(
        self,
        agents: List[TradingAgent],
        market_data: np.ndarray,
        initial_capital: float = 1000000.0,
        transaction_cost: float = 0.001
    ):
        """
        初始化

        Args:
            agents: 智能体列表
            market_data: 市场数据
            initial_capital: 初始资金
            transaction_cost: 交易成本
        """
        self.agents = agents
        self.market_data = market_data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        self.agent_portfolios: Dict[str, Dict[str, Any]] = {}
        self.agent_positions: Dict[str, Dict[str, float]] = {}

        # 初始化智能体组合
        for agent in agents:
            self.agent_portfolios[agent.agent_id] = {
                "cash": initial_capital / len(agents),
                "holdings": {},
                "total_value": initial_capital / len(agents)
            }
            self.agent_positions[agent.agent_id] = {}

        self.current_step = 0
        self.episode_rewards: Dict[str, List[float]] = {
            agent.agent_id: [] for agent in agents
        }

    def reset(self) -> Dict[str, np.ndarray]:
        """
        重置环境

        Returns:
            初始状态字典
        """
        self.current_step = 0

        # 重置组合
        for agent in self.agents:
            self.agent_portfolios[agent.agent_id] = {
                "cash": self.initial_capital / len(self.agents),
                "holdings": {},
                "total_value": self.initial_capital / len(self.agents)
            }
            self.agent_positions[agent.agent_id] = {}
            self.episode_rewards[agent.agent_id] = []

        return self._get_observations()

    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """
        执行一步

        Args:
            actions: 动作字典

        Returns:
            观测、奖励、完成、信息
        """
        # 执行动作
        self._execute_actions(actions)

        # 更新状态
        self.current_step += 1

        # 获取新观测
        observations = self._get_observations()

        # 计算奖励
        rewards = self._compute_rewards()

        # 检查是否结束
        dones = self._get_dones()

        # 额外信息
        infos = self._get_infos()

        return observations, rewards, dones, infos

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        获取观测

        Returns:
            观测字典
        """
        observations = {}

        # 市场数据
        market_features = self.market_data[self.current_step]

        for agent in self.agents:
            # 智能体特定特征
            portfolio_state = self.agent_portfolios[agent.agent_id]

            # 通信消息（来自其他智能体）
            messages = self._get_communication_messages(agent)

            # 组合观测
            obs = np.concatenate([
                market_features,
                self._portfolio_to_features(portfolio_state),
                self._messages_to_features(messages)
            ])

            observations[agent.agent_id] = obs

        return observations

    def _portfolio_to_features(self, portfolio: Dict[str, Any]) -> np.ndarray:
        """
        将组合转换为特征

        Args:
            portfolio: 组合字典

        Returns:
            特征数组
        """
        cash = portfolio["cash"]
        total_value = portfolio["total_value"]

        features = [
            cash / total_value if total_value > 0 else 0,
            (total_value - cash) / total_value if total_value > 0 else 0
        ]

        return np.array(features)

    def _messages_to_features(self, messages: List[Dict[str, Any]]) -> np.ndarray:
        """
        将消息转换为特征

        Args:
            messages: 消息列表

        Returns:
            特征数组
        """
        if not messages:
            return np.zeros(10)

        features = []

        for msg in messages[:5]:  # 最多5条消息
            features.extend([
                msg.get("sentiment", 0),
                msg.get("urgency", 0)
            ])

        # 填充到固定长度
        while len(features) < 10:
            features.extend([0, 0])

        return np.array(features)

    def _get_communication_messages(self, agent: TradingAgent) -> List[Dict[str, Any]]:
        """
        获取通信消息

        Args:
            agent: 智能体

        Returns:
            消息列表
        """
        messages = []

        # 从其他智能体获取消息
        for other_agent in self.agents:
            if other_agent.agent_id != agent.agent_id:
                # 基于其他智能体的持仓生成消息
                position = self.agent_positions.get(other_agent.agent_id, {})
                if position:
                    messages.append({
                        "sender": other_agent.agent_id,
                        "sender_type": other_agent.agent_type.value,
                        "sentiment": np.random.uniform(-1, 1),
                        "urgency": np.random.uniform(0, 1)
                    })

        return messages

    def _execute_actions(self, actions: Dict[str, int]):
        """
        执行动作

        Args:
            actions: 动作字典
        """
        # 获取当前价格
        current_price = self.market_data[self.current_step, 0]  # 假设第一列是价格

        for agent_id, action in actions.items():
            portfolio = self.agent_portfolios[agent_id]

            # 简化的动作执行
            if action == 0:  # BUY
                quantity = min(100, portfolio["cash"] / current_price * (1 - self.transaction_cost))
                if quantity > 0:
                    portfolio["cash"] -= quantity * current_price * (1 + self.transaction_cost)
                    portfolio["holdings"]["default"] = portfolio["holdings"].get("default", 0) + quantity

            elif action == 1:  # SELL
                quantity = portfolio["holdings"].get("default", 0)
                if quantity > 0:
                    portfolio["cash"] += quantity * current_price * (1 - self.transaction_cost)
                    portfolio["holdings"]["default"] = 0

            # 更新组合价值
            portfolio["total_value"] = portfolio["cash"] + sum(
                quantity * current_price
                for quantity in portfolio["holdings"].values()
            )

    def _compute_rewards(self) -> Dict[str, float]:
        """
        计算奖励

        Returns:
            奖励字典
        """
        rewards = {}

        for agent in self.agents:
            portfolio = self.agent_portfolios[agent.agent_id]

            # 基本奖励：收益
            total_value = portfolio["total_value"]
            initial_value = self.initial_capital / len(self.agents)
            reward = (total_value - initial_value) / initial_value

            # 风险调整
            # ...（简化实现）

            # 协作奖励（基于其他智能体的表现）
            cooperation_reward = self._compute_cooperation_reward(agent)

            rewards[agent.agent_id] = reward + cooperation_reward * 0.1

            # 记录
            self.episode_rewards[agent.agent_id].append(reward)

        return rewards

    def _compute_cooperation_reward(self, agent: TradingAgent) -> float:
        """
        计算协作奖励

        Args:
            agent: 智能体

        Returns:
            协作奖励
        """
        # 简化实现：基于整体市场表现
        total_value = sum(
            p["total_value"]
            for p in self.agent_portfolios.values()
        )
        initial_total = self.initial_capital

        return (total_value - initial_total) / initial_total

    def _get_dones(self) -> Dict[str, bool]:
        """
        获取完成标志

        Returns:
            完成标志字典
        """
        done = self.current_step >= len(self.market_data) - 1

        return {agent.agent_id: done for agent in self.agents}

    def _get_infos(self) -> Dict[str, Any]:
        """
        获取额外信息

        Returns:
            信息字典
        """
        return {
            "step": self.current_step,
            "portfolios": self.agent_portfolios.copy()
        }
```

### 52.4 集中训练分散执行

```python
class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient
    集中训练分散执行框架
    """

    def __init__(
        self,
        agents: List[TradingAgent],
        env: MultiAgentTradingEnvironment,
        gamma: float = 0.99,
        tau: float = 0.01,
        policy_freq: int = 2
    ):
        """
        初始化

        Args:
            agents: 智能体列表
            env: 环境
            gamma: 折扣因子
            tau: 软更新参数
            policy_freq: 策略更新频率
        """
        self.agents = agents
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq

        # 为每个智能体创建集中式Critic
        self.central_critics: Dict[str, nn.Module] = {}
        self.central_critics_target: Dict[str, nn.Module] = {}

        for agent in agents:
            # 集中式Critic输入所有智能体的状态和动作
            central_state_dim = sum(a.state_dim for a in agents)
            central_action_dim = sum(a.action_dim for a in agents)

            central_critic = nn.Sequential(
                nn.Linear(central_state_dim + central_action_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

            self.central_critics[agent.agent_id] = central_critic
            self.central_critics_target[agent.agent_id] = self._clone_network(central_critic)

        # 优化器
        self.central_critic_optimizers: Dict[str, optim.Optimizer] = {}
        for agent_id, critic in self.central_critics.items():
            self.central_critic_optimizers[agent_id] = optim.Adam(critic.parameters(), lr=1e-3)

    def _clone_network(self, network: nn.Module) -> nn.Module:
        """
        克隆网络

        Args:
            network: 原网络

        Returns:
            克隆的网络
        """
        import copy
        return copy.deepcopy(network)

    def soft_update(self, local_model: nn.Module, target_model: nn.Module):
        """
        软更新目标网络

        Args:
            local_model: 本地模型
            target_model: 目标模型
        """
        for target_param, local_param in zip(
            target_model.parameters(),
            local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1 - self.tau) * target_param.data
            )

    def train(
        self,
        num_episodes: int = 1000,
        max_steps: int = 1000,
        batch_size: int = 64
    ) -> Dict[str, List[float]]:
        """
        训练

        Args:
            num_episodes: 训练回合数
            max_steps: 最大步数
            batch_size: 批次大小

        Returns:
            训练历史
        """
        training_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "agent_rewards": {agent.agent_id: [] for agent in self.agents}
        }

        for episode in range(num_episodes):
            # 重置环境
            observations = self.env.reset()

            episode_reward = 0
            episode_length = 0

            for step in range(max_steps):
                # 选择动作
                actions = {}
                for agent in self.agents:
                    obs = observations[agent.agent_id]
                    action = agent.select_action(obs, explore=True)
                    actions[agent.agent_id] = action

                # 执行动作
                next_observations, rewards, dones, infos = self.env.step(actions)

                # 存储经验
                for agent in self.agents:
                    agent.replay_buffer.append({
                        "state": observations[agent.agent_id],
                        "action": actions[agent.agent_id],
                        "reward": rewards[agent.agent_id],
                        "next_state": next_observations[agent.agent_id],
                        "done": dones[agent.agent_id]
                    })

                # 更新网络
                if len(self.agents[0].replay_buffer) > batch_size:
                    self.update(batch_size)

                episode_reward += sum(rewards.values())
                episode_length += 1

                observations = next_observations

                if all(dones.values()):
                    break

            # 记录
            training_history["episode_rewards"].append(episode_reward)
            training_history["episode_lengths"].append(episode_length)

            for agent in self.agents:
                agent_rewards = sum(self.env.episode_rewards[agent.agent_id])
                training_history["agent_rewards"][agent.agent_id].append(agent_rewards)

            # 软更新目标网络
            for agent in self.agents:
                self.soft_update(
                    agent.critic,
                    self.central_critics_target[agent.agent_id]
                )

            if episode % 100 == 0:
                avg_reward = np.mean(training_history["episode_rewards"][-100:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

        return training_history

    def update(self, batch_size: int):
        """
        更新网络

        Args:
            batch_size: 批次大小
        """
        # 采样批次
        batch_indices = np.random.choice(
            len(self.agents[0].replay_buffer),
            batch_size,
            replace=False
        )

        # 更新每个智能体
        for agent in self.agents:
            # 准备批次数据
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

            for idx in batch_indices:
                transition = agent.replay_buffer[idx]
                states.append(transition["state"])
                actions.append(transition["action"])
                rewards.append(transition["reward"])
                next_states.append(transition["next_state"])
                dones.append(transition["done"])

            # 转换为张量
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions))
            rewards = torch.FloatTensor(np.array(rewards))
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(np.array(dones))

            # 集中式Critic更新
            # 收集所有智能体的状态和动作
            all_states = []
            all_next_states = []

            for other_agent in self.agents:
                if other_agent.agent_id == agent.agent_id:
                    all_states.append(states)
                    all_next_states.append(next_states)
                else:
                    # 从其他智能体的replay buffer中获取
                    # ...（简化实现）
                    all_states.append(torch.zeros_like(states))
                    all_next_states.append(torch.zeros_like(next_states))

            all_states = torch.cat(all_states, dim=-1)
            all_next_states = torch.cat(all_next_states, dim=-1)

            # 计算目标Q值
            with torch.no_grad():
                target_actions = []
                for other_agent in self.agents:
                    if other_agent.agent_id == agent.agent_id:
                        target_actions.append(
                            torch.argmax(other_agent.actor(next_states), dim=-1, keepdim=True)
                        )
                    else:
                        target_actions.append(torch.zeros((len(states), 1)))

                target_actions = torch.cat(target_actions, dim=-1)

                target_q = self.central_critics_target[agent.agent_id](
                    torch.cat([all_next_states, target_actions], dim=-1)
                ).squeeze()

                targets = rewards + self.gamma * target_q * (1 - dones)

            # 当前Q值
            actions_one_hot = torch.zeros(len(actions), sum(a.action_dim for a in self.agents))
            actions_one_hot.scatter_(1, actions.unsqueeze(1), 1)

            current_q = self.central_critics[agent.agent_id](
                torch.cat([all_states, actions_one_hot], dim=-1)
            ).squeeze()

            # Critic损失
            critic_loss = nn.MSELoss()(current_q, targets)

            self.central_critic_optimizers[agent.agent_id].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.central_critics[agent.agent_id].parameters(),
                1.0
            )
            self.central_critic_optimizers[agent.agent_id].step()

            # Actor更新
            action_probs = agent.actor(states)
            actor_loss = -(action_probs * current_q.detach()).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_optimizer.step()
```

---

## 第53章 实时特征工程

### 53.1 概述

实时特征工程系统以微秒级延迟从原始市场数据中提取有效特征。

**核心特性：**
- 流式处理
- 低延迟计算
- 特征窗口管理
- 特征版本控制
- 在线特征存储

### 53.2 流式特征提取

```python
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class FeatureSpec:
    """特征规范"""
    name: str
    description: str
    compute_fn: Callable
    dependencies: List[str] = field(default_factory=list)
    window_size: Optional[int] = None
    dtype: type = float
    is_online: bool = True
    version: int = 1


@dataclass
class FeatureValue:
    """特征值"""
    name: str
    value: Any
    timestamp: datetime
    version: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureWindow:
    """
    特征窗口
    """

    def __init__(self, size: int, dtype: type = float):
        """
        初始化

        Args:
            size: 窗口大小
            dtype: 数据类型
        """
        self.size = size
        self.dtype = dtype
        self.buffer = deque(maxlen=size)
        self.timestamps = deque(maxlen=size)

    def append(self, value: Any, timestamp: datetime):
        """
        添加值

        Args:
            value: 值
            timestamp: 时间戳
        """
        self.buffer.append(value)
        self.timestamps.append(timestamp)

    def get_values(self) -> List[Any]:
        """
        获取所有值

        Returns:
            值列表
        """
        return list(self.buffer)

    def get_array(self) -> np.ndarray:
        """
        获取数组

        Returns:
            数组
        """
        return np.array(self.buffer, dtype=self.dtype)

    def is_full(self) -> bool:
        """
        是否已满

        Returns:
            是否已满
        """
        return len(self.buffer) >= self.size

    def get_age_seconds(self) -> float:
        """
        获取窗口年龄（秒）

        Returns:
            年龄
        """
        if len(self.timestamps) < 2:
            return 0.0

        return (self.timestamps[-1] - self.timestamps[0]).total_seconds()


class StreamingFeatureExtractor:
    """
    流式特征提取器
    """

    def __init__(
        self,
        max_workers: int = 4,
        feature_ttl_seconds: int = 3600
    ):
        """
        初始化

        Args:
            max_workers: 最大工作线程数
            feature_ttl_seconds: 特征生存时间（秒）
        """
        self.features: Dict[str, FeatureSpec] = {}
        self.feature_windows: Dict[str, FeatureWindow] = {}
        self.feature_values: Dict[str, FeatureValue] = {}
        self.computed_at: Dict[str, datetime] = {}

        self.max_workers = max_workers
        self.feature_ttl_seconds = feature_ttl_seconds
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.computation_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "total_computations": 0,
            "total_time_ms": 0,
            "avg_time_ms": 0,
            "max_time_ms": 0,
            "last_computed": None
        })

    def register_feature(self, spec: FeatureSpec):
        """
        注册特征

        Args:
            spec: 特征规范
        """
        self.features[spec.name] = spec

        if spec.window_size:
            self.feature_windows[spec.name] = FeatureWindow(
                size=spec.window_size,
                dtype=spec.dtype
            )

        logger.info(f"Registered feature: {spec.name}")

    def compute_features(
        self,
        market_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, FeatureValue]:
        """
        计算特征

        Args:
            market_data: 市场数据
            timestamp: 时间戳

        Returns:
            特征值字典
        """
        if timestamp is None:
            timestamp = datetime.now()

        results = {}

        # 按依赖顺序排序特征
        sorted_features = self._topological_sort()

        # 并行计算
        futures = {}
        for feature_name in sorted_features:
            future = self.executor.submit(
                self._compute_single_feature,
                feature_name,
                market_data,
                timestamp
            )
            futures[future] = feature_name

        for future in asyncio.as_completed(futures):
            feature_name = futures[future]
            try:
                feature_value = future.result(timeout=1.0)
                results[feature_name] = feature_value
            except Exception as e:
                logger.error(f"Failed to compute {feature_name}: {e}")
                results[feature_name] = FeatureValue(
                    name=feature_name,
                    value=None,
                    timestamp=timestamp,
                    version=self.features[feature_name].version
                )

        return results

    def _compute_single_feature(
        self,
        feature_name: str,
        market_data: Dict[str, Any],
        timestamp: datetime
    ) -> FeatureValue:
        """
        计算单个特征

        Args:
            feature_name: 特征名称
            market_data: 市场数据
            timestamp: 时间戳

        Returns:
            特征值
        """
        spec = self.features[feature_name]
        start_time = datetime.now()

        # 准备依赖特征
        kwargs = {}
        for dep in spec.dependencies:
            if dep in self.feature_values:
                kwargs[dep] = self.feature_values[dep].value
            elif dep in market_data:
                kwargs[dep] = market_data[dep]
            else:
                kwargs[dep] = None

        # 添加原始数据
        kwargs["raw_data"] = market_data
        kwargs["timestamp"] = timestamp

        # 计算特征
        try:
            value = spec.compute_fn(**kwargs)

            # 更新窗口
            if feature_name in self.feature_windows:
                self.feature_windows[feature_name].append(value, timestamp)

            # 创建特征值
            feature_value = FeatureValue(
                name=feature_name,
                value=value,
                timestamp=timestamp,
                version=spec.version
            )

            self.feature_values[feature_name] = feature_value

        except Exception as e:
            logger.error(f"Error computing {feature_name}: {e}")
            value = None

            feature_value = FeatureValue(
                name=feature_name,
                value=value,
                timestamp=timestamp,
                version=spec.version,
                metadata={"error": str(e)}
            )

        # 更新统计
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        stats = self.computation_stats[feature_name]
        stats["total_computations"] += 1
        stats["total_time_ms"] += elapsed_ms
        stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_computations"]
        stats["max_time_ms"] = max(stats["max_time_ms"], elapsed_ms)
        stats["last_computed"] = timestamp

        return feature_value

    def _topological_sort(self) -> List[str]:
        """
        拓扑排序特征

        Returns:
            排序后的特征名称列表
        """
        visited = set()
        result = []

        def visit(feature_name: str):
            if feature_name in visited:
                return

            visited.add(feature_name)

            spec = self.features.get(feature_name)
            if spec:
                for dep in spec.dependencies:
                    visit(dep)

            result.append(feature_name)

        for feature_name in self.features:
            visit(feature_name)

        return result

    def get_feature(self, feature_name: str) -> Optional[FeatureValue]:
        """
        获取特征值

        Args:
            feature_name: 特征名称

        Returns:
            特征值
        """
        return self.feature_values.get(feature_name)

    def get_feature_window(self, feature_name: str) -> Optional[FeatureWindow]:
        """
        获取特征窗口

        Args:
            feature_name: 特征名称

        Returns:
            特征窗口
        """
        return self.feature_windows.get(feature_name)

    def clear_stale_features(self):
        """清理过期特征"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.feature_ttl_seconds)

        stale_features = [
            name for name, fv in self.feature_values.items()
            if fv.timestamp < cutoff
        ]

        for name in stale_features:
            del self.feature_values[name]

        logger.info(f"Cleared {len(stale_features)} stale features")
```

### 53.3 技术指标特征

```python
class TechnicalIndicators:
    """
    技术指标
    """

    @staticmethod
    def sma(prices: np.ndarray, period: int) -> float:
        """
        简单移动平均

        Args:
            prices: 价格数组
            period: 周期

        Returns:
            SMA值
        """
        if len(prices) < period:
            return np.nan

        return float(np.mean(prices[-period:]))

    @staticmethod
    def ema(prices: np.ndarray, period: int) -> float:
        """
        指数移动平均

        Args:
            prices: 价格数组
            period: 周期

        Returns:
            EMA值
        """
        if len(prices) < period:
            return np.nan

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return float(ema)

    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """
        相对强弱指数

        Args:
            prices: 价格数组
            period: 周期

        Returns:
            RSI值
        """
        if len(prices) < period + 1:
            return np.nan

        deltas = np.diff(prices[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    @staticmethod
    def bollinger_bands(
        prices: np.ndarray,
        period: int = 20,
        num_std: float = 2.0
    ) -> Dict[str, float]:
        """
        布林带

        Args:
            prices: 价格数组
            period: 周期
            num_std: 标准差倍数

        Returns:
            布林带字典
        """
        if len(prices) < period:
            return {
                "upper": np.nan,
                "middle": np.nan,
                "lower": np.nan,
                "bandwidth": np.nan
            }

        middle = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        upper = middle + num_std * std
        lower = middle - num_std * std
        bandwidth = (upper - lower) / middle if middle != 0 else 0

        return {
            "upper": float(upper),
            "middle": float(middle),
            "lower": float(lower),
            "bandwidth": float(bandwidth)
        }

    @staticmethod
    def macd(
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, float]:
        """
        MACD

        Args:
            prices: 价格数组
            fast_period: 快速周期
            slow_period: 慢速周期
            signal_period: 信号周期

        Returns:
            MACD字典
        """
        if len(prices) < slow_period:
            return {
                "macd": np.nan,
                "signal": np.nan,
                "histogram": np.nan
            }

        ema_fast = TechnicalIndicators.ema(prices, fast_period)
        ema_slow = TechnicalIndicators.ema(prices, slow_period)

        macd_line = ema_fast - ema_slow

        # 简化：不计算信号线
        signal_line = macd_line
        histogram = macd_line - signal_line

        return {
            "macd": float(macd_line),
            "signal": float(signal_line),
            "histogram": float(histogram)
        }

    @staticmethod
    def stochastic(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, float]:
        """
        随机指标

        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组
            k_period: K周期
            d_period: D周期

        Returns:
            随机指标字典
        """
        if len(close) < k_period:
            return {"k": np.nan, "d": np.nan}

        recent_high = np.max(high[-k_period:])
        recent_low = np.min(low[-k_period:])

        if recent_high == recent_low:
            k = 50.0
        else:
            k = 100 * (close[-1] - recent_low) / (recent_high - recent_low)

        # 简化：不计算D
        d = k

        return {"k": float(k), "d": float(d)}

    @staticmethod
    def atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> float:
        """
        平均真实波幅

        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组
            period: 周期

        Returns:
            ATR值
        """
        if len(close) < period + 1:
            return np.nan

        tr_list = []

        for i in range(1, len(close)):
            high_low = high[i] - low[i]
            high_close = abs(high[i] - close[i - 1])
            low_close = abs(low[i] - close[i - 1])

            tr = max(high_low, high_close, low_close)
            tr_list.append(tr)

        atr = np.mean(tr_list[-period:])

        return float(atr)

    @staticmethod
    def obv(prices: np.ndarray, volumes: np.ndarray) -> float:
        """
        能量潮

        Args:
            prices: 价格数组
            volumes: 成交量数组

        Returns:
            OBV值
        """
        if len(prices) < 2:
            return 0.0

        obv = 0.0

        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                obv += volumes[i]
            elif prices[i] < prices[i - 1]:
                obv -= volumes[i]

        return float(obv)


class TradingFeatureFactory:
    """
    交易特征工厂
    """

    @staticmethod
    def create_standard_features(extractor: StreamingFeatureExtractor):
        """
        创建标准特征

        Args:
            extractor: 特征提取器
        """
        # 价格相关特征
        extractor.register_feature(FeatureSpec(
            name="price_sma_5",
            description="5周期简单移动平均",
            compute_fn=lambda raw_data, **kwargs: TechnicalIndicators.sma(
                raw_data.get("prices", np.array([])),
                5
            ),
            window_size=5
        ))

        extractor.register_feature(FeatureSpec(
            name="price_sma_20",
            description="20周期简单移动平均",
            compute_fn=lambda raw_data, **kwargs: TechnicalIndicators.sma(
                raw_data.get("prices", np.array([])),
                20
            ),
            window_size=20
        ))

        extractor.register_feature(FeatureSpec(
            name="price_ema_12",
            description="12周期指数移动平均",
            compute_fn=lambda raw_data, **kwargs: TechnicalIndicators.ema(
                raw_data.get("prices", np.array([])),
                12
            ),
            window_size=12
        ))

        # RSI
        extractor.register_feature(FeatureSpec(
            name="rsi_14",
            description="14周期RSI",
            compute_fn=lambda raw_data, **kwargs: TechnicalIndicators.rsi(
                raw_data.get("prices", np.array([])),
                14
            ),
            window_size=15
        ))

        # 布林带
        extractor.register_feature(FeatureSpec(
            name="bb_upper",
            description="布林带上轨",
            compute_fn=lambda raw_data, **kwargs: TechnicalIndicators.bollinger_bands(
                raw_data.get("prices", np.array([]))
            )["upper"],
            window_size=20
        ))

        extractor.register_feature(FeatureSpec(
            name="bb_lower",
            description="布林带下轨",
            compute_fn=lambda raw_data, **kwargs: TechnicalIndicators.bollinger_bands(
                raw_data.get("prices", np.array([]))
            )["lower"],
            window_size=20
        ))

        extractor.register_feature(FeatureSpec(
            name="bb_bandwidth",
            description="布林带宽度",
            compute_fn=lambda raw_data, **kwargs: TechnicalIndicators.bollinger_bands(
                raw_data.get("prices", np.array([]))
            )["bandwidth"],
            window_size=20
        ))

        # MACD
        extractor.register_feature(FeatureSpec(
            name="macd",
            description="MACD",
            compute_fn=lambda raw_data, **kwargs: TechnicalIndicators.macd(
                raw_data.get("prices", np.array([]))
            )["macd"],
            window_size=26
        ))

        # ATR
        extractor.register_feature(FeatureSpec(
            name="atr_14",
            description="14周期ATR",
            compute_fn=lambda raw_data, **kwargs: TechnicalIndicators.atr(
                raw_data.get("high", np.array([])),
                raw_data.get("low", np.array([])),
                raw_data.get("close", np.array([])),
                14
            ),
            window_size=15
        ))

        # 成交量特征
        extractor.register_feature(FeatureSpec(
            name="volume_sma_20",
            description="20周期成交量均线",
            compute_fn=lambda raw_data, **kwargs: TechnicalIndicators.sma(
                raw_data.get("volumes", np.array([])),
                20
            ),
            window_size=20
        ))

        # 价格动量
        extractor.register_feature(FeatureSpec(
            name="price_momentum_5",
            description="5周期价格动量",
            compute_fn=lambda raw_data, **kwargs: (
                (raw_data.get("prices", np.array([]))[-1] -
                 raw_data.get("prices", np.array([]))[-6]) /
                raw_data.get("prices", np.array([]))[-6]
                if len(raw_data.get("prices", np.array([]))) >= 6
                else np.nan
            ),
            window_size=6
        ))

        # 波动率
        extractor.register_feature(FeatureSpec(
            name="volatility_20",
            description="20周期波动率",
            compute_fn=lambda raw_data, **kwargs: np.std(
                raw_data.get("returns", np.array([]))[-20:]
            ) * np.sqrt(252) if len(raw_data.get("returns", np.array([]))) >= 20 else np.nan,
            window_size=20
        ))
```

### 53.4 在线特征存储

```python
class OnlineFeatureStore:
    """
    在线特征存储
    """

    def __init__(
        self,
        max_size: int = 1000000,
        ttl_seconds: int = 3600
    ):
        """
        初始化

        Args:
            max_size: 最大大小
            ttl_seconds: 生存时间（秒）
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        self.features: Dict[str, Dict[str, FeatureValue]] = defaultdict(dict)
        self.indices: Dict[str, Dict[Any, Set[str]]] = defaultdict(lambda: defaultdict(set))

        self.access_count: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, datetime] = {}

    def put(
        self,
        entity_id: str,
        feature_name: str,
        feature_value: FeatureValue
    ):
        """
        存储特征

        Args:
            entity_id: 实体ID
            feature_name: 特征名称
            feature_value: 特征值
        """
        key = f"{entity_id}:{feature_name}"

        # 删除旧值
        if feature_name in self.features[entity_id]:
            old_value = self.features[entity_id][feature_name]
            self._remove_from_indices(entity_id, feature_name, old_value)

        # 存储新值
        self.features[entity_id][feature_name] = feature_value

        # 更新索引
        self._add_to_indices(entity_id, feature_name, feature_value)

        # 更新访问统计
        self.access_count[key] += 1
        self.last_access[key] = datetime.now()

        # 检查大小限制
        if len(self.features) > self.max_size:
            self._evict_lru()

    def get(
        self,
        entity_id: str,
        feature_name: str
    ) -> Optional[FeatureValue]:
        """
        获取特征

        Args:
            entity_id: 实体ID
            feature_name: 特征名称

        Returns:
            特征值
        """
        feature_value = self.features[entity_id].get(feature_name)

        if feature_value:
            key = f"{entity_id}:{feature_name}"
            self.access_count[key] += 1
            self.last_access[key] = datetime.now()

            # 检查是否过期
            age = (datetime.now() - feature_value.timestamp).total_seconds()
            if age > self.ttl_seconds:
                del self.features[entity_id][feature_name]
                return None

        return feature_value

    def get_multi(
        self,
        entity_id: str,
        feature_names: List[str]
    ) -> Dict[str, FeatureValue]:
        """
        批量获取特征

        Args:
            entity_id: 实体ID
            feature_names: 特征名称列表

        Returns:
            特征值字典
        """
        result = {}

        for feature_name in feature_names:
            feature_value = self.get(entity_id, feature_name)
            if feature_value:
                result[feature_name] = feature_value

        return result

    def _add_to_indices(
        self,
        entity_id: str,
        feature_name: str,
        feature_value: FeatureValue
    ):
        """
        添加到索引

        Args:
            entity_id: 实体ID
            feature_name: 特征名称
            feature_value: 特征值
        """
        # 时间索引
        timestamp_key = feature_value.timestamp.strftime("%Y%m%d%H%M")
        self.indices["timestamp"][timestamp_key].add(f"{entity_id}:{feature_name}")

    def _remove_from_indices(
        self,
        entity_id: str,
        feature_name: str,
        feature_value: FeatureValue
    ):
        """
        从索引删除

        Args:
            entity_id: 实体ID
            feature_name: 特征名称
            feature_value: 特征值
        """
        timestamp_key = feature_value.timestamp.strftime("%Y%m%d%H%M")
        self.indices["timestamp"][timestamp_key].discard(f"{entity_id}:{feature_name}")

    def _evict_lru(self):
        """淘汰最少使用的特征"""
        # 找到最少使用的键
        lru_key = min(self.last_access, key=self.last_access.get)

        # 删除
        entity_id, feature_name = lru_key.split(":")
        if feature_name in self.features[entity_id]:
            del self.features[entity_id][feature_name]

        del self.access_count[lru_key]
        del self.last_access[lru_key]

    def cleanup_stale(self):
        """清理过期特征"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.ttl_seconds)

        stale_keys = []

        for entity_id, features in self.features.items():
            for feature_name, feature_value in features.items():
                if feature_value.timestamp < cutoff:
                    stale_keys.append((entity_id, feature_name))

        for entity_id, feature_name in stale_keys:
            del self.features[entity_id][feature_name]

        logger.info(f"Cleaned up {len(stale_keys)} stale features")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计字典
        """
        return {
            "total_features": sum(len(features) for features in self.features.values()),
            "unique_entities": len(self.features),
            "avg_access_count": np.mean(list(self.access_count.values())) if self.access_count else 0,
            "total_accesses": sum(self.access_count.values())
        }
```

---

## 第54章 高级异常检测

### 54.1 概述

高级异常检测系统用于识别交易数据中的异常模式和潜在风险。

**核心特性：**
- 多种异常检测算法
- 实时异常评分
- 异常解释
- 自适应阈值
- 异常聚类分析

### 54.2 异常检测算法

```python
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """异常类型"""
    STATISTICAL = "statistical"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class AnomalySeverity(Enum):
    """异常严重程度"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Anomaly:
    """异常"""
    anomaly_id: str
    type: AnomalyType
    severity: AnomalySeverity
    score: float
    timestamp: int
    entity_id: str
    features: Dict[str, Any]
    explanation: Dict[str, Any]
    confidence: float
    related_anomalies: List[str] = None


class BaseAnomalyDetector(ABC):
    """
    异常检测器基类
    """

    def __init__(self, threshold: float = 0.5):
        """
        初始化

        Args:
            threshold: 异常阈值
        """
        self.threshold = threshold
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseAnomalyDetector":
        """
        拟合模型

        Args:
            X: 训练数据

        Returns:
            自身
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常分数

        Args:
            X: 数据

        Returns:
            异常分数
        """
        pass

    def detect(
        self,
        X: np.ndarray,
        entity_ids: Optional[List[str]] = None
    ) -> List[Anomaly]:
        """
        检测异常

        Args:
            X: 数据
            entity_ids: 实体ID列表

        Returns:
            异常列表
        """
        scores = self.predict(X)

        anomalies = []

        for i, score in enumerate(scores):
            if score > self.threshold:
                entity_id = entity_ids[i] if entity_ids else f"entity_{i}"

                anomaly = Anomaly(
                    anomaly_id=f"{self.__class__.__name__}_{i}_{int(time.time())}",
                    type=self.get_anomaly_type(),
                    severity=self._get_severity(score),
                    score=float(score),
                    timestamp=int(time.time()),
                    entity_id=entity_id,
                    features={"data_point": X[i].tolist()},
                    explanation=self._explain(X[i], score),
                    confidence=min(score / self.threshold, 1.0)
                )

                anomalies.append(anomaly)

        return anomalies

    @abstractmethod
    def get_anomaly_type(self) -> AnomalyType:
        """获取异常类型"""
        pass

    def _get_severity(self, score: float) -> AnomalySeverity:
        """
        获取严重程度

        Args:
            score: 异常分数

        Returns:
            异常严重程度
        """
        if score > self.threshold * 3:
            return AnomalySeverity.CRITICAL
        elif score > self.threshold * 2:
            return AnomalySeverity.HIGH
        elif score > self.threshold * 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def _explain(self, x: np.ndarray, score: float) -> Dict[str, Any]:
        """
        解释异常

        Args:
            x: 数据点
            score: 异常分数

        Returns:
            解释字典
        """
        return {
            "score": float(score),
            "threshold": self.threshold,
            "features": x.tolist()
        }


class ZScoreDetector(BaseAnomalyDetector):
    """
    Z-Score异常检测器
    """

    def __init__(self, threshold: float = 3.0):
        """
        初始化

        Args:
            threshold: Z-score阈值
        """
        super().__init__(threshold)
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        """
        拟合模型

        Args:
            X: 训练数据

        Returns:
            自身
        """
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常分数

        Args:
            X: 数据

        Returns:
            异常分数（Z-score）
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted")

        z_scores = np.abs((X - self.means) / (self.stds + 1e-8))
        max_z_scores = np.max(z_scores, axis=1)

        return max_z_scores

    def get_anomaly_type(self) -> AnomalyType:
        """获取异常类型"""
        return AnomalyType.STATISTICAL


class IsolationForestDetector(BaseAnomalyDetector):
    """
    隔离森林异常检测器
    """

    def __init__(
        self,
        threshold: float = 0.5,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Union[int, float] = "auto"
    ):
        """
        初始化

        Args:
            threshold: 异常阈值
            contamination: 污染比例
            n_estimators: 树数量
            max_samples: 最大样本数
        """
        super().__init__(threshold)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """
        拟合模型

        Args:
            X: 训练数据

        Returns:
            自身
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常分数

        Args:
            X: 数据

        Returns:
            异常分数
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted")

        X_scaled = self.scaler.transform(X)
        # 转换为0-1范围（1表示异常）
        scores = -self.model.score_samples(X_scaled)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        return scores

    def get_anomaly_type(self) -> AnomalyType:
        """获取异常类型"""
        return AnomalyType.COLLECTIVE


class OneClassSVMDetector(BaseAnomalyDetector):
    """
    单类SVM异常检测器
    """

    def __init__(
        self,
        threshold: float = 0.5,
        kernel: str = "rbf",
        gamma: str = "scale",
        nu: float = 0.1
    ):
        """
        初始化

        Args:
            threshold: 异常阈值
            kernel: 核函数
            gamma: gamma参数
            nu: nu参数
        """
        super().__init__(threshold)
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu

        self.model = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        """
        拟合模型

        Args:
            X: 训练数据

        Returns:
            自身
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常分数

        Args:
            X: 数据

        Returns:
            异常分数
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted")

        X_scaled = self.scaler.transform(X)
        # 转换为0-1范围
        scores = -self.model.score_samples(X_scaled)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        return scores

    def get_anomaly_type(self) -> AnomalyType:
        """获取异常类型"""
        return AnomalyType.CONTEXTUAL


class AutoEncoderAnomalyDetector(BaseAnomalyDetector):
    """
    自编码器异常检测器
    """

    def __init__(
        self,
        threshold: float = 0.5,
        encoding_dim: int = 32,
        hidden_dims: List[int] = None,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        初始化

        Args:
            threshold: 异常阈值
            encoding_dim: 编码维度
            hidden_dims: 隐藏层维度
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批次大小
        """
        super().__init__(threshold)
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.encoder: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None

    def _build_model(self, input_dim: int):
        """
        构建模型

        Args:
            input_dim: 输入维度
        """
        # 编码器
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, self.encoding_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器（对称）
        decoder_layers = []
        prev_dim = self.encoding_dim

        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

        # 优化器
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate
        )

    def fit(self, X: np.ndarray) -> "AutoEncoderAnomalyDetector":
        """
        拟合模型

        Args:
            X: 训练数据

        Returns:
            自身
        """
        self._build_model(X.shape[1])

        # 转换为张量
        X_tensor = torch.FloatTensor(X)

        # 训练
        for epoch in range(self.epochs):
            # 前向传播
            encoded = self.encoder(X_tensor)
            decoded = self.decoder(encoded)

            # 计算损失
            loss = nn.MSELoss()(decoded, X_tensor)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常分数

        Args:
            X: 数据

        Returns:
            异常分数（重构误差）
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted")

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            encoded = self.encoder(X_tensor)
            decoded = self.decoder(encoded)

            # 计算重构误差
            reconstruction_errors = torch.mean((X_tensor - decoded) ** 2, dim=1)

            # 归一化到0-1
            errors = reconstruction_errors.numpy()

        return errors

    def get_anomaly_type(self) -> AnomalyType:
        """获取异常类型"""
        return AnomalyType.CONTEXTUAL


class LSTMAnomalyDetector(BaseAnomalyDetector):
    """
    LSTM异常检测器（时序）
    """

    def __init__(
        self,
        threshold: float = 0.5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        sequence_length: int = 10,
        learning_rate: float = 0.001,
        epochs: int = 100
    ):
        """
        初始化

        Args:
            threshold: 异常阈值
            hidden_dim: 隐藏维度
            num_layers: LSTM层数
            sequence_length: 序列长度
            learning_rate: 学习率
            epochs: 训练轮数
        """
        super().__init__(threshold)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None

    def _build_model(self, input_dim: int):
        """
        构建模型

        Args:
            input_dim: 输入维度
        """
        self.model = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(self.hidden_dim, input_dim)
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.fc.parameters()),
            lr=self.learning_rate
        )

    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        创建序列

        Args:
            X: 数据

        Returns:
            序列数组
        """
        sequences = []

        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i + self.sequence_length])

        return np.array(sequences)

    def fit(self, X: np.ndarray) -> "LSTMAnomalyDetector":
        """
        拟合模型

        Args:
            X: 训练数据

        Returns:
            自身
        """
        self._build_model(X.shape[1])

        # 创建序列
        sequences = self._create_sequences(X)

        # 转换为张量
        X_tensor = torch.FloatTensor(sequences)

        # 训练
        for epoch in range(self.epochs):
            # 前向传播
            lstm_out, _ = self.model(X_tensor)
            predictions = self.fc(lstm_out[:, -1, :])

            # 目标是序列的下一个值
            targets = X_tensor[:, -1, :]

            # 计算损失
            loss = nn.MSELoss()(predictions, targets)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常分数

        Args:
            X: 数据

        Returns:
            异常分数（预测误差）
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted")

        self.model.eval()

        errors = []

        with torch.no_grad():
            for i in range(len(X) - self.sequence_length):
                sequence = torch.FloatTensor(X[i:i + self.sequence_length]).unsqueeze(0)

                # 预测
                lstm_out, _ = self.model(sequence)
                prediction = self.fc(lstm_out[:, -1, :])

                # 计算误差
                actual = X[i + self.sequence_length]
                error = np.mean((prediction.squeeze().numpy() - actual) ** 2)

                errors.append(error)

        # 前sequence_length个点无法预测，填充均值
        avg_error = np.mean(errors)
        errors = [avg_error] * self.sequence_length + errors

        # 归一化到0-1
        errors = np.array(errors)
        errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

        return errors

    def get_anomaly_type(self) -> AnomalyType:
        """获取异常类型"""
        return AnomalyType.TEMPORAL
```

### 54.3 集成异常检测

```python
class EnsembleAnomalyDetector:
    """
    集成异常检测器
    """

    def __init__(
        self,
        detectors: List[BaseAnomalyDetector],
        voting: str = "soft",
        weights: Optional[List[float]] = None
    ):
        """
        初始化

        Args:
            detectors: 检测器列表
            voting: 投票方式（hard/soft）
            weights: 权重列表
        """
        self.detectors = detectors
        self.voting = voting
        self.weights = weights or [1.0] * len(detectors)

        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def fit(self, X: np.ndarray) -> "EnsembleAnomalyDetector":
        """
        拟合所有检测器

        Args:
            X: 训练数据

        Returns:
            自身
        """
        for detector in self.detectors:
            detector.fit(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常分数

        Args:
            X: 数据

        Returns:
            集成异常分数
        """
        # 获取所有检测器的分数
        all_scores = np.array([detector.predict(X) for detector in self.detectors])

        if self.voting == "hard":
            # 硬投票：多数投票
            votes = (all_scores > np.array([d.threshold for d in self.detectors]).reshape(-1, 1)).astype(int)
            ensemble_scores = np.mean(votes, axis=0)

        else:  # soft voting
            # 软投票：加权平均
            ensemble_scores = np.average(all_scores, axis=0, weights=self.weights)

        return ensemble_scores

    def detect(
        self,
        X: np.ndarray,
        entity_ids: Optional[List[str]] = None
    ) -> List[Anomaly]:
        """
        检测异常

        Args:
            X: 数据
            entity_ids: 实体ID列表

        Returns:
            异常列表
        """
        scores = self.predict(X)
        all_anomalies = []

        # 收集所有检测器的异常
        for detector in self.detectors:
            anomalies = detector.detect(X, entity_ids)
            all_anomalies.extend(anomalies)

        # 按实体ID分组
        anomaly_groups: Dict[str, List[Anomaly]] = defaultdict(list)
        for anomaly in all_anomalies:
            anomaly_groups[anomaly.entity_id].append(anomaly)

        # 合并同一实体的异常
        merged_anomalies = []

        for entity_id, group in anomaly_groups.items():
            # 计算集成分数
            entity_idx = list(entity_ids).index(entity_id) if entity_ids else -1
            if entity_idx >= 0:
                ensemble_score = scores[entity_idx]

                # 获取最常见的类型和最高严重程度
                types = [a.type for a in group]
                most_common_type = max(set(types), key=types.count)

                severities = [a.severity for a in group]
                max_severity = max(severities)

                # 合并解释
                explanations = [a.explanation for a in group]
                merged_explanation = {
                    "detectors": len(group),
                    "detector_types": [d.__class__.__name__ for d in self.detectors],
                    "individual_scores": [a.score for a in group],
                    "ensemble_score": float(ensemble_score)
                }

                merged_anomaly = Anomaly(
                    anomaly_id=f"ensemble_{entity_id}_{int(time.time())}",
                    type=most_common_type,
                    severity=max_severity,
                    score=float(ensemble_score),
                    timestamp=group[0].timestamp,
                    entity_id=entity_id,
                    features=group[0].features,
                    explanation=merged_explanation,
                    confidence=min(ensemble_score / 0.5, 1.0),
                    related_anomalies=[a.anomaly_id for a in group]
                )

                merged_anomalies.append(merged_anomaly)

        return merged_anomalies
```

---

## 第55章 超参数优化

### 55.1 概述

超参数优化系统自动搜索模型的最佳超参数配置。

**核心特性：**
- 多种优化算法
- 并行评估
- 早停机制
- 超参数重要性分析
- 可视化结果

### 55.2 优化算法

```python
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
import logging
import time

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """优化策略"""
    TPE = "tpe"  # Tree-structured Parzen Estimator
    CMA_ES = "cma_es"  # Covariance Matrix Adaptation Evolution Strategy
    RANDOM = "random"
    GRID = "grid"
    BAYESIAN = "bayesian"


@dataclass
class HyperparameterSpace:
    """超参数空间"""
    name: str
    type: str  # float, int, categorical, log_float
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    step: Optional[float] = None


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_value: float
    best_trial: int
    n_trials: int
    study_summary: Dict[str, Any]
    importance: Dict[str, float]
    history: List[Dict[str, Any]]


class HyperparameterOptimizer:
    """
    超参数优化器
    """

    def __init__(
        self,
        objective_fn: Callable,
        search_space: List[HyperparameterSpace],
        direction: str = "maximize",
        strategy: OptimizationStrategy = OptimizationStrategy.TPE,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ):
        """
        初始化

        Args:
            objective_fn: 目标函数
            search_space: 搜索空间
            direction: 优化方向
            strategy: 优化策略
            n_trials: 试验次数
            timeout: 超时时间（秒）
            n_jobs: 并行作业数
        """
        self.objective_fn = objective_fn
        self.search_space = {sp.name: sp for sp in search_space}
        self.direction = direction
        self.strategy = strategy
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs

        self.study = None
        self.result: Optional[OptimizationResult] = None

    def optimize(self) -> OptimizationResult:
        """
        执行优化

        Returns:
            优化结果
        """
        # 创建sampler
        sampler = self._create_sampler()

        # 创建study
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )

        # 定义目标函数包装器
        def objective(trial: optuna.Trial) -> float:
            # 采样超参数
            params = self._sample_params(trial)

            # 调用目标函数
            return self.objective_fn(trial, params)

        # 执行优化
        start_time = time.time()

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )

        elapsed_time = time.time() - start_time

        # 收集结果
        self.result = self._collect_results(elapsed_time)

        return self.result

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """
        创建采样器

        Returns:
            采样器
        """
        if self.strategy == OptimizationStrategy.TPE:
            return TPESampler(
                n_startup_trials=10,
                multivariate=True,
                seed=42
            )

        elif self.strategy == OptimizationStrategy.CMA_ES:
            return CmaEsSampler(
                n_startup_trials=10,
                seed=42
            )

        elif self.strategy == OptimizationStrategy.RANDOM:
            return RandomSampler(seed=42)

        else:
            return TPESampler()

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        采样超参数

        Args:
            trial: 试验对象

        Returns:
            超参数字典
        """
        params = {}

        for name, space in self.search_space.items():
            if space.type == "float":
                params[name] = trial.suggest_float(
                    name,
                    space.low,
                    space.high,
                    step=space.step
                )

            elif space.type == "log_float":
                params[name] = trial.suggest_float(
                    name,
                    space.low,
                    space.high,
                    log=True
                )

            elif space.type == "int":
                params[name] = trial.suggest_int(
                    name,
                    int(space.low),
                    int(space.high),
                    step=int(space.step) if space.step else 1
                )

            elif space.type == "categorical":
                params[name] = trial.suggest_categorical(
                    name,
                    space.choices
                )

        return params

    def _collect_results(self, elapsed_time: float) -> OptimizationResult:
        """
        收集结果

        Args:
            elapsed_time: 经过时间

        Returns:
            优化结果
        """
        best_trial = self.study.best_trial

        # 计算超参数重要性
        importance = optuna.importance.get_param_importances(self.study)

        # 收集历史
        history = []
        for trial in self.study.trials:
            history.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name
            })

        return OptimizationResult(
            best_params=best_trial.params,
            best_value=best_trial.value,
            best_trial=best_trial.number,
            n_trials=len(self.study.trials),
            study_summary={
                "elapsed_time": elapsed_time,
                "direction": self.direction,
                "strategy": self.strategy.value
            },
            importance=importance,
            history=history
        )

    def get_best_params(self) -> Dict[str, Any]:
        """
        获取最佳参数

        Returns:
            最佳参数字典
        """
        if self.result:
            return self.result.best_params
        elif self.study:
            return self.study.best_params
        else:
            raise RuntimeError("No optimization results available")

    def visualize_results(self) -> Dict[str, Any]:
        """
        可视化结果

        Returns:
            可视化数据
        """
        import matplotlib.pyplot as plt
        import io
        import base64

        figures = {}

        # 优化历史
        fig, ax = plt.subplots(figsize=(12, 6))

        values = [t.value for t in self.study.trials if t.value is not None]
        ax.plot(values, 'o-', alpha=0.5)
        ax.plot(np.maximum.accumulate(values), 'g-', label='Best')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Objective Value')
        ax.set_title('Optimization History')
        ax.legend()
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        figures["history"] = base64.b64encode(buf.read()).decode()
        plt.close()

        # 超参数重要性
        if self.result and self.result.importance:
            fig, ax = plt.subplots(figsize=(10, 6))

            importance = self.result.importance
            names = list(importance.keys())
            values = list(importance.values())

            ax.barh(names, values)
            ax.set_xlabel('Importance')
            ax.set_title('Hyperparameter Importance')
            ax.grid(True, alpha=0.3, axis='x')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            figures["importance"] = base64.b64encode(buf.read()).decode()
            plt.close()

        # 参数关系
        if len(self.study.trials) > 10:
            fig = optuna.visualization.plot_param_importances(self.study)
            figures["param_importances_plot"] = fig.to_html()

            fig = optuna.visualization.plot_optimization_history(self.study)
            figures["optimization_history_plot"] = fig.to_html()

        return figures
```

### 55.3 多目标优化

```python
class MultiObjectiveHyperparameterOptimizer:
    """
    多目标超参数优化器
    """

    def __init__(
        self,
        objective_fn: Callable,
        search_space: List[HyperparameterSpace],
        directions: List[str],
        strategy: OptimizationStrategy = OptimizationStrategy.TPE,
        n_trials: int = 100
    ):
        """
        初始化

        Args:
            objective_fn: 目标函数（返回多个值）
            search_space: 搜索空间
            directions: 优化方向列表
            strategy: 优化策略
            n_trials: 试验次数
        """
        self.objective_fn = objective_fn
        self.search_space = {sp.name: sp for sp in search_space}
        self.directions = directions
        self.strategy = strategy
        self.n_trials = n_trials

        self.study = None
        self.result: Optional[OptimizationResult] = None

    def optimize(self) -> OptimizationResult:
        """
        执行优化

        Returns:
            优化结果
        """
        # 创建sampler
        sampler = self._create_sampler()

        # 创建多目标study
        self.study = optuna.create_study(
            directions=self.directions,
            sampler=sampler
        )

        # 定义目标函数包装器
        def objective(trial: optuna.Trial) -> float:
            # 采样超参数
            params = self._sample_params(trial)

            # 调用目标函数
            values = self.objective_fn(trial, params)

            # 返回第一个值（用于单目标接口）
            return values[0]

        # 执行优化
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        # 收集结果
        self.result = self._collect_results()

        return self.result

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """创建采样器"""
        if self.strategy == OptimizationStrategy.TPE:
            return TPESampler(
                n_startup_trials=10,
                multivariate=True,
                seed=42
            )
        else:
            return TPESampler()

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """采样超参数"""
        params = {}

        for name, space in self.search_space.items():
            if space.type == "float":
                params[name] = trial.suggest_float(name, space.low, space.high)
            elif space.type == "log_float":
                params[name] = trial.suggest_float(name, space.low, space.high, log=True)
            elif space.type == "int":
                params[name] = trial.suggest_int(name, int(space.low), int(space.high))
            elif space.type == "categorical":
                params[name] = trial.suggest_categorical(name, space.choices)

        return params

    def _collect_results(self) -> OptimizationResult:
        """收集结果"""
        best_trials = self.study.best_trials

        # 选择帕累托前沿中的第一个作为"最佳"
        best_trial = best_trials[0]

        # 简化的重要性分析（基于第一个目标）
        importance = optuna.importance.get_param_importances(self.study)

        history = []
        for trial in self.study.trials:
            history.append({
                "number": trial.number,
                "values": trial.values,
                "params": trial.params,
                "state": trial.state.name
            })

        return OptimizationResult(
            best_params=best_trial.params,
            best_value=best_trial.values[0],  # 主要目标
            best_trial=best_trial.number,
            n_trials=len(self.study.trials),
            study_summary={
                "n_pareto_trials": len(best_trials),
                "strategy": self.strategy.value
            },
            importance=importance,
            history=history
        )

    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """
        获取帕累托前沿

        Returns:
            帕累托前沿列表
        """
        best_trials = self.study.best_trials

        pareto_front = []
        for trial in best_trials:
            pareto_front.append({
                "params": trial.params,
                "values": trial.values,
                "number": trial.number
            })

        return pareto_front

    def visualize_pareto_front(self) -> str:
        """
        可视化帕累托前沿

        Returns:
            HTML字符串
        """
        import optuna.visualization

        fig = optuna.visualization.plot_pareto_front(
            self.study,
            target_names=self.directions
        )

        return fig.to_html()
```

---

## 结语

本传说级实现细节文档涵盖了AI交易系统的最前沿技术和未来方向：

**第51章 - 神经架构搜索（NAS）：**
- 搜索空间定义与采样
- 架构变异与进化
- ENAS高效优化
- 多目标架构搜索

**第52章 - 多智能体强化学习：**
- 交易智能体架构
- 多智能体环境
- MADDPG集中训练分散执行
- 智能体通信与协作

**第53章 - 实时特征工程：**
- 流式特征提取
- 技术指标特征库
- 特征窗口管理
- 在线特征存储

**第54章 - 高级异常检测：**
- 多种异常检测算法（Z-Score、隔离森林、单类SVM、自编码器、LSTM）
- 集成异常检测
- 异常解释与评分

**第55章 - 超参数优化：**
- 多种优化策略（TPE、CMA-ES、随机搜索）
- 多目标优化
- 帕累托前沿分析
- 可视化结果

这些技术代表了AI交易系统的最前沿研究和应用方向。结合之前的文档，我们现在拥有了从基础到传说级的完整技术栈。

---

**文档版本：** v1.0
**最后更新：** 2025年
**作者：** Claude AI Trading System Team

---

## 附录

### A. 相关技术栈

- **NAS：** AutoKeras, NNI, Ray Tune
- **MARL：** Ray RLlib, MAgent, StarCraft II Learning Environment
- **实时处理：** Apache Flink, Apache Storm, Spark Streaming
- **异常检测：** PyOD, Alibi-Detect, DeepAnomaly
- **超参数优化：** Optuna, Hyperopt, Ray Tune

### B. 参考资源

- "Neural Architecture Search: A Survey" (2022)
- "Multi-Agent Reinforcement Learning: A Selective Overview" (2023)
- "Real-Time Feature Engineering for Machine Learning" (2023)
- "Hands-On Anomaly Detection" (2022)
- "Hyperparameter Optimization for Machine Learning" (2023)

---

**完成！**
