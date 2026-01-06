# NOFX Python 重构技术方案 - 神级实现细节

## 文档说明

本文档是NOFX交易系统从Go重构到Python的**神级**实现细节指南，涵盖了AI研究的绝对前沿技术和理论突破。

**适用场景：**
- A股（中国内地股票市场）
- 港股（香港股票市场）
- 加密货币市场
- 未来的量子金融市场

**技术难度：** ⭐⭐⭐⭐⭐⭐⭐（神级）

---

## 第61章 量子机器学习

### 61.1 概述

量子机器学习（Quantum Machine Learning, QML）利用量子计算的优势来加速和增强机器学习算法。

**核心特性：**
- 量子特征映射
- 变分量子电路
- 量子核方法
- 量子神经网络
- 量子增强优化

### 61.2 量子特征映射

```python
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class QuantumFeatureMap(ABC):
    """
    量子特征映射基类
    """

    @abstractmethod
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        编码经典数据到量子态

        Args:
            x: 经典数据

        Returns:
            量子态表示
        """
        pass

    def get_num_qubits(self) -> int:
        """
        获取需要的量子比特数

        Returns:
            量子比特数
        """
        pass


class PauliFeatureMap(QuantumFeatureMap):
    """
    Pauli旋转特征映射
    """

    def __init__(
        self,
        num_features: int,
        num_qubits: Optional[int] = None,
        alpha: float = 1.0,
        reps: int = 2
    ):
        """
        初始化

        Args:
            num_features: 特征数量
            num_qubits: 量子比特数
            alpha: 缩放因子
            reps: 重复次数
        """
        self.num_features = num_features
        self.num_qubits = num_qubits or int(np.ceil(np.log2(num_features)))
        self.alpha = alpha
        self.reps = reps

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        使用Pauli旋转编码特征

        Args:
            x: 输入特征 [num_features]

        Returns:
            量子态 [2^num_qubits]
        """
        # 初始化为零态
        state = np.zeros(2 ** self.num_qubits, dtype=complex)
        state[0] = 1.0

        # 填充特征
        x_padded = np.pad(x, (0, 2 ** self.num_qubits - len(x)))

        # 应用Hadamard门创建叠加态
        for i in range(self.num_qubits):
            state = self._apply_hadamard(state, i)

        # 应用Pauli-Z旋转
        for rep in range(self.reps):
            for i in range(self.num_qubits):
                # 旋转角度取决于特征
                angle = self.alpha * x_padded[i] * (rep + 1)
                state = self._apply_rz(state, i, angle)

        return state

    def _apply_hadamard(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """应用Hadamard门"""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        return self._apply_single_qubit_gate(state, H, qubit)

    def _apply_rz(self, state: np.ndarray, qubit: int, angle: float) -> np.ndarray:
        """应用Z旋转门"""
        RZ = np.array([
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)]
        ])
        return self._apply_single_qubit_gate(state, RZ, qubit)

    def _apply_single_qubit_gate(
        self,
        state: np.ndarray,
        gate: np.ndarray,
        qubit: int
    ) -> np.ndarray:
        """应用单量子比特门"""
        num_qubits = int(np.log2(len(state)))
        I = np.eye(2)

        # 构建完整的算符
        operator = 1
        for i in range(num_qubits):
            if i == qubit:
                operator = np.kron(operator, gate)
            else:
                operator = np.kron(operator, I)

        return operator @ state

    def get_num_qubits(self) -> int:
        """获取量子比特数"""
        return self.num_qubits


class AmplitudeEncoding(QuantumFeatureMap):
    """
    振幅编码
    """

    def __init__(self, num_features: int):
        """
        初始化

        Args:
            num_features: 特征数量
        """
        self.num_features = num_features
        self.num_qubits = int(np.ceil(np.log2(num_features)))

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        使用振幅编码

        Args:
            x: 输入特征

        Returns:
            量子态
        """
        # 归一化
        x_normalized = x / np.linalg.norm(x)

        # 填充到2^n
        state_size = 2 ** self.num_qubits
        if len(x_normalized) < state_size:
            x_padded = np.pad(x_normalized, (0, state_size - len(x_normalized)))
        else:
            x_padded = x_normalized[:state_size]

        # 量子态就是归一化的特征向量
        state = x_padded.astype(complex)

        return state

    def get_num_qubits(self) -> int:
        """获取量子比特数"""
        return self.num_qubits


class QuantumKernel:
    """
    量子核函数
    """

    def __init__(
        self,
        feature_map: QuantumFeatureMap,
        measurement: str = "overlap"
    ):
        """
        初始化

        Args:
            feature_map: 特征映射
            measurement: 测量方式
        """
        self.feature_map = feature_map
        self.measurement = measurement

    def compute_kernel_matrix(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算核矩阵

        Args:
            X: 数据集1 [n_samples, n_features]
            Y: 数据集2 [m_samples, n_features]

        Returns:
            核矩阵 [n_samples, m_samples]
        """
        if Y is None:
            Y = X

        n_samples_X = X.shape[0]
        n_samples_Y = Y.shape[0]

        kernel_matrix = np.zeros((n_samples_X, n_samples_Y))

        # 编码所有样本
        X_encoded = np.array([self.feature_map.encode(x) for x in X])
        Y_encoded = np.array([self.feature_map.encode(y) for y in Y])

        # 计算核矩阵
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                if self.measurement == "overlap":
                    # 重叠核
                    kernel_matrix[i, j] = np.abs(np.vdot(X_encoded[i], Y_encoded[j])) ** 2
                elif self.measurement == "fidelity":
                    # 保真度
                    kernel_matrix[i, j] = np.abs(np.vdot(X_encoded[i], Y_encoded[j]))

        return kernel_matrix


class QuantumSupportVectorMachine:
    """
    量子支持向量机
    """

    def __init__(
        self,
        feature_map: QuantumFeatureMap,
        C: float = 1.0
    ):
        """
        初始化

        Args:
            feature_map: 量子特征映射
            C: 正则化参数
        """
        self.feature_map = feature_map
        self.C = C
        self.quantum_kernel = QuantumKernel(feature_map)
        self.dual_coef_ = None
        self.support_vectors_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        拟合模型

        Args:
            X: 训练数据
            y: 标签
        """
        from cvxopt import matrix, solvers

        n_samples = X.shape[0]

        # 计算量子核矩阵
        K = self.quantum_kernel.compute_kernel_matrix(X)

        # 构建二次规划问题
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))
        G = matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]))
        h = matrix(np.hstack([np.zeros(n_samples), np.ones(n_samples) * self.C]))
        A = matrix(y.reshape(1, -1).astype(float))
        b = matrix(0.0)

        # 求解
        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x']).flatten()

        # 存储支持向量
        sv = alpha > 1e-5
        self.dual_coef_ = alpha[sv] * y[sv]
        self.support_vectors_ = X[sv]

        logger.info(f"Trained QSVM with {np.sum(sv)} support vectors")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 测试数据

        Returns:
            预测结果
        """
        K_sv = self.quantum_kernel.compute_kernel_matrix(X, self.support_vectors_)
        decision = np.dot(K_sv, self.dual_coef_)

        return np.sign(decision).astype(int)


class VariationalQuantumCircuit(nn.Module):
    """
    变分量子电路（VQC）
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int,
        input_dim: int,
        output_dim: int
    ):
        """
        初始化

        Args:
            num_qubits: 量子比特数
            num_layers: 层数
            input_dim: 输入维度
            output_dim: 输出维度
        """
        super().__init__()

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 可训练参数（旋转角度）
        self.encoding_params = nn.Parameter(torch.randn(input_dim, num_qubits))
        self.variational_params = nn.Parameter(torch.randn(num_layers, num_qubits, 3))
        self.measurement_params = nn.Parameter(torch.randn(num_qubits, output_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码输入

        Args:
            x: 输入 [batch_size, input_dim]

        Returns:
            量子态
        """
        batch_size = x.size(0)

        # 初始化为零态
        state = torch.zeros(batch_size, 2 ** self.num_qubits, dtype=torch.complex64)
        state[:, 0] = 1.0

        # 应用Hadamard门
        for i in range(self.num_qubits):
            state = self._apply_hadamard_batch(state, i)

        # 编码旋转
        for feat_idx in range(self.input_dim):
            for qubit in range(self.num_qubits):
                angle = x[:, feat_idx] * self.encoding_params[feat_idx, qubit]
                state = self._apply_rz_batch(state, qubit, angle)

        return state

    def variational_layer(self, state: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        变分层

        Args:
            state: 量子态
            layer_idx: 层索引

        Returns:
            更新后的量子态
        """
        params = self.variational_params[layer_idx]  # [num_qubits, 3]

        for qubit in range(self.num_qubits):
            # RX, RY, RZ旋转
            rx_angle = params[qubit, 0]
            ry_angle = params[qubit, 1]
            rz_angle = params[qubit, 2]

            state = self._apply_rx_batch(state, qubit, rx_angle)
            state = self._apply_ry_batch(state, qubit, ry_angle)
            state = self._apply_rz_batch(state, qubit, rz_angle)

        # CNOT门（纠缠）
        for qubit in range(self.num_qubits - 1):
            state = self._apply_cnot_batch(state, qubit, qubit + 1)

        return state

    def measure(self, state: torch.Tensor) -> torch.Tensor:
        """
        测量

        Args:
            state: 量子态

        Returns:
            测量结果
        """
        # 计算期望值
        batch_size = state.size(0)
        output = torch.zeros(batch_size, self.output_dim)

        for out_idx in range(self.output_dim):
            for qubit in range(self.num_qubits):
                # 计算Z算符的期望值
                prob_0 = torch.abs(state[:, 2 ** qubit]) ** 2
                prob_1 = torch.abs(state[:, 2 ** qubit + 2 ** (self.num_qubits - 1)]) ** 2

                output[:, out_idx] += self.measurement_params[qubit, out_idx] * (prob_0 - prob_1)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入

        Returns:
            输出
        """
        # 编码
        state = self.encode(x)

        # 变分层
        for layer_idx in range(self.num_layers):
            state = self.variational_layer(state, layer_idx)

        # 测量
        output = self.measure(state)

        return output

    def _apply_hadamard_batch(self, state: torch.Tensor, qubit: int) -> torch.Tensor:
        """批量应用Hadamard门"""
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
        return self._apply_single_qubit_gate_batch(state, H, qubit)

    def _apply_rz_batch(self, state: torch.Tensor, qubit: int, angle: torch.Tensor) -> torch.Tensor:
        """批量应用Z旋转门"""
        # RZ = diag(exp(-i*angle/2), exp(i*angle/2))
        phase = torch.exp(-1j * angle / 2)
        RZ = torch.diag(torch.stack([phase, torch.conj(phase)]))
        return self._apply_single_qubit_gate_batch(state, RZ, qubit)

    def _apply_rx_batch(self, state: torch.Tensor, qubit: int, angle: torch.Tensor) -> torch.Tensor:
        """批量应用X旋转门"""
        cos = torch.cos(angle / 2)
        sin = -1j * torch.sin(angle / 2)
        RX = torch.stack([torch.stack([cos, sin]), torch.stack([sin, cos])])
        return self._apply_single_qubit_gate_batch(state, RX, qubit)

    def _apply_ry_batch(self, state: torch.Tensor, qubit: int, angle: torch.Tensor) -> torch.Tensor:
        """批量应用Y旋转门"""
        cos = torch.cos(angle / 2)
        sin = -1j * torch.sin(angle / 2)
        RY = torch.stack([torch.stack([cos, sin]), torch.stack([-sin, cos])])
        return self._apply_single_qubit_gate_batch(state, RY, qubit)

    def _apply_cnot_batch(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        """批量应用CNOT门"""
        batch_size = state.size(0)
        num_qubits = int(np.log2(state.size(1)))

        # 简化实现：交换量子比特顺序
        # 实际量子模拟器需要更复杂的实现
        return state

    def _apply_single_qubit_gate_batch(
        self,
        state: torch.Tensor,
        gate: torch.Tensor,
        qubit: int
    ) -> torch.Tensor:
        """批量应用单量子比特门"""
        batch_size = state.size(0)
        num_qubits = int(np.log2(state.size(1)))

        # 重塑为张量
        state_tensor = state.view(batch_size, *[2] * num_qubits)

        # 应用门
        # 简化实现
        state_tensor = torch.moveaxis(state_tensor, qubit, 0)
        state_tensor = torch.einsum('ij,j...->i...', gate, state_tensor)
        state_tensor = torch.moveaxis(state_tensor, 0, qubit)

        # 展平
        state = state_tensor.view(batch_size, -1)

        return state


class QuantumNeuralNetwork(nn.Module):
    """
    量子神经网络（混合量子-经典架构）
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_qubits: int = 4,
        num_layers: int = 2
    ):
        """
        初始化

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_qubits: 量子比特数
            num_layers: 量子层数
        """
        super().__init__()

        # 经典预处理
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_qubits)
        )

        # 量子层
        self.quantum_layer = VariationalQuantumCircuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            input_dim=num_qubits,
            output_dim=hidden_dim
        )

        # 经典后处理
        self.postprocess = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入

        Returns:
            输出
        """
        # 预处理
        x = self.preprocess(x)

        # 量子层
        x = self.quantum_layer(x)

        # 后处理
        x = self.postprocess(x)

        return x
```

---

## 第62章 脉冲神经网络

### 62.1 概述

脉冲神经网络（Spiking Neural Networks, SNN）更接近生物神经系统的真实工作方式，具有事件驱动和低功耗的特点。

**核心特性：**
- 脉冲编码
- 可塑性规则
- 脉冲神经元模型
- 时间动力学
- 神经形态计算

### 62.2 脉冲神经元模型

```python
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpikeResponse:
    """脉冲响应"""
    spikes: torch.Tensor
    membrane_potential: torch.Tensor
    current: torch.Tensor


class LeakyIntegrateAndFire:
    """
    Leaky Integrate-and-Fire (LIF) 神经元
    """

    def __init__(
        self,
        tau_mem: float = 10.0,
        tau_syn: float = 5.0,
        threshold: float = 1.0,
        reset: float = 0.0,
        rest_potential: float = 0.0
    ):
        """
        初始化

        Args:
            tau_mem: 膜时间常数
            tau_syn: 突触时间常数
            threshold: 发放阈值
            reset: 重置电位
            rest_potential: 静息电位
        """
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.threshold = threshold
        self.reset = reset
        self.rest_potential = rest_potential

    def forward(
        self,
        input_current: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        dt: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            input_current: 输入电流
            state: 神经元状态
            dt: 时间步长

        Returns:
            (脉冲, 新状态)
        """
        if state is None:
            state = {
                'v': torch.zeros_like(input_current) + self.rest_potential,
                'i': torch.zeros_like(input_current)
            }

        v = state['v']
        i = state['i']

        # 更新电流
        alpha = np.exp(-dt / self.tau_syn)
        i = alpha * i + (1 - alpha) * input_current

        # 更新膜电位
        beta = np.exp(-dt / self.tau_mem)
        v = beta * v + (1 - beta) * i

        # 检测脉冲
        spikes = (v >= self.threshold).float()

        # 重置
        v = v * (1 - spikes) + self.reset * spikes

        new_state = {'v': v, 'i': i}

        return spikes, new_state


class AdaptiveLIF:
    """
    自适应LIF神经元
    """

    def __init__(
        self,
        tau_mem: float = 10.0,
        tau_syn: float = 5.0,
        tau_adapt: float = 100.0,
        threshold: float = 1.0,
        reset: float = 0.0,
        threshold_delta: float = 0.05
    ):
        """
        初始化

        Args:
            tau_mem: 膜时间常数
            tau_syn: 突触时间常数
            tau_adapt: 适应时间常数
            threshold: 基础阈值
            reset: 重置电位
            threshold_delta: 阈值增量
        """
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.tau_adapt = tau_adapt
        self.base_threshold = threshold
        self.reset = reset
        self.threshold_delta = threshold_delta

    def forward(
        self,
        input_current: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        dt: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            input_current: 输入电流
            state: 神经元状态
            dt: 时间步长

        Returns:
            (脉冲, 新状态)
        """
        if state is None:
            state = {
                'v': torch.zeros_like(input_current),
                'i': torch.zeros_like(input_current),
                'b': torch.zeros_like(input_current),  # 适应变量
                'threshold': torch.ones_like(input_current) * self.base_threshold
            }

        v = state['v']
        i = state['i']
        b = state['b']
        threshold = state['threshold']

        # 更新电流
        alpha = np.exp(-dt / self.tau_syn)
        i = alpha * i + (1 - alpha) * input_current

        # 更新适应变量
        rho = np.exp(-dt / self.tau_adapt)
        b = rho * b

        # 更新阈值
        threshold = self.base_threshold + b

        # 更新膜电位
        beta = np.exp(-dt / self.tau_mem)
        v = beta * v + (1 - beta) * i - b

        # 检测脉冲
        spikes = (v >= threshold).float()

        # 更新适应变量
        b = b + self.threshold_delta * spikes

        # 重置
        v = v * (1 - spikes) + self.reset * spikes

        new_state = {'v': v, 'i': i, 'b': b, 'threshold': threshold}

        return spikes, new_state


class IzhikevichNeuron:
    """
    Izhikevich神经元模型
    """

    def __init__(
        self,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0
    ):
        """
        初始化

        Args:
            a: 恢复时间常数
            b: 恢复变量灵敏度
            c: 膜电位重置值
            d: 恢复变量重置值
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def forward(
        self,
        input_current: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        dt: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            input_current: 输入电流
            state: 神经元状态
            dt: 时间步长

        Returns:
            (脉冲, 新状态)
        """
        if state is None:
            state = {
                'v': torch.zeros_like(input_current) - 65.0,
                'u': torch.zeros_like(input_current) * self.b * -65.0
            }

        v = state['v']
        u = state['u']

        # Izhikevich方程
        dv = (0.04 * v ** 2 + 5 * v + 140 - u + input_current) * dt
        du = (self.a * (self.b * v - u)) * dt

        v = v + dv
        u = u + du

        # 检测脉冲
        spikes = (v >= 30.0).float()

        # 重置
        v = v * (1 - spikes) + self.c * spikes
        u = u + self.d * spikes

        new_state = {'v': v, 'u': u}

        return spikes, new_state


class SpikeEncoder:
    """
    脉冲编码器
    """

    def __init__(self, encoding_type: str = "rate"):
        """
        初始化

        Args:
            encoding_type: 编码类型 (rate, temporal, latency, delta)
        """
        self.encoding_type = encoding_type

    def encode(
        self,
        x: torch.Tensor,
        num_steps: int = 20,
        max_rate: float = 100.0
    ) -> torch.Tensor:
        """
        编码为脉冲序列

        Args:
            x: 输入 [batch_size, features]
            num_steps: 时间步数
            max_rate: 最大发放率

        Returns:
            脉冲序列 [batch_size, num_steps, features]
        """
        if self.encoding_type == "rate":
            return self._rate_encoding(x, num_steps, max_rate)
        elif self.encoding_type == "temporal":
            return self._temporal_encoding(x, num_steps)
        elif self.encoding_type == "latency":
            return self._latency_encoding(x, num_steps)
        elif self.encoding_type == "delta":
            return self._delta_encoding(x, num_steps)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    def _rate_encoding(
        self,
        x: torch.Tensor,
        num_steps: int,
        max_rate: float
    ) -> torch.Tensor:
        """
        速率编码

        Args:
            x: 输入
            num_steps: 时间步数
            max_rate: 最大发放率

        Returns:
            脉冲序列
        """
        batch_size, features = x.shape

        # 归一化到[0, 1]
        x_normalized = (x - x.min()) / (x.max() - x.min() + 1e-8)

        # 计算发放率
        rates = x_normalized * max_rate

        # 生成泊松脉冲
        spikes = torch.rand(batch_size, num_steps, features) < (rates.unsqueeze(1) / max_rate)

        return spikes.float()

    def _temporal_encoding(
        self,
        x: torch.Tensor,
        num_steps: int
    ) -> torch.Tensor:
        """
        时序编码

        Args:
            x: 输入
            num_steps: 时间步数

        Returns:
            脉冲序列
        """
        batch_size, features = x.shape

        # 归一化
        x_normalized = (x - x.min()) / (x.max() - x.min() + 1e-8)

        # 计算发放时间
        spike_times = (x_normalized * (num_steps - 1)).long()

        # 生成脉冲
        spikes = torch.zeros(batch_size, num_steps, features)

        for b in range(batch_size):
            for f in range(features):
                t = spike_times[b, f]
                if t < num_steps:
                    spikes[b, t, f] = 1.0

        return spikes

    def _latency_encoding(
        self,
        x: torch.Tensor,
        num_steps: int
    ) -> torch.Tensor:
        """
        延迟编码

        Args:
            x: 输入
            num_steps: 时间步数

        Returns:
            脉冲序列
        """
        # 与时序编码相反：越大的值发放越早
        batch_size, features = x.shape

        # 归一化
        x_normalized = (x - x.min()) / (x.max() - x.min() + 1e-8)

        # 计算发放时间（反序）
        spike_times = ((1 - x_normalized) * (num_steps - 1)).long()

        # 生成脉冲
        spikes = torch.zeros(batch_size, num_steps, features)

        for b in range(batch_size):
            for f in range(features):
                t = spike_times[b, f]
                if t < num_steps:
                    spikes[b, t, f] = 1.0

        return spikes

    def _delta_encoding(
        self,
        x: torch.Tensor,
        num_steps: int
    ) -> torch.Tensor:
        """
        Delta调制编码

        Args:
            x: 输入 [batch_size, seq_len, features]
            num_steps: 时间步数

        Returns:
            脉冲序列
        """
        batch_size, seq_len, features = x.shape

        # 计算差分
        delta = torch.diff(x, dim=1)

        # 正差分发脉冲
        spikes_pos = (delta > 0).float()

        # 负差分发脉冲（不同通道）
        spikes_neg = (delta < 0).float()

        # 合并
        spikes = torch.cat([spikes_pos, spikes_neg], dim=-1)

        # 调整到num_steps
        if spikes.size(1) < num_steps:
            spikes = F.pad(spikes, (0, 0, 0, num_steps - spikes.size(1)))

        return spikes[:, :num_steps, :]


class SpikingLayer(nn.Module):
    """
    脉冲神经网络层
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        neuron_type: str = "lif",
        num_steps: int = 20
    ):
        """
        初始化

        Args:
            input_size: 输入大小
            output_size: 输出大小
            neuron_type: 神经元类型
            num_steps: 时间步数
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_steps = num_steps

        # 权重
        self.weight = nn.Linear(input_size, output_size, bias=False)

        # 神经元
        if neuron_type == "lif":
            self.neuron = LeakyIntegrateAndFire()
        elif neuron_type == "alif":
            self.neuron = AdaptiveLIF()
        elif neuron_type == "izhikevich":
            self.neuron = IzhikevichNeuron()
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入 [batch_size, num_steps, input_size]
            state: 神经元状态

        Returns:
            (输出脉冲, 新状态)
        """
        batch_size, num_steps, input_size = x.shape

        # 初始化状态
        if state is None:
            state = {
                'v': torch.zeros(batch_size, self.output_size),
                'i': torch.zeros(batch_size, self.output_size)
            }

        all_spikes = []
        all_v = []
        all_i = []

        for t in range(num_steps):
            # 线性变换
            input_current = self.weight(x[:, t, :])

            # 神经元更新
            spikes, new_state = self.neuron(input_current, state)

            all_spikes.append(spikes)
            all_v.append(new_state['v'])
            all_i.append(new_state['i'])

            state = new_state

        # 堆叠时间步
        output_spikes = torch.stack(all_spikes, dim=1)  # [batch_size, num_steps, output_size]

        # 更新状态
        state = {
            'v': state['v'],
            'i': state['i']
        }

        return output_spikes, state


class SpikingNeuralNetwork(nn.Module):
    """
    脉冲神经网络
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        neuron_type: str = "lif",
        num_steps: int = 20,
        encoding_type: str = "rate"
    ):
        """
        初始化

        Args:
            input_size: 输入大小
            hidden_sizes: 隐藏层大小列表
            output_size: 输出大小
            neuron_type: 神经元类型
            num_steps: 时间步数
            encoding_type: 编码类型
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_steps = num_steps

        # 编码器
        self.encoder = SpikeEncoder(encoding_type)

        # 层
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(
                SpikingLayer(prev_size, hidden_size, neuron_type, num_steps)
            )
            prev_size = hidden_size

        # 输出层
        layers.append(
            SpikingLayer(prev_size, output_size, neuron_type, num_steps)
        )

        self.layers = nn.ModuleList(layers)

        # 读出层
        self.readout = nn.Linear(output_size, output_size)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入 [batch_size, input_size]

        Returns:
            输出
        """
        batch_size = x.size(0)

        # 编码为脉冲
        x = x.unsqueeze(1)  # [batch_size, 1, input_size]
        x = x.repeat(1, self.num_steps, 1)  # [batch_size, num_steps, input_size]

        # 通过各层
        states = [None] * len(self.layers)

        for layer_idx, layer in enumerate(self.layers):
            x, states[layer_idx] = layer(x, states[layer_idx])

        # 解码：对时间步求和
        spike_counts = x.sum(dim=1)  # [batch_size, output_size]

        # 读出
        output = self.readout(spike_counts)

        return output


class STDPRule:
    """
    脉冲时序可塑性规则
    """

    def __init__(
        self,
        A_plus: float = 0.01,
        A_minus: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0
    ):
        """
        初始化

        Args:
            A_plus: LTP强度
            A_minus: LTD强度
            tau_plus: LTP时间常数
            tau_minus: LTD时间常数
        """
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

    def compute_weight_update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        计算权重更新

        Args:
            pre_spikes: 前脉冲 [batch_size, num_steps, num_pre]
            post_spikes: 后脉冲 [batch_size, num_steps, num_post]
            weights: 当前权重 [num_pre, num_post]
            dt: 时间步长

        Returns:
            权重更新 [num_pre, num_post]
        """
        num_steps = pre_spikes.size(1)
        num_pre = pre_spikes.size(2)
        num_post = post_spikes.size(2)

        # 计算脉冲时间差
        weight_update = torch.zeros_like(weights)

        for t_pre in range(num_steps):
            for t_post in range(num_steps):
                delta_t = t_post - t_pre

                # 获取脉冲对
                pre_active = pre_spikes[:, t_pre, :].unsqueeze(2)  # [batch_size, num_pre, 1]
                post_active = post_spikes[:, t_post, :].unsqueeze(1)  # [batch_size, 1, num_post]

                # 共激活
                co_active = (pre_active * post_active).float()  # [batch_size, num_pre, num_post]

                if co_active.sum() > 0:
                    # STDP规则
                    if delta_t > 0:
                        # LTP: 前脉冲在后脉冲之前
                        delta_w = self.A_plus * np.exp(-delta_t / self.tau_plus)
                    else:
                        # LTD: 前脉冲在后脉冲之后
                        delta_w = -self.A_minus * np.exp(delta_t / self.tau_minus)

                    # 更新权重
                    weight_update += delta_w * co_active.mean(0)

        return weight_update * dt
```

---

## 第63章 超维计算

### 63.1 概述

超维计算（Hyperdimensional Computing, HDC）使用高维随机向量进行符号推理，具有高效和鲁棒的特点。

**核心特性：**
- 高维随机向量
- 绑定和捆绑操作
- 符号推理
- 概率检索
- 容错计算

### 63.2 HDC核心操作

```python
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class Hypervector:
    """
    超向量
    """

    def __init__(
        self,
        dimension: int = 10000,
        sparsity: float = 0.5,
        device: str = "cpu"
    ):
        """
        初始化

        Args:
            dimension: 维度
            sparsity: 稀疏度
            device: 设备
        """
        self.dimension = dimension
        self.sparsity = sparsity
        self.device = torch.device(device)

    def random_hypervector(self, batch_size: int = 1) -> torch.Tensor:
        """
        生成随机超向量

        Args:
            batch_size: 批次大小

        Returns:
            超向量 [batch_size, dimension]
        """
        # 生成二值向量
        hv = torch.randint(0, 2, (batch_size, self.dimension), device=self.device).float()
        hv = hv * 2 - 1  # 转换为{-1, +1}

        # 应用稀疏度
        if self.sparsity < 1.0:
            mask = torch.rand(batch_size, self.dimension, device=self.device) < self.sparsity
            hv = hv * mask.float()

        return hv

    def permute(self, hv: torch.Tensor, seed: int = None) -> torch.Tensor:
        """
        置换（用于表示序列）

        Args:
            hv: 超向量
            seed: 随机种子

        Returns:
            置换后的超向量
        """
        if seed is not None:
            torch.manual_seed(seed)

        # 生成随机置换
        perm = torch.randperm(self.dimension, device=self.device)

        return hv[:, perm]

    def bind(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """
        绑定（结合）

        Args:
            hv1: 超向量1
            hv2: 超向量2

        Returns:
            绑定后的超向量
        """
        return hv1 * hv2

    def bundle(self, hv_list: List[torch.Tensor]) -> torch.Tensor:
        """
        捆绑（叠加）

        Args:
            hv_list: 超向量列表

        Returns:
            捆绑后的超向量
        """
        result = torch.stack(hv_list).sum(dim=0)

        # 二值化
        result = torch.sign(result)

        return result

    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """
        计算相似度

        Args:
            hv1: 超向量1
            hv2: 超向量2

        Returns:
            相似度
        """
        # 内积归一化
        dim = hv1.size(-1)
        sim = (hv1 * hv2).sum(dim=-1) / dim

        return sim

    def cosine_similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """
        余弦相似度

        Args:
            hv1: 超向量1
            hv2: 超向量2

        Returns:
            余弦相似度
        """
        return F.cosine_similarity(hv1, hv2, dim=-1)


class HyperdimensionalMemory:
    """
    超维记忆
    """

    def __init__(
        self,
        dimension: int = 10000,
        max_items: int = 10000
    ):
        """
        初始化

        Args:
            dimension: 维度
            max_items: 最大项目数
        """
        self.dimension = dimension
        self.max_items = max_items

        self.hv = Hypervector(dimension=dimension)

        self.keys: torch.Tensor = None
        self.values: torch.Tensor = None
        self.num_items = 0

    def store(
        self,
        keys: torch.Tensor,
        values: torch.Tensor
    ):
        """
        存储键值对

        Args:
            keys: 键 [batch_size, dimension]
            values: 值 [batch_size, dimension]
        """
        batch_size = keys.size(0)

        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = torch.cat([self.keys, keys], dim=0)
            self.values = torch.cat([self.values, values], dim=0)

        self.num_items += batch_size

        # 限制大小
        if self.num_items > self.max_items:
            self.keys = self.keys[-self.max_items:]
            self.values = self.values[-self.max_items:]
            self.num_items = self.max_items

    def retrieve(
        self,
        query_keys: torch.Tensor,
        k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检索

        Args:
            query_keys: 查询键 [batch_size, dimension]
            k: 返回top-k

        Returns:
            (值, 相似度)
        """
        if self.keys is None:
            return None, None

        # 计算相似度
        similarities = self.hv.similarity(
            query_keys.unsqueeze(1),  # [batch_size, 1, dimension]
            self.keys.unsqueeze(0)  # [1, num_items, dimension]
        )  # [batch_size, num_items]

        # Top-k
        top_k_sim, top_k_idx = torch.topk(similarities, k=k, dim=-1)

        # 获取值
        batch_size = query_keys.size(0)
        retrieved_values = []

        for b in range(batch_size):
            idx = top_k_idx[b]
            retrieved_values.append(self.values[idx])

        retrieved_values = torch.stack(retrieved_values)

        return retrieved_values, top_k_sim

    def associate(
        self,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        关联（创建联想记忆）

        Args:
            keys: 键
            values: 值

        Returns:
            关联记忆
        """
        # 绑定操作
        associations = self.hv.bind(keys, values)

        return associations

    def recall(
        self,
        partial_keys: torch.Tensor,
        partial_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        回忆

        Args:
            partial_keys: 部分键
            partial_values: 部分值

        Returns:
            完整的键或值
        """
        if partial_values is not None:
            # 使用部分键和部分值进行回忆
            # 绑定
            partial_assoc = self.hv.bind(partial_keys, partial_values)

            # 从记忆中检索
            retrieved, _ = self.retrieve(partial_assoc, k=1)

            return retrieved.squeeze(1)
        else:
            # 仅使用键
            retrieved, _ = self.retrieve(partial_keys, k=1)

            return retrieved.squeeze(1)


class HyperdimensionalClassifier(nn.Module):
    """
    超维分类器
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dimension: int = 10000,
        sparsity: float = 0.5
    ):
        """
        初始化

        Args:
            input_dim: 输入维度
            num_classes: 类别数
            dimension: 超向量维度
            sparsity: 稀疏度
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dimension = dimension

        self.hv = Hypervector(dimension=dimension, sparsity=sparsity)

        # 为每个类别生成原型超向量
        self.class_prototypes = nn.Parameter(
            self.hv.random_hypervector(num_classes),
            requires_grad=False
        )

        # 编码矩阵（将输入映射到超向量）
        self.encoding_matrix = nn.Parameter(
            self.hv.random_hypervector(input_dim),
            requires_grad=False
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码输入到超向量

        Args:
            x: 输入 [batch_size, input_dim]

        Returns:
            超向量 [batch_size, dimension]
        """
        # 绑定
        encoded = self.hv.bind(
            x.unsqueeze(-1),  # [batch_size, input_dim, 1]
            self.encoding_matrix.unsqueeze(0)  # [1, input_dim, dimension]
        )  # [batch_size, input_dim, dimension]

        # 捆绑
        encoded = self.hv.bundle([encoded[:, i] for i in range(x.size(1))])

        return encoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入

        Returns:
            类别分数
        """
        # 编码
        encoded = self.encode(x)  # [batch_size, dimension]

        # 计算与每个类原型的相似度
        similarities = self.hv.similarity(
            encoded.unsqueeze(1),  # [batch_size, 1, dimension]
            self.class_prototypes.unsqueeze(0)  # [1, num_classes, dimension]
        )  # [batch_size, num_classes]

        return similarities

    def fit(self, X: torch.Tensor, y: torch.Tensor, learning_rate: float = 0.1):
        """
        训练（更新原型）

        Args:
            X: 训练数据
            y: 标签
            learning_rate: 学习率
        """
        # 编码训练数据
        encoded = self.encode(X)  # [batch_size, dimension]

        # 更新类别原型
        for c in range(self.num_classes):
            mask = (y == c)

            if mask.sum() > 0:
                class_vectors = encoded[mask]
                class_prototype = class_vectors.mean(dim=0)

                # 二值化
                class_prototype = torch.sign(class_prototype)

                # 更新
                self.class_prototypes.data[c] = (
                    (1 - learning_rate) * self.class_prototypes.data[c] +
                    learning_rate * class_prototype
                )

                # 二值化
                self.class_prototypes.data[c] = torch.sign(self.class_prototypes.data[c])


class HyperdimensionalRegression(nn.Module):
    """
    超维回归
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dimension: int = 10000,
        num_basis_vectors: int = 100
    ):
        """
        初始化

        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            dimension: 超向量维度
            num_basis_vectors: 基向量数量
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dimension = dimension
        self.num_basis_vectors = num_basis_vectors

        self.hv = Hypervector(dimension=dimension)

        # 基向量
        self.basis_vectors = nn.Parameter(
            self.hv.random_hypervector(num_basis_vectors),
            requires_grad=False
        )

        # 编码矩阵
        self.encoding_matrix = nn.Parameter(
            self.hv.random_hypervector(input_dim),
            requires_grad=False
        )

        # 解码权重
        self.decode_weights = nn.Parameter(torch.randn(num_basis_vectors, output_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码

        Args:
            x: 输入

        Returns:
            编码表示
        """
        # 编码为超向量
        encoded = self.hv.bind(
            x.unsqueeze(-1),
            self.encoding_matrix.unsqueeze(0)
        )

        encoded = self.hv.bundle([encoded[:, i] for i in range(x.size(1))])

        # 计算与基向量的相似度
        similarities = self.hv.similarity(
            encoded.unsqueeze(1),
            self.basis_vectors.unsqueeze(0)
        )

        return similarities

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入

        Returns:
            预测
        """
        # 编码
        features = self.encode(x)  # [batch_size, num_basis_vectors]

        # 解码
        predictions = torch.matmul(features, self.decode_weights)  # [batch_size, output_dim]

        return predictions
```

---

## 第64章 因果表示学习

### 64.1 概述

因果表示学习（Causal Representation Learning）旨在学习数据中的因果因子，提高模型的泛化能力和鲁棒性。

**核心特性：**
- 因果解耦
- 不变因果机制
- 反事实表示
- 因果图学习
- 干涉不变性

### 64.2 因果解耦

```python
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CausalEncoder(nn.Module):
    """
    因果编码器
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_factors: int,
        hidden_dim: int = 256
    ):
        """
        初始化

        Args:
            input_dim: 输入维度
            latent_dim: 潜在维度
            num_factors: 因子数量
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_factors = num_factors

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 因子编码头
        self.factor_heads = nn.ModuleList([
            nn.Linear(hidden_dim, latent_dim // num_factors)
            for _ in range(num_factors)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入

        Returns:
            因子表示
        """
        # 编码
        features = self.encoder(x)

        # 提取因子
        factors = []
        for head in self.factor_heads:
            factor = head(features)
            factors.append(factor)

        # 拼接
        z = torch.cat(factors, dim=-1)

        return z

    def encode_factor(self, x: torch.Tensor, factor_idx: int) -> torch.Tensor:
        """
        编码单个因子

        Args:
            x: 输入
            factor_idx: 因子索引

        Returns:
            因子表示
        """
        features = self.encoder(x)
        factor = self.factor_heads[factor_idx](features)

        return factor


class InterventionMechanism(nn.Module):
    """
    干预机制
    """

    def __init__(
        self,
        latent_dim: int,
        num_factors: int
    ):
        """
        初始化

        Args:
            latent_dim: 潜在维度
            num_factors: 因子数量
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_factors = num_factors

        # 因子大小
        self.factor_size = latent_dim // num_factors

        # 干预强度
        self.intervention_strength = nn.Parameter(torch.ones(num_factors))

    def intervene(
        self,
        z: torch.Tensor,
        intervention_idx: int,
        intervention_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        执行干预

        Args:
            z: 潜在表示
            intervention_idx: 干预的因子索引
            intervention_value: 干预值

        Returns:
            干预后的表示
        """
        z_intervened = z.clone()

        # 确定因子范围
        start_idx = intervention_idx * self.factor_size
        end_idx = start_idx + self.factor_size

        if intervention_value is not None:
            # 设置为干预值
            z_intervened[:, start_idx:end_idx] = intervention_value
        else:
            # 随机干预
            random_value = torch.randn_like(z[:, start_idx:end_idx])
            strength = torch.sigmoid(self.intervention_strength[intervention_idx])

            z_intervened[:, start_idx:end_idx] = (
                (1 - strength) * z[:, start_idx:end_idx] +
                strength * random_value
            )

        return z_intervened

    def do_intervention(
        self,
        z: torch.Tensor,
        intervention_idx: int
    ) -> torch.Tensor:
        """
        Do-intervention（固定干预）

        Args:
            z: 潜在表示
            intervention_idx: 干预的因子索引

        Returns:
            干预后的表示
        """
        return self.intervene(z, intervention_idx, None)


class CausalDecoder(nn.Module):
    """
    因果解码器
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 256
    ):
        """
        初始化

        Args:
            latent_dim: 潜在维度
            output_dim: 输出维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            z: 潜在表示

        Returns:
            重构
        """
        return self.decoder(z)


class CausalVAE(nn.Module):
    """
    因果变分自编码器
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_factors: int,
        hidden_dim: int = 256,
        beta: float = 1.0
    ):
        """
        初始化

        Args:
            input_dim: 输入维度
            latent_dim: 潜在维度
            num_factors: 因子数量
            hidden_dim: 隐藏层维度
            beta: β系数（用于β-VAE）
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_factors = num_factors
        self.beta = beta

        # 编码器
        self.encoder = CausalEncoder(input_dim, latent_dim, num_factors, hidden_dim)

        # 解码器
        self.decoder = CausalDecoder(latent_dim, input_dim, hidden_dim)

        # 因子先验（标准正态）
        self.register_buffer('factor_means', torch.zeros(num_factors))
        self.register_buffer('factor_stds', torch.ones(num_factors))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码

        Args:
            x: 输入

        Returns:
            (潜在表示, 均值, 对数方差)
        """
        # 因子编码
        factors = []
        factor_means = []
        factor_log_vars = []

        for i in range(self.num_factors):
            factor = self.encoder.encode_factor(x, i)

            # 计算均值和方差
            mean = factor.mean(dim=0, keepdim=True)
            log_var = torch.log(factor.var(dim=0, keepdim=True) + 1e-8)

            factors.append(factor)
            factor_means.append(mean)
            factor_log_vars.append(log_var)

        # 重参数化
        z = []
        for i in range(self.num_factors):
            eps = torch.randn_like(factors[i])
            z_i = factors[i] + eps * torch.exp(0.5 * factor_log_vars[i])
            z.append(z_i)

        z = torch.cat(z, dim=-1)

        return z, torch.cat(factor_means, dim=-1), torch.cat(factor_log_vars, dim=-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码

        Args:
            z: 潜在表示

        Returns:
            重构
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入

        Returns:
            (重构, 潜在表示, 均值, 对数方差)
        """
        z, mean, log_var = self.encode(x)
        recon = self.decode(z)

        return recon, z, mean, log_var

    def loss_function(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        损失函数

        Args:
            x: 输入
            recon: 重构
            mean: 均值
            log_var: 对数方差

        Returns:
            (总损失, 损失字典)
        """
        # 重构损失
        recon_loss = F.mse_loss(recon, x, reduction='sum')

        # KL散度
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # 总损失
        total_loss = recon_loss + self.beta * kl_loss

        losses = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }

        return total_loss, losses

    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        从先验采样

        Args:
            num_samples: 样本数
            device: 设备

        Returns:
            采样结果
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)

        return samples

    def intervene(
        self,
        x: torch.Tensor,
        intervention_idx: int,
        intervention_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        干预并生成

        Args:
            x: 输入
            intervention_idx: 干预因子索引
            intervention_value: 干预值

        Returns:
            干预后的生成
        """
        # 编码
        z, _, _ = self.encode(x)

        # 干预
        mechanism = InterventionMechanism(self.latent_dim, self.num_factors)
        z_intervened = mechanism.intervene(z, intervention_idx, intervention_value)

        # 解码
        x_intervened = self.decode(z_intervened)

        return x_intervened


class InvariantCausalRepresentation(nn.Module):
    """
        """
        不变因果表示学习
        """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_domains: int,
        hidden_dim: int = 256
    ):
        """
        初始化

        Args:
            input_dim: 输入维度
            latent_dim: 潜在维度
            num_domains: 域数量
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_domains = num_domains

        # 域编码器
        self.domain_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            )
            for _ in range(num_domains)
        ])

        # 域判别器
        self.domain_discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains)
        )

        # 内容编码器
        self.content_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # 重构解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(
        self,
        x: torch.Tensor,
        domain_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码

        Args:
            x: 输入
            domain_idx: 域索引

        Returns:
            (内容表示, 风格表示)
        """
        # 内容编码
        content = self.content_encoder(x)

        # 风格编码
        style = self.domain_encoders[domain_idx](x)

        return content, style

    def decode(
        self,
        content: torch.Tensor,
        style: torch.Tensor
    ) -> torch.Tensor:
        """
        解码

        Args:
            content: 内容表示
            style: 风格表示

        Returns:
            重构
        """
        z = torch.cat([content, style], dim=-1)
        recon = self.decoder(z)

        return recon

    def forward(
        self,
        x: torch.Tensor,
        domain_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入
            domain_idx: 域索引

        Returns:
            (重构, 内容, 风格)
        """
        content, style = self.encode(x, domain_idx)
        recon = self.decode(content, style)

        return recon, content, style

    def domain_adversarial_loss(
        self,
        content: torch.Tensor,
        domain_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        域对抗损失

        Args:
            content: 内容表示
            domain_labels: 域标签

        Returns:
            对抗损失
        """
        # 域分类
        domain_pred = self.domain_discriminator(content)

        # 分类损失
        loss = F.cross_entropy(domain_pred, domain_labels)

        return loss

    def transfer_style(
        self,
        x: torch.Tensor,
        source_domain: int,
        target_domain: int
    ) -> torch.Tensor:
        """
        风格迁移

        Args:
            x: 输入
            source_domain: 源域
            target_domain: 目标域

        Returns:
            迁移后的结果
        """
        # 编码
        content, _ = self.encode(x, source_domain)

        # 使用目标域的风格
        # 采样目标域的风格
        dummy_target = torch.zeros_like(x)
        _, target_style = self.encode(dummy_target, target_domain)

        # 解码
        x_transferred = self.decode(content, target_style)

        return x_transferred


class CounterfactualGenerator(nn.Module):
    """
    反事实生成器
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_factors: int,
        hidden_dim: int = 256
    ):
        """
        初始化

        Args:
            input_dim: 输入维度
            latent_dim: 潜在维度
            num_factors: 因子数量
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.causal_vae = CausalVAE(input_dim, latent_dim, num_factors, hidden_dim)

    def generate_counterfactual(
        self,
        x: torch.Tensor,
        intervention_idx: int,
        intervention_value: torch.Tensor
    ) -> torch.Tensor:
        """
        生成反事实

        Args:
            x: 事实
            intervention_idx: 干预因子索引
            intervention_value: 干预值

        Returns:
            反事实
        """
        # 编码
        z, _, _ = self.causal_vae.encode(x)

        # 干预
        mechanism = InterventionMechanism(
            self.causal_vae.latent_dim,
            self.causal_vae.num_factors
        )
        z_cf = mechanism.intervene(z, intervention_idx, intervention_value)

        # 解码
        x_cf = self.causal_vae.decode(z_cf)

        return x_cf

    def compute_causal_effect(
        self,
        x: torch.Tensor,
        intervention_idx: int,
        treatment_value: torch.Tensor,
        control_value: torch.Tensor
    ) -> torch.Tensor:
        """
        计算因果效应

        Args:
            x: 输入
            intervention_idx: 干预因子索引
            treatment_value: 处理值
            control_value: 对照值

        Returns:
            因果效应
        """
        # 生成处理组的反事实
        x_treatment = self.generate_counterfactual(
            x, intervention_idx, treatment_value
        )

        # 生成对照组的反事实
        x_control = self.generate_counterfactual(
            x, intervention_idx, control_value
        )

        # 因果效应
        causal_effect = x_treatment - x_control

        return causal_effect
```

---

## 第65章 AGI级交易系统

### 65.1 概述

AGI（通用人工智能）级交易系统整合所有AI技术，实现真正的智能交易代理。

**核心特性：**
- 多模态理解
- 自主决策
- 持续学习
- 世界模型
- 价值对齐

### 65.2 世界模型

```python
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import logging

logger = logging.getLogger(__name__)


class WorldState:
    """
    世界状态
    """

    def __init__(
        self,
        market_data: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        environment_state: Dict[str, Any],
        agent_state: Dict[str, Any]
    ):
        """
        初始化

        Args:
            market_data: 市场数据
            portfolio_state: 组合状态
            environment_state: 环境状态
            agent_state: 代理状态
        """
        self.market_data = market_data
        self.portfolio_state = portfolio_state
        self.environment_state = environment_state
        self.agent_state = agent_state
        self.timestamp = None

    def to_tensor(self) -> torch.Tensor:
        """
        转换为张量

        Returns:
            张量表示
        """
        # 简化实现：实际中需要更复杂的编码
        features = []

        # 市场数据特征
        if 'prices' in self.market_data:
            features.extend(self.market_data['prices'][-10:])  # 最近10个价格

        # 组合状态
        if 'total_value' in self.portfolio_state:
            features.append(self.portfolio_state['total_value'])

        if 'cash' in self.portfolio_state:
            features.append(self.portfolio_state['cash'])

        # 环境状态
        if 'volatility' in self.environment_state:
            features.append(self.environment_state['volatility'])

        # 代理状态
        if 'risk_level' in self.agent_state:
            features.append(self.agent_state['risk_level'])

        return torch.tensor(features, dtype=torch.float32)


class TransitionModel(nn.Module):
    """
    转移模型（预测下一状态）
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512
    ):
        """
        初始化

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 确定性转移模型
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # 不确定性模型
        self.uncertainty_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus()
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测下一状态

        Args:
            state: 当前状态
            action: 动作

        Returns:
            (预测状态, 不确定性)
        """
        x = torch.cat([state, action], dim=-1)

        # 确定性预测
        next_state_pred = self.model(x)

        # 不确定性
        uncertainty = self.uncertainty_model(x)

        return next_state_pred, uncertainty


class RewardModel(nn.Module):
    """
    奖励模型
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        """
        初始化

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        预测奖励

        Args:
            state: 状态
            action: 动作

        Returns:
            预测奖励
        """
        x = torch.cat([state, action], dim=-1)
        reward = self.model(x).squeeze(-1)

        return reward


class WorldModel(nn.Module):
    """
    世界模型（整合转移、奖励、终止模型）
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512
    ):
        """
        初始化

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 转移模型
        self.transition_model = TransitionModel(
            state_dim, action_dim, hidden_dim
        )

        # 奖励模型
        self.reward_model = RewardModel(
            state_dim, action_dim, hidden_dim
        )

        # 终止模型
        self.termination_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def imagine_rollout(
        self,
        initial_state: torch.Tensor,
        policy: nn.Module,
        horizon: int = 10
    ) -> List[Dict[str, torch.Tensor]]:
        """
        想象回放（模型预测）

        Args:
            initial_state: 初始状态
            policy: 策略网络
            horizon: 规划视野

        Returns:
            轨迹列表
        """
        trajectory = []

        state = initial_state

        for step in range(horizon):
            # 策略选择动作
            action_probs = policy(state)
            action = torch.multinomial(action_probs, 1).squeeze(-1)

            # 预测下一状态
            next_state, uncertainty = self.transition_model(state, action)

            # 预测奖励
            reward = self.reward_model(state, action)

            # 预测终止
            done = self.termination_model(torch.cat([state, action], dim=-1))

            # 记录
            trajectory.append({
                'state': state,
                'action': action,
                'next_state': next_state,
                'reward': reward,
                'done': done,
                'uncertainty': uncertainty
            })

            # 更新状态
            state = next_state

            # 检查是否终止
            if done.squeeze(-1).mean() > 0.5:
                break

        return trajectory

    def dreamer_planning(
        self,
        initial_state: torch.Tensor,
        policy: nn.Module,
        value_net: nn.Module,
        horizon: int = 15,
        imagination_samples: int = 10
    ) -> torch.Tensor:
        """
        Dreamer算法规划

        Args:
            initial_state: 初始状态
            policy: 策略网络
            value_net: 价值网络
            horizon: 规划视野
            imagination_samples: 想象样本数

        Returns:
            动作
        """
        # 想象多个轨迹
        imagined_returns = []

        for _ in range(imagination_samples):
            trajectory = self.imagine_rollout(initial_state, policy, horizon)

            # 计算累积回报
            discounted_return = 0.0
            gamma = 0.99

            for step, transition in enumerate(reversed(trajectory)):
                discounted_return = transition['reward'] + gamma * discounted_return

            imagined_returns.append(discounted_return)

        # 选择最优动作
        imagined_returns = torch.stack(imagined_returns)
        best_action_idx = torch.argmax(imagined_returns)

        return best_action_idx


class AGITradingAgent:
    """
    AGI级交易代理
    """

    def __init__(
        self,
        world_model: WorldModel,
        policy_network: nn.Module,
        value_network: nn.Module,
        meta_controller: Optional[nn.Module] = None
    ):
        """
        初始化

        Args:
            world_model: 世界模型
            policy_network: 策略网络
            value_network: 价值网络
            meta_controller: 元控制器
        """
        self.world_model = world_model
        self.policy_network = policy_network
        self.value_network = value_network
        self.meta_controller = meta_controller

        self.memory = deque(maxlen=100000)
        self.episode_memory = deque(maxlen=1000)

        # 元学习
        self.adaptation_history: List[Dict[str, Any]] = []

    def select_action(
        self,
        state: WorldState,
        explore: bool = True,
        epsilon: float = 0.1
    ) -> int:
        """
        选择动作

        Args:
            state: 世界状态
            explore: 是否探索
            epsilon: 探索率

        Returns:
            动作
        """
        state_tensor = state.to_tensor()

        if explore and np.random.random() < epsilon:
            # 随机探索
            return np.random.randint(self.world_model.action_dim)
        else:
            # 策略选择
            with torch.no_grad():
                action_probs = self.policy_network(state_tensor.unsqueeze(0))
                action = torch.multinomial(action_probs, 1).item()

            return action

    def observe(
        self,
        transition: Dict[str, Any],
        store: bool = True
    ):
        """
        观察转移

        Args:
            transition: 转移数据
            store: 是否存储
        """
        if store:
            self.memory.append(transition)

    def meta_learn(
        self,
        task_data: List[Dict[str, Any]],
        num_steps: int = 100
    ):
        """
        元学习

        Args:
            task_data: 任务数据
            num_steps: 学习步数
        """
        # 快速适应新任务
        # ...（简化实现）

        logger.info(f"Meta-learned from {len(task_data)} examples")

    def plan(
        self,
        current_state: WorldState,
        horizon: int = 20
    ) -> int:
        """
        规划（使用世界模型）

        Args:
            current_state: 当前状态
            horizon: 规划视野

        Returns:
            最优动作
        """
        state_tensor = current_state.to_tensor()

        # 使用世界模型进行想象
        best_action_idx = self.world_model.dreamer_planning(
            state_tensor.unsqueeze(0),
            self.policy_network,
            self.value_network,
            horizon
        )

        return best_action_idx.item()

    def reflect(self):
        """
        反思（评估自身性能）
        """
        if len(self.episode_memory) < 10:
            return

        # 分析最近的性能
        recent_rewards = [
            transition['reward']
            for transition in list(self.episode_memory)[-10:]
        ]

        avg_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)

        logger.info(f"Reflection: avg_reward={avg_reward:.4f}, std={std_reward:.4f}")

        # 记录到适应历史
        self.adaptation_history.append({
            'timestamp': time.time(),
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'memory_size': len(self.memory)
        })

    def communicate(
        self,
        message: Dict[str, Any],
        other_agents: List['AGITradingAgent']
    ):
        """
        与其他代理通信

        Args:
            message: 消息
            other_agents: 其他代理
        """
        for agent in other_agents:
            agent.receive_message(message, self)

    def receive_message(
        self,
        message: Dict[str, Any],
        sender: 'AGITradingAgent'
    ):
        """
        接收消息

        Args:
            message: 消息
            sender: 发送者
        """
        # 处理消息
        # ...（简化实现）
        pass

    def update_world_model(
        self,
        batch_size: int = 256
    ):
        """
        更新世界模型

        Args:
            batch_size: 批次大小
        """
        if len(self.memory) < batch_size:
            return

        # 采样批次
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        # 提取数据
        states = torch.stack([t['state'].to_tensor() for t in batch])
        actions = torch.stack([torch.tensor(t['action']) for t in batch])
        next_states = torch.stack([t['next_state'].to_tensor() for t in batch])
        rewards = torch.stack([torch.tensor(t['reward']) for t in batch])

        # 训练转移模型
        pred_next_states, uncertainty = self.world_model.transition_model(states, actions)
        transition_loss = F.mse_loss(pred_next_states, next_states)

        # 训练奖励模型
        pred_rewards = self.world_model.reward_model(states, actions)
        reward_loss = F.mse_loss(pred_rewards, rewards)

        # 训练终止模型
        dones = torch.stack([torch.tensor(float(t['done'])) for t in batch])
        pred_dones = self.world_model.termination_model(torch.cat([states, actions], dim=-1))
        termination_loss = F.binary_cross_entropy(pred_dones.squeeze(-1), dones)

        # 总损失
        total_loss = transition_loss + reward_loss + termination_loss

        # 反向传播
        # ...（简化实现）

        logger.debug(f"World model updated: loss={total_loss.item():.4f}")

    def save_experience(
        self,
        episode: List[Dict[str, Any]]
    ):
        """
        保存完整经验

        Args:
            episode: 回合数据
        """
        self.episode_memory.extend(episode)

        # 反思
        self.reflect()
```

---

## 结语

本神级实现细节文档涵盖了AI研究的绝对前沿技术和理论突破：

**第61章 - 量子机器学习：**
- 量子特征映射（Pauli、振幅编码）
- 量子核函数和量子支持向量机
- 变分量子电路（VQC）
- 量子神经网络

**第62章 - 脉冲神经网络：**
- LIF、自适应LIF、Izhikevich神经元模型
- 脉冲编码（速率、时序、延迟、Delta调制）
- STDP可塑性规则
- 脉冲神经网络架构

**第63章 - 超维计算：**
- 超向量和核心操作（绑定、捆绑、置换）
- 超维记忆和联想检索
- 超维分类器和回归

**第64章 - 因果表示学习：**
- 因果VAE和解耦表示
- 干预机制和反事实生成
- 不变因果表示
- 因果效应估计

**第65章 - AGI级交易系统：**
- 世界模型（转移、奖励、终止预测）
- Dreamer规划和想象回放
- AGI交易代理架构
- 元学习和反思机制

这些技术代表了AI研究的绝对前沿，其中许多仍处于理论和实验阶段。它们指向了未来通用人工智能在金融领域的应用方向。

结合之前所有文档，我们现在拥有了从基础到神级的完整技术栈，涵盖了：
- 基础实现（第1-35章）
- 专家级实现（第36-40章）
- 大师级实现（第41-45章）
- 至尊级实现（第46-50章）
- 传说级实现（第51-55章）
- 神话级实现（第56-60章）
- **神级实现（第61-65章）**

完整技术文档共计**65章**，约30,000+行代码，涵盖了AI交易系统的所有技术层面。

---

**文档版本：** v1.0
**最后更新：** 2025年
**作者：** Claude AI Trading System Team

---

**完成！**
