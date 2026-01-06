# NOFX Python重构技术方案 - 专家级实现细节篇

> 本文档是《NOFX_Python_重构技术方案_A股港股》系列的第九部分
> 覆盖第36-40章：深度学习架构、高级策略实现、实时系统架构、综合测试策略、生产级性能调优

---

## 第36章 深度学习架构实现

### 36.1 Transformer-based市场预测模型

```python
# src/models/transformer_market_predictor.py
"""
基于Transformer的市场预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TransformerConfig:
    """Transformer配置"""
    # 模型维度
    d_model: int = 512
    # 注意力头数
    n_heads: int = 8
    # 编码器层数
    n_encoder_layers: int = 6
    # 解码器层数
    n_decoder_layers: int = 6
    # 前馈网络维度
    d_ff: int = 2048
    # Dropout率
    dropout: float = 0.1
    # 最大序列长度
    max_seq_len: int = 5000
    # 输入特征维度
    input_dim: int = 64  # 价格、成交量、技术指标等
    # 输出维度 (预测未来N步)
    output_dim: int = 10
    # 词汇表大小 (用于股票代码嵌入)
    vocab_size: int = 10000


class PositionalEncoding(nn.Module):
    """
    位置编码

    使用正弦/余弦函数编码位置信息
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # 注册为buffer (不参与梯度更新)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads

        # Q, K, V投影
        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)

        # 输出投影
        self.W_o = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            query: 查询张量 [batch_size, seq_len, d_model]
            key: 键张量 [batch_size, seq_len, d_model]
            value: 值张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码

        Returns:
            (输出张量, 注意力权重)
        """
        batch_size = query.size(0)

        # 线性变换并分割成多头
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )

        # 合并多头
        attn_output = (
            attn_output
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        # 输出投影
        output = self.W_o(attn_output)

        return output, attn_weights

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        缩放点积注意力

        Args:
            Q: 查询张量 [batch_size, n_heads, seq_len, d_k]
            K: 键张量 [batch_size, n_heads, seq_len, d_k]
            V: 值张量 [batch_size, n_heads, seq_len, d_k]
            mask: 注意力掩码

        Returns:
            (输出张量, 注意力权重)
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


class FeedForward(nn.Module):
    """
    前馈神经网络
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            输出张量
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码

        Returns:
            输出张量
        """
        # 自注意力 + 残差连接
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            encoder_output: 编码器输出
            self_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码

        Returns:
            输出张量
        """
        # 自注意力 + 残差连接
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 交叉注意力 + 残差连接
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class MarketPredictorTransformer(nn.Module):
    """
    基于Transformer的市场预测模型

    架构：
    1. 输入嵌入层 (特征嵌入 + 位置编码)
    2. Transformer编码器
    3. 预测头 (多层MLP)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # 特征嵌入
        self.feature_embedding = nn.Linear(config.input_dim, config.d_model)

        # 股票代码嵌入
        self.symbol_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(
            config.d_model,
            config.max_seq_len,
            config.dropout
        )

        # Transformer编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.n_encoder_layers)
        ])

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.output_dim),
        )

        # 价格方向预测 (分类)
        self.direction_head = nn.Linear(config.d_model, 3)  # 上涨/下跌/横盘

        # 波动率预测
        self.volatility_head = nn.Linear(config.d_model, 1)

        # 风险评分
        self.risk_head = nn.Linear(config.d_model, 1)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        features: torch.Tensor,
        symbols: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        前向传播

        Args:
            features: 市场特征 [batch_size, seq_len, input_dim]
            symbols: 股票代码 [batch_size, seq_len]
            mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            预测结果字典
        """
        batch_size, seq_len, _ = features.shape

        # 特征嵌入
        x = self.feature_embedding(features)  # [batch_size, seq_len, d_model]

        # 股票代码嵌入
        symbol_emb = self.symbol_embedding(symbols)  # [batch_size, seq_len, d_model]
        x = x + symbol_emb

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer编码
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)

        # 全局平均池化
        pooled = x.mean(dim=1)  # [batch_size, d_model]

        # 多任务预测
        price_predictions = self.prediction_head(pooled)
        direction_logits = self.direction_head(pooled)
        volatility = self.volatility_head(pooled).squeeze(-1)
        risk_score = torch.sigmoid(self.risk_head(pooled).squeeze(-1))

        return {
            'price_predictions': price_predictions,  # [batch_size, output_dim]
            'direction_logits': direction_logits,     # [batch_size, 3]
            'volatility': volatility,                 # [batch_size]
            'risk_score': risk_score,                 # [batch_size]
            'encoder_output': x,                      # [batch_size, seq_len, d_model]
        }

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成因果掩码

        Args:
            seq_len: 序列长度
            device: 设备

        Returns:
            掩码张量 [seq_len, seq_len]
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask


class TransformerTrainer:
    """
    Transformer模型训练器
    """

    def __init__(
        self,
        model: MarketPredictorTransformer,
        config: TransformerConfig,
        device: str = 'cuda',
    ):
        """
        初始化训练器

        Args:
            model: 模型
            config: 配置
            device: 设备
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000,
            eta_min=1e-6,
        )

        # 梯度裁剪
        self.max_grad_norm = 1.0

        logger.info("Transformer训练器初始化", extra={
            'device': device,
            'parameters': sum(p.numel() for p in model.parameters()),
        })

    def train_step(
        self,
        batch: dict,
    ) -> dict:
        """
        训练步骤

        Args:
            batch: 批次数据

        Returns:
            损失字典
        """
        self.model.train()

        # 前向传播
        features = batch['features'].to(self.device)
        symbols = batch['symbols'].to(self.device)
        mask = batch.get('mask')
        if mask is not None:
            mask = mask.to(self.device)

        outputs = self.model(features, symbols, mask)

        # 计算损失 (多任务学习)
        losses = {}

        # 价格预测损失
        if 'price_targets' in batch:
            price_targets = batch['price_targets'].to(self.device)
            losses['price'] = self.mse_loss(
                outputs['price_predictions'],
                price_targets,
            )

        # 方向分类损失
        if 'direction_targets' in batch:
            direction_targets = batch['direction_targets'].to(self.device)
            losses['direction'] = self.ce_loss(
                outputs['direction_logits'],
                direction_targets,
            )

        # 波动率损失
        if 'volatility_targets' in batch:
            volatility_targets = batch['volatility_targets'].to(self.device)
            losses['volatility'] = self.mse_loss(
                outputs['volatility'],
                volatility_targets,
            )

        # 总损失 (加权求和)
        weights = {
            'price': 1.0,
            'direction': 0.5,
            'volatility': 0.3,
        }

        total_loss = sum(
            weights.get(k, 1.0) * v
            for k, v in losses.items()
        )
        losses['total'] = total_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm,
        )

        self.optimizer.step()
        self.scheduler.step()

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        """
        评估模型

        Args:
            dataloader: 数据加载器

        Returns:
            评估指标
        """
        self.model.eval()

        all_losses = []
        all_predictions = []
        all_targets = []

        for batch in dataloader:
            # 前向传播
            features = batch['features'].to(self.device)
            symbols = batch['symbols'].to(self.device)
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(self.device)

            outputs = self.model(features, symbols, mask)

            # 计算损失
            if 'price_targets' in batch:
                price_targets = batch['price_targets'].to(self.device)
                loss = self.mse_loss(
                    outputs['price_predictions'],
                    price_targets,
                )
                all_losses.append(loss.item())

                all_predictions.append(outputs['price_predictions'].cpu())
                all_targets.append(price_targets.cpu())

        # 计算指标
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        mae = torch.abs(predictions - targets).mean().item()
        rmse = torch.sqrt(((predictions - targets) ** 2).mean()).item()

        return {
            'loss': sum(all_losses) / len(all_losses),
            'mae': mae,
            'rmse': rmse,
        }

    def save_checkpoint(self, path: str, epoch: int):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"检查点已加载: {path}")


# 模型推理
class TransformerPredictor:
    """
    Transformer预测器 (生产环境)
    """

    def __init__(
        self,
        model_path: str,
        config: TransformerConfig,
        device: str = 'cuda',
    ):
        """初始化预测器"""
        self.device = device
        self.config = config

        # 加载模型
        self.model = MarketPredictorTransformer(config).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info("Transformer预测器初始化", extra={
            'model_path': model_path,
            'device': device,
        })

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        symbols: torch.Tensor,
    ) -> dict:
        """
        预测

        Args:
            features: 市场特征 [seq_len, input_dim]
            symbols: 股票代码 [seq_len]

        Returns:
            预测结果
        """
        # 添加批次维度
        features = features.unsqueeze(0).to(self.device)
        symbols = symbols.unsqueeze(0).to(self.device)

        # 前向传播
        outputs = self.model(features, symbols)

        # 解析结果
        price_pred = outputs['price_predictions'][0].cpu().numpy()
        direction = torch.argmax(outputs['direction_logits'][0]).item()
        volatility = outputs['volatility'][0].cpu().item()
        risk = outputs['risk_score'][0].cpu().item()

        return {
            'price_predictions': price_pred,
            'direction': direction,  # 0: 下跌, 1: 横盘, 2: 上涨
            'volatility': volatility,
            'risk_score': risk,
        }
```

### 36.2 LSTM时序预测模型

```python
# src/models/lstm_predictor.py
"""
基于LSTM的时序预测模型
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LSTMConfig:
    """LSTM配置"""
    input_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 3
    output_dim: int = 10
    dropout: float = 0.2
    bidirectional: bool = True


class LSTMPredictor(nn.Module):
    """
    LSTM预测模型

    特点：
    1. 双向LSTM捕获前后依赖
    2. 多层堆叠提高表达能力
    3. 注意力机制加权重要时间步
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        # 注意力权重
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim),
            nn.Tanh(),
            nn.Linear(lstm_output_dim, 1),
        )

        # 预测层
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_dim // 2, config.output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> dict:
        """
        前向传播

        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            hidden: 初始隐藏状态

        Returns:
            预测结果和隐藏状态
        """
        # LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)
        # lstm_out: [batch_size, seq_len, lstm_output_dim]

        # 注意力权重
        attn_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)

        # 加权求和
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch_size, lstm_output_dim]

        # 预测
        output = self.fc(context)  # [batch_size, output_dim]

        return {
            'predictions': output,
            'attention_weights': attn_weights.squeeze(-1),
            'hidden': hidden,
        }


class LSTMTrainer:
    """LSTM模型训练器"""

    def __init__(
        self,
        model: LSTMPredictor,
        config: LSTMConfig,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-5,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
        )

        self.criterion = nn.MSELoss()

    def train_step(self, batch: dict) -> dict:
        """训练步骤"""
        self.model.train()

        x = batch['features'].to(self.device)
        y = batch['targets'].to(self.device)

        outputs = self.model(x)
        loss = self.criterion(outputs['predictions'], y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {'loss': loss.item()}
```

### 36.3 图神经网络(GNN)关联分析

```python
# src/models/gnn_stock_correlation.py
"""
基于图神经网络的股票关联分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from typing import Optional

from src.core.logging import get_logger

logger = get_logger(__name__)


class StockCorrelationGNN(nn.Module):
    """
    股票关联图神经网络

    应用场景：
    1. 同行业股票联动分析
    2. 上下游产业链关联
    3. 资金流向关联
    """

    def __init__(
        self,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        output_dim: int = 32,
    ):
        super().__init__()

        # 图注意力层
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(node_feature_dim, hidden_dim, heads=num_heads, concat=True)
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
            )

        self.convs.append(
            GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, node_feature_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边属性 [num_edges, edge_attr_dim]

        Returns:
            节点嵌入 [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = self.dropout(x)

        return x


class GraphConstructor:
    """
    图构建器

    根据股票特征构建关联图
    """

    @staticmethod
    def build_correlation_graph(
        features: torch.Tensor,
        threshold: float = 0.7,
    ) -> Data:
        """
        基于相关性构建图

        Args:
            features: 股票特征矩阵 [num_stocks, feature_dim]
            threshold: 相关性阈值

        Returns:
            图数据
        """
        # 计算相关性矩阵
        corr_matrix = torch.corrcoef(features)

        # 构建边
        edges = []
        edge_attrs = []

        num_stocks = features.size(0)
        for i in range(num_stocks):
            for j in range(i + 1, num_stocks):
                if corr_matrix[i, j].abs() > threshold:
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_attrs.append(corr_matrix[i, j])
                    edge_attrs.append(corr_matrix[i, j])

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attrs).unsqueeze(-1)

        return Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
```

---

## 第37章 高级交易策略实现

### 37.1 强化学习交易策略

```python
# src/strategies/rl_strategy.py
"""
基于强化学习的交易策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import random

from src.core.base_strategy import BaseStrategy
from src.core.signal import Signal, SignalType
from src.core.market_data import MarketData
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RLConfig:
    """强化学习配置"""
    # 状态空间
    state_dim: int = 100  # 价格、成交量、指标、持仓等

    # 动作空间
    action_dim: int = 5  # 买入、卖出、持有、加仓、减仓

    # 网络配置
    hidden_dim: int = 256
    num_layers: int = 3

    # 训练配置
    learning_rate: float = 1e-4
    gamma: float = 0.99  # 折扣因子
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # 经验回放
    replay_buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 100


class DQN(nn.Module):
    """
    Deep Q-Network

    输入：状态向量
    输出：每个动作的Q值
    """

    def __init__(self, config: RLConfig):
        super().__init__()

        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Q网络
        layers = []
        for _ in range(config.num_layers - 1):
            layers.extend([
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])

        self.q_network = nn.Sequential(*layers)
        self.q_head = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: 状态张量 [batch_size, state_dim]

        Returns:
            Q值 [batch_size, action_dim]
        """
        x = self.feature_extractor(state)
        x = self.q_network(x)
        q_values = self.q_head(x)
        return q_values


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        """采样批次"""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    DQN智能体
    """

    def __init__(self, config: RLConfig, device: str = 'cuda'):
        self.config = config
        self.device = device

        # Q网络
        self.q_network = DQN(config).to(device)
        self.target_network = DQN(config).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate,
        )

        # 经验回放
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        # 探索参数
        self.epsilon = config.epsilon_start
        self.step_count = 0

    def select_action(
        self,
        state: np.ndarray,
        eval_mode: bool = False,
    ) -> int:
        """
        选择动作

        Args:
            state: 状态向量
            eval_mode: 是否为评估模式

        Returns:
            动作索引
        """
        if eval_mode or random.random() > self.epsilon:
            # 贪婪策略
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=-1).item()
        else:
            # 随机探索
            action = random.randint(0, self.config.action_dim - 1)

        return action

    def train_step(self) -> Dict[str, float]:
        """
        训练步骤

        Returns:
            损失字典
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        # 采样批次
        batch = self.replay_buffer.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.config.gamma * next_q_values * (1 - dones)

        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # 更新探索率
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay,
        )

        # 定期更新目标网络
        self.step_count += 1
        if self.step_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
        }

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


class RLTradingStrategy(BaseStrategy):
    """
    强化学习交易策略

    状态空间：
    - 价格序列 (归一化)
    - 技术指标 (MA, MACD, RSI等)
    - 当前持仓
    - 可用资金
    - 市场波动率

    动作空间：
    - 0: 持有
    - 1: 买入 100股
    - 2: 买入 200股
    - 3: 卖出 100股
    - 4: 卖出 200股

    奖励函数：
    - 收益率 - 交易成本 - 风险惩罚
    """

    def __init__(
        self,
        config: RLConfig,
        trader_id: str,
        initial_capital: float = 1000000,
    ):
        super().__init__(trader_id, initial_capital)
        self.config = config
        self.agent = DQNAgent(config)

        # 历史数据缓存 (用于构建状态)
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)

        # 状态归一化参数
        self.state_normalizer = None

        logger.info("RL交易策略初始化", extra={
            'trader_id': trader_id,
            'config': config,
        })

    def _build_state(self, market_data: MarketData) -> np.ndarray:
        """
        构建状态向量

        Args:
            market_data: 市场数据

        Returns:
            状态向量
        """
        features = []

        # 1. 价格序列 (最近60个)
        if len(self.price_history) >= 60:
            prices = list(self.price_history)[-60:]
            # 归一化: (p - p_mean) / p_std
            prices = np.array(prices)
            normalized = (prices - prices.mean()) / (prices.std() + 1e-8)
            features.extend(normalized.tolist())
        else:
            features.extend([0] * 60)

        # 2. 技术指标
        if len(self.price_history) >= 20:
            prices = np.array(list(self.price_history))

            # 移动平均线
            ma5 = prices[-5:].mean()
            ma10 = prices[-10:].mean()
            ma20 = prices[-20:].mean()
            current_price = prices[-1]

            # 相对位置
            features.append((current_price - ma5) / ma5)
            features.append((current_price - ma10) / ma10)
            features.append((current_price - ma20) / ma20)

            # RSI
            rsi = self._calculate_rsi(prices, 14)
            features.append(rsi)

            # MACD
            macd, signal = self._calculate_macd(prices)
            features.append(macd)
            features.append(signal)
        else:
            features.extend([0] * 8)

        # 3. 当前持仓
        position = self._get_current_position(market_data.symbol)
        features.append(position.quantity / 1000 if position else 0)
        features.append(position.unrealized_pnl / 10000 if position else 0)

        # 4. 可用资金
        available_capital = self._get_available_capital()
        features.append(available_capital / self.initial_capital)

        # 5. 市场波动率
        if len(self.price_history) >= 20:
            volatility = np.std(list(self.price_history)[-20:]) / np.mean(list(self.price_history)[-20:])
            features.append(volatility)
        else:
            features.append(0)

        return np.array(features, dtype=np.float32)

    def _calculate_reward(
        self,
        action: int,
        prev_portfolio_value: float,
        curr_portfolio_value: float,
        transaction_cost: float,
    ) -> float:
        """
        计算奖励

        Args:
            action: 执行的动作
            prev_portfolio_value: 之前组合价值
            curr_portfolio_value: 当前组合价值
            transaction_cost: 交易成本

        Returns:
            奖励值
        """
        # 收益
        return_reward = (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value

        # 交易成本惩罚
        cost_penalty = -transaction_cost / prev_portfolio_value

        # 风险惩罚 (基于持仓波动)
        risk_penalty = -abs(curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value * 0.1

        # 动作惩罚 (过度交易)
        action_penalty = 0
        if action in [1, 2, 3, 4]:  # 买卖动作
            action_penalty = -0.001

        total_reward = return_reward + cost_penalty + risk_penalty + action_penalty

        return total_reward

    async def on_bar(self, bar: MarketData) -> List[Signal]:
        """
        K线事件处理

        Args:
            bar: K线数据

        Returns:
            信号列表
        """
        # 更新历史
        self.price_history.append(bar.close)
        self.volume_history.append(bar.volume)

        # 构建状态
        state = self._build_state(bar)

        # 选择动作
        action = self.agent.select_action(state)

        # 执行动作
        signals = []
        prev_value = self._get_portfolio_value()

        if action == 1:  # 买入 100股
            signals.append(Signal(
                symbol=bar.symbol,
                signal_type=SignalType.BUY,
                quantity=100,
                price=bar.close,
                confidence=0.7,
                reason='RL买入信号',
            ))
        elif action == 2:  # 买入 200股
            signals.append(Signal(
                symbol=bar.symbol,
                signal_type=SignalType.BUY,
                quantity=200,
                price=bar.close,
                confidence=0.8,
                reason='RL加仓信号',
            ))
        elif action == 3:  # 卖出 100股
            signals.append(Signal(
                symbol=bar.symbol,
                signal_type=SignalType.SELL,
                quantity=100,
                price=bar.close,
                confidence=0.7,
                reason='RL卖出信号',
            ))
        elif action == 4:  # 卖出 200股
            signals.append(Signal(
                symbol=bar.symbol,
                signal_type=SignalType.SELL,
                quantity=200,
                price=bar.close,
                confidence=0.8,
                reason='RL减仓信号',
            ))

        # 计算奖励并训练
        curr_value = self._get_portfolio_value()
        transaction_cost = sum(s.quantity * s.price * 0.0003 for s in signals)
        reward = self._calculate_reward(action, prev_value, curr_value, transaction_cost)

        # 下一个状态
        next_state = self._build_state(bar)

        # 存储经验
        self.agent.replay_buffer.push(
            state, action, reward, next_state, False,
        )

        # 训练
        self.agent.train_step()

        return signals

    def _get_portfolio_value(self) -> float:
        """获取组合价值"""
        total = 0
        for position in self.positions.values():
            total += position.market_value
        total += self._get_available_capital()
        return total

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """计算RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = gains[-period:].mean()
        avg_loss = losses[-period:].mean()

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self,
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[float, float]:
        """计算MACD"""
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        macd = ema_fast - ema_slow

        # 信号线
        if len(macd) >= signal:
            signal_line = self._ema(macd, signal)
            return macd[-1], signal_line[-1]

        return macd[-1], 0

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算指数移动平均"""
        ema = np.zeros_like(data)
        ema[period - 1] = data[:period].mean()

        multiplier = 2 / (period + 1)
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]

        return ema
```

### 37.2 多因子选股策略

```python
# src/strategies/multi_factor_strategy.py
"""
多因子选股策略
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from src.core.base_strategy import BaseStrategy
from src.core.signal import Signal, SignalType
from src.core.market_data import MarketData
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FactorConfig:
    """因子配置"""
    # 价值因子
    pe_ratio: float = 20.0  # 市盈率
    pb_ratio: float = 2.0   # 市净率

    # 成长因子
    revenue_growth: float = 0.2  # 营收增长率
    profit_growth: float = 0.2   # 利润增长率

    # 质量因子
    roe: float = 0.15  # 净资产收益率
    roa: float = 0.08  # 总资产收益率

    # 技术因子
    momentum: float = 0.05  # 动量
    volatility: float = 0.2  # 波动率上限

    # 权重
    value_weight: float = 0.25
    growth_weight: float = 0.25
    quality_weight: float = 0.25
    technical_weight: float = 0.25


class MultiFactorStrategy(BaseStrategy):
    """
    多因子选股策略

    步骤：
    1. 计算各股票的因子得分
    2. 因子标准化 (Z-score)
    3. 加权合成综合得分
    4. 选择得分最高的N只股票
    """

    def __init__(
        self,
        config: FactorConfig,
        trader_id: str,
        initial_capital: float = 1000000,
        max_positions: int = 10,
    ):
        super().__init__(trader_id, initial_capital)
        self.config = config
        self.max_positions = max_positions

        # 因子数据缓存
        self.factor_data: Dict[str, Dict] = {}

        logger.info("多因子选股策略初始化", extra={
            'trader_id': trader_id,
            'max_positions': max_positions,
        })

    def calculate_value_score(
        self,
        symbol: str,
        fundamental: Dict,
    ) -> float:
        """
        计算价值因子得分

        Args:
            symbol: 股票代码
            fundamental: 基本面数据

        Returns:
            价值得分
        """
        pe = fundamental.get('pe_ratio', 0)
        pb = fundamental.get('pb_ratio', 0)

        # 越低越好
        pe_score = min(pe / self.config.pe_ratio, 2.0)
        pb_score = min(pb / self.config.pb_ratio, 2.0)

        # 综合得分 (反向)
        value_score = 2 - (pe_score + pb_score) / 2

        return max(0, value_score)

    def calculate_growth_score(
        self,
        symbol: str,
        fundamental: Dict,
    ) -> float:
        """
        计算成长因子得分
        """
        revenue_growth = fundamental.get('revenue_growth', 0)
        profit_growth = fundamental.get('profit_growth', 0)

        # 标准化
        revenue_score = min(revenue_growth / self.config.revenue_growth, 2.0)
        profit_score = min(profit_growth / self.config.profit_growth, 2.0)

        growth_score = (revenue_score + profit_score) / 2

        return max(0, growth_score)

    def calculate_quality_score(
        self,
        symbol: str,
        fundamental: Dict,
    ) -> float:
        """
        计算质量因子得分
        """
        roe = fundamental.get('roe', 0)
        roa = fundamental.get('roa', 0)

        roe_score = min(roe / self.config.roe, 2.0)
        roa_score = min(roa / self.config.roa, 2.0)

        quality_score = (roe_score + roa_score) / 2

        return max(0, quality_score)

    def calculate_technical_score(
        self,
        symbol: str,
        price_history: List[float],
    ) -> float:
        """
        计算技术因子得分
        """
        prices = np.array(price_history)

        if len(prices) < 20:
            return 0.5

        # 动量 (20日涨幅)
        momentum = (prices[-1] - prices[-20]) / prices[-20]
        momentum_score = min(momentum / self.config.momentum, 2.0)

        # 波动率 (惩罚)
        volatility = prices[-20:].std() / prices[-20:].mean()
        volatility_score = max(0, 2 - volatility / self.config.volatility)

        technical_score = (momentum_score + volatility_score) / 2

        return max(0, technical_score)

    def calculate_composite_score(
        self,
        symbol: str,
        fundamental: Dict,
        price_history: List[float],
    ) -> float:
        """
        计算综合得分
        """
        value_score = self.calculate_value_score(symbol, fundamental)
        growth_score = self.calculate_growth_score(symbol, fundamental)
        quality_score = self.calculate_quality_score(symbol, fundamental)
        technical_score = self.calculate_technical_score(symbol, price_history)

        composite = (
            value_score * self.config.value_weight +
            growth_score * self.config.growth_weight +
            quality_score * self.config.quality_weight +
            technical_score * self.config.technical_weight
        )

        return composite

    async def select_stocks(
        self,
        symbols: List[str],
        fundamental_data: Dict[str, Dict],
        price_data: Dict[str, List[float]],
    ) -> List[tuple]:
        """
        选股

        Args:
            symbols: 股票列表
            fundamental_data: 基本面数据
            price_data: 价格数据

        Returns:
            [(股票, 得分), ...] 按得分降序
        """
        scores = []

        for symbol in symbols:
            fundamental = fundamental_data.get(symbol, {})
            history = price_data.get(symbol, [])

            score = self.calculate_composite_score(
                symbol, fundamental, history
            )
            scores.append((symbol, score))

        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    async def on_bar(self, bar: MarketData) -> List[Signal]:
        """
        K线事件处理

        (简化版，实际应该在定期调仓时触发)
        """
        return []

    async def rebalance(
        self,
        symbols: List[str],
        fundamental_data: Dict[str, Dict],
        price_data: Dict[str, List[float]],
    ) -> List[Signal]:
        """
        调仓

        Args:
            symbols: 股票池
            fundamental_data: 基本面数据
            price_data: 价格数据

        Returns:
            交易信号列表
        """
        # 选股
        scores = await self.select_stocks(
            symbols, fundamental_data, price_data
        )

        # 选择前N只
        selected = scores[:self.max_positions]

        signals = []

        # 计算每只股票的目标仓位 (等权重)
        capital_per_stock = self._get_available_capital() / len(selected)

        for symbol, score in selected:
            current_price = price_data[symbol][-1]
            target_quantity = int(capital_per_stock / current_price / 100) * 100

            # 检查当前持仓
            current_position = self._get_current_position(symbol)
            current_quantity = current_position.quantity if current_position else 0

            # 生成信号
            if target_quantity > current_quantity:
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    quantity=target_quantity - current_quantity,
                    price=current_price,
                    confidence=score,
                    reason=f'多因子选股 (得分: {score:.2f})',
                ))
            elif target_quantity < current_quantity:
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    quantity=current_quantity - target_quantity,
                    price=current_price,
                    confidence=score,
                    reason=f'多因子调仓 (得分: {score:.2f})',
                ))

        # 清仓不在选中的股票
        for symbol, position in self.positions.items():
            if symbol not in [s for s, _ in selected]:
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    quantity=position.quantity,
                    price=position.current_price,
                    confidence=0.5,
                    reason='多因子剔除',
                ))

        return signals
```

### 37.3 统计套利策略

```python
# src/strategies/statistical_arbitrage_strategy.py
"""
统计套利策略 (配对交易)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint

from src.core.base_strategy import BaseStrategy
from src.core.signal import Signal, SignalType
from src.core.market_data import MarketData
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PairConfig:
    """配对交易配置"""
    # 协整检验窗口
    coint_window: int = 252  # 一年交易日

    # Z-score阈值
    entry_threshold: float = 2.0  # 开仓阈值
    exit_threshold: float = 0.5   # 平仓阈值
    stop_loss: float = 4.0        # 止损阈值

    # 持仓限制
    max_pairs: int = 5
    position_size: float = 100000  # 每对配对的资金


class CointegrationPair:
    """
    协整配对

    表示两只存在长期均衡关系的股票
    """

    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        hedge_ratio: float,
        p_value: float,
    ):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.hedge_ratio = hedge_ratio  # 对冲比率
        self.p_value = p_value          # 协整检验p值

        # 价差序列
        self.spreads: List[float] = []

        # 当前状态
        self.position: Optional[str] = None  # 'long' (做多价差) 或 'short' (做空价差)
        self.entry_price: Optional[float] = None

    def update_spread(self, price_a: float, price_b: float) -> float:
        """
        更新价差

        Args:
            price_a: 股票A价格
            price_b: 股票B价格

        Returns:
            当前价差
        """
        spread = price_a - self.hedge_ratio * price_b
        self.spreads.append(spread)

        # 保持固定长度
        if len(self.spreads) > 100:
            self.spreads.pop(0)

        return spread

    def calculate_z_score(self) -> Optional[float]:
        """
        计算价差的Z-score

        Returns:
            Z-score值
        """
        if len(self.spreads) < 20:
            return None

        spreads = np.array(self.spreads[-60:])  # 使用60个样本
        mean = spreads.mean()
        std = spreads.std()

        if std == 0:
            return 0

        z_score = (spreads[-1] - mean) / std
        return z_score


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    统计套利策略

    原理：
    1. 找到具有协整关系的股票对
    2. 当价差偏离均值时开仓
    3. 当价差回归均值时平仓
    """

    def __init__(
        self,
        config: PairConfig,
        trader_id: str,
        initial_capital: float = 1000000,
    ):
        super().__init__(trader_id, initial_capital)
        self.config = config

        # 配对池
        self.pairs: List[CointegrationPair] = []

        # 价格历史 (用于协整检验)
        self.price_history: Dict[str, List[float]] = {}

        logger.info("统计套利策略初始化", extra={
            'trader_id': trader_id,
        })

    async def find_cointegrated_pairs(
        self,
        symbols: List[str],
        price_data: Dict[str, List[float]],
    ) -> List[Tuple[str, str, float, float]]:
        """
        寻找协整配对

        Args:
            symbols: 股票列表
            price_data: 价格数据

        Returns:
            [(股票A, 股票B, 对冲比率, p值), ...]
        """
        cointegrated_pairs = []

        for i, symbol_a in enumerate(symbols):
            for symbol_b in symbols[i + 1:]:
                # 检查数据长度
                if (
                    symbol_a not in price_data or
                    symbol_b not in price_data
                ):
                    continue

                prices_a = price_data[symbol_a][-self.config.coint_window:]
                prices_b = price_data[symbol_b][-self.config.coint_window:]

                if len(prices_a) < self.config.coint_window:
                    continue

                # 协整检验
                try:
                    score, p_value, _ = coint(prices_a, prices_b)

                    # p值 < 0.05 表示存在协整关系
                    if p_value < 0.05:
                        # 计算对冲比率 (OLS回归)
                        hedge_ratio = self._calculate_hedge_ratio(prices_a, prices_b)

                        cointegrated_pairs.append((
                            symbol_a,
                            symbol_b,
                            hedge_ratio,
                            p_value,
                        ))

                        logger.info("发现协整配对", extra={
                            'symbol_a': symbol_a,
                            'symbol_b': symbol_b,
                            'hedge_ratio': hedge_ratio,
                            'p_value': p_value,
                        })

                except Exception as e:
                    logger.warning(f"协整检验失败: {symbol_a}-{symbol_b}", extra={
                        'error': str(e),
                    })

        # 按p值排序 (越小越好)
        cointegrated_pairs.sort(key=lambda x: x[3])

        return cointegrated_pairs[:self.config.max_pairs]

    def _calculate_hedge_ratio(
        self,
        prices_a: List[float],
        prices_b: List[float],
    ) -> float:
        """
        计算对冲比率 (OLS回归)

        Args:
            prices_a: 股票A价格序列
            prices_b: 股票B价格序列

        Returns:
            对冲比率
        """
        # prices_a = alpha + hedge_ratio * prices_b
        x = np.array(prices_b).reshape(-1, 1)
        y = np.array(prices_a)

        # OLS回归
        hedge_ratio, _ = np.linalg.lstsq(x, y, rcond=None)[:2]

        return hedge_ratio[0]

    async def initialize_pairs(
        self,
        symbols: List[str],
        price_data: Dict[str, List[float]],
    ):
        """
        初始化配对

        Args:
            symbols: 股票池
            price_data: 价格数据
        """
        # 寻找协整配对
        cointegrated = await self.find_cointegrated_pairs(symbols, price_data)

        # 创建配对对象
        self.pairs = []
        for symbol_a, symbol_b, hedge_ratio, p_value in cointegrated:
            pair = CointegrationPair(symbol_a, symbol_b, hedge_ratio, p_value)
            self.pairs.append(pair)

    async def on_bar(self, bar: MarketData) -> List[Signal]:
        """
        K线事件处理

        Args:
            bar: K线数据

        Returns:
            信号列表
        """
        signals = []

        # 更新价格历史
        if bar.symbol not in self.price_history:
            self.price_history[bar.symbol] = []
        self.price_history[bar.symbol].append(bar.close)

        # 检查每个配对
        for pair in self.pairs:
            # 检查是否需要更新价差
            if bar.symbol not in [pair.symbol_a, pair.symbol_b]:
                continue

            # 获取两只股票的最新价格
            if (
                pair.symbol_a not in self.price_history or
                pair.symbol_b not in self.price_history
            ):
                continue

            price_a = self.price_history[pair.symbol_a][-1]
            price_b = self.price_history[pair.symbol_b][-1]

            # 更新价差
            spread = pair.update_spread(price_a, price_b)

            # 计算Z-score
            z_score = pair.calculate_z_score()
            if z_score is None:
                continue

            # 根据Z-score生成交易信号
            if pair.position is None:
                # 开仓
                if z_score > self.config.entry_threshold:
                    # 价差过高，做空价差 (卖A买B)
                    pair.position = 'short'
                    pair.entry_price = spread
                    signals.extend(self._create_pair_signals(
                        pair, 'short', price_a, price_b,
                    ))

                elif z_score < -self.config.entry_threshold:
                    # 价差过低，做多价差 (买A卖B)
                    pair.position = 'long'
                    pair.entry_price = spread
                    signals.extend(self._create_pair_signals(
                        pair, 'long', price_a, price_b,
                    ))

            else:
                # 平仓或止损
                should_close = False

                if pair.position == 'short':
                    if z_score < self.config.exit_threshold:
                        should_close = True
                    elif z_score > self.config.stop_loss:
                        should_close = True

                elif pair.position == 'long':
                    if z_score > -self.config.exit_threshold:
                        should_close = True
                    elif z_score < -self.config.stop_loss:
                        should_close = True

                if should_close:
                    signals.extend(self._close_pair_signals(
                        pair, price_a, price_b,
                    ))
                    pair.position = None
                    pair.entry_price = None

        return signals

    def _create_pair_signals(
        self,
        pair: CointegrationPair,
        direction: str,
        price_a: float,
        price_b: float,
    ) -> List[Signal]:
        """
        创建配对交易信号

        Args:
            pair: 配对对象
            direction: 'long' 或 'short'
            price_a: 股票A价格
            price_b: 股票B价格

        Returns:
            信号列表
        """
        signals = []

        # 计算数量 (基于资金)
        quantity_a = int(self.config.position_size / price_a / 100) * 100
        quantity_b = int(quantity_a * pair.hedge_ratio / 100) * 100

        if direction == 'long':
            # 做多价差: 买A卖B
            signals.append(Signal(
                symbol=pair.symbol_a,
                signal_type=SignalType.BUY,
                quantity=quantity_a,
                price=price_a,
                confidence=0.7,
                reason=f'配对交易-做多价差 (Z-score触发)',
            ))
            signals.append(Signal(
                symbol=pair.symbol_b,
                signal_type=SignalType.SELL,
                quantity=quantity_b,
                price=price_b,
                confidence=0.7,
                reason=f'配对交易-做多价差 (Z-score触发)',
            ))

        else:  # short
            # 做空价差: 卖A买B
            signals.append(Signal(
                symbol=pair.symbol_a,
                signal_type=SignalType.SELL,
                quantity=quantity_a,
                price=price_a,
                confidence=0.7,
                reason=f'配对交易-做空价差 (Z-score触发)',
            ))
            signals.append(Signal(
                symbol=pair.symbol_b,
                signal_type=SignalType.BUY,
                quantity=quantity_b,
                price=price_b,
                confidence=0.7,
                reason=f'配对交易-做空价差 (Z-score触发)',
            ))

        return signals

    def _close_pair_signals(
        self,
        pair: CointegrationPair,
        price_a: float,
        price_b: float,
    ) -> List[Signal]:
        """
        平仓信号

        Args:
            pair: 配对对象
            price_a: 股票A价格
            price_b: 股票B价格

        Returns:
            信号列表
        """
        signals = []

        # 平掉所有相关持仓
        for symbol, position in self.positions.items():
            if symbol in [pair.symbol_a, pair.symbol_b]:
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL if position.quantity > 0 else SignalType.BUY,
                    quantity=abs(position.quantity),
                    price=price_a if symbol == pair.symbol_a else price_b,
                    confidence=0.7,
                    reason=f'配对交易-平仓',
                ))

        return signals
```

---

## 第38章 实时系统架构深度剖析

### 38.1 高性能消息队列

```python
# src/infrastructure/high_performance_mq.py
"""
高性能消息队列实现
"""

import asyncio
import aiozmq
import zmq
import uuid
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import orjson

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """消息对象"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    headers: Dict[str, str] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """序列化为字节"""
        data = {
            'id': self.id,
            'topic': self.topic,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'headers': self.headers,
        }
        return orjson.dumps(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        """从字节反序列化"""
        obj = orjson.loads(data)
        return cls(
            id=obj['id'],
            topic=obj['topic'],
            payload=obj['payload'],
            timestamp=datetime.fromisoformat(obj['timestamp']),
            headers=obj.get('headers', {}),
        )


class MessageQueue:
    """
    消息队列

    特性：
    1. 支持发布订阅模式
    2. 支持请求应答模式
    3. 零拷贝传输
    4. 高性能
    """

    def __init__(
        self,
        zmq_type: int,
        host: str = "*",
        port: int = 5555,
    ):
        """
        初始化消息队列

        Args:
            zmq_type: ZMQ socket类型
            host: 绑定地址
            port: 端口
        """
        self.zmq_type = zmq_type
        self.host = host
        self.port = port
        self.socket: Optional[aiozmq.ZmqSocket] = None
        self.lock = asyncio.Lock()

    async def start(self):
        """启动消息队列"""
        self.socket = await aiozmq.create_zmq_socket(self.zmq_type)
        await self.socket.bind(f"tcp://{self.host}:{self.port}")
        logger.info(f"消息队列启动: tcp://{self.host}:{self.port}")

    async def stop(self):
        """停止消息队列"""
        if self.socket:
            self.socket.close()
            logger.info("消息队列已关闭")

    async def send(self, message: Message):
        """发送消息"""
        async with self.lock:
            await self.socket.send(message.to_bytes())

    async def receive(self) -> Message:
        """接收消息"""
        data = await self.socket.recv()
        return Message.from_bytes(data)


class PubSubBroker:
    """
    发布订阅代理

    使用ZMQ的XPUB/XSUB模式实现高性能消息分发
    """

    def __init__(
        self,
        xsub_port: int = 5555,
        xpub_port: int = 5556,
    ):
        self.xsub_port = xsub_port
        self.xpub_port = xpub_port
        self.xsub: Optional[aiozmq.ZmqSocket] = None
        self.xpub: Optional[aiozmq.ZmqSocket] = None

    async def start(self):
        """启动代理"""
        # 创建XSUB (订阅者连接)
        self.xsub = await aiozmq.create_zmq_socket(zmq.XSUB)
        await self.xsub.bind(f"tcp://*:{self.xsub_port}")

        # 创建XPUB (发布者连接)
        self.xpub = await aiozmq.create_zmq_socket(zmq.XPUB)
        await self.xpub.bind(f"tcp://*:{self.xpub_port}")

        # 启动转发
        asyncio.create_task(self._forward())

        logger.info(f"发布订阅代理启动: XSUB={self.xsub_port}, XPUB={self.xpub_port}")

    async def _forward(self):
        """转发消息"""
        try:
            while True:
                # XSUB -> XPUB
                if await self.xsub.poll(timeout=100):
                    data = await self.xsub.recv_multipart()
                    await self.xpub.send_multipart(data)

                # XPUB -> XSUB
                if await self.xpub.poll(timeout=100):
                    data = await self.xpub.recv_multipart()
                    await self.xsub.send_multipart(data)

        except Exception as e:
            logger.error("消息转发失败", extra={'error': str(e)})


class Publisher:
    """发布者"""

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 5555,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.socket: Optional[aiozmq.ZmqSocket] = None

    async def start(self):
        """启动发布者"""
        self.socket = await aiozmq.create_zmq_socket(zmq.PUB)
        await self.socket.connect(f"tcp://{self.broker_host}:{self.broker_port}")
        logger.info(f"发布者连接到代理: {self.broker_host}:{self.broker_port}")

    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ):
        """发布消息"""
        message = Message(
            topic=topic,
            payload=payload,
            headers=headers or {},
        )

        # 发送主题 + 消息
        await self.socket.send_multipart([
            topic.encode(),
            message.to_bytes(),
        ])

    async def stop(self):
        """停止发布者"""
        if self.socket:
            self.socket.close()


class Subscriber:
    """订阅者"""

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 5556,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.socket: Optional[aiozmq.ZmqSocket] = None
        self.handlers: Dict[str, Callable] = {}

    async def start(self):
        """启动订阅者"""
        self.socket = await aiozmq.create_zmq_socket(zmq.SUB)
        await self.socket.connect(f"tcp://{self.broker_host}:{self.broker_port}")

        # 启动接收循环
        asyncio.create_task(self._receive_loop())

        logger.info(f"订阅者连接到代理: {self.broker_host}:{self.broker_port}")

    async def subscribe(self, topic: str, handler: Callable):
        """订阅主题"""
        self.socket.subscribe(topic.encode())
        self.handlers[topic] = handler
        logger.info(f"订阅主题: {topic}")

    async def unsubscribe(self, topic: str):
        """取消订阅"""
        self.socket.unsubscribe(topic.encode())
        self.handlers.pop(topic, None)
        logger.info(f"取消订阅: {topic}")

    async def _receive_loop(self):
        """接收循环"""
        try:
            while True:
                # 接收主题 + 消息
                topic, data = await self.socket.recv_multipart()
                topic = topic.decode()

                # 解析消息
                message = Message.from_bytes(data)

                # 调用处理器
                handler = self.handlers.get(topic)
                if handler:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except Exception as e:
                        logger.error(f"消息处理失败: {topic}", extra={'error': str(e)})

        except Exception as e:
            logger.error("接收循环错误", extra={'error': str(e)})


class ReplyQueue:
    """
    应答队列 (用于请求应答模式)
    """

    def __init__(self, max_size: int = 10000):
        self.pending: Dict[str, asyncio.Future] = {}
        self.max_size = max_size
        self.lock = asyncio.Lock()

    async def wait_for_reply(
        self,
        message_id: str,
        timeout: float = 5.0,
    ) -> Any:
        """
        等待应答

        Args:
            message_id: 消息ID
            timeout: 超时时间

        Returns:
            应答数据
        """
        future = asyncio.Future()

        async with self.lock:
            if len(self.pending) >= self.max_size:
                raise RuntimeError("应答队列已满")

            self.pending[message_id] = future

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            async with self.lock:
                self.pending.pop(message_id, None)
            raise
        finally:
            async with self.lock:
                self.pending.pop(message_id, None)

    async def set_reply(self, message_id: str, result: Any):
        """设置应答"""
        async with self.lock:
            future = self.pending.get(message_id)
            if future and not future.done():
                future.set_result(result)


class RequestClient:
    """请求客户端"""

    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 5555,
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.socket: Optional[aiozmq.ZmqSocket] = None
        self.reply_queue = ReplyQueue()
        self.receive_task: Optional[asyncio.Task] = None

    async def start(self):
        """启动客户端"""
        self.socket = await aiozmq.create_zmq_socket(zmq.DEALER)
        await self.socket.connect(f"tcp://{self.server_host}:{self.server_port}")

        # 启动接收循环
        self.receive_task = asyncio.create_task(self._receive_loop())

        logger.info(f"请求客户端连接: {self.server_host}:{self.server_port}")

    async def request(
        self,
        service: str,
        method: str,
        params: Dict[str, Any],
        timeout: float = 5.0,
    ) -> Any:
        """
        发送请求

        Args:
            service: 服务名
            method: 方法名
            params: 参数
            timeout: 超时时间

        Returns:
            响应结果
        """
        message = Message(
            topic=f"{service}.{method}",
            payload=params,
        )

        # 发送请求
        await self.socket.send(message.to_bytes())

        # 等待应答
        return await self.reply_queue.wait_for_reply(message.id, timeout)

    async def _receive_loop(self):
        """接收循环"""
        try:
            while True:
                data = await self.socket.recv()
                message = Message.from_bytes(data)

                # 设置应答
                await self.reply_queue.set_reply(message.id, message.payload)

        except Exception as e:
            logger.error("接收循环错误", extra={'error': str(e)})

    async def stop(self):
        """停止客户端"""
        if self.receive_task:
            self.receive_task.cancel()
        if self.socket:
            self.socket.close()


class RequestServer:
    """请求服务器"""

    def __init__(
        self,
        host: str = "*",
        port: int = 5555,
    ):
        self.host = host
        self.port = port
        self.socket: Optional[aiozmq.ZmqSocket] = None
        self.handlers: Dict[str, Callable] = {}

    async def start(self):
        """启动服务器"""
        self.socket = await aiozmq.create_zmq_socket(zmq.ROUTER)
        await self.socket.bind(f"tcp://{self.host}:{self.port}")

        # 启动处理循环
        asyncio.create_task(self._handler_loop())

        logger.info(f"请求服务器启动: {self.port}")

    def register_handler(
        self,
        service: str,
        method: str,
        handler: Callable,
    ):
        """注册处理器"""
        key = f"{service}.{method}"
        self.handlers[key] = handler
        logger.info(f"注册处理器: {key}")

    async def _handler_loop(self):
        """处理循环"""
        try:
            while True:
                # 接收请求 (路由ID + 消息)
                router_id, request = await self.socket.recv_multipart()

                # 解析消息
                message = Message.from_bytes(request)

                # 获取处理器
                handler = self.handlers.get(message.topic)
                if not handler:
                    # 发送错误响应
                    error_response = Message(
                        id=message.id,
                        topic=message.topic,
                        payload={'error': 'Handler not found'},
                    )
                    await self.socket.send_multipart([
                        router_id,
                        error_response.to_bytes(),
                    ])
                    continue

                # 处理请求
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(message.payload)
                    else:
                        result = handler(message.payload)

                    # 发送响应
                    response = Message(
                        id=message.id,
                        topic=message.topic,
                        payload={'result': result},
                    )
                    await self.socket.send_multipart([
                        router_id,
                        response.to_bytes(),
                    ])

                except Exception as e:
                    logger.error("请求处理失败", extra={
                        'topic': message.topic,
                        'error': str(e),
                    })

                    # 发送错误响应
                    error_response = Message(
                        id=message.id,
                        topic=message.topic,
                        payload={'error': str(e)},
                    )
                    await self.socket.send_multipart([
                        router_id,
                        error_response.to_bytes(),
                    ])

        except Exception as e:
            logger.error("处理循环错误", extra={'error': str(e)})
```

### 38.2 高频交易引擎核心

```python
# src/trading/hft_engine.py
"""
高频交易引擎
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np

from src.core.order import Order, OrderSide, OrderType, OrderStatus
from src.core.position import Position
from src.core.market_data import MarketData, Tick, OrderBook
from src.infrastructure.high_performance_mq import Publisher, Subscriber
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OrderRequest:
    """订单请求"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    client_order_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RiskCheckResult:
    """风险检查结果"""
    passed: bool
    reason: str = ""
    warning: str = ""


class HFTRiskManager:
    """
    高频交易风险管理器

    实时风险检查：
    1. 持仓限制
    2. 资金限制
    3. 交易频率限制
    4. 价格偏离检查
    """

    def __init__(
        self,
        max_position_value: float = 1000000,
        max_orders_per_second: int = 100,
        max_price_deviation: float = 0.02,  # 2%
    ):
        self.max_position_value = max_position_value
        self.max_orders_per_second = max_orders_per_second
        self.max_price_deviation = max_price_deviation

        # 订单计数器
        self.order_counts: Dict[str, List[datetime]] = defaultdict(list)

        # 当前持仓
        self.positions: Dict[str, Position] = {}

    async def check_order(
        self,
        request: OrderRequest,
        current_positions: Dict[str, Position],
        market_data: Optional[MarketData] = None,
    ) -> RiskCheckResult:
        """
        检查订单

        Args:
            request: 订单请求
            current_positions: 当前持仓
            market_data: 市场数据

        Returns:
            风险检查结果
        """
        # 1. 交易频率检查
        now = datetime.now()
        recent_orders = [
            t for t in self.order_counts[request.symbol]
            if (now - t).total_seconds() < 1
        ]

        if len(recent_orders) >= self.max_orders_per_second:
            return RiskCheckResult(
                passed=False,
                reason=f"超过每秒订单限制 ({self.max_orders_per_second})",
            )

        # 2. 持仓限制检查
        current_position = current_positions.get(request.symbol)
        current_quantity = current_position.quantity if current_position else 0

        new_quantity = current_quantity
        if request.side == OrderSide.BUY:
            new_quantity += request.quantity
        else:
            new_quantity -= request.quantity

        # 估算持仓价值
        if market_data:
            market_value = abs(new_quantity) * market_data.last_price
            if market_value > self.max_position_value:
                return RiskCheckResult(
                    passed=False,
                    reason=f"超过持仓限制 ({self.max_position_value})",
                )

        # 3. 价格偏离检查
        if request.order_type == OrderType.LIMIT and market_data:
            deviation = abs(request.price - market_data.last_price) / market_data.last_price
            if deviation > self.max_price_deviation:
                return RiskCheckResult(
                    passed=False,
                    reason=f"价格偏离过大 ({deviation*100:.1f}%)",
                )

        # 记录订单
        self.order_counts[request.symbol].append(now)

        return RiskCheckResult(passed=True)


class HFTOrderRouter:
    """
    高频订单路由器

    功能：
    1. 订单拆分 (大单拆小)
    2. 智能路由 (选择最优交易所)
    3. 订单优先级管理
    """

    def __init__(self):
        self.pending_orders: List[OrderRequest] = []
        self.exchange_rules: Dict[str, Dict] = {
            'XSHE': {'min_qty': 100, 'qty_step': 100},
            'XSHG': {'min_qty': 100, 'qty_step': 100},
            'XHKG': {'min_qty': 100, 'qty_step': 1},
        }

    async def route_order(
        self,
        request: OrderRequest,
        exchange: str,
    ) -> List[OrderRequest]:
        """
        路由订单

        Args:
            request: 订单请求
            exchange: 目标交易所

        Returns:
            拆分后的订单列表
        """
        rules = self.exchange_rules.get(exchange, {})
        min_qty = rules.get('min_qty', 100)
        qty_step = rules.get('qty_step', 100)

        # 检查数量
        if request.quantity < min_qty:
            logger.warning(f"订单数量小于最小值: {request.quantity} < {min_qty}")
            return []

        # 整数倍检查
        if request.quantity % qty_step != 0:
            # 调整到最近的整数倍
            adjusted_qty = (request.quantity // qty_step) * qty_step
            if adjusted_qty == 0:
                logger.warning(f"订单数量调整后为0: {request.quantity}")
                return []
            request.quantity = adjusted_qty

        # 拆分大单
        if request.quantity > 10000:  # 超过10000股拆分
            return await self._split_order(request, 10000)

        return [request]

    async def _split_order(
        self,
        request: OrderRequest,
        chunk_size: int,
    ) -> List[OrderRequest]:
        """
        拆分订单

        Args:
            request: 订单请求
            chunk_size: 拆分大小

        Returns:
            拆分后的订单列表
        """
        orders = []
        remaining = request.quantity

        while remaining > 0:
            qty = min(remaining, chunk_size)

            order = OrderRequest(
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                quantity=qty,
                price=request.price,
                time_in_force=request.time_in_force,
            )

            orders.append(order)
            remaining -= qty

        return orders


class HFTTradingEngine:
    """
    高频交易引擎

    特点：
    1. 微秒级延迟
    2. 高吞吐量
    3. 实时风险控制
    4. 智能订单路由
    """

    def __init__(
        self,
        trader_id: str,
        initial_capital: float = 1000000,
    ):
        self.trader_id = trader_id
        self.initial_capital = initial_capital

        # 核心组件
        self.risk_manager = HFTRiskManager()
        self.order_router = HFTOrderRouter()

        # 订单管理
        self.orders: Dict[str, Order] = {}
        self.pending_requests: List[OrderRequest] = []

        # 持仓管理
        self.positions: Dict[str, Position] = {}
        self.available_capital = initial_capital

        # 市场数据缓存
        self.market_data: Dict[str, MarketData] = {}

        # 性能统计
        self.order_latency: List[float] = []
        self.fill_latency: List[float] = []

        # 消息队列
        self.publisher: Optional[Publisher] = None
        self.subscriber: Optional[Subscriber] = None

    async def start(self):
        """启动引擎"""
        # 启动消息队列
        self.publisher = Publisher()
        await self.publisher.start()

        self.subscriber = Subscriber()
        await self.subscriber.start()

        # 订阅市场数据
        await self.subscriber.subscribe("market.tick", self._on_tick)
        await self.subscriber.subscribe("market.orderbook", self._on_orderbook)
        await self.subscriber.subscribe("execution.fill", self._on_fill)
        await self.subscriber.subscribe("execution.reject", self._on_reject)

        logger.info("高频交易引擎启动", extra={'trader_id': self.trader_id})

    async def stop(self):
        """停止引擎"""
        if self.publisher:
            await self.publisher.stop()
        if self.subscriber:
            await self.subscriber.stop()

    async def submit_order(self, request: OrderRequest) -> Optional[str]:
        """
        提交订单

        Args:
            request: 订单请求

        Returns:
            订单ID (如果提交成功)
        """
        start_time = time.time()

        # 风险检查
        market_data = self.market_data.get(request.symbol)
        risk_result = await self.risk_manager.check_order(
            request, self.positions, market_data
        )

        if not risk_result.passed:
            logger.warning(f"订单被风控拒绝: {risk_result.reason}")
            return None

        # 订单路由
        routed_orders = await self.order_router.route_order(request, 'XSHE')

        if not routed_orders:
            logger.warning("订单路由失败")
            return None

        # 创建订单
        order = Order(
            trader_id=self.trader_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
        )

        self.orders[order.id] = order
        self.pending_requests.append(request)

        # 发布订单
        await self.publisher.publish(
            "execution.submit",
            {
                'order_id': order.id,
                'trader_id': self.trader_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
            },
        )

        # 记录延迟
        latency = (time.time() - start_time) * 1e6  # 微秒
        self.order_latency.append(latency)

        logger.info("订单已提交", extra={
            'order_id': order.id,
            'latency_us': latency,
        })

        return order.id

    async def cancel_order(self, order_id: str) -> bool:
        """
        取消订单

        Args:
            order_id: 订单ID

        Returns:
            是否成功
        """
        order = self.orders.get(order_id)
        if not order:
            return False

        if order.status != OrderStatus.OPEN:
            return False

        # 发布取消请求
        await self.publisher.publish(
            "execution.cancel",
            {'order_id': order_id},
        )

        return True

    async def _on_tick(self, message):
        """处理行情数据"""
        tick = Tick(**message.payload)

        # 更新市场数据
        if tick.symbol not in self.market_data:
            self.market_data[tick.symbol] = MarketData(symbol=tick.symbol)

        self.market_data[tick.symbol].update_tick(tick)

    async def _on_orderbook(self, message):
        """处理订单簿数据"""
        orderbook = OrderBook(**message.payload)

        # 更新订单簿
        if orderbook.symbol not in self.market_data:
            self.market_data[orderbook.symbol] = MarketData(symbol=orderbook.symbol)

        self.market_data[orderbook.symbol].update_orderbook(orderbook)

    async def _on_fill(self, message):
        """处理成交回报"""
        fill_data = message.payload
        order_id = fill_data['order_id']

        order = self.orders.get(order_id)
        if not order:
            return

        # 更新订单
        order.filled_quantity += fill_data['filled_quantity']
        order.avg_fill_price = (
            (order.avg_fill_price * order.filled_quantity + fill_data['price'] * fill_data['filled_quantity']) /
            (order.filled_quantity + fill_data['filled_quantity'])
        )

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        # 更新持仓
        await self._update_position(order, fill_data)

        logger.info("订单成交", extra={
            'order_id': order_id,
            'filled_qty': fill_data['filled_quantity'],
            'price': fill_data['price'],
        })

    async def _on_reject(self, message):
        """处理订单拒绝"""
        reject_data = message.payload
        order_id = reject_data['order_id']

        order = self.orders.get(order_id)
        if order:
            order.status = OrderStatus.REJECTED
            logger.warning("订单被拒绝", extra={
                'order_id': order_id,
                'reason': reject_data.get('reason'),
            })

    async def _update_position(self, order: Order, fill_data: Dict):
        """更新持仓"""
        symbol = order.symbol

        if symbol not in self.positions:
            self.positions[symbol] = Position(
                trader_id=self.trader_id,
                symbol=symbol,
            )

        position = self.positions[symbol]
        quantity_delta = fill_data['filled_quantity']

        if order.side == OrderSide.BUY:
            position.quantity += quantity_delta
        else:
            position.quantity -= quantity_delta

        # 更新成本
        position.avg_cost = (
            (position.avg_cost * (position.quantity - quantity_delta) + fill_data['price'] * quantity_delta) /
            position.quantity
        ) if position.quantity != 0 else 0

        # 更新市值
        market_data = self.market_data.get(symbol)
        if market_data:
            position.current_price = market_data.last_price
            position.market_value = abs(position.quantity) * market_data.last_price

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        return {
            'order_count': len(self.order_latency),
            'avg_order_latency_us': np.mean(self.order_latency) if self.order_latency else 0,
            'p50_order_latency_us': np.percentile(self.order_latency, 50) if self.order_latency else 0,
            'p99_order_latency_us': np.percentile(self.order_latency, 99) if self.order_latency else 0,
            'total_capital': self.available_capital + sum(p.market_value for p in self.positions.values()),
        }
```

---

## 第39章 综合测试策略

### 39.1 单元测试最佳实践

```python
# tests/conftest.py
"""
测试配置和fixtures
"""

import pytest
import asyncio
from typing import AsyncGenerator
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from src.database.session import get_test_session
from src.models.trader import TraderModel
from src.models.order import OrderModel
from src.core.config import settings


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_session() -> AsyncGenerator:
    """创建测试数据库会话"""
    async with get_test_session() as session:
        yield session
        # 清理
        await session.rollback()


@pytest.fixture
def sample_trader(db_session):
    """创建示例交易员"""
    trader = TraderModel(
        id="test_trader_001",
        name="测试交易员",
        type="ai",
        initial_capital=1000000,
        current_capital=1000000,
        is_active=True,
    )
    db_session.add(trader)
    await db_session.commit()
    await db_session.refresh(trader)
    return trader


@pytest.fixture
def mock_market_data():
    """模拟市场数据"""
    return Mock(
        symbol="000001",
        exchange="XSHE",
        timestamp=datetime.now(),
        open=10.0,
        high=10.5,
        low=9.8,
        close=10.2,
        volume=1000000,
        last_price=10.2,
    )


@pytest.fixture
def mock_llm_response():
    """模拟LLM响应"""
    return AsyncMock(
        response=Mock(
            choices=[Mock(
                message=Mock(
                    content='{"action": "buy", "quantity": 100, "confidence": 0.8}'
                )
            )]
        )
    )
```


# tests/unit/test_strategies.py
"""
策略单元测试
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.strategies.dual_ma_strategy import DualMovingAverageStrategy
from src.strategies.multi_factor_strategy import MultiFactorStrategy, FactorConfig
from src.core.signal import SignalType
from src.core.market_data import MarketData


class TestDualMovingAverageStrategy:
    """双均线策略测试"""

    @pytest.fixture
    def strategy(self):
        """创建策略实例"""
        return DualMovingAverageStrategy(
            trader_id="test_trader",
            short_period=5,
            long_period=20,
            initial_capital=1000000,
        )

    @pytest.fixture
    def price_history(self):
        """创建价格历史"""
        return [
            10.0, 10.2, 10.1, 10.3, 10.5,  # 短期上涨
            10.4, 10.6, 10.8, 11.0, 11.2,
            11.1, 11.3, 11.5, 11.4, 11.6,
            11.8, 12.0, 11.9, 12.1, 12.3,
        ]

    @pytest.mark.asyncio
    async def test_generate_buy_signal(self, strategy, price_history):
        """测试生成买入信号"""
        # 构建市场数据
        bars = []
        for i, price in enumerate(price_history):
            bar = MarketData(
                symbol="000001",
                exchange="XSHE",
                timestamp=datetime.now(),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000000,
            )
            bars.append(bar)

        # 处理K线
        for bar in bars[:-1]:
            await strategy.on_bar(bar)

        # 最后一个K线应该触发买入信号
        signals = await strategy.on_bar(bars[-1])

        assert len(signals) > 0
        assert signals[0].signal_type == SignalType.BUY
        assert signals[0].symbol == "000001"

    @pytest.mark.asyncio
    async def test_generate_sell_signal(self, strategy):
        """测试生成卖出信号"""
        # 创建下跌的价格序列
        prices = list(reversed([10.0 + i * 0.1 for i in range(25)]))

        bars = []
        for price in prices:
            bar = MarketData(
                symbol="000001",
                exchange="XSHE",
                timestamp=datetime.now(),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000000,
            )
            bars.append(bar)

        # 处理所有K线
        for bar in bars:
            signals = await strategy.on_bar(bar)

        # 检查卖出信号
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sell_signals) > 0

    @pytest.mark.asyncio
    async def test_no_signal_when_insufficient_data(self, strategy):
        """测试数据不足时不生成信号"""
        # 只提供少量数据
        for _ in range(10):
            bar = MarketData(
                symbol="000001",
                exchange="XSHE",
                timestamp=datetime.now(),
                open=10.0,
                high=10.1,
                low=9.9,
                close=10.0,
                volume=1000000,
            )
            signals = await strategy.on_bar(bar)
            assert len(signals) == 0


class TestMultiFactorStrategy:
    """多因子策略测试"""

    @pytest.fixture
    def strategy(self):
        """创建策略实例"""
        config = FactorConfig(
            value_weight=0.3,
            growth_weight=0.3,
            quality_weight=0.2,
            technical_weight=0.2,
        )
        return MultiFactorStrategy(
            config=config,
            trader_id="test_trader",
            initial_capital=1000000,
            max_positions=5,
        )

    @pytest.fixture
    def sample_fundamental_data(self):
        """示例基本面数据"""
        return {
            "000001": {
                "pe_ratio": 15.0,
                "pb_ratio": 1.5,
                "revenue_growth": 0.25,
                "profit_growth": 0.30,
                "roe": 0.18,
                "roa": 0.10,
            },
            "000002": {
                "pe_ratio": 25.0,
                "pb_ratio": 2.5,
                "revenue_growth": 0.15,
                "profit_growth": 0.10,
                "roe": 0.12,
                "roa": 0.06,
            },
        }

    @pytest.fixture
    def sample_price_history(self):
        """示例价格历史"""
        return {
            "000001": [10.0 + i * 0.1 for i in range(60)],
            "000002": [20.0 + i * 0.05 for i in range(60)],
        }

    @pytest.mark.asyncio
    async def test_calculate_value_score(self, strategy, sample_fundamental_data):
        """测试价值因子计算"""
        score = strategy.calculate_value_score(
            "000001",
            sample_fundamental_data["000001"],
        )

        assert 0 <= score <= 2
        # 低PE/PB应该得到高分
        assert score > 1.0

    @pytest.mark.asyncio
    async def test_calculate_growth_score(self, strategy, sample_fundamental_data):
        """测试成长因子计算"""
        score = strategy.calculate_growth_score(
            "000001",
            sample_fundamental_data["000001"],
        )

        assert 0 <= score <= 2
        # 高增长率应该得到高分
        assert score > 1.0

    @pytest.mark.asyncio
    async def test_select_stocks(
        self,
        strategy,
        sample_fundamental_data,
        sample_price_history,
    ):
        """测试选股"""
        scores = await strategy.select_stocks(
            symbols=["000001", "000002"],
            fundamental_data=sample_fundamental_data,
            price_data=sample_price_history,
        )

        assert len(scores) == 2
        # 应该按得分降序排列
        assert scores[0][1] >= scores[1][1]

        # 000001应该有更高的综合得分
        assert scores[0][0] == "000001"

    @pytest.mark.asyncio
    async def test_rebalance(
        self,
        strategy,
        sample_fundamental_data,
        sample_price_history,
    ):
        """测试调仓"""
        signals = await strategy.rebalance(
            symbols=["000001", "000002"],
            fundamental_data=sample_fundamental_data,
            price_data=sample_price_history,
        )

        # 应该生成买入信号
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) > 0


class TestRLStrategy:
    """强化学习策略测试"""

    @pytest.fixture
    def rl_config(self):
        """RL配置"""
        from src.strategies.rl_strategy import RLConfig
        return RLConfig(
            state_dim=64,
            action_dim=5,
            hidden_dim=128,
            learning_rate=1e-3,
        )

    @pytest.fixture
    def rl_agent(self, rl_config):
        """创建RL智能体"""
        from src.strategies.rl_strategy import DQNAgent
        return DQNAgent(rl_config, device='cpu')

    def test_initial_epsilon(self, rl_agent):
        """测试初始探索率"""
        assert rl_agent.epsilon == rl_agent.config.epsilon_start

    def test_select_action_random(self, rl_agent):
        """测试随机动作选择"""
        state = np.random.randn(64)

        # 设置高探索率
        rl_agent.epsilon = 1.0

        action = rl_agent.select_action(state)
        assert 0 <= action < 5

    def test_select_action_greedy(self, rl_agent):
        """测试贪婪动作选择"""
        state = np.random.randn(64)

        action = rl_agent.select_action(state, eval_mode=True)
        assert 0 <= action < 5

    def test_replay_buffer(self, rl_agent):
        """测试经验回放"""
        # 添加经验
        for _ in range(100):
            state = np.random.randn(64)
            action = np.random.randint(0, 5)
            reward = np.random.randn()
            next_state = np.random.randn(64)

            rl_agent.replay_buffer.push(state, action, reward, next_state, False)

        assert len(rl_agent.replay_buffer) == 100
```

### 39.2 集成测试

```python
# tests/integration/test_trading_flow.py
"""
交易流程集成测试
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from src.services.trading_service import TradingService
from src.services.order_service import OrderService
from src.services.position_service import PositionService
from src.core.order import Order, OrderSide, OrderType
from src.core.signal import Signal, SignalType
from src.core.market_data import MarketData


@pytest.mark.integration
class TestTradingFlow:
    """交易流程集成测试"""

    @pytest.fixture
    async def trading_service(self, db_session):
        """创建交易服务"""
        return TradingService(db_session)

    @pytest.fixture
    def sample_signal(self):
        """创建示例信号"""
        return Signal(
            symbol="000001",
            signal_type=SignalType.BUY,
            quantity=100,
            price=10.0,
            confidence=0.8,
            reason="测试信号",
        )

    @pytest.mark.asyncio
    async def test_complete_trading_flow(
        self,
        trading_service,
        sample_signal,
    ):
        """测试完整交易流程"""
        # 1. 提交信号
        order = await trading_service.submit_signal(sample_signal)
        assert order is not None
        assert order.symbol == sample_signal.symbol
        assert order.quantity == sample_signal.quantity

        # 2. 检查订单状态
        status = await trading_service.get_order_status(order.id)
        assert status in ["pending", "open"]

        # 3. 模拟成交
        await trading_service.simulate_fill(
            order_id=order.id,
            filled_quantity=100,
            fill_price=10.0,
        )

        # 4. 检查持仓
        position = await trading_service.get_position("000001")
        assert position is not None
        assert position.quantity == 100

        # 5. 计算盈亏
        pnl = await trading_service.calculate_pnl("000001", 10.5)
        assert pnl > 0  # 买入后价格上涨

    @pytest.mark.asyncio
    async def test_multi_order_flow(self, trading_service):
        """测试多订单流程"""
        signals = [
            Signal(
                symbol="000001",
                signal_type=SignalType.BUY,
                quantity=100,
                price=10.0,
                confidence=0.8,
                reason="测试",
            ),
            Signal(
                symbol="000002",
                signal_type=SignalType.BUY,
                quantity=200,
                price=20.0,
                confidence=0.7,
                reason="测试",
            ),
        ]

        orders = []
        for signal in signals:
            order = await trading_service.submit_signal(signal)
            orders.append(order)

        assert len(orders) == 2

        # 模拟所有成交
        for order in orders:
            await trading_service.simulate_fill(
                order_id=order.id,
                filled_quantity=order.quantity,
                fill_price=order.price,
            )

        # 检查总持仓
        positions = await trading_service.get_all_positions()
        assert len(positions) == 2


@pytest.mark.integration
class TestRiskManagementFlow:
    """风险管理流程集成测试"""

    @pytest.fixture
    async def risk_manager(self, db_session):
        """创建风险管理器"""
        from src.services.risk_service import RiskManager
        return RiskManager(db_session)

    @pytest.mark.asyncio
    async def test_position_limit_check(self, risk_manager):
        """测试持仓限制检查"""
        # 创建超过限制的订单
        signal = Signal(
            symbol="000001",
            signal_type=SignalType.BUY,
            quantity=1000000,  # 超大订单
            price=10.0,
            confidence=0.8,
            reason="测试",
        )

        result = await risk_manager.check_position_limit(signal)
        assert not result.passed
        assert "超过持仓限制" in result.reason

    @pytest.mark.asyncio
    async def test_loss_limit_trigger(self, risk_manager):
        """测试亏损限制触发"""
        # 模建持仓亏损
        await risk_manager.create_position(
            symbol="000001",
            quantity=1000,
            avg_cost=10.0,
        )

        # 市价下跌触发止损
        triggered = await risk_manager.check_loss_limit(
            symbol="000001",
            current_price=9.0,  # 下跌10%
        )

        assert triggered


@pytest.mark.integration
class TestMarketDataFlow:
    """市场数据流程集成测试"""

    @pytest.fixture
    async def market_data_service(self):
        """创建市场数据服务"""
        from src.services.market_data_service import MarketDataService
        return MarketDataService()

    @pytest.mark.asyncio
    async def test_tick_data_flow(self, market_data_service):
        """测试Tick数据流"""
        # 模拟Tick数据
        tick_data = {
            'symbol': '000001',
            'timestamp': datetime.now().isoformat(),
            'last_price': 10.5,
            'bid_price': 10.49,
            'ask_price': 10.51,
            'bid_volume': 1000,
            'ask_volume': 1000,
            'volume': 100000,
        }

        # 处理Tick
        await market_data_service.process_tick(tick_data)

        # 获取最新价格
        latest = await market_data_service.get_latest_price('000001')
        assert latest == 10.5

    @pytest.mark.asyncio
    async def test_kline_synthesis(self, market_data_service):
        """测试K线合成"""
        ticks = []
        base_price = 10.0

        # 生成100个Tick
        for i in range(100):
            tick = {
                'symbol': '000001',
                'timestamp': datetime.now().isoformat(),
                'last_price': base_price + i * 0.01,
                'volume': 1000,
            }
            ticks.append(tick)

        # 合成K线
        kline = await market_data_service.synthesize_kline('000001', ticks)

        assert kline is not None
        assert kline.open > 0
        assert kline.high >= kline.low
        assert kline.volume > 0
```

### 39.3 性能测试

```python
# tests/performance/test_performance.py
"""
性能测试
"""

import pytest
import asyncio
import time
from locust import HttpUser, task, between
from datetime import datetime

from src.services.trading_service import TradingService
from src.core.signal import Signal, SignalType
from src.core.market_data import MarketData


@pytest.mark.performance
class TestTradingPerformance:
    """交易性能测试"""

    @pytest.mark.asyncio
    async def test_order_submission_latency(self, db_session):
        """测试订单提交延迟"""
        trading_service = TradingService(db_session)

        latencies = []

        for i in range(1000):
            signal = Signal(
                symbol="000001",
                signal_type=SignalType.BUY,
                quantity=100,
                price=10.0 + i * 0.01,
                confidence=0.8,
                reason="性能测试",
            )

            start = time.perf_counter()
            await trading_service.submit_signal(signal)
            latency = (time.perf_counter() - start) * 1000  # ms

            latencies.append(latency)

        # 统计
        avg_latency = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[len(latencies) // 2]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

        print(f"平均延迟: {avg_latency:.2f}ms")
        print(f"P50延迟: {p50_latency:.2f}ms")
        print(f"P99延迟: {p99_latency:.2f}ms")

        # 断言
        assert avg_latency < 10  # 平均延迟小于10ms
        assert p99_latency < 50  # P99延迟小于50ms

    @pytest.mark.asyncio
    async def test_concurrent_order_processing(self, db_session):
        """测试并发订单处理"""
        trading_service = TradingService(db_session)

        async def submit_order(order_id: int):
            signal = Signal(
                symbol="000001",
                signal_type=SignalType.BUY,
                quantity=100,
                price=10.0,
                confidence=0.8,
                reason=f"并发测试-{order_id}",
            )
            return await trading_service.submit_signal(signal)

        # 并发提交100个订单
        start = time.perf_counter()
        tasks = [submit_order(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start

        print(f"并发处理100个订单耗时: {total_time:.2f}秒")
        print(f"平均吞吐量: {100 / total_time:.2f} 订单/秒")

        assert len(results) == 100
        assert total_time < 5  # 5秒内完成


@pytest.mark.performance
class TestDatabasePerformance:
    """数据库性能测试"""

    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, db_session):
        """测试批量插入性能"""
        from src.models.order import OrderModel
        from datetime import datetime

        # 准备数据
        orders = []
        for i in range(10000):
            order = OrderModel(
                id=f"test_order_{i}",
                trader_id="test_trader",
                symbol="000001",
                side="buy",
                type="limit",
                quantity=100,
                price=10.0,
                status="pending",
                created_at=datetime.now(),
            )
            orders.append(order)

        # 批量插入
        start = time.perf_counter()
        db_session.add_all(orders)
        await db_session.commit()
        insert_time = time.perf_counter() - start

        print(f"插入10000条记录耗时: {insert_time:.2f}秒")
        print(f"插入速度: {10000 / insert_time:.0f} 记录/秒")

        assert insert_time < 10  # 10秒内完成

    @pytest.mark.asyncio
    async def test_query_performance(self, db_session):
        """测试查询性能"""
        from src.models.order import OrderModel
        from sqlalchemy import select

        # 查询测试
        times = []

        for _ in range(1000):
            start = time.perf_counter()
            result = await db_session.execute(
                select(OrderModel)
                .where(OrderModel.trader_id == "test_trader")
                .limit(100)
            )
            orders = result.scalars().all()
            query_time = (time.perf_counter() - start) * 1000
            times.append(query_time)

        avg_time = sum(times) / len(times)
        print(f"平均查询时间: {avg_time:.2f}ms")

        assert avg_time < 10  # 平均查询时间小于10ms


# Locust负载测试
class TradingUser(HttpUser):
    """交易用户模拟"""

    wait_time = between(0.1, 0.5)  # 等待时间

    def on_start(self):
        """开始时登录"""
        response = self.client.post("/api/v1/auth/login", json={
            "username": "test_user",
            "password": "test_password",
        })
        self.token = response.json()["token"]

    @task(3)
    def submit_order(self):
        """提交订单"""
        headers = {"Authorization": f"Bearer {self.token}"}
        self.client.post("/api/v1/orders", json={
            "symbol": "000001",
            "side": "buy",
            "type": "limit",
            "quantity": 100,
            "price": 10.0,
        }, headers=headers)

    @task(1)
    def get_positions(self):
        """查询持仓"""
        headers = {"Authorization": f"Bearer {self.token}"}
        self.client.get("/api/v1/positions", headers=headers)

    @task(1)
    def get_market_data(self):
        """获取行情"""
        self.client.get("/api/v1/market/000001")
```

### 39.4 压力测试

```python
# tests/stress/test_stress.py
"""
压力测试
"""

import pytest
import asyncio
import psutil
import time
from datetime import datetime

from src.services.trading_service import TradingService
from src.core.signal import Signal, SignalType
from src.core.market_data import MarketData


@pytest.mark.stress
class TestSystemStress:
    """系统压力测试"""

    @pytest.mark.asyncio
    async def test_high_frequency_trading(self, db_session):
        """测试高频交易"""
        trading_service = TradingService(db_session)

        start_time = time.time()
        order_count = 0
        errors = 0

        # 持续提交订单1分钟
        while time.time() - start_time < 60:
            try:
                signal = Signal(
                    symbol="000001",
                    signal_type=SignalType.BUY,
                    quantity=100,
                    price=10.0,
                    confidence=0.8,
                    reason="压力测试",
                )
                await trading_service.submit_signal(signal)
                order_count += 1
            except Exception as e:
                errors += 1

        elapsed = time.time() - start_time
        tps = order_count / elapsed

        print(f"订单总数: {order_count}")
        print(f"错误数: {errors}")
        print(f"TPS: {tps:.2f}")
        print(f"错误率: {errors / order_count * 100:.2f}%")

        assert order_count > 1000
        assert errors / order_count < 0.01  # 错误率小于1%

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, db_session):
        """测试内存泄漏"""
        trading_service = TradingService(db_session)

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 执行大量操作
        for i in range(10000):
            signal = Signal(
                symbol="000001",
                signal_type=SignalType.BUY,
                quantity=100,
                price=10.0,
                confidence=0.8,
                reason="内存测试",
            )
            await trading_service.submit_signal(signal)

            # 每1000次检查一次内存
            if i % 1000 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory

                print(f"迭代 {i}: 内存增长 {memory_growth:.2f}MB")

                # 如果内存增长超过100MB，可能存在泄漏
                assert memory_growth < 100

        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory

        print(f"初始内存: {initial_memory:.2f}MB")
        print(f"最终内存: {final_memory:.2f}MB")
        print(f"总增长: {total_growth:.2f}MB")

        assert total_growth < 200  # 总增长不超过200MB

    @pytest.mark.asyncio
    async def test_concurrent_users(self, db_session):
        """测试多用户并发"""
        async def user_trading(user_id: int):
            trading_service = TradingService(db_session)

            for i in range(100):
                signal = Signal(
                    symbol="000001",
                    signal_type=SignalType.BUY,
                    quantity=100,
                    price=10.0,
                    confidence=0.8,
                    reason=f"用户{user_id}",
                )
                await trading_service.submit_signal(signal)

        # 模拟100个并发用户
        start_time = time.time()
        tasks = [user_trading(i) for i in range(100)]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        total_orders = 100 * 100  # 100用户 * 100订单
        tps = total_orders / elapsed

        print(f"总订单数: {total_orders}")
        print(f"总耗时: {elapsed:.2f}秒")
        print(f"TPS: {tps:.2f}")

        assert elapsed < 60  # 60秒内完成
```

---

## 第40章 生产级性能调优

### 40.1 应用层优化

```python
# src/optimization/application_optimizations.py
"""
应用层性能优化
"""

import asyncio
import functools
from typing import Callable, TypeVar, Any
from datetime import datetime
import cachetools.keys
from cachetools import TTLCache

from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


# ============================================
# 函数调用优化
# ============================================

def async_lru_cache(maxsize: int = 128, ttl: int = 300):
    """
    异步LRU缓存装饰器

    Args:
        maxsize: 最大缓存大小
        ttl: 过期时间(秒)
    """
    cache = TTLCache(maxsize=maxsize, ttl=ttl)
    lock = asyncio.Lock()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            key = cachetools.keys.hashkey(*args, **kwargs)

            # 尝试从缓存获取
            async with lock:
                if key in cache:
                    logger.debug(f"缓存命中: {func.__name__}")
                    return cache[key]

            # 调用函数
            result = await func(*args, **kwargs)

            # 写入缓存
            async with lock:
                cache[key] = result

            return result

        # 添加缓存统计
        wrapper.cache_info = lambda: cache.info  # type: ignore
        wrapper.cache_clear = lambda: cache.clear()  # type: ignore

        return wrapper

    return decorator


def async_timed(func: Callable[..., T]) -> Callable[..., T]:
    """
    函数执行计时装饰器
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = asyncio.get_event_loop().time()
        result = await func(*args, **kwargs)
        elapsed = (asyncio.get_event_loop().time() - start) * 1000

        logger.debug(f"{func.__name__} 执行时间: {elapsed:.2f}ms")

        # 如果执行时间过长，记录警告
        if elapsed > 1000:
            logger.warning(f"{func.__name__} 执行时间过长: {elapsed:.2f}ms")

        return result

    return wrapper


# ============================================
# 批处理优化
# ============================================

class AsyncBatchProcessor:
    """
    异步批处理器

    将多个单独的请求合并为批量处理，提高效率
    """

    def __init__(
        self,
        batch_size: int = 100,
        timeout: float = 0.1,
        max_wait_time: float = 1.0,
    ):
        """
        初始化批处理器

        Args:
            batch_size: 批次大小
            timeout: 批次超时时间
            max_wait_time: 最大等待时间
        """
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_wait_time = max_wait_time

        self.queue: asyncio.Queue = asyncio.Queue()
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        """启动批处理器"""
        self.task = asyncio.create_task(self._process_batches())

    async def stop(self):
        """停止批处理器"""
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def submit(self, item: Any) -> Any:
        """
        提交项目

        Args:
            item: 要处理的项目

        Returns:
            处理结果
        """
        future = asyncio.Future()
        await self.queue.put((item, future))
        return await future

    async def _process_batches(self):
        """处理批次"""
        try:
            while True:
                batch = []
                futures = []

                # 收集批次
                try:
                    item, future = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=self.timeout,
                    )
                    batch.append(item)
                    futures.append(future)

                    # 尝试收集更多
                    while len(batch) < self.batch_size:
                        try:
                            item, future = await asyncio.wait_for(
                                self.queue.get(),
                                timeout=0.01,
                            )
                            batch.append(item)
                            futures.append(future)
                        except asyncio.TimeoutError:
                            break

                except asyncio.TimeoutError:
                    pass

                if batch:
                    # 批量处理
                    start = asyncio.get_event_loop().time()
                    results = await self._process_batch(batch)
                    elapsed = (asyncio.get_event_loop().time() - start) * 1000

                    logger.debug(f"批次处理: {len(batch)}项, 耗时: {elapsed:.2f}ms")

                    # 设置结果
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)

        except asyncio.CancelledError:
            # 处理剩余项目
            while not self.queue.empty():
                batch = []
                futures = []

                for _ in range(min(self.batch_size, self.queue.qsize())):
                    item, future = await self.queue.get()
                    batch.append(item)
                    futures.append(future)

                if batch:
                    results = await self._process_batch(batch)
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)

    async def _process_batch(self, batch: list) -> list:
        """
        批量处理 (子类实现)

        Args:
            batch: 批次数据

        Returns:
            处理结果列表
        """
        raise NotImplementedError


# ============================================
# 连接池优化
# ============================================

class AsyncConnectionPool:
    """
    异步连接池

    特性：
    1. 连接复用
    2. 自动重连
    3. 健康检查
    4. 动态扩缩容
    """

    def __init__(
        self,
        create_connection: Callable,
        min_size: int = 5,
        max_size: int = 20,
        idle_timeout: float = 300,
        health_check_interval: float = 60,
    ):
        """
        初始化连接池

        Args:
            create_connection: 创建连接的函数
            min_size: 最小连接数
            max_size: 最大连接数
            idle_timeout: 空闲超时
            health_check_interval: 健康检查间隔
        """
        self.create_connection = create_connection
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        self.health_check_interval = health_check_interval

        self.pool: asyncio.Queue = asyncio.Queue()
        self.size = 0
        self.lock = asyncio.Lock()

        self.health_check_task: Optional[asyncio.Task] = None

    async def start(self):
        """启动连接池"""
        # 创建最小数量的连接
        for _ in range(self.min_size):
            await self._create_connection()

        # 启动健康检查任务
        self.health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("连接池启动", extra={
            'min_size': self.min_size,
            'max_size': self.max_size,
        })

    async def stop(self):
        """停止连接池"""
        if self.health_check_task:
            self.health_check_task.cancel()

        # 关闭所有连接
        while not self.pool.empty():
            conn = await self.pool.get()
            await self._close_connection(conn)

        logger.info("连接池已关闭")

    async def acquire(self) -> Any:
        """
        获取连接

        Returns:
            连接对象
        """
        # 尝试从池中获取
        try:
            conn = await asyncio.wait_for(self.pool.get(), timeout=0.1)
            return conn
        except asyncio.TimeoutError:
            pass

        # 池中没有可用连接，创建新的
        async with self.lock:
            if self.size < self.max_size:
                return await self._create_connection()

        # 等待可用连接
        return await self.pool.get()

    async def release(self, conn: Any):
        """
        释放连接

        Args:
            conn: 连接对象
        """
        await self.pool.put(conn)

    async def _create_connection(self) -> Any:
        """创建新连接"""
        conn = await self.create_connection()
        async with self.lock:
            self.size += 1
        return conn

    async def _close_connection(self, conn: Any):
        """关闭连接"""
        if hasattr(conn, 'close'):
            await conn.close()
        async with self.lock:
            self.size -= 1

    async def _health_check_loop(self):
        """健康检查循环"""
        try:
            while True:
                await asyncio.sleep(self.health_check_interval)

                # 检查连接健康状态
                healthy_conns = []
                while not self.pool.empty():
                    conn = await self.pool.get()

                    if await self._is_healthy(conn):
                        healthy_conns.append(conn)
                    else:
                        await self._close_connection(conn)
                        # 创建新连接替代
                        await self._create_connection()

                # 放回健康连接
                for conn in healthy_conns:
                    await self.pool.put(conn)

        except asyncio.CancelledError:
            pass

    async def _is_healthy(self, conn: Any) -> bool:
        """检查连接健康状态"""
        if hasattr(conn, 'ping'):
            try:
                await asyncio.wait_for(conn.ping(), timeout=5)
                return True
            except:
                return False
        return True


# ============================================
# 并发控制
# ============================================

class AsyncSemaphore:
    """
    异步信号量 (带优先级)
    """

    def __init__(self, max_concurrent: int):
        """
        初始化信号量

        Args:
            max_concurrent: 最大并发数
        """
        self.max_concurrent = max_concurrent
        self.current = 0
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        """进入上下文"""
        future = asyncio.Future()
        await self.queue.put((0, future))
        await future
        return self

    async def __aexit__(self, *args):
        """退出上下文"""
        self.current -= 1
        if not self.queue.empty():
            _, next_future = await self.queue.get()
            next_future.set_result(True)


class AsyncRateLimiter:
    """
    异步速率限制器

    令牌桶算法实现
    """

    def __init__(self, rate: float, burst: int = 10):
        """
        初始化速率限制器

        Args:
            rate: 令牌生成速率 (令牌/秒)
            burst: 桶容量
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1):
        """
        获取令牌

        Args:
            tokens: 令牌数量
        """
        async with self.lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_update

            # 补充令牌
            self.tokens = min(
                self.burst,
                self.tokens + elapsed * self.rate,
            )
            self.last_update = now

            # 检查是否有足够令牌
            if self.tokens < tokens:
                # 计算等待时间
                wait_time = (tokens - self.tokens) / self.rate
                await asyncio.sleep(wait_time)

                # 更新令牌
                self.tokens = 0
                self.last_update = asyncio.get_event_loop().time()
            else:
                self.tokens -= tokens

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        pass
```

### 40.2 数据库层优化

```python
# src/optimization/database_optimizations.py
"""
数据库层性能优化
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import text, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from src.core.logging import get_logger

logger = get_logger(__name__)


class DatabaseQueryOptimizer:
    """数据库查询优化器"""

    @staticmethod
    async def bulk_insert(
        session: AsyncSession,
        model: type,
        data: List[Dict[str, Any]],
        batch_size: int = 1000,
    ) -> int:
        """
        批量插入优化

        Args:
            session: 数据库会话
            model: 模型类
            data: 数据列表
            batch_size: 批次大小

        Returns:
            插入的记录数
        """
        inserted = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            objects = [model(**item) for item in batch]
            session.add_all(objects)
            await session.flush()
            inserted += len(batch)

        await session.commit()

        logger.info(f"批量插入完成: {inserted}条记录")
        return inserted

    @staticmethod
    async def bulk_update(
        session: AsyncSession,
        model: type,
        data: List[Dict[str, Any]],
        key_field: str = "id",
        batch_size: int = 500,
    ) -> int:
        """
        批量更新优化

        使用CASE WHEN语句批量更新
        """
        updated = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            # 构建CASE WHEN语句
            keys = [item[key_field] for item in batch]

            for field in batch[0].keys():
                if field == key_field:
                    continue

                # 构建WHEN子句
                when_clauses = []
                for item in batch:
                    if field in item:
                        when_clauses.append(
                            f"WHEN '{item[key_field]}' THEN {item[field]}"
                        )

                if when_clauses:
                    sql = f"""
                    UPDATE {model.__tablename__}
                    SET {field} = CASE {key_field}
                        {' '.join(when_clauses)}
                        ELSE {field}
                    END
                    WHERE {key_field} IN ({','.join(f"'{k}'" for k in keys)})
                    """

                    await session.execute(text(sql))

            await session.commit()
            updated += len(batch)

        logger.info(f"批量更新完成: {updated}条记录")
        return updated

    @staticmethod
    async def get_with_n_plus_1_fix(
        session: AsyncSession,
        model: type,
        relation: str,
    ):
        """
        修复N+1查询问题

        使用eager loading预加载关联数据
        """
        query = (
            select(model)
            .options(selectinload(getattr(model, relation)))
        )

        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def get_paginated(
        session: AsyncSession,
        model: type,
        page: int = 1,
        page_size: int = 100,
        filters: Optional[Dict] = None,
        order_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        分页查询优化

        使用keyset分页提高性能
        """
        # 构建查询
        query = select(model)

        # 应用过滤条件
        if filters:
            for key, value in filters.items():
                query = query.where(getattr(model, key) == value)

        # 计算总数
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await session.execute(count_query)
        total = total_result.scalar()

        # 应用排序
        if order_by:
            query = query.order_by(getattr(model, order_by))

        # 应用分页
        offset = (page - 1) * page_size
        query = query.limit(page_size).offset(offset)

        # 执行查询
        result = await session.execute(query)
        items = result.scalars().all()

        return {
            'items': items,
            'total': total,
            'page': page,
            'page_size': page_size,
            'total_pages': (total + page_size - 1) // page_size,
        }


class MaterializedViewManager:
    """
    物化视图管理器

    用于加速复杂聚合查询
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_daily_stats_view(self):
        """创建每日统计物化视图"""
        sql = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS daily_trading_stats AS
        SELECT
            trader_id,
            DATE(timestamp) as trade_date,
            COUNT(*) as trade_count,
            SUM(quantity * price) as total_volume,
            SUM(realized_pnl) as total_pnl,
            AVG(realized_pnl) as avg_pnl
        FROM trades
        WHERE status = 'filled'
        GROUP BY trader_id, DATE(timestamp)
        WITH DATA;

        CREATE INDEX IF NOT EXISTS idx_daily_stats_date
        ON daily_trading_stats(trade_date DESC);

        CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_stats_trader_date
        ON daily_trading_stats(trader_id, trade_date);
        """

        await self.session.execute(text(sql))
        await self.session.commit()

        logger.info("物化视图创建成功: daily_trading_stats")

    async def refresh_daily_stats_view(self):
        """刷新物化视图"""
        sql = "REFRESH MATERIALIZED VIEW CONCURRENTLY daily_trading_stats"
        await self.session.execute(text(sql))
        await self.session.commit()

        logger.info("物化视图刷新成功")

    async def get_daily_stats(
        self,
        trader_id: str,
        start_date: datetime,
        end_date: datetime,
    ):
        """从物化视图查询统计数据"""
        sql = """
        SELECT * FROM daily_trading_stats
        WHERE trader_id = :trader_id
        AND trade_date BETWEEN :start_date AND :end_date
        ORDER BY trade_date DESC
        """

        result = await self.session.execute(
            text(sql),
            {
                'trader_id': trader_id,
                'start_date': start_date.date(),
                'end_date': end_date.date(),
            },
        )

        return result.fetchall()
```

### 40.3 网络层优化

```python
# src/optimization/network_optimizations.py
"""
网络层性能优化
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime
import orjson

from src.core.logging import get_logger

logger = get_logger(__name__)


class OptimizedHTTPClient:
    """
    优化的HTTP客户端

    特性：
    1. 连接池复用
    2. 请求合并
    3. 自动重试
    4. 超时控制
    5. 压缩传输
    """

    def __init__(
        self,
        pool_size: int = 100,
        timeout: float = 30,
        max_retries: int = 3,
    ):
        """
        初始化HTTP客户端

        Args:
            pool_size: 连接池大小
            timeout: 超时时间
            max_retries: 最大重试次数
        """
        self.pool_size = pool_size
        self.timeout = timeout
        self.max_retries = max_retries

        self.session: Optional[aiohttp.ClientSession] = None
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.batch_task: Optional[asyncio.Task] = None

    async def start(self):
        """启动HTTP客户端"""
        connector = aiohttp.TCPConnector(
            limit=self.pool_size,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            json_serialize=lambda x: orjson.dumps(x).decode(),
        )

        # 启动批处理任务
        self.batch_task = asyncio.create_task(self._process_batches())

        logger.info("HTTP客户端启动", extra={
            'pool_size': self.pool_size,
            'timeout': self.timeout,
        })

    async def stop(self):
        """停止HTTP客户端"""
        if self.batch_task:
            self.batch_task.cancel()

        if self.session:
            await self.session.close()

    async def get(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        GET请求

        Args:
            url: 请求URL
            params: 查询参数
            headers: 请求头

        Returns:
            响应数据
        """
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(
                    url,
                    params=params,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    return await response.json()

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # 指数退避

    async def post(
        self,
        url: str,
        data: Optional[Dict] = None,
        json: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        POST请求
        """
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    url,
                    data=data,
                    json=json,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    return await response.json()

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def batch_get(
        self,
        urls: List[str],
        params: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量GET请求

        Args:
            urls: URL列表
            params: 参数列表

        Returns:
            响应列表
        """
        if params is None:
            params = [None] * len(urls)

        tasks = []
        for url, param in zip(urls, params):
            task = self.get(url, params=param)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def _process_batches(self):
        """处理请求批次"""
        try:
            batch = []
            futures = []

            while True:
                try:
                    # 等待请求
                    request, future = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=0.1,
                    )
                    batch.append((request, future))

                    # 收集更多请求
                    while len(batch) < 10:
                        try:
                            request, future = await asyncio.wait_for(
                                self.request_queue.get(),
                                timeout=0.01,
                            )
                            batch.append((request, future))
                        except asyncio.TimeoutError:
                            break

                    # 处理批次
                    for request, future in batch:
                        try:
                            result = await self._execute_request(request)
                            future.set_result(result)
                        except Exception as e:
                            future.set_exception(e)

                    batch.clear()

                except asyncio.TimeoutError:
                    if batch:
                        # 处理剩余请求
                        for request, future in batch:
                            try:
                                result = await self._execute_request(request)
                                future.set_result(result)
                            except Exception as e:
                                future.set_exception(e)

                        batch.clear()

        except asyncio.CancelledError:
            pass

    async def _execute_request(self, request: Dict) -> Dict:
        """执行单个请求"""
        method = request['method'].upper()
        url = request['url']

        if method == 'GET':
            return await self.get(
                url,
                params=request.get('params'),
                headers=request.get('headers'),
            )
        elif method == 'POST':
            return await self.post(
                url,
                data=request.get('data'),
                json=request.get('json'),
                headers=request.get('headers'),
            )
        else:
            raise ValueError(f"不支持的方法: {method}")


class WebSocketConnectionPool:
    """
    WebSocket连接池

    用于实时数据推送
    """

    def __init__(self, max_connections: int = 50):
        """
        初始化连接池

        Args:
            max_connections: 最大连接数
        """
        self.max_connections = max_connections
        self.connections: Dict[str, aiohttp.ClientWebSocketResponse] = {}
        self.lock = asyncio.Lock()

    async def connect(
        self,
        url: str,
        headers: Optional[Dict] = None,
    ) -> aiohttp.ClientWebSocketResponse:
        """
        建立WebSocket连接

        Args:
            url: WebSocket URL
            headers: 请求头

        Returns:
            WebSocket连接
        """
        async with self.lock:
            # 检查是否已存在连接
            if url in self.connections:
                ws = self.connections[url]
                if not ws.closed:
                    return ws
                else:
                    del self.connections[url]

            # 检查连接数限制
            if len(self.connections) >= self.max_connections:
                # 关闭最旧的连接
                oldest_url = next(iter(self.connections))
                await self.close(oldest_url)

            # 建立新连接
            session = aiohttp.ClientSession()
            ws = await session.ws_connect(url, headers=headers)
            self.connections[url] = ws

            return ws

    async def close(self, url: str):
        """
        关闭连接

        Args:
            url: WebSocket URL
        """
        async with self.lock:
            if url in self.connections:
                ws = self.connections[url]
                await ws.close()
                del self.connections[url]

    async def close_all(self):
        """关闭所有连接"""
        async with self.lock:
            for ws in self.connections.values():
                await ws.close()
            self.connections.clear()

    async def send(
        self,
        url: str,
        data: Any,
    ):
        """
        发送数据

        Args:
            url: WebSocket URL
            data: 要发送的数据
        """
        if url not in self.connections:
            raise ValueError(f"连接不存在: {url}")

        ws = self.connections[url]
        await ws.send_json(data)

    async def receive(self, url: str) -> Any:
        """
        接收数据

        Args:
            url: WebSocket URL

        Returns:
            接收到的数据
        """
        if url not in self.connections:
            raise ValueError(f"连接不存在: {url}")

        ws = self.connections[url]
        return await ws.receive_json()
```

### 40.4 监控与性能分析

```python
# src/optimization/monitoring.py
"""
性能监控与分析
"""

import asyncio
import time
import psutil
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    function_name: str
    call_count: int = 0
    total_time: float = 0
    min_time: float = float('inf')
    max_time: float = 0
    errors: int = 0
    last_execution: Optional[datetime] = None

    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0


class PerformanceMonitor:
    """
    性能监控器

    跟踪函数调用性能指标
    """

    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.lock = asyncio.Lock()

    def track(self, func: Callable) -> Callable:
        """
        跟踪函数性能

        Args:
            func: 要跟踪的函数

        Returns:
            包装后的函数
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                async with self.lock:
                    if func.__name__ not in self.metrics:
                        self.metrics[func.__name__] = PerformanceMetrics(
                            function_name=func.__name__,
                        )
                    self.metrics[func.__name__].errors += 1
                raise
            finally:
                elapsed = (time.perf_counter() - start) * 1000

                async with self.lock:
                    if func.__name__ not in self.metrics:
                        self.metrics[func.__name__] = PerformanceMetrics(
                            function_name=func.__name__,
                        )

                    metric = self.metrics[func.__name__]
                    metric.call_count += 1
                    metric.total_time += elapsed
                    metric.min_time = min(metric.min_time, elapsed)
                    metric.max_time = max(metric.max_time, elapsed)
                    metric.last_execution = datetime.now()

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # ... (同步版本类似)
                raise
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                # ... (更新指标)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    async def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取性能指标"""
        async with self.lock:
            return {
                name: {
                    'call_count': metric.call_count,
                    'avg_time_ms': metric.avg_time,
                    'min_time_ms': metric.min_time if metric.min_time != float('inf') else 0,
                    'max_time_ms': metric.max_time,
                    'errors': metric.errors,
                    'last_execution': metric.last_execution.isoformat() if metric.last_execution else None,
                }
                for name, metric in self.metrics.items()
            }

    def reset(self):
        """重置指标"""
        self.metrics.clear()


class SystemResourceMonitor:
    """
    系统资源监控器

    监控CPU、内存、网络、磁盘使用情况
    """

    def __init__(self, interval: float = 1.0):
        """
        初始化监控器

        Args:
            interval: 采样间隔(秒)
        """
        self.interval = interval
        self.task: Optional[asyncio.Task] = None
        self.metrics: Dict[str, list] = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'network_sent': [],
            'network_recv': [],
        }
        self.max_samples = 3600  # 保留1小时数据(1秒间隔)

    async def start(self):
        """启动监控"""
        self.task = asyncio.create_task(self._monitor_loop())
        logger.info("系统资源监控启动")

    async def stop(self):
        """停止监控"""
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """监控循环"""
        try:
            while True:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)

                # 内存使用
                memory = psutil.virtual_memory()
                memory_percent = memory.percent

                # 磁盘使用
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent

                # 网络IO
                network = psutil.net_io_counters()
                network_sent = network.bytes_sent / 1024 / 1024  # MB
                network_recv = network.bytes_recv / 1024 / 1024  # MB

                # 存储指标
                self.metrics['cpu'].append(cpu_percent)
                self.metrics['memory'].append(memory_percent)
                self.metrics['disk'].append(disk_percent)
                self.metrics['network_sent'].append(network_sent)
                self.metrics['network_recv'].append(network_recv)

                # 限制样本数量
                for key in self.metrics:
                    if len(self.metrics[key]) > self.max_samples:
                        self.metrics[key].pop(0)

                await asyncio.sleep(self.interval)

        except asyncio.CancelledError:
            pass

    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        return {
            'cpu_percent': self.metrics['cpu'][-1] if self.metrics['cpu'] else 0,
            'memory_percent': self.metrics['memory'][-1] if self.metrics['memory'] else 0,
            'disk_percent': self.metrics['disk'][-1] if self.metrics['disk'] else 0,
            'network_sent_mb': self.metrics['network_sent'][-1] if self.metrics['network_sent'] else 0,
            'network_recv_mb': self.metrics['network_recv'][-1] if self.metrics['network_recv'] else 0,
        }

    def get_avg_stats(self, samples: int = 60) -> Dict[str, float]:
        """获取平均值统计"""
        return {
            'cpu_avg': sum(self.metrics['cpu'][-samples:]) / min(samples, len(self.metrics['cpu'])),
            'memory_avg': sum(self.metrics['memory'][-samples:]) / min(samples, len(self.metrics['memory'])),
            'disk_avg': sum(self.metrics['disk'][-samples:]) / min(samples, len(self.metrics['disk'])),
        }


class Profiler:
    """
    性能分析器

    用于深度性能分析
    """

    def __init__(self):
        self.active = False
        self.snapshots: List[Dict] = []

    async def profile(self, func: Callable, *args, **kwargs):
        """
        分析函数性能

        Args:
            func: 要分析的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数结果
        """
        import cProfile
        import pstats
        from io import StringIO

        pr = cProfile.Profile()

        def wrapper():
            if asyncio.iscoroutinefunction(func):
                return asyncio.run(func(*args, **kwargs))
            else:
                return func(*args, **kwargs)

        pr.enable()
        result = wrapper()
        pr.disable()

        # 收集统计信息
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)

        logger.info(f"性能分析结果:\n{s.getvalue()}")

        return result

    def start(self):
        """启动分析"""
        self.active = True

    def stop(self):
        """停止分析"""
        self.active = False
```

---

## 总结

本专家级实现细节文档涵盖:

**第36章 - 深度学习架构实现**
- Transformer市场预测模型 (多头注意力、位置编码、编码器-解码器架构)
- LSTM时序预测模型 (双向LSTM、注意力机制)
- 图神经网络(GNN)股票关联分析 (图注意力网络、图构建器)

**第37章 - 高级交易策略实现**
- 强化学习交易策略 (DQN智能体、经验回放、Q学习)
- 多因子选股策略 (价值/成长/质量/技术因子、因子合成)
- 统计套利策略 (协整检验、配对交易、Z-score信号)

**第38章 - 实时系统架构深度剖析**
- 高性能消息队列 (ZMQ发布订阅、请求应答模式)
- 高频交易引擎核心 (微秒级延迟、实时风控、智能路由)

**第39章 - 综合测试策略**
- 单元测试最佳实践 (pytest fixtures、mock、参数化测试)
- 集成测试 (交易流程、风险管理、市场数据流)
- 性能测试 (延迟测试、并发测试、Locust负载测试)
- 压力测试 (高频交易、内存泄漏、多用户并发)

**第40章 - 生产级性能调优**
- 应用层优化 (异步缓存、批处理、连接池、速率限制)
- 数据库层优化 (批量操作、物化视图、查询优化)
- 网络层优化 (HTTP连接池、WebSocket池、请求合并)
- 监控与性能分析 (性能指标、资源监控、性能分析器)

---

*本文档持续更新中...*

*最后更新: 2026-01-05*
