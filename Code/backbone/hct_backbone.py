"""HCT 特征主干：仅提取时间序列特征，不依赖预测目标或分位数。"""

import math
import torch
from torch import nn

try:
    from torchvision.ops import stochastic_depth
except ImportError:
    def stochastic_depth(x: torch.Tensor, p: float, mode: str, training: bool = True) -> torch.Tensor:
        """在训练阶段随机丢弃残差分支，并按存活概率缩放；row 模式逐样本采样。"""
        if p == 0.0 or not training:
            return x

        survival_rate = 1.0 - p

        if mode == "row":
            shape = [x.shape[0]] + [1] * (x.ndim - 1)
        else:
            shape = [1] * x.ndim

        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(survival_rate)
        return x * noise.div(survival_rate)


class RHDC1D(nn.Module):
    """保持时间长度的残差空洞卷积块。

    三层卷积的膨胀率依次为 1、2、4，残差相加后再映射到 out_channels。
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        """创建空洞卷积、通道映射及 GroupNorm、GELU、Dropout 层。"""
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.norm1 = nn.GroupNorm(1, in_channels)

        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.norm2 = nn.GroupNorm(1, in_channels)

        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4)
        self.norm3 = nn.GroupNorm(1, in_channels)

        self.conv4 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm4 = nn.GroupNorm(1, out_channels)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 (B, in_channels, T) 映射为 (B, out_channels, T)。"""
        residual = x

        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.act(self.norm3(self.conv3(x)))

        x = x + residual

        x = self.act(self.norm4(self.conv4(x)))
        return self.drop(x)


class PosEncSinDynamic(nn.Module):
    """根据当前序列长度生成正余弦位置编码，无需预设最大时间长度。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """为 (B, T, C) 特征叠加位置编码，计算后恢复输入 dtype。"""
        _, t_len, dim = x.shape
        device = x.device
        dtype = x.dtype

        pos = torch.arange(t_len, dtype=torch.float32, device=device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / dim)
        )

        pe = torch.zeros(t_len, dim, dtype=torch.float32, device=device)
        pe[:, 0::2] = torch.sin(pos * div)

        if dim > 1:
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])

        return x + pe.to(dtype=dtype).unsqueeze(0)


class DropPath(nn.Module):
    """训练时按样本丢弃残差路径，评估时原样返回输入。"""

    def __init__(self, p: float = 0.0):
        """记录残差路径的丢弃概率 p。"""
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """仅在训练且 p 非零时调用逐样本 stochastic_depth。"""
        if self.p == 0.0 or not self.training:
            return x

        return stochastic_depth(x, self.p, mode="row", training=True)


class ExplicitViTBlock(nn.Module):
    """Pre-LN Transformer 块：多头自注意力和四倍宽度 MLP 各带残差连接。"""

    def __init__(self, embed_dim: int, nhead: int, dropout: float, drop_path_rate: float):
        """构建 batch_first 注意力、前馈网络和两个残差 DropPath。"""
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path_rate)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先归一化再计算注意力和 MLP，保持 (B, T, embed_dim) 形状。"""

        residual1 = x
        x_norm1 = self.norm1(x)
        attn_output = self.attn(x_norm1, x_norm1, x_norm1, need_weights=False)[0]
        x = residual1 + self.drop_path1(attn_output)

        residual2 = x
        x_norm2 = self.norm2(x)
        mlp_output = self.mlp(x_norm2)
        x = residual2 + self.drop_path2(mlp_output)

        return x


class HCTBackbone(nn.Module):
    """将 (B,T,input_dim) 地震序列转换为 (B,T,output_dim) 特征。"""

    def __init__(self, input_dim=3, base_channels=16, embed_dim=256,
                 nhead=4, depth=3, dropout=0.2, drop_path_rate=0.0):
        super().__init__()
        self.output_dim = embed_dim
        self._config = dict(input_dim=input_dim, base_channels=base_channels,
                            embed_dim=embed_dim, nhead=nhead, depth=depth,
                            dropout=dropout, drop_path_rate=drop_path_rate)
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(input_dim, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, base_channels),
            nn.GELU(),
            RHDC1D(base_channels, 2 * base_channels, dropout),
            RHDC1D(2 * base_channels, 4 * base_channels, dropout),
            RHDC1D(4 * base_channels, 8 * base_channels, dropout),
        )

        self.feat_proj = nn.Conv1d(8 * base_channels, embed_dim, kernel_size=1)

        self.posenc = PosEncSinDynamic()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.transformer_blocks = nn.Sequential(
            *[ExplicitViTBlock(embed_dim, nhead, dropout, dpr[i]) for i in range(depth)]
        )

        self.final_norm = nn.LayerNorm(embed_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 (B, T, input_dim) 输入映射为 (B, T, embed_dim) 共享特征。

        Conv1d 使用通道优先布局；投影后切回时间优先布局供 Transformer 使用。
        """
        x = x.transpose(1, 2)

        x = self.cnn_extractor(x)

        x = self.feat_proj(x).transpose(1, 2)

        x = self.posenc(x)
        x = self.transformer_blocks(x)

        return self.final_norm(x)

    def get_config(self):
        """返回可写入 checkpoint 的构造配置副本。"""
        return dict(self._config)
