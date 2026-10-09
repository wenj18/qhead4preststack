"""独立 QHead：分位数变换、特征增广和 MLP；不包含主干或训练策略。"""

import torch
from torch import nn

class TauFusion(nn.Module):
    """将分位数 tau 作为一个附加特征拼到每个时间点的主干表示。"""

    def __init__(self, center=0.5, scale=12.0):
        super().__init__()
        self.center = float(center)
        self.scale = float(scale)

    def forward(self, z: torch.Tensor, tau) -> torch.Tensor:
        """将 tau 广播为 (B, T, 1)，缩放为 scale * (tau - center) 后拼接到 z。

        支持标量、长度 T 或 B 的向量、(B,T)/(B,1)/(1,T) 矩阵，
        以及可广播到 (B,T,1) 的三维张量。一维长度同时等于 B 和 T 时按 T 解释。
        """
        batch, t_len, _ = z.shape

        if isinstance(tau, (float, int)):
            tau_feat = torch.full((batch, t_len, 1), float(tau), device=z.device, dtype=z.dtype)
        else:
            tau_feat = torch.as_tensor(tau, device=z.device, dtype=z.dtype)

            if tau_feat.ndim == 0:
                tau_feat = torch.full(
                    (batch, t_len, 1),
                    float(tau_feat.item()),
                    device=z.device,
                    dtype=z.dtype,
                )

            elif tau_feat.ndim == 1:
                if tau_feat.shape[0] == t_len:
                    tau_feat = tau_feat.view(1, t_len, 1).expand(batch, t_len, 1)
                elif tau_feat.shape[0] == batch:
                    tau_feat = tau_feat.view(batch, 1, 1).expand(batch, t_len, 1)
                else:
                    raise ValueError(f"Unsupported tau shape: {tuple(tau_feat.shape)}")

            elif tau_feat.ndim == 2:
                if tau_feat.shape == (batch, t_len):
                    tau_feat = tau_feat.unsqueeze(-1)
                elif tau_feat.shape == (batch, 1):
                    tau_feat = tau_feat.view(batch, 1, 1).expand(batch, t_len, 1)
                elif tau_feat.shape == (1, t_len):
                    tau_feat = tau_feat.view(1, t_len, 1).expand(batch, t_len, 1)
                else:
                    raise ValueError(f"Unsupported tau shape: {tuple(tau_feat.shape)}")

            elif tau_feat.ndim == 3:
                if tau_feat.shape[-1] != 1:
                    raise ValueError(f"Unsupported tau shape: {tuple(tau_feat.shape)}")
                tau_feat = tau_feat.expand(batch, t_len, 1)

            else:
                raise ValueError(f"Unsupported tau shape: {tuple(tau_feat.shape)}")

        return torch.cat([z, (tau_feat - self.center) * self.scale], dim=-1)


class QHead(nn.Module):
    """预测给定 tau 的条件分位数，输入 (B,T,C)，输出 (B,T,out_dim)。

    feature_dim 由主干输出决定，out_dim 由调用方的标签决定。
    同一个时间点的输出参数共享 tau；tau 的广播规则与原 TauFusion 一致。
    center、scale 是固定配置，不是可学习参数。输出不进行分位数排序或裁剪。
    """

    def __init__(self, feature_dim, out_dim=3, hidden_dim=None, dropout=0.2,
                 tau_center=0.5, tau_scale=12.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        hidden_dim = feature_dim if hidden_dim is None else hidden_dim
        self._config = dict(feature_dim=feature_dim, out_dim=out_dim,
                            hidden_dim=hidden_dim, dropout=dropout,
                            tau_center=float(tau_center), tau_scale=float(tau_scale))
        self.tau_fusion = TauFusion(tau_center, tau_scale)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z, tau):
        """融合显式分位数条件，逐时间点输出各目标的预测。"""
        if z.ndim != 3 or z.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected features (B,T,{self.feature_dim}), got {tuple(z.shape)}")
        return self.decoder(self.tau_fusion(z, tau))

    def get_config(self):
        return dict(self._config)
