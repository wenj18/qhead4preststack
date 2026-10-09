"""PHead（Point Prediction Head）：HCT-PHead 的确定性预测头，与 QHead 使用相同的特征输入约定。"""

from torch import nn


class PointHead(nn.Module):
    """将 (B,T,C) 特征映射为 (B,T,out_dim) 点预测。"""

    def __init__(self, feature_dim, out_dim=3, hidden_dim=None, dropout=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        hidden_dim = feature_dim if hidden_dim is None else hidden_dim
        self._config = dict(feature_dim=feature_dim, out_dim=out_dim,
                            hidden_dim=hidden_dim, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        if z.ndim != 3 or z.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected features (B,T,{self.feature_dim}), got {tuple(z.shape)}")
        return self.decoder(z)

    def get_config(self):
        return dict(self._config)
