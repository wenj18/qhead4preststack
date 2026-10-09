"""三参数加权均方误差损失。"""

import torch
from torch import nn

def normalize_weights(weights, out_dim):
    """校验输出权重，并归一化到和为 1；保持旧模型的计算顺序。"""
    w = torch.as_tensor(weights, dtype=torch.float32)
    if w.ndim != 1 or w.numel() != out_dim:
        raise ValueError(f"target_loss_weights must have length {out_dim}.")
    if torch.any(w < 0):
        raise ValueError("target_loss_weights must be non-negative.")
    if float(w.sum()) <= 0:
        raise ValueError("target_loss_weights must contain at least one positive value.")
    return w / w.sum()

class MSELoss(nn.Module):
    """权重按输出通道排列并归一化；先沿目标维求和，再平均批次和时间。"""

    def __init__(self, target_weights=(2.0, 1.0, 1.0)):
        super().__init__()
        self.register_buffer("weights", normalize_weights(target_weights, len(target_weights)))
        self.config = {"target_weights": tuple(float(w) for w in target_weights)}

    def forward(self, prediction, target):
        if prediction.shape != target.shape or target.ndim != 3:
            raise ValueError("预测与标签必须具有相同的 (B,T,out_dim) 形状。")
        if target.shape[-1] != self.weights.numel():
            raise ValueError("损失权重个数必须等于输出通道数。")
        weights = self.weights.to(device=target.device, dtype=target.dtype)
        element = nn.functional.mse_loss(prediction, target, reduction="none")
        return (element * weights.view(1, 1, -1)).sum(dim=-1).mean()
