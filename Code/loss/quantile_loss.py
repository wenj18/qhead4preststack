"""三参数加权 Huber 分位数损失。"""

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

class HuberQuantileLoss(nn.Module):
    """使用 Huber 平滑误差的分位数损失。

    diff = yhat - y；高估乘 1 - tau，低估乘 tau。
    beta 控制二次段宽度；默认逐元素返回，可再按目标加权。
    """

    def __init__(self, beta: float = 0.1, reduction: str = "none"):
        """设置平滑阈值 beta 和归约方式 none、mean 或 sum。"""
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, yhat: torch.Tensor, y: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """计算可广播 tau 的平滑分位数损失；输入通常为 (B, T, out_dim)。"""
        diff = yhat - y
        abs_diff = torch.abs(diff)

        huber = torch.where(
            abs_diff < self.beta,
            0.5 * diff.pow(2) / self.beta,
            abs_diff - 0.5 * self.beta,
        )

        loss = torch.where(diff >= 0, (1.0 - tau) * huber, tau * huber)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

class QuantileLoss(nn.Module):
    """权重按输出通道排列并归一化；先沿目标维求和，再平均批次和时间。"""

    def __init__(self, target_weights=(2.0, 1.0, 1.0), beta=0.1):
        super().__init__()
        self.register_buffer("weights", normalize_weights(target_weights, len(target_weights)))
        self.config = {"target_weights": tuple(float(w) for w in target_weights)}
        self.config["beta"] = float(beta)
        self.element_loss = HuberQuantileLoss(beta=beta, reduction="none")

    def forward(self, prediction, target, tau):
        if prediction.shape != target.shape or target.ndim != 3:
            raise ValueError("预测与标签必须具有相同的 (B,T,out_dim) 形状。")
        if target.shape[-1] != self.weights.numel():
            raise ValueError("损失权重个数必须等于输出通道数。")
        weights = self.weights.to(device=target.device, dtype=target.dtype)
        element = self.element_loss(prediction, target, tau)
        return (element * weights.view(1, 1, -1)).sum(dim=-1).mean()
