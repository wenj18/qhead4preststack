"""按 backbone、head、loss 组装网络；数据处理和训练流程由 Notebook 负责。"""

import torch
from torch import nn
from backbone.hct_backbone import HCTBackbone
from head.phead import PointHead
from head.qhead import QHead
from loss.quantile_loss import QuantileLoss
from loss.mse_loss import MSELoss


class Model(nn.Module):
    """主干返回 (B,T,C)，预测头返回 (B,T,out_dim)。"""

    def __init__(self, backbone, head, is_quantile):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.is_quantile = is_quantile

    def forward_from_features(self, features, tau=None):
        if self.is_quantile:
            if tau is None:
                raise ValueError("QHead 预测需要分位数 tau。")
            return self.head(features, tau)
        if tau is not None:
            raise ValueError("PHead 不接收分位数 tau。")
        return self.head(features)

    def forward(self, x, tau=None):
        return self.forward_from_features(self.backbone(x), tau)

    @torch.no_grad()
    def predict_quantiles(self, x, taus=(0.025, 0.5, 0.975)):
        """评估时共享一次主干特征，返回 (B,T,Q,out_dim)，保持 taus 顺序。"""
        if not self.is_quantile or self.training:
            raise RuntimeError("多分位数推理需要 QHead 模型，并先调用 model.eval()。")
        taus = tuple(float(t) for t in taus)
        if not taus or any(not 0 < t < 1 for t in taus):
            raise ValueError("分位数必须是非空的 (0,1) 概率列表。")
        features = self.backbone(x)
        return torch.stack([self.forward_from_features(features, t) for t in taus], dim=2)


def build_model(backbone="hct", head="qhead", loss="quantile",
                backbone_params=None, head_params=None, loss_params=None):
    """返回 (model, criterion)，省略的参数使用组件默认值。

    HCT-QHead：hct + qhead + quantile；HCT-PHead：hct + phead + mse（MSE 损失）。
    主干特征维度自动传入预测头；model.config 记录可用于再次构建的完整配置。
    """
    if backbone != "hct":
        raise ValueError(f"未知主干：{backbone}；目前支持 hct。")
    if (head, loss) not in (("qhead", "quantile"), ("phead", "mse")):
        raise ValueError("预测头与损失应组合为 qhead + quantile 或 phead + mse。")
    bp, hp, lp = dict(backbone_params or {}), dict(head_params or {}), dict(loss_params or {})
    criterion = QuantileLoss(**lp) if loss == "quantile" else MSELoss(**lp)
    feature_net = HCTBackbone(**bp)
    feature_dim = hp.pop("feature_dim", feature_net.output_dim)
    if feature_dim != feature_net.output_dim:
        raise ValueError("预测头 feature_dim 必须等于主干输出维度。")
    prediction_head = (QHead if head == "qhead" else PointHead)(feature_dim=feature_dim, **hp)
    if criterion.weights.numel() != prediction_head.out_dim:
        raise ValueError("损失权重个数必须等于预测头 out_dim。")
    model = Model(feature_net, prediction_head, is_quantile=head == "qhead")
    model.config = dict(backbone=backbone, head=head, loss=loss,
                        backbone_params=feature_net.get_config(),
                        head_params=prediction_head.get_config(), loss_params=criterion.config.copy())
    return model, criterion
