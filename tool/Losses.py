import torch
from torch import nn, Tensor
import torch.nn.functional as F

class ContrastiveChangeLoss(nn.Module):
    """
    Contrastive loss for change detection.

    输入:
        c_weights: (B,1,H,W)  —— 特征相似度图 (值越小表示越相似)
        label:     (B,1,H,W)  —— 变化标签 (0=unchange, 1=change)

    设计思想:
        - unchange 区域: 相似度应高 (希望 c_weights 小)
        - change   区域: 相似度应低 (希望 c_weights 大)
    """
    def __init__(self, margin: float = 0.0, temperature: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.T = temperature
        self.reduction = reduction

    def forward(self, c_weights: Tensor, label: Tensor) -> Tensor:
        """
        计算 contrastive-style loss。
        """
        label = label.unsqueeze(1) 
        assert c_weights.ndim == 4 and label.ndim == 4, \
            f"Expect c_weights and label in shape (B,1,H,W), got {c_weights.shape}, {label.shape}"

        B, _, H, W = c_weights.shape
        # 确保 label 尺寸一致
        if label.shape[2:] != (H, W):
            label = F.interpolate(label.float(), size=(H, W), mode="nearest")

        # 相似度图归一化，越相似越接近 1
        sims = torch.sigmoid((1 - c_weights) / self.T)

        change_mask = (label == 1).float()
        unchange_mask = 1.0 - change_mask

        # ---- Unchange 区域：希望相似 (sims→1) ----
        pos_loss = (1 - sims) * unchange_mask
        # loss = pos_loss

        # ---- Change 区域：希望不相似 (sims→0)，使用 margin 控制分离 ----
        neg_loss = torch.clamp(sims - self.margin, min=0.0) * change_mask

        # ---- 聚合 ----
        loss = pos_loss + neg_loss
        if self.reduction == "mean":
            loss = loss.sum() / (unchange_mask.sum() + change_mask.sum() + 1e-6)
        elif self.reduction == "sum":
            loss = loss.sum()

        # B, _, H, W = c_weights.shape
        # sims = c_weights.view(B, -1)  # (B,HW)

        # # 动态阈值：取每个 batch 的相似度分布下分位数
        # thresh = torch.quantile(sims, 0.75, dim=1, keepdim=True)

        # pos_mask = (sims > thresh).float()
        # neg_mask = 1 - pos_mask

        # # 正样本：希望接近 1，负样本：希望接近 0
        # pos_loss = ((1 - sims) * pos_mask).sum() / (pos_mask.sum() + 1e-6)
        # neg_loss = (sims * neg_mask).sum() / (neg_mask.sum() + 1e-6)

        # loss = pos_loss + neg_loss

        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum(self.criterion(p, labels) for p in preds)
        return self.criterion(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _compute_loss(self, pred: Tensor, label: Tensor) -> Tensor:
        loss = self.criterion(pred, label).view(-1)
        valid_loss = loss[loss > self.thresh]
        n_min = label[label != self.ignore_label].numel() // 16
        if valid_loss.numel() < n_min:
            valid_loss, _ = loss.topk(n_min)
        return valid_loss.mean()

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum(w * self._compute_loss(p, labels) for p, w in zip(preds, self.aux_weights))
        return self._compute_loss(preds, labels)


class FocalLoss(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _compute_loss(self, preds: Tensor, labels: Tensor) -> Tensor:
        ce_loss = self.criterion(preds, labels)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum(self._compute_loss(p, labels) for p in preds)
        return self._compute_loss(preds, labels)


class DiceLoss(nn.Module):
    def __init__(self, ignore_label: int = 255) -> None:
        super().__init__()
        self.ignore_label = ignore_label

    def _compute_loss(self, preds: Tensor, labels: Tensor) -> Tensor:
        probs = torch.softmax(preds, dim=1)
        smooth = 1e-5
        inter = (probs[:, 1] * (labels == 1).float()).sum()
        union = probs[:, 1].sum() + (labels == 1).float().sum() + smooth
        return 1 - (2. * inter + smooth) / union

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum(self._compute_loss(p, labels) for p in preds)
        return self._compute_loss(preds, labels)

class BCELoss(nn.Module):
    def __init__(self, ignore_label: int = 255) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')  # 可以用 logits 更稳定

    def _compute_loss(self, preds: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            preds: (B, 2, H, W) 或 (B, H, W)
            labels: (B, H, W), 0/1 二值标签
        """
        if preds.dim() == 4 and preds.size(1) == 2:
            # 取正类 logit
            preds = preds[:, 1, :, :]
        preds = preds.float()
        labels = labels.float()

        mask = (labels != self.ignore_label)
        preds = preds[mask]
        labels = labels[mask]

        if preds.numel() == 0:
            return torch.tensor(0., device=preds.device)

        loss = self.bce(preds, labels) / mask.sum()
        return loss

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum(self._compute_loss(p, labels) for p in preds)
        return self._compute_loss(preds, labels)