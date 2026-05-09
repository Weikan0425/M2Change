import torch
import numpy as np
from torch import Tensor
from typing import Tuple

class Metrics:
    def __init__(self, num_classes: int, ignore_label, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        if self.ignore_label is not None:
            keep = target != self.ignore_label
            target = target[keep]
            pred = pred[keep]
        self.hist += torch.bincount(target * self.num_classes + pred, minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)

    def compute_oa(self) -> float:
        oa = self.hist.diag().sum() / self.hist.sum()
        return round(oa.item() * 100, 2)

    def compute_recall(self) -> Tuple[Tensor, Tensor]:
        recall = self.hist.diag() / self.hist.sum(0)
        mrecall = recall[~recall.isnan()].mean().item()
        recall *= 100
        mrecall *= 100
        return recall.cpu().numpy().round(2).tolist(), round(mrecall, 2)
    
    def compute_kappa(self) -> float:
        hist = self.hist.float()  # 混淆矩阵 [C, C]
        n = hist.sum()

        # 观测一致率
        po = torch.trace(hist) / n

        # 期望一致率
        row_marginals = hist.sum(dim=1)  # 每行求和
        col_marginals = hist.sum(dim=0)  # 每列求和
        pe = (row_marginals * col_marginals).sum() / (n * n)

        # kappa
        kappa = (po - pe) / (1 - pe)

        return kappa * 100
    
    def get_confusion_matrix_tensor(self) -> Tensor:
        return self.hist.cpu()
    
    