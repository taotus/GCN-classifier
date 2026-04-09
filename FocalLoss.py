import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测概率

        # Focal Loss公式
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for binary classification/segmentation.

    Args:
        alpha (float): weight for false positives (FP). Default: 0.5
        beta (float): weight for false negatives (FN). Default: 0.5
        smooth (float): smoothing factor to avoid division by zero. Default: 1e-7
        reduction (str): 'mean' or 'sum' over batch. Default: 'mean'
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): logits of shape (batch, ...)  (no sigmoid yet)
            y_true (torch.Tensor): binary labels of shape (batch, ...), values 0 or 1.

        Returns:
            loss (torch.Tensor): Tversky loss value.
        """
        # Convert logits to probabilities
        y_pred = F.softmax(y_pred, dim=1)[:, 1]
        y_true = y_true.float()

        # Soft counts: TP, FP, FN
        tp = torch.sum(y_pred * y_true)
        fp = torch.sum(y_pred * (1 - y_true))
        fn = torch.sum(1 - y_pred)

        # Tversky index per sample
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1 - tversky_index

        # Aggregate over batch
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss  # return per-sample loss if reduction='none'