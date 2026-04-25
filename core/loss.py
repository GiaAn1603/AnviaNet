import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


class TverskyLoss(_Loss):
    def __init__(self, mode, alpha, beta, gamma):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_score(self, predictions, targets, reduction_dimensions):
        intersection = torch.sum(predictions * targets, dim=reduction_dimensions)
        false_positive = torch.sum(predictions * (1.0 - targets), dim=reduction_dimensions)
        false_negative = torch.sum((1.0 - predictions) * targets, dim=reduction_dimensions)
        scores = intersection / (intersection + self.alpha * false_positive + self.beta * false_negative).clamp_min(min=1e-7)

        return scores

    def forward(self, predictions, targets):
        batch_size = targets.size(dim=0)
        class_count = predictions.size(dim=1)
        reduction_dimensions = (0, 2)

        if self.mode == MULTICLASS_MODE:
            predictions = predictions.log_softmax(dim=1).exp()
        else:
            predictions = F.logsigmoid(predictions).exp()

        if self.mode == BINARY_MODE:
            targets = targets.view(batch_size, 1, -1)
            predictions = predictions.view(batch_size, 1, -1)
        elif self.mode == MULTICLASS_MODE:
            targets = targets.view(batch_size, -1)
            targets = F.one_hot(targets, num_classes=class_count)
            targets = targets.permute(0, 2, 1)
            predictions = predictions.view(batch_size, class_count, -1)
        elif self.mode == MULTILABEL_MODE:
            targets = targets.view(batch_size, class_count, -1)
            predictions = predictions.view(batch_size, class_count, -1)

        scores = self._compute_score(predictions, targets, reduction_dimensions=reduction_dimensions)
        losses = 1.0 - scores

        mask = targets.sum(dim=reduction_dimensions) > 0
        losses *= mask.to(dtype=losses.dtype)

        tversky_loss = losses.mean() ** self.gamma

        return tversky_loss


class FocalLoss(_Loss):
    def __init__(self, mode, alpha, gamma):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.gamma = gamma

    def _compute_loss(self, predictions, targets):
        targets = targets.to(dtype=predictions.dtype)
        log_probabilities = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        probabilities = torch.exp(-log_probabilities)

        focal_term = (1.0 - probabilities).pow(self.gamma)
        losses = focal_term * log_probabilities
        losses *= self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = losses.mean()

        return loss

    def forward(self, predictions, targets):
        focal_loss = 0.0

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            targets = targets.view(-1)
            predictions = predictions.view(-1)
            focal_loss = self._compute_loss(predictions, targets)
        elif self.mode == MULTICLASS_MODE:
            class_count = predictions.size(dim=1)

            for class_index in range(class_count):
                class_targets = (targets == class_index).to(dtype=torch.int64)
                class_predictions = predictions[:, class_index, ...]
                focal_loss += self._compute_loss(class_predictions, class_targets)

        return focal_loss


class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()

        hyperparams = {
            "drivable_area_tversky_alpha": 0.5,
            "drivable_area_tversky_gamma": 1.0,
            "lane_line_tversky_alpha": 0.5,
            "lane_line_tversky_gamma": 1.0,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
        }

        drivable_area_tversky_alpha, drivable_area_tversky_gamma = hyperparams["drivable_area_tversky_alpha"], hyperparams["drivable_area_tversky_gamma"]
        lane_line_tversky_alpha, lane_line_tversky_gamma = hyperparams["lane_line_tversky_alpha"], hyperparams["lane_line_tversky_gamma"]
        focal_alpha, focal_gamma = hyperparams["focal_alpha"], hyperparams["focal_gamma"]

        self.drivable_area_tversky_loss = TverskyLoss(mode=MULTICLASS_MODE, alpha=drivable_area_tversky_alpha, beta=1.0 - drivable_area_tversky_alpha, gamma=drivable_area_tversky_gamma)
        self.lane_line_tversky_loss = TverskyLoss(mode=MULTICLASS_MODE, alpha=lane_line_tversky_alpha, beta=1.0 - lane_line_tversky_alpha, gamma=lane_line_tversky_gamma)

        self.drivable_area_focal_loss = FocalLoss(mode=MULTICLASS_MODE, alpha=focal_alpha, gamma=focal_gamma)
        self.lane_line_focal_loss = FocalLoss(mode=MULTICLASS_MODE, alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, predictions, drivable_area_targets, lane_line_targets):
        drivable_area_targets = drivable_area_targets.to(dtype=torch.int64)
        lane_line_targets = lane_line_targets.to(dtype=torch.int64)
        drivable_area_predictions, lane_line_predictions = predictions[0], predictions[1]

        drivable_area_tversky = self.drivable_area_tversky_loss(drivable_area_predictions, drivable_area_targets)
        lane_line_tversky = self.lane_line_tversky_loss(lane_line_predictions, lane_line_targets)

        drivable_area_focal = self.drivable_area_focal_loss(drivable_area_predictions, drivable_area_targets)
        lane_line_focal = self.lane_line_focal_loss(lane_line_predictions, lane_line_targets)

        tversky_loss = drivable_area_tversky + lane_line_tversky
        focal_loss = drivable_area_focal + lane_line_focal

        total_loss = tversky_loss + focal_loss

        return {
            "total": total_loss,
            "tversky_loss": tversky_loss,
            "focal_loss": focal_loss,
        }
