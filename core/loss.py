import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


class TverskyLoss(_Loss):
    def __init__(self, mode, alpha, beta, gamma, classes=None):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.classes = classes

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

        losses = losses[self.classes] if self.classes is not None else losses
        tversky_loss = losses.mean() ** self.gamma

        return tversky_loss


class FocalLoss(_Loss):
    def __init__(self, mode, alpha, gamma, ohem_ratio):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.gamma = gamma
        self.ohem_ratio = ohem_ratio

    def _compute_loss(self, predictions, targets):
        targets = targets.to(dtype=predictions.dtype)
        log_probabilities = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        probabilities = torch.exp(-log_probabilities)

        focal_term = (1.0 - probabilities).pow(self.gamma)
        losses = focal_term * log_probabilities
        losses *= self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        if self.ohem_ratio < 1.0:
            flattened_losses = losses.view(-1)
            keep_count = int(self.ohem_ratio * flattened_losses.numel())
            losses = flattened_losses.topk(k=keep_count)[0] if keep_count > 0 else losses

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


class LovaszLoss(_Loss):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes

    def _compute_gradient(self, sorted_targets):
        element_count = len(sorted_targets)
        total_targets = sorted_targets.sum()

        intersection = total_targets - sorted_targets.to(dtype=torch.float32).cumsum(dim=0)
        union = total_targets + (1 - sorted_targets).to(dtype=torch.float32).cumsum(dim=0)
        jaccard = 1.0 - intersection / union
        jaccard[1:element_count] = jaccard[1:element_count] - jaccard[0:-1]

        return jaccard

    def forward(self, predictions, targets):
        targets = targets.view(-1)

        probabilities = F.softmax(predictions, dim=1)
        _, channels, _, _ = probabilities.shape
        probabilities = probabilities.permute(0, 2, 3, 1).contiguous().view(-1, channels)

        losses = []

        for class_index in self.classes:
            class_targets = (targets == class_index).to(dtype=torch.float32)

            if class_targets.sum() == 0:
                continue

            class_probabilities = probabilities[:, class_index]
            errors = (class_targets - class_probabilities).abs()

            sorted_errors, permutation = torch.sort(errors, dim=0, descending=True)
            sorted_targets = class_targets[permutation.data]

            class_loss = torch.dot(sorted_errors, self._compute_gradient(sorted_targets))
            losses.append(class_loss)

        lovasz_loss = sum(losses) / len(losses) if losses else probabilities.new_tensor(0.0)

        return lovasz_loss


class ClDiceLoss(_Loss):
    def __init__(self, classes, iterations):
        super().__init__()
        self.classes = classes
        self.iterations = iterations

    def _extract_soft_skeleton(self, tensor):
        temporary_tensor = tensor
        skeleton = torch.zeros_like(tensor)

        for _ in range(self.iterations):
            eroded_tensor = -F.max_pool2d(-temporary_tensor, kernel_size=3, stride=1, padding=1)
            opened_tensor = F.max_pool2d(eroded_tensor, kernel_size=3, stride=1, padding=1)
            skeleton = skeleton + F.relu(temporary_tensor - opened_tensor)
            temporary_tensor = eroded_tensor

        return skeleton

    def forward(self, predictions, targets):
        class_count = predictions.size(dim=1)

        onehot_targets = F.one_hot(targets, num_classes=class_count).permute(0, 3, 1, 2).to(dtype=predictions.dtype)
        probabilities = F.softmax(predictions, dim=1)

        foreground_targets = onehot_targets[:, self.classes]
        foreground_probabilities = probabilities[:, self.classes]

        skeleton_targets = self._extract_soft_skeleton(foreground_targets)
        skeleton_predictions = self._extract_soft_skeleton(foreground_probabilities)

        precision = (torch.sum(skeleton_predictions * foreground_targets, dim=(1, 2, 3)) + 1e-7) / (torch.sum(skeleton_predictions, dim=(1, 2, 3)) + 1e-7)
        sensitivity = (torch.sum(skeleton_targets * foreground_probabilities, dim=(1, 2, 3)) + 1e-7) / (torch.sum(skeleton_targets, dim=(1, 2, 3)) + 1e-7)
        scores = 2.0 * precision * sensitivity / (precision + sensitivity + 1e-7)
        cldice_loss = 1.0 - scores.mean()

        return cldice_loss


class TotalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        drivable_area_tversky_alpha, drivable_area_tversky_gamma = config.drivable_area_tversky_alpha, config.drivable_area_tversky_gamma
        lane_line_tversky_alpha, lane_line_tversky_gamma = config.lane_line_tversky_alpha, config.lane_line_tversky_gamma
        focal_alpha, focal_gamma = config.focal_alpha, config.focal_gamma

        self.drivable_area_tversky_loss = TverskyLoss(mode=MULTICLASS_MODE, alpha=drivable_area_tversky_alpha, beta=1.0 - drivable_area_tversky_alpha, gamma=drivable_area_tversky_gamma)
        self.lane_line_tversky_loss = TverskyLoss(mode=MULTICLASS_MODE, alpha=lane_line_tversky_alpha, beta=1.0 - lane_line_tversky_alpha, gamma=lane_line_tversky_gamma, classes=[config.lane_line_class_id])

        self.drivable_area_focal_loss = FocalLoss(mode=MULTICLASS_MODE, alpha=focal_alpha, gamma=focal_gamma, ohem_ratio=config.drivable_area_ohem_ratio)
        self.lane_line_focal_loss = FocalLoss(mode=MULTICLASS_MODE, alpha=focal_alpha, gamma=focal_gamma, ohem_ratio=config.lane_line_ohem_ratio)

        self.drivable_area_lovasz_loss = LovaszLoss(classes=[config.drivable_area_class_id])
        self.lane_line_cldice_loss = ClDiceLoss(classes=[config.lane_line_class_id], iterations=config.lane_line_cldice_iterations)

    def forward(self, predictions, drivable_area_targets, lane_line_targets, epoch):
        drivable_area_targets = drivable_area_targets.to(dtype=torch.int64)
        lane_line_targets = lane_line_targets.to(dtype=torch.int64)
        drivable_area_predictions, lane_line_predictions = predictions[0], predictions[1]

        drivable_area_tversky = self.drivable_area_tversky_loss(drivable_area_predictions, drivable_area_targets)
        lane_line_tversky = self.lane_line_tversky_loss(lane_line_predictions, lane_line_targets)

        drivable_area_focal = self.drivable_area_focal_loss(drivable_area_predictions, drivable_area_targets)
        lane_line_focal = self.lane_line_focal_loss(lane_line_predictions, lane_line_targets)

        tversky_loss = drivable_area_tversky + lane_line_tversky
        focal_loss = drivable_area_focal + lane_line_focal

        drivable_area_lovasz = self.drivable_area_lovasz_loss(drivable_area_predictions, drivable_area_targets)
        lane_line_cldice = self.lane_line_cldice_loss(lane_line_predictions, lane_line_targets)

        lovasz_weight = self.config.drivable_area_lovasz_weight
        cldice_weight = self.config.lane_line_cldice_weight

        warmup_factor = min(1.0, max(0.0, (epoch - 1) / self.config.warmup_epochs))
        lovasz_weight *= warmup_factor
        cldice_weight *= warmup_factor

        total_loss = tversky_loss + focal_loss + lovasz_weight * drivable_area_lovasz + cldice_weight * lane_line_cldice

        if len(predictions) == 4:
            drivable_area_float_targets = drivable_area_targets.unsqueeze(dim=1).to(dtype=torch.float32)
            lane_line_float_targets = lane_line_targets.unsqueeze(dim=1).to(dtype=torch.float32)
            drivable_area_auxiliary_predictions, lane_line_auxiliary_predictions = predictions[2], predictions[3]

            interpolated_drivable_area = F.interpolate(drivable_area_float_targets, size=drivable_area_auxiliary_predictions.shape[2:], mode="nearest")
            interpolated_lane_line = F.interpolate(lane_line_float_targets, size=lane_line_auxiliary_predictions.shape[2:], mode="nearest")

            drivable_area_auxiliary_targets = interpolated_drivable_area.squeeze(dim=1).to(dtype=torch.int64)
            lane_line_auxiliary_targets = interpolated_lane_line.squeeze(dim=1).to(dtype=torch.int64)

            drivable_area_auxiliary_loss = self.drivable_area_tversky_loss(drivable_area_auxiliary_predictions, drivable_area_auxiliary_targets) + self.drivable_area_focal_loss(drivable_area_auxiliary_predictions, drivable_area_auxiliary_targets)
            lane_line_auxiliary_loss = self.lane_line_tversky_loss(lane_line_auxiliary_predictions, lane_line_auxiliary_targets) + self.lane_line_focal_loss(lane_line_auxiliary_predictions, lane_line_auxiliary_targets)

            auxiliary_loss = drivable_area_auxiliary_loss + lane_line_auxiliary_loss
            auxiliary_weight = self.config.auxiliary_weight

            total_loss += auxiliary_weight * auxiliary_loss

        return {
            "total": total_loss,
            "tversky_loss": tversky_loss,
            "focal_loss": focal_loss,
            "drivable_area_lovasz": drivable_area_lovasz,
            "lane_line_cldice": lane_line_cldice,
        }
