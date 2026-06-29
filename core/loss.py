import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from core.config import LossConfig

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


def to_tensor(data, dtype=None):
    if isinstance(data, torch.Tensor):
        if dtype is not None:
            data = data.type(dtype)

        return data

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

        if dtype is not None:
            data = data.type(dtype)

        return data

    if isinstance(data, (list, tuple)):
        data = np.array(data)
        data = torch.from_numpy(data)

        if dtype is not None:
            data = data.type(dtype)

        return data


def focal_loss_with_logits(
    output, target, gamma=2.0, alpha=0.25, reduction="mean", normalized=False, reduced_threshold=None, eps=1e-6, ohem_ratio=1.0
):
    target = target.type(output.type())
    log_prob = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    prob = torch.exp(-log_prob)

    if reduced_threshold is None:
        focal_term = (1.0 - prob).pow(gamma)
    else:
        focal_term = ((1.0 - prob) / reduced_threshold).pow(gamma)
        focal_term[prob < reduced_threshold] = 1

    loss = focal_term * log_prob

    if alpha is not None:
        loss *= alpha * target + (1.0 - alpha) * (1.0 - target)

    if ohem_ratio < 1.0:
        loss_flat = loss.view(-1)
        num_keep = int(ohem_ratio * loss_flat.numel())

        if num_keep > 0:
            loss_flat, _ = loss_flat.topk(num_keep)
            loss = loss_flat

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def soft_dice_score(output, target, smooth=0.0, eps=1e-7, dims=None):
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)

    return dice_score


def soft_tversky_score(output, target, alpha, beta, smooth=0.0, eps=1e-7, dims=None):
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        false_positives = torch.sum(output * (1.0 - target), dim=dims)
        false_negatives = torch.sum((1.0 - output) * target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        false_positives = torch.sum(output * (1.0 - target))
        false_negatives = torch.sum((1.0 - output) * target)

    tversky_score = (intersection + smooth) / (intersection + alpha * false_positives + beta * false_negatives + smooth).clamp_min(eps)

    return tversky_score


class FocalLoss(_Loss):
    def __init__(self, mode, alpha=0.25, gamma=2.0, ignore_index=None, reduction="mean", normalized=False, reduced_threshold=None, ohem_ratio=1.0):
        super().__init__()
        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
            ohem_ratio=ohem_ratio,
        )

    def forward(self, y_pred, y_true):
        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:
            num_classes = y_pred.size(1)
            loss = 0.0

            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                target_class = (y_true == cls).long()
                pred_class = y_pred[:, cls, ...]

                if self.ignore_index is not None:
                    target_class = target_class[not_ignored]
                    pred_class = pred_class[not_ignored]

                loss += self.focal_loss_fn(pred_class, target_class)

        return loss


class DiceLoss(_Loss):
    def __init__(self, mode, classes=None, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-7):
        super().__init__()
        self.mode = mode

        if classes is not None:
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        batch_size = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(batch_size, 1, -1)
            y_pred = y_pred.view(batch_size, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(batch_size, -1)
            y_pred = y_pred.view(batch_size, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)
                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = F.one_hot(y_true, num_classes)
                y_true = y_true.permute(0, 2, 1)

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(batch_size, num_classes, -1)
            y_pred = y_pred.view(batch_size, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None):
        return soft_dice_score(output, target, smooth, eps, dims)


class TverskyLoss(DiceLoss):
    def __init__(self, mode, classes=None, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-7, alpha=0.5, beta=0.5, gamma=1.0):
        super().__init__(
            mode=mode,
            classes=classes,
            log_loss=log_loss,
            from_logits=from_logits,
            smooth=smooth,
            ignore_index=ignore_index,
            eps=eps,
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None):
        return soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)


class LovaszLoss(_Loss):
    def __init__(self, classes=[1], reduction='mean'):
        super().__init__(reduction=reduction)
        self.classes = classes

    def lovasz_grad(self, gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union

        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

        return jaccard

    def forward(self, probas, labels):
        B, C, H, W = probas.shape
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)

        losses = []
        for c in self.classes:
            fg = (labels == c).float()

            if fg.sum() == 0:
                continue

            class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, self.lovasz_grad(fg_sorted)))

        return sum(losses) / len(losses) if losses else probas.new_tensor(0.0)


class clDiceLoss(_Loss):
    def __init__(self, iterations=5, reduction='mean'):
        super().__init__(reduction=reduction)
        self.iterations = iterations

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
        elif len(img.shape) == 3:
            p1 = -F.max_pool1d(-img, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError(f"Unsupported dimensions: {img.shape}")

        return p1

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            p1 = F.max_pool2d(img, kernel_size=3, stride=1, padding=1)
        elif len(img.shape) == 3:
            p1 = F.max_pool1d(img, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError(f"Unsupported dimensions: {img.shape}")

        return p1

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skeletonize(self, img):
        img_temp = img
        skel = torch.zeros_like(img)

        for _ in range(self.iterations):
            eroded = self.soft_erode(img_temp)
            opened = self.soft_dilate(eroded)
            skel = skel + F.relu(img_temp - opened)
            img_temp = eroded

        return skel

    def forward(self, v_p, v_t):
        s_p = self.soft_skeletonize(v_p)
        s_t = self.soft_skeletonize(v_t)
        t_prec = (torch.sum(s_p * v_t, dim=(1, 2, 3)) + 1e-7) / (torch.sum(s_p, dim=(1, 2, 3)) + 1e-7)
        t_sens = (torch.sum(s_t * v_p, dim=(1, 2, 3)) + 1e-7) / (torch.sum(s_t, dim=(1, 2, 3)) + 1e-7)
        cl_dice = 2.0 * t_prec * t_sens / (t_prec + t_sens + 1e-7)

        return 1.0 - cl_dice.mean()


class SingleLoss(nn.Module):
    def __init__(self, config: LossConfig = None, task=None):
        super().__init__()

        if config is None:
            config = LossConfig()

        self.config = config
        self.task = task

        tversky_da_alpha, tversky_da_gamma = (config.tversky_da_alpha, config.tversky_da_gamma)
        tversky_ll_alpha, tversky_ll_gamma = (config.tversky_ll_alpha, config.tversky_ll_gamma)
        focal_alpha, focal_gamma = config.focal_alpha, config.focal_gamma

        if task == "DA":
            self.tver = TverskyLoss(
                mode=MULTICLASS_MODE,
                alpha=tversky_da_alpha,
                beta=1.0 - tversky_da_alpha,
                gamma=tversky_da_gamma,
                from_logits=True,
            )
            self.focal = FocalLoss(
                mode=MULTICLASS_MODE,
                alpha=focal_alpha,
                gamma=focal_gamma,
                ohem_ratio=config.ohem_ratio_da,
            )
            self.lovasz = LovaszLoss(classes=[1])

        if task == "LL":
            self.tver = TverskyLoss(
                mode=MULTICLASS_MODE,
                classes=[1],
                alpha=tversky_ll_alpha,
                beta=1.0 - tversky_ll_alpha,
                gamma=tversky_ll_gamma,
                from_logits=True,
            )
            self.focal = FocalLoss(
                mode=MULTICLASS_MODE,
                alpha=focal_alpha,
                gamma=focal_gamma,
                ohem_ratio=config.ohem_ratio_ll,
            )
            self.cldice = clDiceLoss(iterations=5)

    def forward(self, outputs, targets, epoch=None):
        targets = targets.long()
        tversky_loss = self.tver(outputs, targets)
        focal_loss = self.focal(outputs, targets)

        if self.task == "DA":
            pred_probs = F.softmax(outputs, dim=1)
            loss_lovasz_da = self.lovasz(pred_probs, targets)

            lovasz_weight = getattr(self.config, "lovasz_da_weight", 0.1)

            if epoch is not None:
                warmup_factor = min(1.0, max(0.0, (epoch - 1) / 5.0))
                lovasz_weight *= warmup_factor

            total_loss = focal_loss + tversky_loss + lovasz_weight * loss_lovasz_da

            return {
                "total": total_loss,
                "focal": focal_loss,
                "tversky": tversky_loss,
                "lovasz_da": loss_lovasz_da,
            }

        if self.task == "LL":
            pred_probs = F.softmax(outputs, dim=1)
            target_onehot = F.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()
            loss_cldice_ll = self.cldice(pred_probs[:, 1:2], target_onehot[:, 1:2])

            cldice_weight = getattr(self.config, "cldice_ll_weight", 0.1)

            if epoch is not None:
                warmup_factor = min(1.0, max(0.0, (epoch - 1) / 5.0))
                cldice_weight *= warmup_factor

            total_loss = focal_loss + tversky_loss + cldice_weight * loss_cldice_ll

            return {
                "total": total_loss,
                "focal": focal_loss,
                "tversky": tversky_loss,
                "cldice_ll": loss_cldice_ll,
            }


class TotalLoss(nn.Module):
    def __init__(self, config: LossConfig = None):
        super().__init__()

        if config is None:
            config = LossConfig()

        self.config = config

        tversky_da_alpha, tversky_da_gamma = (config.tversky_da_alpha, config.tversky_da_gamma)
        tversky_ll_alpha, tversky_ll_gamma = (config.tversky_ll_alpha, config.tversky_ll_gamma)
        focal_alpha, focal_gamma = config.focal_alpha, config.focal_gamma

        self.tver_da = TverskyLoss(
            mode=MULTICLASS_MODE,
            alpha=tversky_da_alpha,
            beta=1.0 - tversky_da_alpha,
            gamma=tversky_da_gamma,
            from_logits=True,
        )
        self.tver_ll = TverskyLoss(
            mode=MULTICLASS_MODE,
            classes=[1],
            alpha=tversky_ll_alpha,
            beta=1.0 - tversky_ll_alpha,
            gamma=tversky_ll_gamma,
            from_logits=True,
        )

        self.focal_da = FocalLoss(mode=MULTICLASS_MODE, alpha=focal_alpha, gamma=focal_gamma, ohem_ratio=config.ohem_ratio_da)
        self.focal_ll = FocalLoss(mode=MULTICLASS_MODE, alpha=focal_alpha, gamma=focal_gamma, ohem_ratio=config.ohem_ratio_ll)

        self.lovasz_da = LovaszLoss(classes=[1])
        self.cldice_ll = clDiceLoss(iterations=5)

    def forward(self, outputs, targets, epoch=None):
        out_da, out_ll = outputs
        target_da, target_ll = targets

        target_da = target_da.long()
        target_ll = target_ll.long()

        loss_tver_da = self.tver_da(out_da, target_da)
        loss_tver_ll = self.tver_ll(out_ll, target_ll)

        loss_focal_da = self.focal_da(out_da, target_da)
        loss_focal_ll = self.focal_ll(out_ll, target_ll)

        pred_da_probs = F.softmax(out_da, dim=1)
        loss_lovasz_da = self.lovasz_da(pred_da_probs, target_da)

        pred_ll_probs = F.softmax(out_ll, dim=1)
        target_ll_onehot = F.one_hot(target_ll, num_classes=2).permute(0, 3, 1, 2).float()
        loss_cldice_ll = self.cldice_ll(pred_ll_probs[:, 1:2], target_ll_onehot[:, 1:2])

        tversky_loss = loss_tver_da + loss_tver_ll
        focal_loss = loss_focal_da + loss_focal_ll

        lovasz_weight = getattr(self.config, "lovasz_da_weight", 0.1)
        cldice_weight = getattr(self.config, "cldice_ll_weight", 0.1)

        if epoch is not None:
            warmup_factor = min(1.0, max(0.0, (epoch - 1) / 5.0))
            lovasz_weight *= warmup_factor
            cldice_weight *= warmup_factor

        total_loss = focal_loss + tversky_loss + lovasz_weight * loss_lovasz_da + cldice_weight * loss_cldice_ll

        return {
            "total": total_loss,
            "focal": focal_loss,
            "tversky": tversky_loss,
            "lovasz_da": loss_lovasz_da,
            "cldice_ll": loss_cldice_ll,
        }
