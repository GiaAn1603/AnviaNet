import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_value = 0
        self.average = 0
        self.total_sum = 0
        self.count = 0

    def update(self, value, batch_size=1):
        self.current_value = value
        self.total_sum += value * batch_size
        self.count += batch_size
        self.average = self.total_sum / self.count if self.count != 0 else 0


class SegmentationMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = None

    def reset(self):
        self.confusion_matrix = None

    def add_batch(self, predictions, targets):
        if self.confusion_matrix is None:
            self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=predictions.device)

        with torch.no_grad():
            valid_mask = (targets >= 0) & (targets < self.num_classes)
            flat_indices = self.num_classes * targets[valid_mask] + predictions[valid_mask]
            self.confusion_matrix += torch.bincount(flat_indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def intersection_over_union(self):
        intersection = torch.diag(self.confusion_matrix)
        ground_truth_sum = self.confusion_matrix.sum(dim=1)
        prediction_sum = self.confusion_matrix.sum(dim=0)
        union = ground_truth_sum + prediction_sum - intersection
        iou = intersection / (union + 1e-15)

        return iou

    def mean_intersection_over_union(self):
        iou = self.intersection_over_union()

        return iou.mean().item()
