import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_value = 0
        self.total_sum = 0
        self.count = 0
        self.average = 0

    def update(self, value, batch_size):
        self.current_value = value
        self.total_sum += value * batch_size
        self.count += batch_size
        self.average = self.total_sum / self.count if self.count != 0 else 0


class SegmentationMetric:
    def __init__(self, class_count):
        self.class_count = class_count
        self.confusion_matrix = None

    def reset(self):
        self.confusion_matrix = None

    @torch.no_grad()
    def add_batch(self, predictions, targets):
        if self.confusion_matrix is None:
            self.confusion_matrix = torch.zeros((self.class_count, self.class_count), dtype=torch.int64, device=predictions.device)

        valid_mask = (targets >= 0) & (targets < self.class_count)
        flat_indices = self.class_count * targets[valid_mask] + predictions[valid_mask]

        batch_confusion = torch.bincount(flat_indices, minlength=self.class_count**2).reshape(self.class_count, self.class_count)
        self.confusion_matrix += batch_confusion

    def intersection_over_union(self):
        if self.confusion_matrix is None:
            return torch.zeros(self.class_count, dtype=torch.float32)

        float_confusion_matrix = self.confusion_matrix.to(dtype=torch.float32)

        intersection = torch.diag(float_confusion_matrix)
        ground_truth_sum = float_confusion_matrix.sum(dim=1)
        prediction_sum = float_confusion_matrix.sum(dim=0)

        union = ground_truth_sum + prediction_sum - intersection
        intersection_over_union = intersection / (union + 1e-15)

        return intersection_over_union

    def mean_intersection_over_union(self):
        intersection_over_union = self.intersection_over_union()
        mean_intersection_over_union = intersection_over_union.mean().item()

        return mean_intersection_over_union
