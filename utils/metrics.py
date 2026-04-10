import torch
from thop import profile


@torch.no_grad()
def get_model_complexity(model, batch_size, channels, height, width, device):
    model.eval()
    dummy_input = torch.randn(batch_size, channels, height, width).to(device)

    flops, parameters = profile(model, (dummy_input,), verbose=False)
    formatted_flops = f"{flops / 1e9:.2f}G"
    formatted_parameters = f"{parameters / 1e6:.2f}M"

    return formatted_flops, formatted_parameters


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

    def class_intersection_over_union(self, class_id):
        intersection_over_union = self.intersection_over_union()
        class_intersection_over_union = intersection_over_union[class_id].item() if class_id < len(intersection_over_union) else 0.0

        return class_intersection_over_union

    def class_accuracy(self, class_id):
        if self.confusion_matrix is None:
            return 0.0

        float_confusion_matrix = self.confusion_matrix.to(dtype=torch.float32)

        true_positive = float_confusion_matrix[class_id, class_id]
        false_negative = float_confusion_matrix[class_id, :].sum() - true_positive
        false_positive = float_confusion_matrix[:, class_id].sum() - true_positive
        true_negative = float_confusion_matrix.sum() - (true_positive + false_positive + false_negative)

        sensitivity = true_positive / (true_positive + false_negative + 1e-15)
        specificity = true_negative / (true_negative + false_positive + 1e-15)
        accuracy = ((sensitivity + specificity) / 2).item()

        return accuracy
