import math
from copy import deepcopy

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from utils.metrics import AverageMeter, SegmentationMetric


class EMAModel:
    def __init__(self, model, initial_decay, initial_updates):
        self.ema_model = deepcopy(model).eval()
        self.base_decay = initial_decay
        self.updates_count = initial_updates

        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    def _calculate_decay(self, current_updates):
        decay_ramp = 1.0 - math.exp(-current_updates / 2000.0)
        current_decay = self.base_decay * decay_ramp

        return current_decay

    @torch.no_grad()
    def update(self, active_model):
        self.updates_count += 1
        current_decay = self._calculate_decay(self.updates_count)
        active_state_dict = active_model.state_dict()

        for name, value in self.ema_model.state_dict().items():
            if value.dtype.is_floating_point:
                value *= current_decay

                active_value = active_state_dict[name].detach()
                residual_weight = 1.0 - current_decay
                value += residual_weight * active_value


class PolynomialDecayScheduler(LambdaLR):
    def __init__(self, optimizer, max_epochs, power):
        self.max_epochs = max_epochs
        self.power = power

        super().__init__(optimizer, lr_lambda=self._calculate_decay)

    def _calculate_decay(self, current_epoch):
        decay_ratio = current_epoch / self.max_epochs
        decay_factor = (1.0 - decay_ratio) ** self.power

        return decay_factor


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, max_epochs, ema=None, scheduler=None):
    model.train()
    loss_meter = AverageMeter()
    progress_bar = tqdm(dataloader, total=len(dataloader), bar_format="{l_bar}{bar:10}{r_bar}")

    for images, drivable_area_targets, lane_line_targets in progress_bar:
        images = images.to(device)
        drivable_area_targets = drivable_area_targets.to(device)
        lane_line_targets = lane_line_targets.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            predictions = model(images)
            loss_dict = criterion(predictions, drivable_area_targets, lane_line_targets, epoch=epoch)
            loss = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        loss_meter.update(loss.item(), images.size(dim=0))
        progress_bar.set_description(f"Epoch [{epoch}/{max_epochs}] | Total Loss: {loss_meter.average:.4f} | Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    if scheduler:
        scheduler.step()

    return loss_meter.average


@torch.no_grad()
def evaluate(model, dataloader, device, class_count, lane_line_class_id):
    model.eval()
    drivable_area_metric = SegmentationMetric(class_count=class_count)
    lane_line_metric = SegmentationMetric(class_count=class_count)
    progress_bar = tqdm(dataloader, total=len(dataloader), desc="Evaluating")

    drivable_area_miou_sum = 0.0
    lane_line_accuracy_sum = 0.0
    lane_line_iou_sum = 0.0
    total_samples = 0

    for images, drivable_area_targets, lane_line_targets in progress_bar:
        images = images.to(device)
        drivable_area_targets = drivable_area_targets.to(device)
        lane_line_targets = lane_line_targets.to(device)
        batch_size = images.size(dim=0)

        drivable_area_predictions, lane_line_predictions = model(images)
        drivable_area_predictions = torch.argmax(drivable_area_predictions, dim=1)
        lane_line_predictions = torch.argmax(lane_line_predictions, dim=1)

        drivable_area_metric.reset()
        lane_line_metric.reset()

        drivable_area_metric.add_batch(drivable_area_predictions, drivable_area_targets)
        lane_line_metric.add_batch(lane_line_predictions, lane_line_targets)

        drivable_area_miou_sum += drivable_area_metric.mean_intersection_over_union() * batch_size
        lane_line_accuracy_sum += lane_line_metric.class_accuracy(lane_line_class_id) * batch_size
        lane_line_iou_sum += lane_line_metric.class_intersection_over_union(lane_line_class_id) * batch_size
        total_samples += batch_size

    drivable_area_miou = drivable_area_miou_sum / total_samples if total_samples > 0 else 0.0
    lane_line_accuracy = lane_line_accuracy_sum / total_samples if total_samples > 0 else 0.0
    lane_line_iou = lane_line_iou_sum / total_samples if total_samples > 0 else 0.0

    print("\n" + "=" * 50)
    print(f"[EVAL] Results Summary")
    print("-" * 50)
    print(f"Drivable Area mIoU: {drivable_area_miou * 100:>10.2f}%")
    print(f"Lane Line Accuracy: {lane_line_accuracy * 100:>10.2f}%")
    print(f"Lane Line IoU     : {lane_line_iou * 100:>10.2f}%")
    print("=" * 50 + "\n")

    return drivable_area_miou, lane_line_accuracy, lane_line_iou
