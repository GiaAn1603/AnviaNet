import torch
from tqdm import tqdm

from utils.metrics import AverageMeter, SegmentationMetric


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, max_epochs, scheduler=None):
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
            loss_dict = criterion(predictions, drivable_area_targets, lane_line_targets)
            loss = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

    for images, drivable_area_targets, lane_line_targets in progress_bar:
        images = images.to(device)
        drivable_area_targets = drivable_area_targets.to(device)
        lane_line_targets = lane_line_targets.to(device)

        drivable_area_predictions, lane_line_predictions = model(images)
        drivable_area_predictions = torch.argmax(drivable_area_predictions, dim=1)
        lane_line_predictions = torch.argmax(lane_line_predictions, dim=1)

        drivable_area_metric.add_batch(drivable_area_predictions, drivable_area_targets)
        lane_line_metric.add_batch(lane_line_predictions, lane_line_targets)

    drivable_area_miou = drivable_area_metric.mean_intersection_over_union()
    lane_line_accuracy = lane_line_metric.class_accuracy(lane_line_class_id)
    lane_line_iou = lane_line_metric.class_intersection_over_union(lane_line_class_id)

    print("\n" + "=" * 50)
    print(f"[EVAL] Results Summary")
    print("-" * 50)
    print(f"Drivable Area mIoU: {drivable_area_miou * 100:>10.2f}%")
    print(f"Lane Line Accuracy: {lane_line_accuracy * 100:>10.2f}%")
    print(f"Lane Line IoU     : {lane_line_iou * 100:>10.2f}%")
    print("=" * 50 + "\n")

    return drivable_area_miou, lane_line_accuracy, lane_line_iou
