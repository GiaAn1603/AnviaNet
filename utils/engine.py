import torch
from tqdm import tqdm
from utils.metrics import AverageMeter, SegmentationMetric


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, max_epochs, scheduler=None):
    model.train()
    loss_meter = AverageMeter()
    pbar = tqdm(dataloader, total=len(dataloader), bar_format="{l_bar}{bar:10}{r_bar}")

    for images, targets_da, targets_ll in pbar:
        images = images.to(device)
        targets_da = targets_da.to(device)
        targets_ll = targets_ll.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out_da, out_ll = model(images)
            loss_dict = criterion((out_da, out_ll), (targets_da, targets_ll))
            loss = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))
        pbar.set_description(f"Epoch [{epoch}/{max_epochs}] | Total Loss: {loss_meter.average:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if scheduler:
        scheduler.step()

    return loss_meter.average


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes=2, lane_class_id=1):
    model.eval()
    da_metric = SegmentationMetric(num_classes=num_classes)
    ll_metric = SegmentationMetric(num_classes=num_classes)
    pbar = tqdm(dataloader, total=len(dataloader), desc="Evaluating")

    for images, targets_da, targets_ll in pbar:
        images = images.to(device)
        targets_da = targets_da.to(device)
        targets_ll = targets_ll.to(device)

        out_da, out_ll = model(images)

        preds_da = torch.argmax(out_da, dim=1)
        preds_ll = torch.argmax(out_ll, dim=1)

        da_metric.add_batch(preds_da, targets_da)
        ll_metric.add_batch(preds_ll, targets_ll)

    da_miou = da_metric.mean_intersection_over_union()

    ll_acc = ll_metric.class_accuracy(lane_class_id)
    ll_iou = ll_metric.class_iou(lane_class_id)

    print("\n" + "=" * 50)
    print(f"[EVAL] Results Summary")
    print("-" * 50)
    print(f"Drivable Area mIoU: {da_miou * 100:>10.2f}%")
    print(f"Lane Line Accuracy: {ll_acc * 100:>10.2f}%")
    print(f"Lane Line IoU     : {ll_iou * 100:>10.2f}%")
    print("=" * 50 + "\n")

    return da_miou, ll_acc, ll_iou
