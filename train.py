import argparse
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from core.config import AnviaNetConfig
from core.dataset import BDD100KDataset
from core.loss import TotalLoss
from core.model import AnviaNet
from utils.engine import train_one_epoch, evaluate
from utils.metrics import get_model_complexity


def parse_arguments():
    parser = argparse.ArgumentParser(description="AnviaNet Training Pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data_group = parser.add_argument_group("Dataset & IO")
    data_group.add_argument("--data_root_path", type=str, required=True, help="Path to BDD100K dataset")
    data_group.add_argument("--checkpoint_directory", type=str, default="./checkpoints", help="Directory to save checkpoints")
    data_group.add_argument("--worker_count", type=int, default=4, help="Data loader workers")
    data_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--image_height", type=int, default=360, help="Target image height")
    model_group.add_argument("--image_width", type=int, default=640, help="Target image width")
    model_group.add_argument("--resume", type=str, default="", help="Resume from checkpoint")

    optimization_group = parser.add_argument_group("Optimization Strategy")
    optimization_group.add_argument("--epochs", type=int, default=100, help="Total epochs")
    optimization_group.add_argument("--batch_size", type=int, default=16, help="Batch size")
    optimization_group.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    optimization_group.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")

    arguments, _ = parser.parse_known_args()

    return arguments


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    arguments = parse_arguments()
    seed_everything(arguments.seed)
    os.makedirs(arguments.checkpoint_directory, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Device: {device.type.upper()}")

    print("[DATA] Loading BDD100K dataset...")
    train_loader = DataLoader(
        BDD100KDataset(arguments.data_root_path, is_train=True, image_size=(arguments.image_height, arguments.image_width)),
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.worker_count,
        pin_memory=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        BDD100KDataset(arguments.data_root_path, is_train=False, image_size=(arguments.image_height, arguments.image_width)),
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.worker_count,
        pin_memory=True,
    )

    print("[MODEL] Assembling AnviaNet model")
    config = AnviaNetConfig(image_height=arguments.image_height, image_width=arguments.image_width)
    model = AnviaNet(config).to(device)
    criterion = TotalLoss(config.loss)
    optimizer = AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=arguments.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda")

    start_epoch = 1
    best_miou = 0.0
    best_drivable_area_miou = 0.0
    best_lane_line_accuracy = 0.0
    best_lane_line_iou = 0.0

    if arguments.resume and os.path.exists(arguments.resume):
        print(f"[RESUME] Loading: {arguments.resume}")
        checkpoint = torch.load(arguments.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_miou = checkpoint.get("miou", 0.0)
        best_drivable_area_miou = checkpoint.get("drivable_area_miou", 0.0)
        best_lane_line_accuracy = checkpoint.get("lane_line_accuracy", 0.0)
        best_lane_line_iou = checkpoint.get("lane_line_iou", 0.0)

        print(f"[RESUME] Resumed from Epoch {start_epoch} (Best mIoU: {best_miou:.4f} | Learning Rate: {optimizer.param_groups[0]['lr']:.6f})")
    elif arguments.resume:
        print("[WARN] Checkpoint not found. Starting from scratch.")

    print("[TRAIN] Beginning training process...")
    for epoch in range(start_epoch, arguments.epochs + 1):
        print(f"\n--- Epoch [{epoch}/{arguments.epochs}] ---")

        average_train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch=epoch,
            max_epochs=arguments.epochs,
            scheduler=scheduler,
        )

        drivable_area_miou, lane_line_accuracy, lane_line_iou = evaluate(model, validation_loader, device, class_count=config.class_count, lane_line_class_id=config.loss.lane_line_class_id)
        current_miou = (drivable_area_miou + lane_line_iou) / 2.0

        if current_miou > best_miou:
            best_miou = current_miou
            best_drivable_area_miou = drivable_area_miou
            best_lane_line_accuracy = lane_line_accuracy
            best_lane_line_iou = lane_line_iou

            save_path = os.path.join(arguments.checkpoint_directory, "best_anvianet_model.pth")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "miou": best_miou,
                "drivable_area_miou": drivable_area_miou,
                "lane_line_accuracy": lane_line_accuracy,
                "lane_line_iou": lane_line_iou,
                "average_train_loss": average_train_loss,
            }
            torch.save(checkpoint, save_path)
            print(f"[SAVE] Best mIoU: {best_miou:.4f} -> {save_path}")

        last_save_path = os.path.join(arguments.checkpoint_directory, "last_anvianet_model.pth")
        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "miou": best_miou,
            "drivable_area_miou": drivable_area_miou,
            "lane_line_accuracy": lane_line_accuracy,
            "lane_line_iou": lane_line_iou,
            "average_train_loss": average_train_loss,
        }
        torch.save(last_checkpoint, last_save_path)

    print("\n" + "=" * 50)
    print(f"{'TRAINING COMPLETED':^50}")
    print("-" * 50)
    print(f"Best mIoU              : {best_miou:10.4f}")
    print(f"Best Drivable Area mIoU: {best_drivable_area_miou*100:10.2f}%")
    print(f"Best Lane Line Accuracy: {best_lane_line_accuracy*100:10.2f}%")
    print(f"Best Lane Line IoU     : {best_lane_line_iou*100:10.2f}%")
    print("-" * 50)
    flops, parameters = get_model_complexity(model, input_size=(1, 3, arguments.image_height, arguments.image_width), device=device)
    print(f"Complexity             : FLOPs: {flops} | Parameters: {parameters}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
