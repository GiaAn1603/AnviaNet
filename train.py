import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from core.dataset import BDD100KDataset
from core.loss import TotalLoss
from core.model import AnviaNet
from utils.engine import train_one_epoch, evaluate


def parse_arguments():
    parser = argparse.ArgumentParser(description="AnviaNet Training Pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data_group = parser.add_argument_group("Dataset & IO")
    data_group.add_argument("--data_root_path", type=str, required=True, help="Path to BDD100K dataset")
    data_group.add_argument("--checkpoint_directory", type=str, default="./checkpoints", help="Directory to save checkpoints")
    data_group.add_argument("--worker_count", type=int, default=4, help="Data loader workers")

    optimization_group = parser.add_argument_group("Optimization Strategy")
    optimization_group.add_argument("--epochs", type=int, default=100, help="Total epochs")
    optimization_group.add_argument("--batch_size", type=int, default=16, help="Batch size")
    optimization_group.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    optimization_group.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")

    return parser.parse_args()


def main():
    arguments = parse_arguments()
    os.makedirs(arguments.checkpoint_directory, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Device: {device.type.upper()}")

    print("[DATA] Loading BDD100K dataset...")
    train_loader = DataLoader(
        BDD100KDataset(arguments.data_root_path, is_train=True, image_size=(360, 640)),
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.worker_count,
        pin_memory=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        BDD100KDataset(arguments.data_root_path, is_train=False, image_size=(360, 640)),
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.worker_count,
        pin_memory=True,
    )

    print("[MODEL] Assembling AnviaNet model")
    model = AnviaNet().to(device)
    criterion = TotalLoss()
    optimizer = AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay)
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda")

    best_miou = 0.0
    best_drivable_area_miou = 0.0
    best_lane_line_miou = 0.0

    print("[TRAIN] Beginning training process...")
    for epoch in range(1, arguments.epochs + 1):
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
        )

        drivable_area_miou, lane_line_miou = evaluate(model, validation_loader, device)
        current_miou = (drivable_area_miou + lane_line_miou) / 2.0

        if current_miou > best_miou:
            best_miou = current_miou
            best_drivable_area_miou = drivable_area_miou
            best_lane_line_miou = lane_line_miou

            save_path = os.path.join(arguments.checkpoint_directory, "best_anvianet_model.pth")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "miou": best_miou,
                "drivable_area_miou": drivable_area_miou,
                "lane_line_iou": lane_line_miou,
                "average_train_loss": average_train_loss,
            }
            torch.save(checkpoint, save_path)
            print(f"[SAVE] Best mIoU: {best_miou:.4f} -> {save_path}")

    last_save_path = os.path.join(arguments.checkpoint_directory, "last_anvianet_model.pth")
    last_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "miou": best_miou,
        "drivable_area_miou": drivable_area_miou,
        "lane_line_iou": lane_line_miou,
        "average_train_loss": average_train_loss,
    }
    torch.save(last_checkpoint, last_save_path)

    print("\n" + "=" * 50)
    print(f"{'TRAINING COMPLETED':^50}")
    print("-" * 50)
    print(f"Best mIoU              : {best_miou:10.4f}")
    print(f"Best Drivable Area mIoU: {best_drivable_area_miou*100:10.2f}%")
    print(f"Best Lane Line mIoU    : {best_lane_line_miou*100:10.2f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
