import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from core.dataset import BDD100KDataset
from core.model import AnviaNet
from core.loss import TotalLoss
from utils.engine import train_one_epoch, evaluate


def parse_arguments():
    parser = argparse.ArgumentParser(description="AnviaNet Training Pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data_group = parser.add_argument_group("Dataset & IO")
    data_group.add_argument("--data_root", type=str, required=True, help="Path to BDD100K dataset")
    data_group.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    data_group.add_argument("--num_workers", type=int, default=4, help="Data loader workers")

    opt_group = parser.add_argument_group("Optimization Strategy")
    opt_group.add_argument("--epochs", type=int, default=50, help="Total epochs")
    opt_group.add_argument("--batch_size", type=int, default=16, help="Batch size")
    opt_group.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    opt_group.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")

    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Initializing AnviaNet pipeline on: {device.type.upper()}")

    best_mean_iou = 0.0
    best_da_miou = 0.0
    best_ll_miou = 0.0

    print("[DATA] Loading BDD100K multi-task dataset...")
    train_loader = DataLoader(
        dataset=BDD100KDataset(data_root=args.data_root, is_train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=BDD100KDataset(data_root=args.data_root, is_train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("[MODEL] Assembling AnviaNet architecture...")
    model = AnviaNet().to(device)
    criterion = TotalLoss()
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print("[TRAIN] Beginning training process...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch [{epoch}/{args.epochs}] ---")

        average_train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            max_epochs=args.epochs,
        )

        da_miou, ll_miou = evaluate(model=model, dataloader=val_loader, device=device)
        current_mean_iou = (da_miou + ll_miou) / 2.0

        if current_mean_iou > best_mean_iou:
            best_mean_iou = current_mean_iou
            best_da_miou = da_miou
            best_ll_miou = ll_miou

            save_path = os.path.join(args.save_dir, "best_anvianet_model.pth")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_iou": best_mean_iou,
                "da_miou": da_miou,
                "ll_miou": ll_miou,
                "train_loss": average_train_loss,
            }
            torch.save(checkpoint, save_path)
            print(f"[SAVE] New Best Mean IoU: {best_mean_iou:.4f} -> {save_path}")

    last_save_path = os.path.join(args.save_dir, "last_anvianet_model.pth")
    last_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "mean_iou": best_mean_iou,
        "da_miou": da_miou,
        "ll_miou": ll_miou,
        "train_loss": average_train_loss,
    }
    torch.save(last_checkpoint, last_save_path)

    print("\n" + "=" * 50)
    print(f"{'TRAINING COMPLETED':^50}")
    print("-" * 50)
    print(f"Peak Mean IoU: {best_mean_iou:10.4f}")
    print(f"Best DA mIoU : {best_da_miou*100:10.2f}%")
    print(f"Best LL mIoU : {best_ll_miou*100:10.2f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
