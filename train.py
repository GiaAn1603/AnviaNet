import os
import random
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from core.dataset import BDD100KDataset
from core.model import AnviaNet
from core.loss import TotalLoss
from core.config import AnviaNetConfig
from utils.engine import train_one_epoch, evaluate
from utils.metrics import get_model_complexity


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = argparse.ArgumentParser(description="AnviaNet Training Pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data_group = parser.add_argument_group("Dataset & IO")
    data_group.add_argument("--data_root", type=str, required=True, help="Path to BDD100K dataset")
    data_group.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    data_group.add_argument("--num_workers", type=int, default=4, help="Data loader workers")
    data_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--img_height", type=int, default=360, help="Target image height")
    model_group.add_argument("--img_width", type=int, default=640, help="Target image width")
    model_group.add_argument("--num_classes", type=int, default=2, help="Segmentation classes")
    model_group.add_argument("--lane_class_id", type=int, default=1, help="Class ID for lanes")
    model_group.add_argument("--resume", type=str, default="", help="Resume from checkpoint")

    opt_group = parser.add_argument_group("Optimization Strategy")
    opt_group.add_argument("--epochs", type=int, default=50, help="Total epochs")
    opt_group.add_argument("--batch_size", type=int, default=16, help="Batch size")
    opt_group.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    opt_group.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    opt_group.add_argument("--momentum", type=float, default=0.9, help="AdamW beta1 momentum")
    opt_group.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")

    args, _ = parser.parse_known_args()

    return args


def main():
    args = parse_arguments()
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Initializing AnviaNet pipeline on: {device.type.upper()}")

    best_mean_iou = 0.0
    best_da_miou = 0.0
    best_ll_iou = 0.0
    best_ll_acc = 0.0
    start_epoch = 1

    print("[DATA] Loading BDD100K multi-task dataset...")
    train_loader = DataLoader(
        dataset=BDD100KDataset(data_root=args.data_root, is_train=True, img_size=(args.img_height, args.img_width)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=BDD100KDataset(data_root=args.data_root, is_train=False, img_size=(args.img_height, args.img_width)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("[MODEL] Assembling AnviaNet architecture...")
    config = AnviaNetConfig(img_height=args.img_height, img_width=args.img_width, num_classes=args.num_classes, lane_class_id=args.lane_class_id)
    model = AnviaNet(config=config).to(device)
    criterion = TotalLoss(config=config.loss)
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(args.momentum, 0.999), eps=args.eps)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if args.resume and os.path.exists(args.resume):
        print(f"[RESUME] Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_mean_iou = checkpoint.get("mean_iou", 0.0)
        best_da_miou = checkpoint.get("da_miou", 0.0)
        best_ll_iou = checkpoint.get("ll_iou", 0.0)
        best_ll_acc = checkpoint.get("ll_acc", 0.0)

        print(
            f"[RESUME] Success: Continuing from Epoch {start_epoch} (Best Mean IoU: {best_mean_iou:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f})"
        )
    elif args.resume:
        print(f"[WARN] Resume path not found: {args.resume}. Starting from scratch.")

    print("[TRAIN] Beginning training process...")
    for epoch in range(start_epoch, args.epochs + 1):
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
            scheduler=scheduler,
        )

        da_miou, ll_acc, ll_iou = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            num_classes=args.num_classes,
            lane_class_id=args.lane_class_id,
        )

        current_mean_iou = (da_miou + ll_iou) / 2.0

        if current_mean_iou > best_mean_iou:
            best_mean_iou = current_mean_iou
            best_da_miou = da_miou
            best_ll_iou = ll_iou
            best_ll_acc = ll_acc

            save_path = os.path.join(args.save_dir, "best_anvianet_model.pth")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "mean_iou": best_mean_iou,
                "da_miou": da_miou,
                "ll_acc": ll_acc,
                "ll_iou": ll_iou,
                "train_loss": average_train_loss,
            }
            torch.save(checkpoint, save_path)
            print(f"[SAVE] New Best Mean IoU: {best_mean_iou:.4f} -> {save_path}")

        last_save_path = os.path.join(args.save_dir, "last_anvianet_model.pth")
        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "mean_iou": best_mean_iou,
            "da_miou": da_miou,
            "ll_acc": ll_acc,
            "ll_iou": ll_iou,
            "train_loss": average_train_loss,
        }
        torch.save(last_checkpoint, last_save_path)

    print("\n" + "=" * 50)
    print(f"{'TRAINING COMPLETED':^50}")
    print("-" * 50)
    print(f"Peak Mean IoU: {best_mean_iou:10.4f}")
    print(f"Best DA mIoU : {best_da_miou*100:10.2f}%")
    print(f"Best LL Acc  : {best_ll_acc*100:10.2f}%")
    print(f"Best LL IoU  : {best_ll_iou*100:10.2f}%")
    print("-" * 50)
    flops, params = get_model_complexity(model, input_size=(1, 3, args.img_height, args.img_width), device=device)
    print(f"Complexity   : FLOPs: {flops} | Params: {params}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
