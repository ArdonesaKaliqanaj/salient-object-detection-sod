from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from data_loader import create_dataloaders
from sod_model import get_model


def iou_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    preds_bin = (preds >= threshold).float()
    targets_bin = (targets >= threshold).float()
    intersection = (preds_bin * targets_bin).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + targets_bin.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def precision_recall_f1(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds_bin = (preds >= threshold).float()
    targets_bin = (targets >= threshold).float()

    tp = (preds_bin * targets_bin).sum()
    fp = (preds_bin * (1 - targets_bin)).sum()
    fn = ((1 - preds_bin) * targets_bin).sum()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


class SODLoss(nn.Module):
  

    def __init__(self, iou_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.iou_weight = iou_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(preds, targets)
        soft_intersection = (preds * targets).sum(dim=(1, 2, 3))
        soft_union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - soft_intersection
        soft_iou = ((soft_intersection + 1e-7) / (soft_union + 1e-7)).mean()
        return bce_loss + self.iou_weight * (1 - soft_iou)


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    batches = 0

    progress = tqdm(loader, desc="train" if training else "val", leave=False)
    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            preds = model(images)
            loss = criterion(preds, masks)
            if training:
                loss.backward()
                optimizer.step()

        precision, recall, f1 = precision_recall_f1(preds.detach(), masks)
        total_loss += loss.item()
        total_iou += iou_score(preds.detach(), masks).item()
        total_precision += precision.item()
        total_recall += recall.item()
        total_f1 += f1.item()
        batches += 1

        progress.set_postfix(loss=loss.item())

    return {
        "loss": total_loss / batches,
        "iou": total_iou / batches,
        "precision": total_precision / batches,
        "recall": total_recall / batches,
        "f1": total_f1 / batches,
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val_loss: float, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path: Path, model, optimizer, device) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint["best_val_loss"])
    print(f"Resumed training from {path} at epoch {start_epoch}")
    return start_epoch, best_val_loss


def append_log(log_path: Path, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_iou",
                    "train_precision",
                    "train_recall",
                    "train_f1",
                    "val_loss",
                    "val_iou",
                    "val_precision",
                    "val_recall",
                    "val_f1",
                ]
            )
        writer.writerow(
            [
                epoch,
                train_metrics["loss"],
                train_metrics["iou"],
                train_metrics["precision"],
                train_metrics["recall"],
                train_metrics["f1"],
                val_metrics["loss"],
                val_metrics["iou"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
            ]
        )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = create_dataloaders(
        processed_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = get_model(args.model).to(device)
    criterion = SODLoss(iou_weight=args.iou_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    last_checkpoint = checkpoint_dir / f"{args.model}_last.pt"
    best_checkpoint = checkpoint_dir / f"{args.model}_best.pt"
    log_path = Path(args.output_dir) / "metrics" / f"{args.model}_training_log.csv"

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume and last_checkpoint.exists():
        start_epoch, best_val_loss = load_checkpoint(last_checkpoint, model, optimizer, device)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)

        append_log(log_path, epoch, train_metrics, val_metrics)
        print(f"Train: {train_metrics}")
        print(f"Val:   {val_metrics}")

        save_checkpoint(last_checkpoint, model, optimizer, epoch, best_val_loss, args.model)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(best_checkpoint, model, optimizer, epoch, best_val_loss, args.model)
            print("New best model saved.")
        else:
            epochs_without_improvement += 1
            print(f"No validation improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN encoder-decoder SOD model.")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--model", choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--image-size", type=int, choices=[128, 224], default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iou-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
