
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from data_loader import SODDataset, create_dataloaders
from sod_model import get_model
from train import iou_score, precision_recall_f1


def mae_score(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.abs(preds - targets).mean()


def load_trained_model(checkpoint_path: str | Path, model_name: str, device: torch.device):
    model = get_model(model_name).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = np.zeros_like(image)
    heat[..., 0] = mask
    overlay = (1 - alpha) * image + alpha * heat
    return np.clip(overlay, 0, 1)


def save_visualization(
    image_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    output_path: Path,
) -> None:
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    gt = gt_tensor.squeeze(0).cpu().numpy()
    pred = pred_tensor.squeeze(0).cpu().numpy()
    overlay = make_overlay(image, pred)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Prediction")
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader = create_dataloaders(
        processed_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = load_trained_model(args.checkpoint, args.model, device)

    totals = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mae": 0.0}
    batches = 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="test"):
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)

            precision, recall, f1 = precision_recall_f1(preds, masks)
            totals["iou"] += iou_score(preds, masks).item()
            totals["precision"] += precision.item()
            totals["recall"] += recall.item()
            totals["f1"] += f1.item()
            totals["mae"] += mae_score(preds, masks).item()
            batches += 1

    metrics = {name: value / batches for name, value in totals.items()}
    print("Test metrics:", metrics)

    metrics_path = Path(args.output_dir) / "metrics" / f"{args.model}_test_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "iou", "precision", "recall", "f1", "mae"])
        writer.writerow([args.model, metrics["iou"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["mae"]])
    print(f"Saved metrics to {metrics_path}")

    save_sample_visualizations(model, args, device)


def save_sample_visualizations(model, args: argparse.Namespace, device: torch.device) -> None:
    dataset = SODDataset(Path(args.data_root) / "test", image_size=args.image_size, augment=False)
    count = min(args.num_visuals, len(dataset))

    with torch.no_grad():
        for index in range(count):
            image, mask = dataset[index]
            pred = model(image.unsqueeze(0).to(device)).squeeze(0).cpu()
            output_path = Path(args.output_dir) / "visualizations" / f"{args.model}_sample_{index + 1}.png"
            save_visualization(image, mask, pred, output_path)
            print(f"Saved visualization: {output_path}")


def predict_single_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    model_name: str = "baseline",
    image_size: int = 128,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_trained_model(checkpoint_path, model_name, resolved_device)

    original = Image.open(image_path).convert("RGB")
    resized = TF.resize(original, [image_size, image_size])
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(resolved_device)

    start = time.perf_counter()
    with torch.no_grad():
        pred = model(image_tensor)
    inference_time = time.perf_counter() - start

    mask = pred.squeeze().cpu().numpy()
    overlay = make_overlay(image_array, mask)
    return image_array, mask, overlay, inference_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize SOD predictions.")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--model", choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--checkpoint", default="checkpoints/baseline_best.pt")
    parser.add_argument("--image-size", type=int, choices=[128, 224], default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-visuals", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
