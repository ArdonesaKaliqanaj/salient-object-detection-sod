
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
   
    image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    mask_files = {p.stem: p for p in masks_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}

    pairs = []
    missing_masks = []
    for image_path in sorted(image_files):
        mask_path = mask_files.get(image_path.stem)
        if mask_path is None:
            missing_masks.append(image_path.name)
        else:
            pairs.append((image_path, mask_path))

    if missing_masks:
        preview = ", ".join(missing_masks[:5])
        print(f"Warning: {len(missing_masks)} images do not have matching masks. Examples: {preview}")

    return pairs


def inspect_dataset(raw_root: str | Path = "data/raw/ECSSD") -> None:
   
    raw_root = Path(raw_root)
    images_dir = raw_root / "images"
    masks_dir = raw_root / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(
            "Expected data/raw/ECSSD/images and data/raw/ECSSD/masks. "
            "Download ECSSD images and masks, unzip them, and place files there."
        )

    pairs = find_pairs(images_dir, masks_dir)
    print(f"Images folder: {images_dir.resolve()}")
    print(f"Masks folder:  {masks_dir.resolve()}")
    print(f"Matched image/mask pairs: {len(pairs)}")

    if pairs:
        image = Image.open(pairs[0][0])
        mask = Image.open(pairs[0][1])
        print(f"Example image: {pairs[0][0].name}, size={image.size}, mode={image.mode}")
        print(f"Example mask:  {pairs[0][1].name}, size={mask.size}, mode={mask.mode}")


def copy_split(pairs: list[tuple[Path, Path]], split_name: str, output_root: Path) -> None:
    image_out = output_root / split_name / "images"
    mask_out = output_root / split_name / "masks"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in pairs:
        shutil.copy2(image_path, image_out / image_path.name)
        shutil.copy2(mask_path, mask_out / mask_path.name)


def prepare_dataset(
    raw_root: str | Path = "data/raw/ECSSD",
    output_root: str | Path = "data/processed",
    seed: int = 42,
) -> None:
    """Create Train 70%, Validation 15%, Test 15% folders."""
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    pairs = find_pairs(raw_root / "images", raw_root / "masks")

    if len(pairs) == 0:
        raise RuntimeError("No image/mask pairs found. Check your ECSSD folder names.")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    for split_name, split_pairs in splits.items():
        copy_split(split_pairs, split_name, output_root)
        print(f"{split_name}: {len(split_pairs)} pairs")

    print(f"Prepared dataset at: {output_root.resolve()}")


class SODDataset(Dataset):
    

    def __init__(
        self,
        split_root: str | Path,
        image_size: int = 128,
        augment: bool = False,
    ) -> None:
        self.split_root = Path(split_root)
        self.image_size = image_size
        self.augment = augment
        self.pairs = find_pairs(self.split_root / "images", self.split_root / "masks")

        if not self.pairs:
            raise RuntimeError(f"No pairs found in {self.split_root}. Run data_loader.py --prepare first.")

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment_pair(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
   
        larger_size = int(self.image_size * 1.15)
        image = TF.resize(image, [larger_size, larger_size])
        mask = TF.resize(mask, [larger_size, larger_size], interpolation=TF.InterpolationMode.NEAREST)

        top = random.randint(0, larger_size - self.image_size)
        left = random.randint(0, larger_size - self.image_size)
        image = TF.crop(image, top, left, self.image_size, self.image_size)
        mask = TF.crop(mask, top, left, self.image_size, self.image_size)

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

       
        brightness_factor = random.uniform(0.8, 1.2)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self._augment_pair(image, mask)
        else:
            image = TF.resize(image, [self.image_size, self.image_size])
            mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.float32) / 255.0
        mask_array = (mask_array >= 0.5).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor


def create_dataloaders(
    processed_root: str | Path = "data/processed",
    image_size: int = 128,
    batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    processed_root = Path(processed_root)

    train_dataset = SODDataset(processed_root / "train", image_size=image_size, augment=True)
    val_dataset = SODDataset(processed_root / "val", image_size=image_size, augment=False)
    test_dataset = SODDataset(processed_root / "test", image_size=image_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or split the ECSSD SOD dataset.")
    parser.add_argument("--raw-root", default="data/raw/ECSSD")
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()

    if args.inspect:
        inspect_dataset(args.raw_root)
    if args.prepare:
        prepare_dataset(args.raw_root, args.output_root, args.seed)
    if not args.inspect and not args.prepare:
        parser.print_help()


if __name__ == "__main__":
    main()
