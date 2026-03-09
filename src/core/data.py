from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from .config import Config
except ImportError:
    from config import Config

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class NEUDetImageFolder(Dataset):
    """Minimal ImageFolder variant with optional path returns.

    root/
      class_a/xxx.jpg
      class_b/yyy.jpg
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        return_paths: bool = False,
    ):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")

        self.transform = transform
        self.return_paths = return_paths

        self.classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        if not self.classes:
            raise RuntimeError(f"No class folders found under: {self.root_dir}")

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            label = self.class_to_idx[cls_name]
            for img_path in sorted(cls_dir.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"No image files found under: {self.root_dir}")

        self.targets = [y for _, y in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, label, str(img_path)
        return img, label


def build_transforms(cfg: Config, mean=IMG_MEAN, std=IMG_STD):
    train_tfm = transforms.Compose(
        [
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_tfm = transforms.Compose(
        [
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tfm, val_tfm


def build_dataloader(cfg: Config, split: str) -> DataLoader:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    train_tfm, val_tfm = build_transforms(cfg)
    is_val = split == "val"

    root = cfg.val_dir if is_val else cfg.train_dir
    ds = NEUDetImageFolder(
        root, transform=(val_tfm if is_val else train_tfm), return_paths=is_val
    )

    # cuda: pin_memory speeds host->device copies
    pin_memory = str(cfg.device).startswith("cuda") or (
        cfg.device == "auto" and torch.cuda.is_available()
    )
    persistent_workers = cfg.num_workers > 0

    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=not is_val,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


if __name__ == "__main__":
    cfg = Config(
        device=Config.device,
        val_dir=Config.val_dir,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        img_size=Config.img_size,
    )
    loader = build_dataloader(cfg, "val")
    class_names = loader.dataset.classes
    n_class = len(class_names)
    print(f"class_names = {', '.join(class_names)}\nn_class = {n_class}")
