# Config.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class Config:
    # Repro / runtime
    seed: int = 114514
    device: str = "auto"  # auto|cpu|cuda|cuda:0

    # Data
    train_dir: str = "./dataset/NEU-DET/train/images"
    val_dir: str = "./dataset/NEU-DET/validation/images"

    img_size: int = 200  # pixel

    # Loader
    batch_size: int = 128
    num_workers: int = 8

    # Optim
    epochs: int = 20
    lr: float = 0.001
    weight_decay: float = 0.01
    momentum: float = 0.9

    # Output
    run_dir: str = "./results/runs"
    eval_result_dir: str = "./results/evals"
    gate_result_dir: str = "./results/gate"
    run_name: Optional[str] = None  # None => timestamp

    # ... data/train ...
    model_name: str = "resnet18"
    model_pretrained: bool = False
    train_freeze_backbone: bool = False

    # storge
    def __post_init__(self) -> None:
        # Resolve project-local paths against repository root so relative config
        # values work even when scripts are launched from different CWDs.
        self.train_dir = str(resolve_project_path(self.train_dir))
        self.val_dir = str(resolve_project_path(self.val_dir))
        self.run_dir = str(resolve_project_path(self.run_dir))
        self.eval_result_dir = str(resolve_project_path(self.eval_result_dir))
        self.gate_result_dir = str(resolve_project_path(self.gate_result_dir))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / p).resolve()
