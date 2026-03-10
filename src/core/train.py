# train.py
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .config import Config
    from .data import build_dataloader
    from .engine import run_train
    from .models import build_model
except ImportError:
    from config import Config
    from data import build_dataloader
    from engine import run_train
    from models import build_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train classifier")
    p.add_argument("--train-dir", default=Config.train_dir)
    p.add_argument("--val-dir", default=Config.val_dir)

    p.add_argument("--device", default=Config.device)
    p.add_argument("--seed", type=int, default=Config.seed)

    p.add_argument("--img-size", type=int, default=Config.img_size)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--num-workers", type=int, default=Config.num_workers)

    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    p.add_argument("--momentum", type=float, default=Config.momentum)

    p.add_argument("--run-dir", default=Config.run_dir)
    p.add_argument("--run-name", default=None)

    p.add_argument("--model", default=Config.model_name, help="simple|resnet18")

    p.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=True,  # 預設開啟
        help="Use ImageNet pretrained weights (default: ON).",
    )
    p.add_argument(
        "--no-pretrained",
        dest="pretrained",  # dest 修改pretrained變數
        action="store_false",  # 打了就關掉
        help="Disable ImageNet pretrained weights.",
    )
    p.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        default=True,  # 預設開啟
        help="Freeze backbone and train only fc (default: ON).",
    )
    p.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Train full model (disable freeze-backbone).",
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    cfg = Config(
        seed=args.seed,
        device=args.device,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        run_dir=args.run_dir,
        run_name=args.run_name,
        model_name=args.model,
        model_pretrained=args.pretrained,
        train_freeze_backbone=args.freeze_backbone,
    )

    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")

    n_class = len(val_loader.dataset.classes)
    model = build_model(args.model, n_class, pretrained=args.pretrained)

    run_path: Path = run_train(
        cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        freeze_backbone=args.freeze_backbone,
    )
    print(f"\nDone. Run saved to: {run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
