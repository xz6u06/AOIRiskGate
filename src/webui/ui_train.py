from __future__ import annotations

import json
from pathlib import Path

from webui.ui_paths import safe_file
from webui.ui_runner import bool_flag


def build_train_args(
    train_dir: str,
    val_dir: str,
    device: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    model: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    momentum: float,
    run_dir: str,
    run_name: str,
    pretrained: bool,
    freeze_backbone: bool,
) -> list[str]:
    args = [
        "--train-dir",
        train_dir,
        "--val-dir",
        val_dir,
        "--device",
        device,
        "--img-size",
        str(img_size),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--model",
        model,
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--momentum",
        str(momentum),
        "--run-dir",
        run_dir,
    ]
    if run_name.strip():
        args.extend(["--run-name", run_name.strip()])

    args.extend(bool_flag(pretrained, "--pretrained", "--no-pretrained"))
    args.extend(bool_flag(freeze_backbone, "--freeze-backbone", "--no-freeze-backbone"))
    return args


def load_train_outputs(run_dir: str, run_name: str) -> tuple[str | None, dict]:
    root = Path(run_dir) / run_name

    curve = safe_file(root / "loss.png")
    metrics_path = root / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as e:
            metrics = {"error": f"Failed to parse metrics.json: {e}"}
    else:
        metrics = {"hint": "No metrics.json yet."}

    return curve, metrics
