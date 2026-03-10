# eval.py
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

try:
    from .config import Config, ensure_dir, resolve_device, resolve_project_path
    from .data import build_dataloader
    from .engine import evaluate_with_cm
    from .models import build_model
    from .utils import save_json
    from .viz import plot_anomaly_his, plot_confusion_matrix
except ImportError:
    from config import Config, ensure_dir, resolve_device, resolve_project_path
    from data import build_dataloader
    from engine import evaluate_with_cm
    from models import build_model
    from utils import save_json
    from viz import plot_anomaly_his, plot_confusion_matrix


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate classifier")
    p.add_argument("--val-dir", default=Config.val_dir)
    p.add_argument("--device", default=Config.device)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--num-workers", type=int, default=Config.num_workers)
    p.add_argument("--img-size", type=int, default=Config.img_size)

    p.add_argument("--model", default=Config.model_name, help="simple|resnet18")

    # Two input modes:
    #   1) --name <run_id> -> auto-pick ./results/runs/<run_id>/checkpoints/{best.pt,last.pt}
    #   2) --ckpt <path>   -> use the checkpoint directly
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="Run name/id, e.g. 20260202_152211")
    g.add_argument("--ckpt", help="Checkpoint path, e.g. /some/path/best.pt")

    p.add_argument(
        "--run-dir",
        default=Config.run_dir,
        help="Base runs dir (default: ./results/runs)",
    )
    p.add_argument(
        "--eval-result",
        default=Config.eval_result_dir,
        help="Base output dir for eval results (default: ./results/evals)",
    )

    # misclassified outputs
    p.add_argument(
        "--copy-misclassified",
        action="store_true",
        help="Copy misclassified images into <out>/imgs and write a val_misclassified.txt",
    )
    p.add_argument(
        "--max-misclassified",
        type=int,
        default=0,
        help="Limit number of misclassified images to copy (0 = no limit)",
    )

    return p


def _save_misclassified(
    out_root: Path, wrong_items, class_names: list[str], *, max_items: int = 0
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) write list
    txt_path = out_root / "val_misclassified.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("path\ttrue\tpred\n")
        for path, t, p in wrong_items:
            t_name = class_names[t] if 0 <= t < len(class_names) else str(t)
            p_name = class_names[p] if 0 <= p < len(class_names) else str(p)
            f.write(f"{path}\t{t_name}\t{p_name}\n")

    # 2) copy images
    img_dir = out_root / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for path, t, p in wrong_items:
        if max_items and copied >= max_items:
            break

        src_path = Path(path)
        p_name = class_names[p] if 0 <= p < len(class_names) else str(p)

        # keep filenames stable + include pred label for quick scanning
        dst_path = img_dir / f"pred_{p_name}__{src_path.name}"
        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
        except FileNotFoundError:
            print(f"Warning: could not find {src_path}")

    print(f"Saved misclassified list -> {txt_path}")
    print(f"Copied misclassified images: {copied} -> {img_dir}")


def _pick_ckpt_from_run(run_dir: Path, run_name: str) -> Path:
    ckpt_dir = run_dir / run_name / "checkpoints"
    best = ckpt_dir / "best.pt"
    last = ckpt_dir / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found. Tried: {best} and {last}.")


def _infer_run_name_from_ckpt_path(ckpt_path: Path, *, run_dir: Path) -> str:
    """Infer run name if path looks like .../<run_dir>/<run_name>/checkpoints/*.pt."""
    try:
        run_dir = run_dir.resolve()
        ckpt_resolved = ckpt_path.resolve()
    except Exception:
        run_dir = run_dir
        ckpt_resolved = ckpt_path

    parts = list(ckpt_resolved.parts)
    # look for .../results/runs/<run_name>/checkpoints/<file>
    if run_dir.name in parts:
        idx = parts.index(run_dir.name)
        if idx + 2 < len(parts) and parts[idx + 2] == "checkpoints":
            return parts[idx + 1]

    # fallback
    return ckpt_path.stem


def main() -> int:
    # 準備設定 路徑
    args = build_parser().parse_args()
    cfg = Config(
        device=args.device,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    run_dir = resolve_project_path(args.run_dir)

    # 解析 checkpoint + output folder name
    if args.name:
        run_name = args.name
        ckpt_path = _pick_ckpt_from_run(run_dir, run_name)
    else:
        ckpt_path = Path(args.ckpt)
        run_name = _infer_run_name_from_ckpt_path(ckpt_path, run_dir=run_dir)

    # 準備ckpt、modeldevice、loader、class_names、n_class
    device = resolve_device(cfg.device)
    val_loader = build_dataloader(cfg, "val")
    class_names = val_loader.dataset.classes
    n_class = len(class_names)
    model = build_model(args.model, n_class).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # 推論、計算ＣＭ、存檔
    out_root = ensure_dir(resolve_project_path(args.eval_result) / run_name)

    cm, metrics, wrong_items, val_scores, missed_scores = evaluate_with_cm(
        model,
        val_loader,
        n_class=n_class,
        device=device,
        path=out_root,
        class_names=class_names,
    )

    plot_confusion_matrix(
        cm,
        out_root,
        class_names=class_names,
        normalize=False,
        title="Val Confusion Matrix",
    )

    plot_confusion_matrix(
        cm,
        out_root,
        class_names=class_names,
        normalize=True,
        title="Val Confusion Matrix (Normalized)",
    )

    metrics_index = {
        "accuracy": float(metrics["accuracy"]),
        "precision": dict(zip(class_names, metrics["precision"].tolist())),
        "recall": dict(zip(class_names, metrics["recall"].tolist())),
        "f1": dict(zip(class_names, metrics["f1"].tolist())),
        "support": dict(zip(class_names, map(int, metrics["support"].tolist()))),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
    }
    save_json(out_root / "metrics_index.json", metrics_index)

    if args.copy_misclassified:
        _save_misclassified(
            out_root, wrong_items, class_names, max_items=args.max_misclassified
        )

    # plot_anomaly_his(out_root, val_scores, missed_scores)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(
        f"Macro P/R/F1: {metrics['macro_precision']:.4f} / {metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}"
    )
    print(f"Saved outputs to: {out_root}")
    print(f"Misclassified samples collected: {len(wrong_items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
