from __future__ import annotations

from pathlib import Path

from webui.ui_paths import list_images, safe_file


def build_eval_args(
    val_dir: str,
    device: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    model: str,
    run_dir: str,
    eval_result_dir: str,
    source_mode: str,
    run_name: str,
    ckpt_path: str,
    copy_misclassified: bool,
    max_misclassified: int,
) -> list[str]:
    args = [
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
        "--run-dir",
        run_dir,
        "--eval-result",
        eval_result_dir,
    ]

    if source_mode == "Run Name":
        args.extend(["--name", run_name])
    else:
        args.extend(["--ckpt", ckpt_path])

    if copy_misclassified:
        args.append("--copy-misclassified")
    args.extend(["--max-misclassified", str(max_misclassified)])
    return args


def load_eval_outputs(eval_result_dir: str, run_name: str) -> tuple[str | None, str | None, list[str], str]:
    root = Path(eval_result_dir) / run_name
    cm = safe_file(root / "cm.png")
    cm_norm = safe_file(root / "cm_norm.png")
    mis_imgs = list_images(root / "imgs")
    emb_hint = "exists" if (root / "val_embeddings.pt").exists() else "missing"
    return cm, cm_norm, mis_imgs, emb_hint
