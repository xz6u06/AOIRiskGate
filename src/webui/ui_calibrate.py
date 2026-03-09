from __future__ import annotations

import json
from pathlib import Path

from webui.ui_paths import safe_file


def build_calibrate_args(
    val_dir: str,
    device: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    model: str,
    method: str,
    min_correct_per_class: int,
    pca_components: int,
    source_mode: str,
    run_name: str,
    ckpt_path: str,
    run_dir: str,
    eval_result_dir: str,
    gate_result_dir: str,
    target_review: float | None,
    quantile_step: float,
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
        "--method",
        method,
        "--min-correct-per-class",
        str(min_correct_per_class),
        "--pca-components",
        str(pca_components),
        "--run-dir",
        run_dir,
        "--eval-result",
        eval_result_dir,
        "--gate-result",
        gate_result_dir,
        "--quantile-step",
        str(quantile_step),
    ]

    if source_mode == "Run Name":
        args.extend(["--run-name", run_name])
    else:
        args.extend(["--ckpt", ckpt_path])

    if target_review is not None:
        args.extend(["--target-review", str(target_review)])

    return args


def load_calibrate_outputs(gate_result_dir: str, run_name: str) -> tuple[str | None, str | None, dict, dict]:
    root = Path(gate_result_dir) / run_name
    tradeoff = safe_file(root / "gate_tradeoff_curve.png")
    his_true = safe_file(root / "per_class_zscore_his_true.png")

    rec = root / "recommended.json"
    if rec.exists():
        try:
            recommended = json.loads(rec.read_text(encoding="utf-8"))
        except Exception as e:
            recommended = {"error": f"Failed to parse recommended.json: {e}"}
    else:
        recommended = {"hint": "recommended.json not found (target_review may be empty)."}

    stats = root / "gate_stats.json"
    if stats.exists():
        try:
            gate_stats = json.loads(stats.read_text(encoding="utf-8"))
        except Exception as e:
            gate_stats = {"error": f"Failed to parse gate_stats.json: {e}"}
    else:
        gate_stats = {"hint": "No gate_stats.json yet."}

    return tradeoff, his_true, recommended, gate_stats
