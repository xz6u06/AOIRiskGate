from __future__ import annotations

import json
from pathlib import Path

from webui.ui_paths import list_images, safe_file


def build_report_args(
    source_mode: str,
    run_name: str,
    scores_path: str,
    eval_result_dir: str,
    gate_result_dir: str,
    mode: str,
    threshold: float,
    target_review: float,
    top_k: int,
    copy_images: bool,
) -> list[str]:
    args = [
        "--eval-result",
        eval_result_dir,
        "--gate-result",
        gate_result_dir,
        "--top-k",
        str(top_k),
    ]

    if source_mode == "Run Name":
        args.extend(["--run-name", run_name])
    else:
        args.extend(["--scores", scores_path])

    if mode == "Target Review":
        args.extend(["--target-review", str(target_review)])
    else:
        args.extend(["--threshold", str(threshold)])

    if copy_images:
        args.append("--copy-images")

    return args


def load_report_outputs(gate_result_dir: str, run_name: str) -> tuple[dict, str | None, list[str], list[str]]:
    root = Path(gate_result_dir) / run_name
    return load_report_outputs_from_dir(root)


def load_report_outputs_from_dir(root_dir: str | Path) -> tuple[dict, str | None, list[str], list[str]]:
    root = Path(root_dir)
    report_path = root / "report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            report = {"error": f"Failed to parse report.json: {e}"}
    else:
        report = {"hint": "No report.json yet."}

    his_pred = safe_file(root / "per_class_zscore_his_pred.png")
    review_imgs = list_images(root / "review_imgs")
    escape_imgs = list_images(root / "escape_imgs")
    return report, his_pred, review_imgs, escape_imgs
