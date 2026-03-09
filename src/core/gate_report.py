# gate_report.py
# 風險閘門報表（Reporting）CLI：
# - 讀取 gate_calibrate.py 產生的 scores.csv
# - 依 threshold 或 target-review（固定人工比例）計算指標
# - 輸出 Top-K：被攔下樣本 / 漏網之魚
# - 可選擇複製圖片，形成展示用「案例牆」

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from .config import Config, resolve_project_path
    from .gate import (
        compute_risk_scores,
        metrics_at_threshold,
        pick_threshold_for_target_review,
        threshold_to_review_mask,
    )
    from .viz import plot_per_class_zscore_his
except ImportError:
    from config import Config, resolve_project_path
    from gate import (
        compute_risk_scores,
        metrics_at_threshold,
        pick_threshold_for_target_review,
        threshold_to_review_mask,
    )
    from viz import plot_per_class_zscore_his


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Report risk gate metrics (MVP)")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scores", help="Path to scores.csv")
    src.add_argument(
        "--run-name",
        help="Run name; will load results/gate/<run>/scores.csv",
    )

    p.add_argument(
        "--eval-result",
        default=Config.eval_result_dir,
        help="Base vals dir (default: ./results/evals)",
    )
    p.add_argument(
        "--gate-result",
        default=Config.gate_result_dir,
        help="Base gate dir (default: ./results/gate)",
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--threshold", type=float, help="Manual threshold (sigma)")
    mode.add_argument(
        "--target-review", type=float, help="Pick threshold to hit target review rate"
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-K 案例數（0 = 不限制，全部列出）",
    )
    p.add_argument("--copy-images", action="store_true")

    return p


def _load_scores_csv(path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # 必要欄位
    # Backward compatible: old scores.csv may contain signed z-score.
    # Report semantics use deviation magnitude, so convert to absolute value.
    risk = np.asarray([abs(float(row["risk_score"])) for row in rows], dtype=np.float64)
    correct = np.asarray([int(row["correct"]) == 1 for row in rows], dtype=bool)

    paths = [row.get("path", "") for row in rows]
    y_true = np.asarray([int(row["y_true"]) for row in rows], dtype=int)
    y_pred = np.asarray([int(row["y_pred"]) for row in rows], dtype=int)
    true_class = np.asarray([str(row["true_class"]) for row in rows], dtype=str)
    pred_class = np.asarray([str(row.get("pred_class", "")) for row in rows], dtype=str)

    # （可選）若 scores.csv 已包含 risk_centroid，則直接讀取
    risk_centroid = None
    if rows and "risk_centroid" in rows[0]:
        try:
            risk_centroid = np.asarray(
                [float(row["risk_centroid"]) for row in rows], dtype=np.float64
            )
        except Exception:
            risk_centroid = None

    return {
        "rows": rows,
        "risk_score": risk,
        "risk_centroid": risk_centroid,
        "correct": correct,
        "paths": paths,
        "y_true": y_true,
        "y_pred": y_pred,
        "true_class": true_class,
        "pred_class": pred_class,
    }


def _topk_indices(values: np.ndarray, k: int) -> List[int]:
    """回傳 values 最大的前 K 個 index。

    - k > 0：取 Top-K
    - k == 0：不限制（全部列出，依值由大到小排序）
    """

    if values.size == 0:
        return []

    k = int(k)
    if k == 0:
        # 全部排序（可能比較慢，但在本專案 N 通常不大）
        idx = np.argsort(-values)
        return idx.tolist()

    if k < 0:
        return []

    k = min(k, int(values.size))
    # 用 argpartition 取 Top-K，避免完整排序（較省時間）
    idx = np.argpartition(-values, k - 1)[:k]
    idx = idx[np.argsort(-values[idx])]
    return idx.tolist()


def _copy_image(src: str, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except FileNotFoundError:
        return False


def main() -> int:
    args = build_parser().parse_args()

    if args.scores:
        scores_path = resolve_project_path(args.scores)
        run_name = (
            scores_path.parent.name
            if scores_path.name == "scores.csv"
            else "<unknown>"
        )
        out_dir = scores_path.parent
    else:
        run_name = args.run_name
        out_dir = resolve_project_path(args.gate_result) / run_name
        scores_path = out_dir / "scores.csv"

    data = _load_scores_csv(scores_path)
    risk = data["risk_score"]
    correct = data["correct"]
    true_class = data["true_class"]
    pred_class = data["pred_class"]

    # gate z-score 的分佈是依 y_pred（predicted class）做 normalization，
    # 所以這張 per-class 分佈圖用 y_pred 分組最直覺。
    class_for_plot = (
        pred_class
        if (pred_class is not None and np.any(pred_class != ""))
        else data["y_pred"].astype(str)
    )

    # 附加 centroid 分數（用於對照/可解釋性）
    # - 若 scores.csv 已有 risk_centroid 則沿用
    # - 否則嘗試從 results/evals/<run>/val_embeddings.pt 重算（因此需先跑過 gate_calibrate）
    risk_centroid = data.get("risk_centroid")
    if risk_centroid is None:
        emb_path = resolve_project_path(args.eval_result) / run_name / "val_embeddings.pt"
        if run_name != "<unknown>" and emb_path.exists():
            import torch

            emb = torch.load(emb_path, map_location="cpu")
            risk_centroid = compute_risk_scores(emb, method="centroid_pred_class")
        else:
            risk_centroid = None

    if args.target_review is not None:
        thr = pick_threshold_for_target_review(
            risk, target_review_rate=float(args.target_review)
        )
    else:
        thr = float(args.threshold)

    review_mask = threshold_to_review_mask(risk, thr)
    m = metrics_at_threshold(correct=correct, review_mask=review_mask)

    # Top-K（被攔下送人工）：在 review 樣本中挑 risk_score 最大的前 K 筆
    reviewed_scores = np.where(review_mask, risk, -np.inf)
    top_review = _topk_indices(reviewed_scores, args.top_k)
    top_review = [i for i in top_review if np.isfinite(reviewed_scores[i])]

    # Top-K（漏網之魚）：放行但其實判錯的樣本中，挑 risk_score 最大的前 K 筆
    escape_mask = (~correct) & (~review_mask)
    escape_scores = np.where(escape_mask, risk, -np.inf)
    top_escape = _topk_indices(escape_scores, args.top_k)
    top_escape = [i for i in top_escape if np.isfinite(escape_scores[i])]

    # （可選）複製圖片到輸出資料夾，方便展示
    copied_review = 0
    copied_escape = 0
    if args.copy_images:
        for i in top_review:
            src = data["paths"][i]
            if not src:
                continue
            name = Path(src).name
            dst = out_dir / "review_imgs" / f"score_{risk[i]:.4f}__{name}"
            if _copy_image(src, dst):
                copied_review += 1
        for i in top_escape:
            src = data["paths"][i]
            if not src:
                continue
            name = Path(src).name
            dst = out_dir / "escape_imgs" / f"score_{risk[i]:.4f}__{name}"
            if _copy_image(src, dst):
                copied_escape += 1

    report = {
        "meta": {
            "run_name": run_name,
            "scores_path": str(scores_path),
            "threshold": float(thr),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "target_review" if args.target_review is not None else "threshold",
            "target_review": (
                float(args.target_review) if args.target_review is not None else None
            ),
        },
        "metrics": asdict(m),
        "top_review": [
            {
                "idx": int(i),
                "path": data["paths"][i],
                "risk_score": float(risk[i]),
                "risk_centroid": (
                    float(risk_centroid[i])
                    if (risk_centroid is not None and i < len(risk_centroid))
                    else None
                ),
                "y_true": int(data["y_true"][i]),
                "y_pred": int(data["y_pred"][i]),
                "correct": bool(correct[i]),
            }
            for i in top_review
            if np.isfinite(reviewed_scores[i])
        ],
        "top_escape": [
            {
                "idx": int(i),
                "path": data["paths"][i],
                "risk_score": float(risk[i]),
                "risk_centroid": (
                    float(risk_centroid[i])
                    if (risk_centroid is not None and i < len(risk_centroid))
                    else None
                ),
                "y_true": int(data["y_true"][i]),
                "y_pred": int(data["y_pred"][i]),
                "correct": bool(correct[i]),
            }
            for i in top_escape
            if np.isfinite(escape_scores[i])
        ],
        "copied": (
            {"review": copied_review, "escape": copied_escape}
            if args.copy_images
            else None
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[gate_report] run={run_name} threshold={thr:.6f}")
    print(
        f"[gate_report] review_rate={m.review_rate:.4f} escape_rate={m.escape_rate:.4f} "
        f"capture_rate={m.error_capture_rate:.4f} auto_rate={m.automation_rate:.4f}"
    )
    if args.copy_images:
        print(
            f"[gate_report] copied review_imgs={copied_review}, escape_imgs={copied_escape} -> {out_dir}"
        )
    print(f"[gate_report] Saved report -> {out_path}")
    his_out_path = out_dir / "per_class_zscore_his_pred.png"
    # 使用實際使用的 threshold（不論是手動 threshold 或 target-review 推算）來畫圖
    plot_per_class_zscore_his(
        risk,
        class_for_plot,
        correct,
        his_out_path,
        threshold=float(thr),
    )
    print(f"[gate_report] Saved report -> {his_out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
