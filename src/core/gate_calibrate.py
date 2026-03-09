# gate_calibrate.py
# 風險閘門校正（Calibration）CLI：
# - 載入 checkpoint + validation set
# - 抽取 embeddings（透過 engine.evaluate_with_cm 產出 val_embeddings.pt）
# - 計算 risk_score（中心距離 / PCA 重建誤差）
# - [新增] 計算並套用 Z-Score Normalization
# - 產出 trade-off 表；若提供 target-review 則產推薦 threshold

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np

try:
    from .config import Config, ensure_dir, resolve_device, resolve_project_path
    from .data import build_dataloader
    from .engine import evaluate_with_cm
    from .gate import (
        build_tradeoff_points,
        compute_risk_scores,
        metrics_at_threshold,
        pick_threshold_for_target_review,
        threshold_to_review_mask,
    )
    from .models import build_model
    from .viz import plot_tradeoff_curve, plot_per_class_zscore_his
except ImportError:
    from config import Config, ensure_dir, resolve_device, resolve_project_path
    from data import build_dataloader
    from engine import evaluate_with_cm
    from gate import (
        build_tradeoff_points,
        compute_risk_scores,
        metrics_at_threshold,
        pick_threshold_for_target_review,
        threshold_to_review_mask,
    )
    from models import build_model
    from viz import plot_tradeoff_curve, plot_per_class_zscore_his


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibrate risk gate (MVP)")

    p.add_argument("--val-dir", default=Config.val_dir)
    p.add_argument("--device", default=Config.device)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--num-workers", type=int, default=Config.num_workers)
    p.add_argument("--img-size", type=int, default=Config.img_size)

    p.add_argument("--model", default=Config.model_name, help="simple|resnet18")
    p.add_argument(
        "--method",
        default="pca_recon_pred_class",
        help="centroid_pred_class | pca_recon_pred_class",
    )
    p.add_argument("--min-correct-per-class", type=int, default=5)
    p.add_argument(
        "--pca-components",
        type=int,
        default=32,
        help="PCA n_components for pca_recon_pred_class",
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-name", help="Run name/id, e.g. 20260202_152211")
    g.add_argument("--ckpt", help="Checkpoint path, e.g. /some/path/best.pt")

    p.add_argument(
        "--run-dir",
        default=Config.run_dir,
        help="Base runs dir (default: ./results/runs)",
    )
    p.add_argument(
        "--eval-result",
        default=Config.eval_result_dir,
        help="Base output dir (default: ./results/evals)",
    )
    p.add_argument(
        "--gate-result",
        default=Config.gate_result_dir,
        help="Base gate output dir (default: ./results/gate)",
    )

    p.add_argument(
        "--target-review",
        type=float,
        default=None,
        help="(optional) Pick recommended threshold by target review rate",
    )
    p.add_argument("--quantile-step", type=float, default=0.005)

    return p


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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
    try:
        run_dir = run_dir.resolve()  # 轉換為其 絕對路徑
        ckpt_resolved = ckpt_path.resolve()
    except Exception:
        ckpt_resolved = ckpt_path

    parts = list(ckpt_resolved.parts)
    if run_dir.name in parts:
        idx = parts.index(run_dir.name)
        if idx + 2 < len(parts) and parts[idx + 2] == "checkpoints":
            return parts[idx + 1]
    return ckpt_path.stem


def _write_scores_csv(
    path: Path,
    *,
    emb: Dict[str, Any],
    risk_score: np.ndarray,
) -> List[str]:
    paths = emb.get("paths", [])
    y_true = emb["y_true"].detach().cpu().tolist()
    y_pred = emb["y_pred"].detach().cpu().tolist()
    correct = emb["correct_mask"].detach().cpu().tolist()

    class_names = emb.get("class_names")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "idx",
                "path",
                "y_true",
                "y_pred",
                "correct",
                "risk_score",
                "true_class",
                "pred_class",
            ]
        )

        true_class = []
        for i in range(len(y_true)):
            t = int(y_true[i])
            p = int(y_pred[i])
            t_name = class_names[t] if class_names and 0 <= t < len(class_names) else ""
            p_name = class_names[p] if class_names and 0 <= p < len(class_names) else ""
            w.writerow(
                [
                    i,
                    paths[i] if i < len(paths) else "",
                    t,
                    p,
                    1 if bool(correct[i]) else 0,
                    float(risk_score[i]),
                    t_name,
                    p_name,
                ]
            )
            true_class.append(t_name)

    return true_class


def main() -> int:
    args = build_parser().parse_args()

    cfg = Config(
        device=args.device,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    run_dir = resolve_project_path(args.run_dir)

    if args.run_name:
        run_name = args.run_name
        ckpt_path = _pick_ckpt_from_run(run_dir, run_name)
    else:
        ckpt_path = Path(args.ckpt)
        run_name = _infer_run_name_from_ckpt_path(ckpt_path, run_dir=run_dir)

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

    # 1. 重用既有 eval 流程：抽取 embeddings 並輸出 val_embeddings.pt
    out_root = ensure_dir(resolve_project_path(args.eval_result) / run_name)
    _cm, _metrics, _wrong_items, _val_scores, _missed_scores = evaluate_with_cm(
        model,
        val_loader,
        n_class=n_class,
        device=device,
        path=out_root,
        class_names=class_names,
    )

    emb_path = out_root / "val_embeddings.pt"
    emb = torch.load(emb_path, map_location="cpu")

    # 2. 計算原始 Risk Score (Raw Score)
    raw_risk_score = compute_risk_scores(
        emb,
        method=args.method,
        min_correct_per_class=args.min_correct_per_class,
        pca_components=args.pca_components,
    )

    # 3. [關鍵修正] 計算 Z-Score 統計量 (基於正確預測的樣本)
    y_pred = _to_numpy(emb["y_pred"]).astype(int)
    correct_mask = _to_numpy(emb["correct_mask"]).astype(bool)  # 只用正確樣本建立分佈

    stats_dict = {}

    # 建立一個新的陣列存 Normalized Score，初始值複製 raw scores
    norm_risk_score = np.array(raw_risk_score, dtype=np.float64, copy=True)

    for c, cls_name in enumerate(class_names):
        # 選出該類別中「預測正確」的樣本來計算 Mean/Std
        sel_correct = np.where((y_pred == c) & correct_mask)[0]

        # 選出該類別中「所有」樣本來進行轉換 (Normalization)
        sel_all_pred = np.where(y_pred == c)[0]

        if sel_correct.size < 2:
            # 樣本太少無法計算 std，就用該類別所有樣本的 stats (fallback)
            # 或者給一個預設值，避免除以零
            if sel_all_pred.size > 0:
                mu = float(np.mean(raw_risk_score[sel_all_pred]))
                std = float(np.std(raw_risk_score[sel_all_pred])) + 1e-9
            else:
                mu = 0.0
                std = 1.0
        else:
            mu = float(np.mean(raw_risk_score[sel_correct]))
            std = float(np.std(raw_risk_score[sel_correct])) + 1e-9

        stats_dict[cls_name] = {"mean": mu, "std": std}

        # 對所有預測為 c 的樣本進行標準化 (包含錯誤的樣本)
        if sel_all_pred.size > 0:
            norm_risk_score[sel_all_pred] = (raw_risk_score[sel_all_pred] - mu) / std

    gate_dir = ensure_dir(resolve_project_path(args.gate_result) / run_name)

    # 4. [新增] 儲存統計量 gate_stats.json (供推論與檢視使用)
    (gate_dir / "gate_stats.json").write_text(
        json.dumps(stats_dict, indent=2), encoding="utf-8"
    )

    # 5. [修正] 使用 |Z-Score| 進行後續所有操作
    # 需求：Gate 應抓「偏離正常分佈的程度」，與方向無關。
    abs_risk_score = np.abs(norm_risk_score)

    # 注意：現在傳入的是 abs_risk_score
    scores_csv = gate_dir / "scores.csv"
    true_class = _write_scores_csv(scores_csv, emb=emb, risk_score=abs_risk_score)

    points = build_tradeoff_points(
        risk_score=abs_risk_score,  # Use absolute normalized score
        correct=emb["correct_mask"],
        quantile_step=args.quantile_step,
    )

    thr = None
    m = None
    if args.target_review is not None:
        thr = pick_threshold_for_target_review(
            abs_risk_score, target_review_rate=args.target_review  # Use absolute normalized score
        )
        review_mask = threshold_to_review_mask(abs_risk_score, thr)  # Use absolute normalized score
        m = metrics_at_threshold(correct=emb["correct_mask"], review_mask=review_mask)

    meta = {
        "run_name": run_name,
        "ckpt_path": str(ckpt_path),
        "model": args.model,
        "method": args.method,
        "min_correct_per_class": args.min_correct_per_class,
        "target_review": (
            float(args.target_review) if args.target_review is not None else None
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "val_dir": str(cfg.val_dir),
        "img_size": int(args.img_size),
        "score_type": "abs(z-score_normalized)",  # 標記這是絕對值標準化分數
    }

    tradeoff = {"meta": meta, "points": points}
    (gate_dir / "tradeoff.json").write_text(
        json.dumps(tradeoff, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.target_review is not None:
        recommended = {
            "meta": meta,
            "threshold": float(thr),
            "metrics": asdict(m),
        }
        (gate_dir / "recommended.json").write_text(
            json.dumps(recommended, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    his_out_path = gate_dir / "per_class_zscore_his_true.png"
    thr_for_plot = float(thr) if thr is not None else None

    plot_tradeoff_curve(gate_dir, points=points, target_review_rate=args.target_review)
    plot_per_class_zscore_his(
        abs_risk_score,
        true_class,
        correct_mask,
        his_out_path,
        threshold=thr_for_plot,
    )
    print(f"[gate_calibrate] Saved plot -> {his_out_path}")

    print(f"[gate_calibrate] Saved stats: {gate_dir / 'gate_stats.json'}")
    print(f"[gate_calibrate] Saved: {scores_csv}")
    print(f"[gate_calibrate] Saved: {gate_dir / 'tradeoff.json'}")
    if args.target_review is not None:
        print(f"[gate_calibrate] Saved: {gate_dir / 'recommended.json'}")
        print(
            f"[gate_calibrate] Threshold@target_review={args.target_review:.3f}: {thr:.6f} (|Sigma|)"
        )
        print(
            f"[gate_calibrate] review_rate={m.review_rate:.4f} escape_rate={m.escape_rate:.4f} "
            f"capture_rate={m.error_capture_rate:.4f} auto_rate={m.automation_rate:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
