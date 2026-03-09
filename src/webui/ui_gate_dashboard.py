from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from core.gate import (
    metrics_at_threshold,
    pick_threshold_for_target_review,
    threshold_to_review_mask,
)


def load_scores(scores_csv: str) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with Path(scores_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    # Backward compatible: old scores.csv may contain signed z-score.
    # Dashboard semantics use deviation magnitude, so use abs() consistently.
    risk = np.asarray([abs(float(r["risk_score"])) for r in rows], dtype=np.float64)
    correct = np.asarray([int(r["correct"]) == 1 for r in rows], dtype=bool)
    paths = [r.get("path", "") for r in rows]

    return {
        "rows": rows,
        "risk": risk,
        "correct": correct,
        "paths": paths,
    }


def load_tradeoff_points(gate_dir: str, run_name: str) -> list[dict[str, Any]]:
    run_name = (run_name or "").strip()
    if not run_name:
        return []
    path = Path(gate_dir) / run_name / "tradeoff.json"
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        points = obj.get("points", []) if isinstance(obj, dict) else []
        return points if isinstance(points, list) else []
    except Exception:
        return []


def load_tradeoff_points_from_scores(scores_csv: str | Path) -> list[dict[str, Any]]:
    path = Path(scores_csv).parent / "tradeoff.json"
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        points = obj.get("points", []) if isinstance(obj, dict) else []
        return points if isinstance(points, list) else []
    except Exception:
        return []


def topk_indices(values: np.ndarray, k: int) -> list[int]:
    if values.size == 0:
        return []
    k = int(k)
    if k == 0:
        return np.argsort(-values).tolist()
    if k < 0:
        return []
    k = min(k, int(values.size))
    idx = np.argpartition(-values, k - 1)[:k]
    idx = idx[np.argsort(-values[idx])]
    return idx.tolist()


def remap_path(path: str, from_prefix: str, to_prefix: str) -> str:
    if not path:
        return path
    if from_prefix and to_prefix and path.startswith(from_prefix):
        return to_prefix + path[len(from_prefix) :]
    return path


def build_gallery_paths(
    data: dict[str, Any],
    review_mask: np.ndarray,
    *,
    top_k: int,
    from_prefix: str,
    to_prefix: str,
) -> tuple[list[str], list[str]]:
    risk = data["risk"]
    correct = data["correct"]
    paths = data["paths"]

    reviewed_scores = np.where(review_mask, risk, -np.inf)
    top_review = topk_indices(reviewed_scores, top_k)
    top_review = [i for i in top_review if np.isfinite(reviewed_scores[i])]

    escape_mask = (~correct) & (~review_mask)
    escape_scores = np.where(escape_mask, risk, -np.inf)
    top_escape = topk_indices(escape_scores, top_k)
    top_escape = [i for i in top_escape if np.isfinite(escape_scores[i])]

    review_imgs: list[str] = []
    escape_imgs: list[str] = []

    for i in top_review:
        p = remap_path(paths[i], from_prefix, to_prefix)
        if Path(p).exists():
            review_imgs.append(p)

    for i in top_escape:
        p = remap_path(paths[i], from_prefix, to_prefix)
        if Path(p).exists():
            escape_imgs.append(p)

    return review_imgs, escape_imgs


def metrics_from_review_rate(data: dict[str, Any], review_rate: float) -> tuple[float, dict[str, Any]]:
    risk = data["risk"]
    correct = data["correct"]

    thr = pick_threshold_for_target_review(risk, target_review_rate=float(review_rate))
    review_mask = threshold_to_review_mask(risk, thr)
    m = metrics_at_threshold(correct=correct, review_mask=review_mask)
    return float(thr), asdict(m)


def metrics_from_threshold(data: dict[str, Any], threshold: float) -> tuple[float, dict[str, Any]]:
    risk = data["risk"]
    correct = data["correct"]

    review_mask = threshold_to_review_mask(risk, float(threshold))
    m = metrics_at_threshold(correct=correct, review_mask=review_mask)
    return float(m.review_rate), asdict(m)
