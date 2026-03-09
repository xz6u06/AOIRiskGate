from __future__ import annotations

"""NEU classifier 的 Risk Gate（風險閘門）工具。

本模組保持輕量（numpy + torch），方便被 CLI 腳本與未來 UI 重用。

核心概念（MVP）：
- 使用 backbone 的 embedding（已在 `engine.evaluate_with_cm` 內抽取並輸出）
- 為每筆樣本計算一個 `risk_score`（越大代表越可能判錯/越不典型）
- 套用 threshold 形成決策：自動放行（auto-accept）或送人工覆核（send-to-human）

MVP 預設方法：
- `centroid_pred_class`：以 predicted class 為條件，對「預測正確」的樣本建中心，
  分數為樣本到該類中心的距離。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class GateMetrics:
    review_rate: float
    automation_rate: float
    escape_rate: float
    error_capture_rate: float

    n_total: int
    n_review: int
    n_auto: int

    n_wrong_total: int
    n_wrong_review: int
    n_wrong_escape: int


def _to_numpy(x) -> np.ndarray:

    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_risk_scores(
    emb: Dict[str, Any],
    *,  # 強制後續參數只能用名稱指定
    method: str = "centroid_pred_class",
    min_correct_per_class: int = 5,
    pca_components: int = 32,
) -> np.ndarray:
    """計算每筆樣本的風險分數（risk_score）。

    參數：
        emb: `engine.evaluate_with_cm()` 輸出的 embedding dict（通常從 `val_embeddings.pt` 載入）
        method: 分數方法（例如 centroid / pca recon）
        min_correct_per_class: 若某 predicted class 的「預測正確樣本」太少，
            則退回使用 global 的中心 / PCA（避免 per-class 模型不穩）
        pca_components: PCA 重建誤差方法的 n_components

    回傳：
        risk_score: shape = [N]，float64
    """
    # 正則化、分評分方法
    method = method.lower().strip()
    if method in {"centroid", "centroid_pred_class", "center_pred_class"}:
        return risk_centroid_pred_class(
            emb,
            min_correct_per_class=min_correct_per_class,
        )
    if method in {"pca", "pca_recon_pred_class", "pca_reconstruction_pred_class"}:
        return risk_pca_recon_pred_class(
            emb,
            min_correct_per_class=min_correct_per_class,
            n_components=int(pca_components),
        )
    raise ValueError(f"Unknown gate method: {method}")


# 類別中心法(以每個判對樣本的中心)
def risk_centroid_pred_class(
    emb: Dict[str, Any],
    *,
    min_correct_per_class: int = 5,
) -> np.ndarray:
    """類別中心距離（以 predicted class 為條件）。

    對每個 predicted class = c，用下列樣本建立中心（centroid）：
      - `correct_mask == True` 且 `y_pred == c`

    對任一樣本 x 的分數：
      - score = || x - μ_c ||_2

    退回策略（fallback）：
      - 若某類別的 correct 樣本數不足，則使用「全部 correct 樣本」的 global 中心。
    """

    # 讀出evaluate_with_cm產出的emb
    X = _to_numpy(emb["features"])  # [N, D]
    y_pred = _to_numpy(emb["y_pred"]).astype(int)
    y_true = _to_numpy(emb["y_true"]).astype(int)
    correct = _to_numpy(emb["correct_mask"]).astype(bool)  # (y_pred == y_true)
    # 驗證數量
    if X.ndim != 2:
        raise ValueError(f"features must be [N,D], got {X.shape}")

    N, D = X.shape
    if len(y_pred) != N or len(correct) != N:
        raise ValueError("Mismatched lengths between features/y_pred/correct_mask")

    if not correct.any():
        raise ValueError("No correct samples; cannot build centroids")

    # 全局中心 & 各類中心
    global_center = X[correct].mean(axis=0)  # = evaluate_with_cm 中的val_center

    n_class = int(y_pred.max()) + 1 if N else 0
    centers = np.zeros((n_class, D), dtype=np.float64)
    center_ok = np.zeros((n_class,), dtype=bool)

    for c in range(n_class):
        idx = np.where(correct & (y_pred == c))[0]  # 取得預測正確且類別為 c 的樣本索引
        if idx.size >= min_correct_per_class:
            centers[c] = X[idx].mean(axis=0)
            center_ok[c] = True
        else:
            centers[c] = global_center
            center_ok[c] = False

    # 每筆樣本：計算其到 predicted class 中心的距離
    chosen = centers[np.clip(y_pred, 0, n_class - 1)]  # np.clip(a, a_min, a_max)
    diff = X - chosen
    scores = np.linalg.norm(diff, axis=1)
    return scores.astype(np.float64)


def risk_pca_recon_pred_class(
    emb: Dict[str, Any],
    *,
    min_correct_per_class: int = 5,
    n_components: int = 32,
) -> np.ndarray:
    """PCA 重建誤差（以 predicted class 為條件）。

    對每個 predicted class = c，用下列樣本的 embeddings 去 fit PCA：
      - `correct_mask == True` 且 `y_pred == c`

    對任一樣本 x 的分數：
      - 先投影到 PCA 子空間再重建得到 x_recon
      - score = || x - x_recon ||_2

    退回策略（fallback）：
      - 若某類別 correct 樣本數不足，則改用「全部 correct 樣本」fit 出來的 global PCA。

    備註：
      - 若環境有 `sklearn` 則優先使用 sklearn 的 PCA；否則使用 numpy SVD 的簡化替代。
      - `n_components` 會依據每個類別可用樣本數/維度自動 clamp 到合理範圍。
    """

    X = _to_numpy(emb["features"]).astype(np.float64)  # [N, D]
    y_pred = _to_numpy(emb["y_pred"]).astype(int)
    correct = _to_numpy(emb["correct_mask"]).astype(bool)

    if X.ndim != 2:
        raise ValueError(f"features must be [N,D], got {X.shape}")
    if not correct.any():
        raise ValueError("No correct samples; cannot fit PCA")

    N, D = X.shape
    n_class = int(y_pred.max()) + 1 if N else 0

    # 優先嘗試使用 sklearn PCA（若環境有安裝）
    try:
        from sklearn.decomposition import PCA  # type: ignore

        def fit_pca(x: np.ndarray, k: int):
            k = int(
                max(1, min(k, x.shape[1], x.shape[0] - 1))
            )  # （1 ≤ k ≤ min(D, n-1)）
            p = PCA(n_components=k)
            p.fit(x)
            return p

        def recon_err(pca, x: np.ndarray) -> np.ndarray:
            z = pca.transform(x)
            xr = pca.inverse_transform(z)
            return np.linalg.norm(x - xr, axis=1)

        use_sklearn = True
    except Exception:

        def fit_pca(x: np.ndarray, k: int):
            # 對中心化後的資料做 SVD（作為 PCA 的簡化替代）
            mu = x.mean(axis=0)
            xc = x - mu
            # full_matrices=False => Vt shape = [min(n,d), d]
            _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
            k = int(max(1, min(k, vt.shape[0])))
            W = vt[:k].T  # [D, k]
            return mu, W

        def recon_err(pca_obj, x: np.ndarray) -> np.ndarray:
            mu, W = pca_obj
            xc = x - mu
            z = xc @ W
            xr = (z @ W.T) + mu
            return np.linalg.norm(x - xr, axis=1)

        use_sklearn = False

    # global PCA（用所有 correct 樣本 fit），當作 per-class 不足時的 fallback
    global_pca = fit_pca(X[correct], int(n_components))

    # per-class PCA 模型
    pcas: list[Any] = [None] * max(n_class, 1)  # 建立一個 [None, None, ...] 的列表
    pca_ok = np.zeros((max(n_class, 1),), dtype=bool)

    # 建立每個分類的PCA模型
    for c in range(n_class):
        idx = np.where(correct & (y_pred == c))[0]
        if idx.size >= max(min_correct_per_class, 2):
            # PCA 至少需要 2 筆樣本（sklearn/SVD 才有意義）
            pcas[c] = fit_pca(X[idx], int(n_components))
            pca_ok[c] = True
        else:
            pcas[c] = global_pca
            pca_ok[c] = False

    # 計算每筆樣本的分數
    scores = np.zeros((N,), dtype=np.float64)
    for c in range(n_class):
        sel = np.where(y_pred == c)[0]
        if sel.size == 0:
            continue
        scores[sel] = recon_err(pcas[c], X[sel]).astype(np.float64)

    return scores


def threshold_to_review_mask(risk_score: np.ndarray, threshold: float) -> np.ndarray:
    """由 threshold 產生是否送人工覆核的 mask。

    回傳：
        True  => 送人工（review）
        False => 自動放行（auto-accept）
    """

    risk_score = _to_numpy(risk_score).astype(np.float64)
    return risk_score > float(threshold)


def metrics_at_threshold(
    *,
    correct: np.ndarray,
    review_mask: np.ndarray,
) -> GateMetrics:

    correct = _to_numpy(correct).astype(bool)
    review_mask = _to_numpy(review_mask).astype(bool)

    if correct.shape != review_mask.shape:
        raise ValueError("correct and review_mask must have same shape")

    N = int(correct.size)
    wrong = ~correct

    n_review = int(review_mask.sum())
    n_auto = int(N - n_review)

    n_wrong_total = int(wrong.sum())
    n_wrong_review = int((wrong & review_mask).sum())
    n_wrong_escape = int((wrong & ~review_mask).sum())

    review_rate = n_review / max(N, 1)
    automation_rate = 1.0 - review_rate

    # escape_rate：在自動放行的樣本中，有多少比例其實是錯的（放行風險）
    escape_rate = n_wrong_escape / max(n_auto, 1)

    # error_capture_rate：所有錯誤中，有多少比例被 gate 攔下送人工
    error_capture_rate = n_wrong_review / max(n_wrong_total, 1)

    return GateMetrics(
        review_rate=review_rate,
        automation_rate=automation_rate,
        escape_rate=escape_rate,
        error_capture_rate=error_capture_rate,
        n_total=N,
        n_review=n_review,
        n_auto=n_auto,
        n_wrong_total=n_wrong_total,
        n_wrong_review=n_wrong_review,
        n_wrong_escape=n_wrong_escape,
    )


def pick_threshold_for_target_review(
    risk_score: np.ndarray,
    *,
    target_review_rate: float,
) -> float:
    """依據目標人工比例（target_review_rate）選擇 threshold。

    規則：若 `score > t` 就送人工，則
      review_rate ≈ fraction(score > t)

    因此選 t 為 `(1 - target_review_rate)` 的分位數（quantile）。
    """

    risk_score = _to_numpy(risk_score).astype(np.float64)
    r = float(target_review_rate)
    if not (0.0 <= r <= 1.0):
        raise ValueError("target_review_rate must be in [0,1]")
    if risk_score.size == 0:
        return 0.0

    q = 1.0 - r
    return float(np.quantile(risk_score, q))  # 分位數


def build_tradeoff_points(
    *,
    risk_score: np.ndarray,
    correct: np.ndarray,
    quantile_step: float = 0.005,
) -> List[Dict[str, Any]]:
    """掃描不同 threshold，建立 trade-off 表（用分位數切點）。"""

    risk_score = _to_numpy(risk_score).astype(np.float64)
    correct = _to_numpy(correct).astype(bool)

    if risk_score.size == 0:
        return []

    step = float(quantile_step)
    if step <= 0 or step > 1:
        raise ValueError("quantile_step must be in (0,1]")

    qs = np.arange(0.0, 1.0 + 1e-12, step)
    thresholds = np.quantile(
        risk_score, qs
    )  # np.quantile 支援 q 是 array，會一次回傳整個 thresholds 陣列

    # 去重，避免相同分位數得到相同 threshold
    thresholds = np.unique(thresholds)

    points = []
    for t in thresholds.tolist():
        review_mask = threshold_to_review_mask(risk_score, t)
        m = metrics_at_threshold(correct=correct, review_mask=review_mask)
        points.append(
            {
                "threshold": float(t),
                "review_rate": m.review_rate,
                "automation_rate": m.automation_rate,
                "escape_rate": m.escape_rate,
                "error_capture_rate": m.error_capture_rate,
                "n_total": m.n_total,
                "n_review": m.n_review,
                "n_auto": m.n_auto,
                "n_wrong_total": m.n_wrong_total,
                "n_wrong_review": m.n_wrong_review,
                "n_wrong_escape": m.n_wrong_escape,
            }
        )

    # 依 review_rate 由小到大排序（threshold 越小通常 review 會越多）
    points.sort(key=lambda p: (p["review_rate"], p["threshold"]))
    return points
