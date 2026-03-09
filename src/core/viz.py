from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def save_train_curve(path: str | Path, history: List[dict]) -> None:
    path = Path(path)
    if path.suffix != ".png":
        path = path.with_suffix(".png")

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(epochs, train_loss, label="Train Loss", color="tab:blue")
    ax1.plot(epochs, val_loss, label="Val Loss", color="tab:orange", linestyle="--")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, label="Train Acc", color="tab:green")
    ax2.plot(epochs, val_acc, label="Val Acc", color="tab:red", linestyle="--")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(
    cm,
    out_dir: str | Path,
    *,
    class_names: Optional[list[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_np = cm.numpy().astype(float)
    if normalize:
        cm_np = cm_np / np.clip(cm_np.sum(axis=1, keepdims=True), 1e-12, None)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_np, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    n = cm_np.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    if class_names is None:
        class_names = [str(i) for i in range(n)]

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(n):
        for j in range(n):
            txt = f"{cm_np[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if cm_np[i, j] > (cm_np.max() / 2) else "black",
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    file_name = "cm_norm.png" if normalize else "cm.png"
    out_path = out_dir / file_name
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_anomaly_his(path, val_scores: List, missed_scores: List):

    file_name = "his_anomaly_score.png"
    out_path = Path(path) / file_name

    # accept either torch.Tensor or list/np
    if hasattr(val_scores, "detach"):
        val_scores = val_scores.detach().cpu().numpy()
    if hasattr(missed_scores, "detach"):
        missed_scores = missed_scores.detach().cpu().numpy()

    plt.hist(val_scores, bins=50, alpha=0.7, label="match")
    plt.hist(missed_scores, bins=50, alpha=0.7, label="missed")
    plt.legend()
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_tradeoff_curve(
    out_dir: str | Path,
    *,
    points: list[dict],
    target_review_rate: float | None = None,
    title: str = "Gate trade-off: review rate vs escape rate",
) -> Path:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_name = "gate_tradeoff_curve.png"
    out_path = out_dir / file_name

    if not points:
        # create an empty placeholder plot
        plt.figure(figsize=(6, 4))
        plt.title(title)
        plt.xlabel("review_rate")
        plt.ylabel("escape_rate")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_path)
        plt.close()
        return out_path

    xs = [p["review_rate"] for p in points]
    ys = [p["escape_rate"] for p in points]

    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, marker=".", linewidth=1)
    plt.title(title)
    plt.xlabel("review_rate (fraction sent to human)")
    plt.ylabel("escape_rate (wrong among auto-accepted)")
    plt.grid(True, alpha=0.3)

    if target_review_rate is not None:
        tr = float(target_review_rate)
        # find nearest x
        idx = int(np.argmin(np.abs(np.asarray(xs) - tr)))
        plt.scatter(
            [xs[idx]], [ys[idx]], s=80, color="tab:red", label=f"target {tr:.3f}"
        )
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_PCA_analysis(d, path):

    X = d["features"].cpu().numpy()  # [N, D]
    correct = d["correct_mask"].cpu().numpy().astype(bool)
    miss = d["miss_mask"].cpu().numpy().astype(bool)

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    file_name = "PCA_analysis.png"
    out_path = Path(path) / file_name

    plt.figure(figsize=(6, 6))
    plt.scatter(X2[correct, 0], X2[correct, 1], s=10, alpha=0.4, label="correct(match)")
    plt.scatter(X2[miss, 0], X2[miss, 1], s=10, alpha=0.7, label="missed(match)")
    plt.legend()
    plt.title("Val embeddings PCA: correct vs missed")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(out_path)
    plt.close()


def plot_per_class_zscore_his(
    risk_score, class_for_plot, correct, path, threshold=None
):

    risk_score = np.asarray(risk_score)
    class_for_plot = np.asarray(class_for_plot)
    correct = np.asarray(correct)

    if not (risk_score.shape[0] == class_for_plot.shape[0] == correct.shape[0]):
        raise ValueError(
            "plot_per_class_zscore_his: length mismatch: "
            f"len(risk_score)={risk_score.shape[0]}, "
            f"len(class_for_plot)={class_for_plot.shape[0]}, "
            f"len(correct)={correct.shape[0]}"
        )

    data = pd.DataFrame(
        {"risk_score": risk_score, "class_for_plot": class_for_plot, "correct": correct}
    )

    data["correct"] = data["correct"].astype(int).astype(bool)

    classes = sorted(data["class_for_plot"].unique().tolist())
    n_cls = len(classes)
    ncols = 3
    nrows = math.ceil(n_cls / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 5.2, nrows * 4.0),
        # sharex=True,
        # sharey=True,
    )  # 座標軸共用尺度
    axes = np.array(axes).reshape(-1)

    overall_min = data["risk_score"].min()
    overall_max = data["risk_score"].max()
    bins = np.linspace(overall_min, overall_max, 31)

    for ax, cls in zip(axes, classes):
        sub = data[data["class_for_plot"] == cls]
        z_all = sub["risk_score"].dropna().to_numpy()
        z_ok = sub.loc[sub["correct"], "risk_score"].dropna().to_numpy()
        z_bad = sub.loc[~sub["correct"], "risk_score"].dropna().to_numpy()

        ax.hist(
            z_ok,
            bins=bins,
            alpha=0.65,
            color="#2ca02c",
            label=f"correct (n={z_ok.size})",
            # density=True,
        )
        ax.hist(
            z_bad,
            bins=bins,
            alpha=0.65,
            color="#d62728",
            label=f"wrong (n={z_bad.size})",
            # density=True,
        )
        if threshold is not None:
            ax.axvline(threshold, color="#333333", linewidth=1.0, linestyle="--")

        ax.set_title(f"{cls}")
        ax.set_xlabel("|z-score|")
        ax.set_ylabel("counts")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        ax.tick_params(labelbottom=True)
        ax.tick_params(labelleft=True)

    # 關閉空白子圖
    for ax in axes[len(classes) :]:
        ax.axis("off")

    fig.suptitle(
        "Per-class |z-score| distribution of risk_score (colored by correct vs wrong)",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(path)
