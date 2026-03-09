# engine.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

try:
    from .checkpoints import save_checkpoint
    from .config import Config, ensure_dir, resolve_device
    from .utils import AverageMeter, accuracy, now_run_id, save_json, set_seed
    from .viz import save_train_curve, plot_PCA_analysis
    from .features import attach_activation_hook, activation_to_embedding
except ImportError:
    from checkpoints import save_checkpoint
    from config import Config, ensure_dir, resolve_device
    from utils import AverageMeter, accuracy, now_run_id, save_json, set_seed
    from viz import save_train_curve, plot_PCA_analysis
    from features import attach_activation_hook, activation_to_embedding


def train_one_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for img, targets in loader:
        img = img.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        output = model(img)
        loss = loss_fn(output, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(output.detach(), targets), batch_size)

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
):
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch in loader:
        # val loader may return paths
        if len(batch) == 3:
            img, targets, _paths = batch
        else:
            img, targets = batch

        img = img.to(device)
        targets = targets.to(device)

        output = model(img)
        loss = loss_fn(output, targets)

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(output, targets), batch_size)

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def run_train(
    cfg: Config,
    *,
    model: nn.Module,
    train_loader,
    val_loader,
    freeze_backbone: bool = False,
) -> Path:

    set_seed(cfg.seed)
    run_name = cfg.run_name or now_run_id()
    run_path = ensure_dir(Path(cfg.run_dir) / run_name)
    ckpt_dir = ensure_dir(run_path / "checkpoints")

    device = resolve_device(cfg.device)
    model = model.to(device)

    if freeze_backbone:
        # 先全部凍結
        for p in model.parameters():
            p.requires_grad = False
        # 只打開分類頭（ResNet18 的最後一層）
        if hasattr(model, "fc"):
            for p in model.fc.parameters():
                p.requires_grad = True
        else:
            raise ValueError(
                "--freeze-backbone is only supported for models with .fc (e.g. resnet18)"
            )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )  # 印出可調參數
    optimizer = torch.optim.AdamW(
        trainable_params,  # 可調參數
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    history: List[Dict[str, Any]] = []

    # persist config once
    save_json(run_path / "config.json", cfg.to_dict())

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history.append(row)

        is_best = val_metrics["acc"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["acc"]

        save_json(
            run_path / "metrics.json",
            {"history": history, "best_val_acc": best_val_acc},
        )

        save_checkpoint(
            ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_val_acc,
            config=cfg.to_dict(),
        )

        if is_best:
            save_checkpoint(
                ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val_acc,
                config=cfg.to_dict(),
            )

        if (
            epoch == 1
            or cfg.epochs < 10
            or (cfg.epochs < 200 and epoch % 10 == 0)
            or (cfg.epochs >= 200 and epoch % 50 == 0)
        ):
            print(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}"
            )

    save_train_curve(run_path / "loss.png", history)
    return run_path


@torch.no_grad()
def evaluate_with_cm(
    model: nn.Module,
    loader,
    *,
    n_class: int,
    device: torch.device,
    path,
    class_names=None,
):
    model.eval()

    # 設置 hook（不拆模型，直接抓分類頭前的 pooling 特徵）
    if hasattr(model, "avgpool"):
        feat_module = model.avgpool  # resnet18: [B, 512, 1, 1]
    elif hasattr(model, "adaptive_pool"):
        feat_module = model.adaptive_pool  # simplecnn: [B, 32, 7, 7]
    else:
        raise ValueError("No supported pool layer to hook")

    activation, h = attach_activation_hook(model, feat_module, name="avgpool")

    # 準備容器
    all_pred = []
    all_true = []
    all_paths = []
    wrong_items: List[Tuple[str, int, int]] = []

    # 推論過程
    for batch in loader:
        # val loader may return paths
        if len(batch) == 3:
            imgs, labels, paths = batch
        else:
            imgs, labels = batch
            paths = ["<no_path>"] * len(labels)

        # 搬到device
        imgs = imgs.to(device)
        labels = labels.to(device)

        # 推論（forward 會觸發 hook，把 pooling out 存到 activation["avgpool"]）
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        # 存結果
        all_pred.append(preds)
        all_true.append(labels)
        all_paths.extend(list(paths))

        # 存錯誤項
        wrong_mask = preds.ne(labels)
        if wrong_mask.any():
            wrong_idx = wrong_mask.nonzero(as_tuple=False).squeeze(1).cpu().tolist()
            for i in wrong_idx:
                wrong_items.append(
                    (str(paths[i]), int(labels[i].item()), int(preds[i].item()))
                )

    h.remove()  # 拆除hook

    y_pred = torch.cat(all_pred).cpu()
    y_true = torch.cat(all_true).cpu()
    correct_mask = y_pred == y_true  # True = correct match（"val/good"）
    miss_mask = ~correct_mask  # True = missed match

    # [batch activations] -> [N, C, H, W]（shape 依模型而定）
    avgpool_acts = torch.cat(activation["avgpool"], dim=0)
    features = activation_to_embedding(avgpool_acts)  # [N, D]

    # 用「分類正確」樣本的特徵平均當中心；若沒有 correct match，中心無法定義
    if not bool(correct_mask.any().item()):
        raise ValueError("No correct predictions in val; cannot compute val_center")

    val_center = features[correct_mask].mean(dim=0)
    scores = torch.norm(features - val_center, dim=1)  # [N] 每個樣本與中心的距離
    val_scores = scores[correct_mask]
    missed_scores = scores[miss_mask]

    # 存下hook出來的特徵結果, 與進行PCA分析
    emb = {
        "features": features,  # [N, D]
        "correct_mask": correct_mask,  # [N] bool
        "miss_mask": miss_mask,  # [N] bool
        "paths": all_paths,  # len N
        "y_true": y_true,  # [N]
        "y_pred": y_pred,  # [N]
        "class_names": class_names,  # list[str], index -> class name
    }
    # plot_PCA_analysis(emb, path)
    torch.save(emb, path / "val_embeddings.pt")

    # 算ＣＭ
    cm = torch.zeros((n_class, n_class), dtype=torch.long)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    eps = 1e-12
    precision = tp.float() / (tp.float() + fp.float() + eps)
    recall = tp.float() / (tp.float() + fn.float() + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = tp.sum().float() / cm.sum().float().clamp_min(1)

    metrics = {
        "accuracy": acc.item(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": cm.sum(dim=1),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
    }

    return cm, metrics, wrong_items, val_scores, missed_scores


"""
後註:
這邊features[correct_mask].mean(dim=0)
• 只用「所有預測正確的樣本」算一個單一中心 val_center
• 每張圖的分數 = 到這個全域中心的距離
• 然後拿來畫 his_anomaly_score.png：比較 correct vs missed 的分數分布
特性

• 不分 predicted class / true class
• 等於在問：「這張圖的 embedding 離『整體正確樣本的平均長相』有多遠？」
這更像是：用來做快速探索/視覺化（看錯誤樣本是不是比較偏離整體中心），不是一個嚴格的產線 gate 方案。"""
