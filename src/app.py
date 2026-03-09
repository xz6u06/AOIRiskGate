from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from webui.ui_calibrate import build_calibrate_args, load_calibrate_outputs
from webui.ui_eval import build_eval_args, load_eval_outputs
from webui.ui_gate_dashboard import (
    build_gallery_paths,
    load_scores,
    load_tradeoff_points,
    load_tradeoff_points_from_scores,
    metrics_from_review_rate,
    metrics_from_threshold,
)
from webui.ui_paths import (
    PROJECT_ROOT,
    list_images,
    list_run_names,
    pick_default_run_name,
    resolve_ui_path,
    safe_file,
)
from webui.ui_report import (
    build_report_args,
    load_report_outputs,
    load_report_outputs_from_dir,
)
from webui.ui_runner import run_script
from webui.ui_schema import (
    CAL_DEFAULTS,
    CAL_METHOD_CHOICES,
    DEVICE_CHOICES,
    EVAL_DEFAULTS,
    GLOBAL_DEFAULTS,
    MODEL_CHOICES,
    REPORT_DEFAULTS,
    TRAIN_DEFAULTS,
)
from webui.ui_train import build_train_args, load_train_outputs


def _refresh_run_choices(
    run_dir: str, eval_dir: str, gate_dir: str
) -> tuple[dict, dict, dict, dict, dict, dict]:
    run_dir = str(resolve_ui_path(run_dir))
    eval_dir = str(resolve_ui_path(eval_dir))
    gate_dir = str(resolve_ui_path(gate_dir))
    train_runs = list_run_names(run_dir)
    eval_runs = list_run_names(eval_dir)
    gate_runs = list_run_names(gate_dir)

    train_default = pick_default_run_name(run_dir, required_files=["loss.png"])
    eval_default = pick_default_run_name(eval_dir)
    gate_default = pick_default_run_name(gate_dir)

    return (
        gr.update(choices=train_runs, value=train_default),
        gr.update(choices=train_runs, value=train_default),
        gr.update(choices=train_runs, value=train_default),
        gr.update(choices=eval_runs, value=eval_default),
        gr.update(choices=gate_runs, value=gate_default),
        gr.update(choices=gate_runs, value=gate_default),
    )


def _load_train_view(run_dir: str, run_name: str):
    if not run_name:
        return None, {"hint": "No run selected."}
    return load_train_outputs(str(resolve_ui_path(run_dir)), run_name)


def _load_eval_view(eval_dir: str, run_name: str):
    if not run_name:
        return None, None, [], "missing"
    return load_eval_outputs(str(resolve_ui_path(eval_dir)), run_name)


def _load_cal_view(gate_dir: str, run_name: str):
    if not run_name:
        return None, None, {"hint": "No run selected."}, {"hint": "No run selected."}
    return load_calibrate_outputs(str(resolve_ui_path(gate_dir)), run_name)


def _load_report_view(gate_dir: str, run_name: str):
    if not run_name:
        return {"hint": "No run selected."}, None, [], []
    return load_report_outputs(str(resolve_ui_path(gate_dir)), run_name)


def _load_report_view_from_source(
    source_mode: str, gate_dir: str, run_name: str, scores_path: str
):
    if source_mode == "Run Name":
        return _load_report_view(gate_dir, run_name)

    if not scores_path or not scores_path.strip():
        return {"hint": "No scores.csv path provided."}, None, [], []

    scores = Path(str(resolve_ui_path(scores_path.strip())))
    root = scores.parent
    # In scores mode, artifacts are stored beside the selected scores.csv.
    report, his_pred, _, _ = load_report_outputs_from_dir(root)
    return (
        report,
        his_pred,
        list_images(root / "review_imgs"),
        list_images(root / "escape_imgs"),
    )


def _load_report_meta_from_source(
    source_mode: str, gate_dir: str, run_name: str, scores_path: str
):
    report, his_pred, _review_imgs, _escape_imgs = _load_report_view_from_source(
        source_mode, gate_dir, run_name, scores_path
    )
    return report, his_pred


def _run_train(
    train_dir: str,
    val_dir: str,
    device: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    model: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    momentum: float,
    run_dir: str,
    run_name: str | None,
    pretrained: bool,
    freeze_backbone: bool,
):
    train_dir = str(resolve_ui_path(train_dir))
    val_dir = str(resolve_ui_path(val_dir))
    run_dir = str(resolve_ui_path(run_dir))
    run_name_clean = (run_name or "").strip()
    args = build_train_args(
        train_dir,
        val_dir,
        device,
        img_size,
        batch_size,
        num_workers,
        model,
        epochs,
        lr,
        weight_decay,
        momentum,
        run_dir,
        run_name_clean,
        pretrained,
        freeze_backbone,
    )
    ok, log = run_script("train.py", args)

    train_runs = list_run_names(run_dir)
    selected = (
        run_name_clean
        if run_name_clean and (run_name_clean in train_runs)
        else pick_default_run_name(run_dir, required_files=["loss.png"])
    )
    train_update = gr.update(choices=train_runs, value=selected)
    source_update = gr.update(choices=train_runs, value=selected)
    return log, train_update, source_update, source_update


def _run_eval(
    val_dir: str,
    device: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    model: str,
    source_mode: str,
    run_name: str | None,
    ckpt_path: str | None,
    run_dir: str,
    eval_result_dir: str,
    copy_mis: bool,
    max_mis: int,
):
    val_dir = str(resolve_ui_path(val_dir))
    run_dir = str(resolve_ui_path(run_dir))
    eval_result_dir = str(resolve_ui_path(eval_result_dir))
    run_name = run_name or ""
    ckpt_path = (ckpt_path or "").strip()
    if ckpt_path:
        ckpt_path = str(resolve_ui_path(ckpt_path))
    args = build_eval_args(
        val_dir,
        device,
        img_size,
        batch_size,
        num_workers,
        model,
        run_dir,
        eval_result_dir,
        source_mode,
        run_name,
        ckpt_path,
        copy_mis,
        max_mis,
    )
    ok, log = run_script("eval.py", args)

    eval_runs = list_run_names(eval_result_dir)
    selected = (
        run_name
        if source_mode == "Run Name" and run_name in eval_runs
        else pick_default_run_name(eval_result_dir)
    )
    return log, gr.update(choices=eval_runs, value=selected)


def _run_calibrate(
    val_dir: str,
    device: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    model: str,
    method: str,
    min_correct: int,
    pca_components: int,
    source_mode: str,
    run_name: str | None,
    ckpt_path: str | None,
    run_dir: str,
    eval_result_dir: str,
    gate_result_dir: str,
    target_review: float | None,
    quantile_step: float,
):
    val_dir = str(resolve_ui_path(val_dir))
    run_dir = str(resolve_ui_path(run_dir))
    eval_result_dir = str(resolve_ui_path(eval_result_dir))
    gate_result_dir = str(resolve_ui_path(gate_result_dir))
    run_name = run_name or ""
    ckpt_path = (ckpt_path or "").strip()
    if ckpt_path:
        ckpt_path = str(resolve_ui_path(ckpt_path))
    tr = None
    if target_review is not None:
        try:
            tr_value = float(target_review)
        except (TypeError, ValueError):
            tr_value = -1.0
        if tr_value >= 0:
            tr = tr_value
    args = build_calibrate_args(
        val_dir,
        device,
        img_size,
        batch_size,
        num_workers,
        model,
        method,
        min_correct,
        pca_components,
        source_mode,
        run_name,
        ckpt_path,
        run_dir,
        eval_result_dir,
        gate_result_dir,
        tr,
        quantile_step,
    )
    ok, log = run_script("gate_calibrate.py", args)

    gate_runs = list_run_names(gate_result_dir)
    selected = (
        run_name
        if source_mode == "Run Name" and run_name in gate_runs
        else pick_default_run_name(gate_result_dir)
    )
    update = gr.update(choices=gate_runs, value=selected)
    return log, update, update


def _run_report(
    source_mode: str,
    run_name: str | None,
    scores_path: str | None,
    eval_dir: str,
    gate_dir: str,
    mode: str,
    threshold: float,
    target_review: float,
    top_k: int,
    copy_images: bool,
):
    eval_dir = str(resolve_ui_path(eval_dir))
    gate_dir = str(resolve_ui_path(gate_dir))
    run_name = run_name or ""
    scores_path = (scores_path or "").strip()
    if scores_path:
        scores_path = str(resolve_ui_path(scores_path))
    args = build_report_args(
        source_mode,
        run_name,
        scores_path,
        eval_dir,
        gate_dir,
        mode,
        threshold,
        target_review,
        top_k,
        copy_images,
    )
    ok, log = run_script("gate_report.py", args)
    return log


def _toggle_ckpt_mode(mode: str):
    return gr.update(visible=mode == "Run Name"), gr.update(
        visible=mode == "Checkpoint"
    )


def _toggle_report_source_mode(mode: str):
    return gr.update(visible=mode == "Run Name"), gr.update(
        visible=mode == "scores.csv"
    )


def _toggle_report_mode(mode: str):
    return gr.update(visible=mode == "Target Review"), gr.update(
        visible=mode == "Threshold"
    )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _finalize_fig(fig):
    try:
        fig.canvas.draw()
    except Exception:
        pass
    plt.close(fig)
    return fig


def _plot_tradeoff(
    points: list[dict[str, Any]], review_rate: float, escape_rate: float
):
    fig, ax = plt.subplots(figsize=(6, 4))
    if points:
        xs = [p.get("review_rate", 0.0) for p in points]
        ys = [p.get("escape_rate", 0.0) for p in points]
        ax.plot(xs, ys, marker=".", linewidth=1.0)
    ax.scatter([review_rate], [escape_rate], color="red", s=60)
    ax.set_title("Review Rate vs Escape Rate")
    ax.set_xlabel("review_rate")
    ax.set_ylabel("escape_rate")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _finalize_fig(fig)


def _plot_per_class(rows: list[dict[str, str]], threshold: float, class_mode: str):
    if not rows:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_title("No scores loaded")
        ax.axis("off")
        return _finalize_fig(fig)

    risk = np.asarray([abs(float(r["risk_score"])) for r in rows], dtype=np.float64)
    correct = np.asarray([int(r["correct"]) == 1 for r in rows], dtype=bool)
    key = "pred_class" if class_mode == "Predicted Class" else "true_class"
    cls = np.asarray([r.get(key, "") for r in rows], dtype=object)

    classes = sorted(list({str(x) for x in cls if str(x)}))
    if not classes:
        classes = sorted(list({str(r.get("y_pred", "")) for r in rows}))
        cls = np.asarray([str(r.get("y_pred", "")) for r in rows], dtype=object)

    n_cls = len(classes)
    ncols = 3
    nrows = int(math.ceil(n_cls / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.8, nrows * 3.6))
    axes = np.array(axes).reshape(-1)

    lo, hi = float(np.min(risk)), float(np.max(risk))
    if hi <= lo:
        hi = lo + 1e-6
    bins = np.linspace(lo, hi, 31)

    for ax, c in zip(axes, classes):
        sel = cls == c
        z_ok = risk[sel & correct]
        z_bad = risk[sel & ~correct]

        ax.hist(
            z_ok, bins=bins, alpha=0.65, color="#2ca02c", label=f"correct ({z_ok.size})"
        )
        ax.hist(
            z_bad, bins=bins, alpha=0.65, color="#d62728", label=f"wrong ({z_bad.size})"
        )
        ax.axvline(float(threshold), color="#222", linestyle="--", linewidth=1.0)
        ax.set_title(str(c))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    for ax in axes[len(classes) :]:
        ax.axis("off")

    fig.tight_layout()
    return _finalize_fig(fig)


def _load_dashboard(
    source_mode: str,
    gate_dir: str,
    run_name: str | None,
    scores_path: str | None,
    target_review: float,
    top_k: int,
    from_prefix: str,
    to_prefix: str,
    class_mode: str,
):
    gate_dir = str(resolve_ui_path(gate_dir))
    if source_mode == "Run Name":
        run_name = (run_name or "").strip()
        if not run_name:
            hint = {
                "hint": "No Gate Run selected. Choose a run or switch to scores.csv mode."
            }
            return (
                {"data": None, "points": []},
                gr.update(value=""),
                gr.update(minimum=0.0, maximum=1.0, value=0.0),
                hint,
                None,
                None,
                [],
                [],
            )
        selected_scores = str(Path(gate_dir) / run_name / "scores.csv")
    else:
        scores_path = (scores_path or "").strip()
        if not scores_path:
            hint = {"hint": "No scores.csv path provided."}
            return (
                {"data": None, "points": []},
                gr.update(value=""),
                gr.update(minimum=0.0, maximum=1.0, value=0.0),
                hint,
                None,
                None,
                [],
                [],
            )
        selected_scores = str(resolve_ui_path(scores_path))

    p = Path(selected_scores)
    if not p.exists():
        hint = {"hint": f"scores.csv not found: {selected_scores}"}
        return (
            {"data": None, "points": []},
            gr.update(value=selected_scores),
            gr.update(minimum=0.0, maximum=1.0, value=0.0),
            hint,
            None,
            None,
            [],
            [],
        )

    data = load_scores(selected_scores)
    target_review = _clamp(target_review, 0.0, 1.0)
    threshold, m = metrics_from_review_rate(data, target_review)

    points = (
        load_tradeoff_points(gate_dir, run_name)
        if source_mode == "Run Name"
        else load_tradeoff_points_from_scores(selected_scores)
    )
    review_mask = data["risk"] > threshold
    review_imgs, escape_imgs = build_gallery_paths(
        data,
        review_mask,
        top_k=top_k,
        from_prefix=from_prefix,
        to_prefix=to_prefix,
    )

    tmin = float(np.min(data["risk"]))
    tmax = float(np.max(data["risk"]))
    if tmax <= tmin:
        tmax = tmin + 1e-6
    margin = max((tmax - tmin) * 1e-9, 1e-9)
    tmin_adj = tmin - margin
    tmax_adj = tmax + margin
    threshold = _clamp(threshold, tmin_adj, tmax_adj)

    tradeoff_fig = _plot_tradeoff(points, m["review_rate"], m["escape_rate"])
    per_class_fig = _plot_per_class(data["rows"], threshold, class_mode)

    return (
        {"data": data, "points": points},
        gr.update(value=selected_scores),
        gr.update(minimum=tmin_adj, maximum=tmax_adj, value=threshold),
        m,
        tradeoff_fig,
        per_class_fig,
        review_imgs,
        escape_imgs,
    )


def _on_review_change(
    state: dict[str, Any] | None,
    review_rate: float,
    top_k: int,
    from_prefix: str,
    to_prefix: str,
    class_mode: str,
):
    data = (state or {}).get("data")
    points = (state or {}).get("points", [])
    if not data:
        return (
            gr.update(value=0.0),
            {"hint": "Please load scores first."},
            None,
            None,
            [],
            [],
        )

    if review_rate is None:
        review_rate = 0.0
    review_rate = _clamp(review_rate, 0.0, 1.0)
    threshold, m = metrics_from_review_rate(data, review_rate)
    review_mask = data["risk"] > threshold
    review_imgs, escape_imgs = build_gallery_paths(
        data,
        review_mask,
        top_k=top_k,
        from_prefix=from_prefix,
        to_prefix=to_prefix,
    )
    tradeoff_fig = _plot_tradeoff(points, m["review_rate"], m["escape_rate"])
    his_fig = _plot_per_class(data["rows"], threshold, class_mode)
    return (
        gr.update(value=threshold),
        gr.update(value=m),
        tradeoff_fig,
        his_fig,
        review_imgs,
        escape_imgs,
    )


def _on_threshold_change(
    state: dict[str, Any] | None,
    threshold: float,
    top_k: int,
    from_prefix: str,
    to_prefix: str,
    class_mode: str,
):
    data = (state or {}).get("data")
    points = (state or {}).get("points", [])
    if not data:
        return (
            gr.update(value=0.0),
            {"hint": "Please load scores first."},
            None,
            None,
            [],
            [],
        )

    threshold = float(threshold) if threshold is not None else 0.0
    review_rate, m = metrics_from_threshold(data, threshold)
    review_rate = _clamp(review_rate, 0.0, 1.0)
    review_mask = data["risk"] > threshold
    review_imgs, escape_imgs = build_gallery_paths(
        data,
        review_mask,
        top_k=top_k,
        from_prefix=from_prefix,
        to_prefix=to_prefix,
    )
    tradeoff_fig = _plot_tradeoff(points, m["review_rate"], m["escape_rate"])
    his_fig = _plot_per_class(data["rows"], threshold, class_mode)
    return (
        gr.update(value=review_rate),
        gr.update(value=m),
        tradeoff_fig,
        his_fig,
        review_imgs,
        escape_imgs,
    )


def _on_review_change_delayed(
    state: dict[str, Any] | None,
    review_rate: float,
    top_k: int,
    from_prefix: str,
    to_prefix: str,
    class_mode: str,
):
    time.sleep(1.0)
    return _on_review_change(
        state, review_rate, top_k, from_prefix, to_prefix, class_mode
    )


def _on_threshold_change_delayed(
    state: dict[str, Any] | None,
    threshold: float,
    top_k: int,
    from_prefix: str,
    to_prefix: str,
    class_mode: str,
):
    time.sleep(1.0)
    return _on_threshold_change(
        state, threshold, top_k, from_prefix, to_prefix, class_mode
    )


def build_app() -> gr.Blocks:
    run_dir_default = GLOBAL_DEFAULTS.run_dir
    eval_dir_default = GLOBAL_DEFAULTS.eval_dir
    gate_dir_default = GLOBAL_DEFAULTS.gate_dir
    run_dir_default_abs = str(resolve_ui_path(run_dir_default))
    eval_dir_default_abs = str(resolve_ui_path(eval_dir_default))
    gate_dir_default_abs = str(resolve_ui_path(gate_dir_default))

    train_default = pick_default_run_name(
        run_dir_default_abs, required_files=["loss.png"]
    )
    eval_default = pick_default_run_name(eval_dir_default_abs)
    gate_default = pick_default_run_name(gate_dir_default_abs)

    train_curve_init, train_metrics_init = (None, {"hint": "No run selected."})
    eval_cm_init, eval_cm_norm_init, eval_mis_init, eval_emb_init = (
        None,
        None,
        [],
        "missing",
    )
    cal_tradeoff_init, cal_his_init, cal_rec_init, cal_stats_init = (
        None,
        None,
        {"hint": "No run selected."},
        {"hint": "No run selected."},
    )
    report_json_init, report_his_pred_init, report_review_init, report_escape_init = (
        {"hint": "No run selected."},
        None,
        [],
        [],
    )

    if train_default:
        train_curve_init, train_metrics_init = _load_train_view(
            run_dir_default_abs, train_default
        )
    if eval_default:
        eval_cm_init, eval_cm_norm_init, eval_mis_init, eval_emb_init = _load_eval_view(
            eval_dir_default_abs, eval_default
        )
    if gate_default:
        cal_tradeoff_init, cal_his_init, cal_rec_init, cal_stats_init = _load_cal_view(
            gate_dir_default_abs, gate_default
        )
        (
            report_json_init,
            report_his_pred_init,
            report_review_init,
            report_escape_init,
        ) = _load_report_view(gate_dir_default_abs, gate_default)

    with gr.Blocks(title="AOIRiskGate Studio ") as demo:
        gr.Markdown("# AOIRiskGate Studio")
        gr.Markdown("全域執行設定（分頁可覆寫）")

        with gr.Row():
            with gr.Column(scale=3):
                global_img_size = gr.Number(
                    label="影像尺寸 (--img-size)",
                    value=GLOBAL_DEFAULTS.img_size,
                    precision=0,
                )
                global_batch_size = gr.Number(
                    label="批次大小 (--batch-size)",
                    value=GLOBAL_DEFAULTS.batch_size,
                    precision=0,
                )
                global_num_workers = gr.Number(
                    label="載入工人數 (--num-workers)",
                    value=GLOBAL_DEFAULTS.num_workers,
                    precision=0,
                )
            with gr.Column(scale=3):
                global_device = gr.Dropdown(
                    label="裝置 (--device)",
                    choices=DEVICE_CHOICES,
                    value=GLOBAL_DEFAULTS.device,
                    allow_custom_value=True,
                )
                global_model = gr.Dropdown(
                    label="模型 (--model)",
                    choices=MODEL_CHOICES,
                    value=GLOBAL_DEFAULTS.model,
                )
            with gr.Column(scale=4):
                global_run_dir = gr.Textbox(
                    label="Runs 路徑 (--run-dir)", value=run_dir_default_abs
                )
                global_eval_dir = gr.Textbox(
                    label="Evals 路徑 (--eval-result)", value=eval_dir_default_abs
                )
                global_gate_dir = gr.Textbox(
                    label="Gate 路徑 (--gate-result)", value=gate_dir_default_abs
                )
                refresh_runs_btn = gr.Button(
                    "Refresh (scan --run-dir/--eval-result/--gate-result)"
                )

        with gr.Tabs():
            with gr.Tab("訓練"):
                with gr.Row():
                    with gr.Column(scale=4):
                        train_train_dir = gr.Textbox(
                            label="訓練資料夾 (--train-dir)",
                            value=TRAIN_DEFAULTS.train_dir,
                        )
                        train_val_dir = gr.Textbox(
                            label="驗證資料夾 (--val-dir)", value=TRAIN_DEFAULTS.val_dir
                        )
                        train_epochs = gr.Number(
                            label="訓練回合 (--epochs)",
                            value=TRAIN_DEFAULTS.epochs,
                            precision=0,
                        )
                        train_lr = gr.Number(
                            label="學習率 (--lr)", value=TRAIN_DEFAULTS.lr
                        )
                        train_weight_decay = gr.Number(
                            label="權重衰減 (--weight-decay)",
                            value=TRAIN_DEFAULTS.weight_decay,
                        )
                        with gr.Accordion("Advanced", open=False):
                            train_momentum = gr.Number(
                                label="動量 (--momentum, AdamW下通常不生效)",
                                value=TRAIN_DEFAULTS.momentum,
                            )
                        train_pretrained = gr.Checkbox(
                            label="使用預訓練權重 (--pretrained/--no-pretrained)(simple model 勿勾選)",
                            value=TRAIN_DEFAULTS.pretrained,
                        )
                        train_freeze_backbone = gr.Checkbox(
                            label="凍結 backbone (--freeze-backbone/--no-freeze-backbone)(simple model 勿勾選)",
                            value=TRAIN_DEFAULTS.freeze_backbone,
                        )
                        train_run_name = gr.Textbox(
                            label="Run 名稱 (--run-name, 空白=timestamp)", value=""
                        )
                        train_run_btn = gr.Button("開始訓練 (train.py)")
                    with gr.Column(scale=6):
                        train_stdout = gr.Textbox(label="執行日誌", lines=14)
                        train_run_selector = gr.Dropdown(
                            label="選擇 Run (from --run-dir)",
                            choices=list_run_names(run_dir_default_abs),
                            value=train_default,
                        )
                        train_refresh_btn = gr.Button("刷新訓練 Runs")
                        train_curve = gr.Image(
                            label="Loss & Accuracy Curve (results/runs/<run>/loss.png)",
                            value=train_curve_init,
                        )
                        train_metrics = gr.JSON(
                            label="訓練摘要 (metrics.json)", value=train_metrics_init
                        )

            with gr.Tab("驗證"):
                with gr.Row():
                    with gr.Column(scale=4):
                        eval_val_dir = gr.Textbox(
                            label="驗證資料夾 (--val-dir)", value=EVAL_DEFAULTS.val_dir
                        )
                        eval_mode = gr.Radio(
                            label="模型來源 (--name | --ckpt)",
                            choices=["Run Name", "Checkpoint"],
                            value="Run Name",
                        )
                        with gr.Group(visible=True) as eval_group_run:
                            eval_run_name = gr.Dropdown(
                                label="Run Name (--name)",
                                choices=list_run_names(run_dir_default_abs),
                                value=train_default,
                            )
                            eval_source_refresh_btn = gr.Button("刷新可用訓練 Runs")
                        with gr.Group(visible=False) as eval_group_ckpt:
                            eval_ckpt_path = gr.Textbox(
                                label="Checkpoint 路徑 (--ckpt)", value=""
                            )
                        eval_copy_mis = gr.Checkbox(
                            label="匯出誤判影像 (--copy-misclassified)",
                            value=EVAL_DEFAULTS.copy_misclassified,
                        )
                        eval_max_mis = gr.Number(
                            label="誤判影像上限 (--max-misclassified, 0=不限)",
                            value=EVAL_DEFAULTS.max_misclassified,
                            precision=0,
                        )
                        eval_run_btn = gr.Button("開始驗證 (eval.py)")
                    with gr.Column(scale=6):
                        eval_stdout = gr.Textbox(label="執行日誌", lines=14)
                        eval_run_selector = gr.Dropdown(
                            label="結果 Run (from --eval-result)",
                            choices=list_run_names(eval_dir_default_abs),
                            value=eval_default,
                        )
                        eval_refresh_btn = gr.Button("刷新驗證 Runs")
                        eval_cm = gr.Image(
                            label="Confusion Matrix (cm.png)", value=eval_cm_init
                        )
                        eval_cm_norm = gr.Image(
                            label="Confusion Matrix Normalized (cm_norm.png)",
                            value=eval_cm_norm_init,
                        )
                        eval_emb_hint = gr.Textbox(
                            label="Embeddings 狀態", value=eval_emb_init
                        )
                        eval_mis_gallery = gr.Gallery(
                            label="Misclassified Gallery (results/evals/<run>/imgs/*)",
                            columns=6,
                            value=eval_mis_init,
                        )

            with gr.Tab("Gate 校正"):
                with gr.Row():
                    with gr.Column(scale=4):
                        cal_val_dir = gr.Textbox(
                            label="驗證資料夾 (--val-dir)", value=CAL_DEFAULTS.val_dir
                        )
                        cal_mode = gr.Radio(
                            label="模型來源 (--run-name | --ckpt)",
                            choices=["Run Name", "Checkpoint"],
                            value="Run Name",
                        )
                        with gr.Group(visible=True) as cal_group_run:
                            cal_run_name = gr.Dropdown(
                                label="Run Name (--run-name)",
                                choices=list_run_names(run_dir_default_abs),
                                value=train_default,
                            )
                            cal_source_refresh_btn = gr.Button("刷新可用訓練 Runs")
                        with gr.Group(visible=False) as cal_group_ckpt:
                            cal_ckpt_path = gr.Textbox(
                                label="Checkpoint 路徑 (--ckpt)", value=""
                            )
                        cal_method = gr.Dropdown(
                            label="風險評估方法 (--method)",
                            choices=CAL_METHOD_CHOICES,
                            value=CAL_DEFAULTS.method,
                        )
                        cal_min_correct = gr.Number(
                            label="每類最少正確樣本數 (--min-correct-per-class)",
                            value=CAL_DEFAULTS.min_correct_per_class,
                            precision=0,
                        )
                        cal_pca_components = gr.Number(
                            label="PCA 維度 (--pca-components)",
                            value=CAL_DEFAULTS.pca_components,
                            precision=0,
                        )
                        cal_target_review = gr.Number(
                            label="目標人工比例 (--target-review, 可空用 -1)", value=-1
                        )
                        cal_quantile_step = gr.Number(
                            label="分位掃描步長 (--quantile-step)",
                            value=CAL_DEFAULTS.quantile_step,
                        )
                        cal_run_btn = gr.Button("開始校正 (gate_calibrate.py)")
                    with gr.Column(scale=6):
                        cal_stdout = gr.Textbox(label="執行日誌", lines=14)
                        cal_run_selector = gr.Dropdown(
                            label="Gate Run (from --gate-result)",
                            choices=list_run_names(gate_dir_default_abs),
                            value=gate_default,
                        )
                        cal_refresh_btn = gr.Button("刷新 Gate Runs")
                        cal_tradeoff = gr.Image(
                            label="Review vs Escape 曲線 (gate_tradeoff_curve.png)",
                            value=cal_tradeoff_init,
                        )
                        cal_his_true = gr.Image(
                            label="Per-class |Z-score| (per_class_zscore_his_true.png)",
                            value=cal_his_init,
                        )
                        cal_recommended = gr.JSON(
                            label="推薦門檻 (recommended.json)", value=cal_rec_init
                        )
                        cal_stats = gr.JSON(
                            label="類別標準化統計 (gate_stats.json)",
                            value=cal_stats_init,
                        )

            with gr.Tab("Gate 儀表與報表"):
                dashboard_state = gr.State(value=None)
                with gr.Row():
                    with gr.Column(scale=4):
                        report_source_mode = gr.Radio(
                            label="分數來源 (--run-name | --scores)",
                            choices=["Run Name", "scores.csv"],
                            value="Run Name",
                        )
                        with gr.Group(visible=True) as report_group_run:
                            report_run_name = gr.Dropdown(
                                label="Run Name (--run-name)",
                                choices=list_run_names(gate_dir_default_abs),
                                value=gate_default,
                            )
                            report_refresh_btn = gr.Button("刷新 Gate Runs")
                        with gr.Group(visible=False) as report_group_scores:
                            report_scores_path = gr.Textbox(
                                label="scores.csv 路徑 (--scores)",
                                value=safe_file(
                                    Path(gate_dir_default_abs)
                                    / (gate_default or "")
                                    / "scores.csv"
                                )
                                or "",
                            )

                        report_mode = gr.Radio(
                            label="門檻模式 (--target-review | --threshold)",
                            choices=["Target Review", "Threshold"],
                            value="Target Review",
                        )
                        with gr.Group(visible=True) as report_group_target:
                            report_target = gr.Slider(
                                label="review_rate (--target-review)",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.001,
                                value=REPORT_DEFAULTS.target_review,
                            )
                        with gr.Group(visible=False) as report_group_threshold:
                            report_threshold = gr.Slider(
                                label="threshold (--threshold)",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.001,
                                value=REPORT_DEFAULTS.threshold,
                            )

                        report_top_k = gr.Slider(
                            label="Top-K (--top-k, 0=全部)",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=REPORT_DEFAULTS.top_k,
                        )
                        report_copy_images = gr.Checkbox(
                            label="複製案例圖片 (--copy-images)",
                            value=REPORT_DEFAULTS.copy_images,
                        )
                        path_from = gr.Textbox(label="Path remap from_prefix", value="")
                        path_to = gr.Textbox(label="Path remap to_prefix", value="")
                        class_mode = gr.Radio(
                            label="Histogram 分組",
                            choices=["Predicted Class", "True Class"],
                            value="Predicted Class",
                        )

                        load_scores_btn = gr.Button(
                            "載入 scores / 更新儀表 (dashboard)"
                        )
                        run_report_btn = gr.Button("產生報表 (gate_report.py)")

                    with gr.Column(scale=6):
                        report_stdout = gr.Textbox(label="執行日誌", lines=8)
                        report_metrics = gr.JSON(label="指標卡")
                        report_tradeoff_plot = gr.Plot(label="Trade-off + 當下點")
                        report_his_plot = gr.Plot(label="Per-class |Z-score| + 門檻線")
                        report_review_gallery = gr.Gallery(
                            label="Top reviewed（送人工）",
                            columns=6,
                            value=report_review_init,
                        )
                        report_escape_gallery = gr.Gallery(
                            label="Top escapes（錯誤但放行）",
                            columns=6,
                            value=report_escape_init,
                        )
                        report_json = gr.JSON(
                            label="report.json（按「產生報表」後更新）",
                            value=report_json_init,
                        )
                        report_his_pred_img = gr.Image(
                            label="per_class_zscore_his_pred.png（|Z-score|，按「產生報表」後更新）",
                            value=report_his_pred_init,
                        )

        refresh_outputs = [
            train_run_selector,
            eval_run_name,
            cal_run_name,
            eval_run_selector,
            cal_run_selector,
            report_run_name,
        ]

        refresh_runs_btn.click(
            fn=_refresh_run_choices,
            inputs=[global_run_dir, global_eval_dir, global_gate_dir],
            outputs=refresh_outputs,
        )
        global_run_dir.change(
            fn=_refresh_run_choices,
            inputs=[global_run_dir, global_eval_dir, global_gate_dir],
            outputs=refresh_outputs,
        )
        global_eval_dir.change(
            fn=_refresh_run_choices,
            inputs=[global_run_dir, global_eval_dir, global_gate_dir],
            outputs=refresh_outputs,
        )
        global_gate_dir.change(
            fn=_refresh_run_choices,
            inputs=[global_run_dir, global_eval_dir, global_gate_dir],
            outputs=refresh_outputs,
        )
        train_refresh_btn.click(
            fn=_refresh_run_choices,
            inputs=[global_run_dir, global_eval_dir, global_gate_dir],
            outputs=refresh_outputs,
        )
        train_refresh_btn.click(
            fn=_load_train_view,
            inputs=[global_run_dir, train_run_selector],
            outputs=[train_curve, train_metrics],
        )
        eval_source_refresh_btn.click(
            fn=_refresh_run_choices,
            inputs=[global_run_dir, global_eval_dir, global_gate_dir],
            outputs=refresh_outputs,
        )
        eval_refresh_btn.click(
            fn=_refresh_run_choices,
            inputs=[global_run_dir, global_eval_dir, global_gate_dir],
            outputs=refresh_outputs,
        )
        eval_refresh_btn.click(
            fn=_load_eval_view,
            inputs=[global_eval_dir, eval_run_selector],
            outputs=[eval_cm, eval_cm_norm, eval_mis_gallery, eval_emb_hint],
        )
        cal_source_refresh_btn.click(
            fn=_refresh_run_choices,
            inputs=[global_run_dir, global_eval_dir, global_gate_dir],
            outputs=refresh_outputs,
        )
        cal_refresh_btn.click(
            fn=_refresh_run_choices,
            inputs=[global_run_dir, global_eval_dir, global_gate_dir],
            outputs=refresh_outputs,
        )
        cal_refresh_btn.click(
            fn=_load_cal_view,
            inputs=[global_gate_dir, cal_run_selector],
            outputs=[cal_tradeoff, cal_his_true, cal_recommended, cal_stats],
        )
        report_refresh_btn.click(
            fn=_refresh_run_choices,
            inputs=[global_run_dir, global_eval_dir, global_gate_dir],
            outputs=refresh_outputs,
        )
        report_refresh_btn.click(
            fn=_load_report_view_from_source,
            inputs=[
                report_source_mode,
                global_gate_dir,
                report_run_name,
                report_scores_path,
            ],
            outputs=[
                report_json,
                report_his_pred_img,
                report_review_gallery,
                report_escape_gallery,
            ],
        )
        report_refresh_btn.click(
            fn=_load_dashboard,
            inputs=[
                report_source_mode,
                global_gate_dir,
                report_run_name,
                report_scores_path,
                report_target,
                report_top_k,
                path_from,
                path_to,
                class_mode,
            ],
            outputs=[
                dashboard_state,
                report_scores_path,
                report_threshold,
                report_metrics,
                report_tradeoff_plot,
                report_his_plot,
                report_review_gallery,
                report_escape_gallery,
            ],
        )

        train_run_selector.change(
            fn=_load_train_view,
            inputs=[global_run_dir, train_run_selector],
            outputs=[train_curve, train_metrics],
        )

        train_run_btn.click(
            fn=_run_train,
            inputs=[
                train_train_dir,
                train_val_dir,
                global_device,
                global_img_size,
                global_batch_size,
                global_num_workers,
                global_model,
                train_epochs,
                train_lr,
                train_weight_decay,
                train_momentum,
                global_run_dir,
                train_run_name,
                train_pretrained,
                train_freeze_backbone,
            ],
            outputs=[train_stdout, train_run_selector, eval_run_name, cal_run_name],
        ).then(
            fn=_load_train_view,
            inputs=[global_run_dir, train_run_selector],
            outputs=[train_curve, train_metrics],
        )

        eval_mode.change(
            fn=_toggle_ckpt_mode,
            inputs=[eval_mode],
            outputs=[eval_group_run, eval_group_ckpt],
        )
        cal_mode.change(
            fn=_toggle_ckpt_mode,
            inputs=[cal_mode],
            outputs=[cal_group_run, cal_group_ckpt],
        )

        eval_run_selector.change(
            fn=_load_eval_view,
            inputs=[global_eval_dir, eval_run_selector],
            outputs=[eval_cm, eval_cm_norm, eval_mis_gallery, eval_emb_hint],
        )

        eval_run_btn.click(
            fn=_run_eval,
            inputs=[
                eval_val_dir,
                global_device,
                global_img_size,
                global_batch_size,
                global_num_workers,
                global_model,
                eval_mode,
                eval_run_name,
                eval_ckpt_path,
                global_run_dir,
                global_eval_dir,
                eval_copy_mis,
                eval_max_mis,
            ],
            outputs=[eval_stdout, eval_run_selector],
        ).then(
            fn=_load_eval_view,
            inputs=[global_eval_dir, eval_run_selector],
            outputs=[eval_cm, eval_cm_norm, eval_mis_gallery, eval_emb_hint],
        )

        cal_run_selector.change(
            fn=_load_cal_view,
            inputs=[global_gate_dir, cal_run_selector],
            outputs=[cal_tradeoff, cal_his_true, cal_recommended, cal_stats],
        )

        cal_run_btn.click(
            fn=_run_calibrate,
            inputs=[
                cal_val_dir,
                global_device,
                global_img_size,
                global_batch_size,
                global_num_workers,
                global_model,
                cal_method,
                cal_min_correct,
                cal_pca_components,
                cal_mode,
                cal_run_name,
                cal_ckpt_path,
                global_run_dir,
                global_eval_dir,
                global_gate_dir,
                cal_target_review,
                cal_quantile_step,
            ],
            outputs=[cal_stdout, cal_run_selector, report_run_name],
        ).then(
            fn=_load_cal_view,
            inputs=[global_gate_dir, cal_run_selector],
            outputs=[cal_tradeoff, cal_his_true, cal_recommended, cal_stats],
        ).then(
            fn=_load_report_view_from_source,
            inputs=[
                report_source_mode,
                global_gate_dir,
                report_run_name,
                report_scores_path,
            ],
            outputs=[
                report_json,
                report_his_pred_img,
                report_review_gallery,
                report_escape_gallery,
            ],
        ).then(
            fn=_load_dashboard,
            inputs=[
                report_source_mode,
                global_gate_dir,
                report_run_name,
                report_scores_path,
                report_target,
                report_top_k,
                path_from,
                path_to,
                class_mode,
            ],
            outputs=[
                dashboard_state,
                report_scores_path,
                report_threshold,
                report_metrics,
                report_tradeoff_plot,
                report_his_plot,
                report_review_gallery,
                report_escape_gallery,
            ],
        )

        report_source_mode.change(
            fn=_toggle_report_source_mode,
            inputs=[report_source_mode],
            outputs=[report_group_run, report_group_scores],
        )
        report_source_mode.change(
            fn=_load_report_view_from_source,
            inputs=[
                report_source_mode,
                global_gate_dir,
                report_run_name,
                report_scores_path,
            ],
            outputs=[
                report_json,
                report_his_pred_img,
                report_review_gallery,
                report_escape_gallery,
            ],
        )
        report_source_mode.change(
            fn=_load_dashboard,
            inputs=[
                report_source_mode,
                global_gate_dir,
                report_run_name,
                report_scores_path,
                report_target,
                report_top_k,
                path_from,
                path_to,
                class_mode,
            ],
            outputs=[
                dashboard_state,
                report_scores_path,
                report_threshold,
                report_metrics,
                report_tradeoff_plot,
                report_his_plot,
                report_review_gallery,
                report_escape_gallery,
            ],
        )
        report_mode.change(
            fn=_toggle_report_mode,
            inputs=[report_mode],
            outputs=[report_group_target, report_group_threshold],
        )

        report_run_name.change(
            fn=_load_report_view_from_source,
            inputs=[
                report_source_mode,
                global_gate_dir,
                report_run_name,
                report_scores_path,
            ],
            outputs=[
                report_json,
                report_his_pred_img,
                report_review_gallery,
                report_escape_gallery,
            ],
        )
        report_run_name.change(
            fn=_load_dashboard,
            inputs=[
                report_source_mode,
                global_gate_dir,
                report_run_name,
                report_scores_path,
                report_target,
                report_top_k,
                path_from,
                path_to,
                class_mode,
            ],
            outputs=[
                dashboard_state,
                report_scores_path,
                report_threshold,
                report_metrics,
                report_tradeoff_plot,
                report_his_plot,
                report_review_gallery,
                report_escape_gallery,
            ],
        )
        report_scores_path.change(
            fn=_load_report_view_from_source,
            inputs=[
                report_source_mode,
                global_gate_dir,
                report_run_name,
                report_scores_path,
            ],
            outputs=[
                report_json,
                report_his_pred_img,
                report_review_gallery,
                report_escape_gallery,
            ],
        )
        report_scores_path.change(
            fn=_load_dashboard,
            inputs=[
                report_source_mode,
                global_gate_dir,
                report_run_name,
                report_scores_path,
                report_target,
                report_top_k,
                path_from,
                path_to,
                class_mode,
            ],
            outputs=[
                dashboard_state,
                report_scores_path,
                report_threshold,
                report_metrics,
                report_tradeoff_plot,
                report_his_plot,
                report_review_gallery,
                report_escape_gallery,
            ],
        )
        demo.load(
            fn=_load_dashboard,
            inputs=[
                report_source_mode,
                global_gate_dir,
                report_run_name,
                report_scores_path,
                report_target,
                report_top_k,
                path_from,
                path_to,
                class_mode,
            ],
            outputs=[
                dashboard_state,
                report_scores_path,
                report_threshold,
                report_metrics,
                report_tradeoff_plot,
                report_his_plot,
                report_review_gallery,
                report_escape_gallery,
            ],
        )

        load_scores_btn.click(
            fn=_load_dashboard,
            inputs=[
                report_source_mode,
                global_gate_dir,
                report_run_name,
                report_scores_path,
                report_target,
                report_top_k,
                path_from,
                path_to,
                class_mode,
            ],
            outputs=[
                dashboard_state,
                report_scores_path,
                report_threshold,
                report_metrics,
                report_tradeoff_plot,
                report_his_plot,
                report_review_gallery,
                report_escape_gallery,
            ],
        )

        report_target.release(
            fn=_on_review_change_delayed,
            inputs=[
                dashboard_state,
                report_target,
                report_top_k,
                path_from,
                path_to,
                class_mode,
            ],
            outputs=[
                report_threshold,
                report_metrics,
                report_tradeoff_plot,
                report_his_plot,
                report_review_gallery,
                report_escape_gallery,
            ],
        )

        report_threshold.release(
            fn=_on_threshold_change_delayed,
            inputs=[
                dashboard_state,
                report_threshold,
                report_top_k,
                path_from,
                path_to,
                class_mode,
            ],
            outputs=[
                report_target,
                report_metrics,
                report_tradeoff_plot,
                report_his_plot,
                report_review_gallery,
                report_escape_gallery,
            ],
        )

        run_report_btn.click(
            fn=_run_report,
            inputs=[
                report_source_mode,
                report_run_name,
                report_scores_path,
                global_eval_dir,
                global_gate_dir,
                report_mode,
                report_threshold,
                report_target,
                report_top_k,
                report_copy_images,
            ],
            outputs=[report_stdout],
        ).then(
            fn=_load_report_meta_from_source,
            inputs=[
                report_source_mode,
                global_gate_dir,
                report_run_name,
                report_scores_path,
            ],
            outputs=[report_json, report_his_pred_img],
        )

    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AOIRiskGate Studio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument(
        "--allow-path",
        action="append",
        default=[],
        help="Additional allowed path for Gradio file serving. Can be used multiple times.",
    )
    p.add_argument(
        "--allow-home",
        dest="allow_home",
        action="store_true",
        default=True,
        help="Allow the current user's HOME directory (default: on).",
    )
    p.add_argument(
        "--no-allow-home",
        dest="allow_home",
        action="store_false",
        help="Disable allowing HOME directory.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    demo = build_app()
    allowed_paths = set(
        [
            str(PROJECT_ROOT),
            str(resolve_ui_path(GLOBAL_DEFAULTS.run_dir)),
            str(resolve_ui_path(GLOBAL_DEFAULTS.eval_dir)),
            str(resolve_ui_path(GLOBAL_DEFAULTS.gate_dir)),
        ]
    )
    # Also allow configured dataset roots for gallery/image preview.
    for p in [TRAIN_DEFAULTS.train_dir, TRAIN_DEFAULTS.val_dir]:
        try:
            allowed_paths.add(str(Path(p).expanduser().resolve()))
        except Exception:
            pass
    # Optional broad allow for dynamic local path switching in UI.
    if args.allow_home:
        try:
            allowed_paths.add(str(Path.home().resolve()))
        except Exception:
            pass
    # User supplied extra allowed paths.
    for p in args.allow_path:
        try:
            allowed_paths.add(str(Path(p).expanduser().resolve()))
        except Exception:
            pass

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        allowed_paths=sorted(allowed_paths),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
