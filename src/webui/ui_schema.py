from __future__ import annotations

from dataclasses import dataclass

from core.config import Config


@dataclass(frozen=True)
class GlobalDefaults:
    img_size: int = Config.img_size
    batch_size: int = Config.batch_size
    num_workers: int = Config.num_workers
    device: str = Config.device
    model: str = Config.model_name

    run_dir: str = Config.run_dir
    eval_dir: str = Config.eval_result_dir
    gate_dir: str = Config.gate_result_dir


@dataclass(frozen=True)
class TrainDefaults:
    train_dir: str = Config.train_dir
    val_dir: str = Config.val_dir
    epochs: int = Config.epochs
    lr: float = Config.lr
    weight_decay: float = Config.weight_decay
    momentum: float = Config.momentum
    pretrained: bool = True
    freeze_backbone: bool = True


@dataclass(frozen=True)
class EvalDefaults:
    val_dir: str = Config.val_dir
    copy_misclassified: bool = False
    max_misclassified: int = 0


@dataclass(frozen=True)
class CalibrateDefaults:
    val_dir: str = Config.val_dir
    method: str = "pca_recon_pred_class"
    min_correct_per_class: int = 5
    pca_components: int = 32
    target_review: float | None = None
    quantile_step: float = 0.005


@dataclass(frozen=True)
class ReportDefaults:
    target_review: float = 0.05
    threshold: float = 0.0
    top_k: int = 24
    copy_images: bool = False


GLOBAL_DEFAULTS = GlobalDefaults()
TRAIN_DEFAULTS = TrainDefaults()
EVAL_DEFAULTS = EvalDefaults()
CAL_DEFAULTS = CalibrateDefaults()
REPORT_DEFAULTS = ReportDefaults()

MODEL_CHOICES = ["resnet18", "simple"]
DEVICE_CHOICES = ["auto", "cpu", "cuda", "cuda:0"]
CAL_METHOD_CHOICES = ["pca_recon_pred_class", "centroid_pred_class"]
