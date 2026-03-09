from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from core.config import Config

FALLBACK_RUN_NAME = "baseline"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class UiPaths:
    run_dir: Path
    eval_dir: Path
    gate_dir: Path


def build_ui_paths(
    run_dir: str | None = None,
    eval_dir: str | None = None,
    gate_dir: str | None = None,
) -> UiPaths:
    return UiPaths(
        run_dir=resolve_ui_path(run_dir or Config.run_dir),
        eval_dir=resolve_ui_path(eval_dir or Config.eval_result_dir),
        gate_dir=resolve_ui_path(gate_dir or Config.gate_result_dir),
    )


def resolve_ui_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _list_subdirs(path: Path) -> List[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()], key=lambda x: x.name)


def list_run_names(path: str | Path) -> list[str]:
    root = resolve_ui_path(path)
    return [p.name for p in _list_subdirs(root)]


def latest_run_name(path: str | Path) -> Optional[str]:
    root = resolve_ui_path(path)
    dirs = _list_subdirs(root)
    if not dirs:
        return None
    newest = max(dirs, key=lambda p: p.stat().st_mtime)
    return newest.name


def pick_default_run_name(
    root: str | Path,
    required_files: Iterable[str] | None = None,
    fallback_run: str = FALLBACK_RUN_NAME,
) -> Optional[str]:
    root_path = resolve_ui_path(root)
    required = list(required_files or [])

    def has_required(run_name: str) -> bool:
        run_dir = root_path / run_name
        return all((run_dir / rel).exists() for rel in required)

    latest = latest_run_name(root_path)
    if latest is not None:
        if has_required(latest):
            return latest

    if (root_path / fallback_run).exists() and has_required(fallback_run):
        return fallback_run

    if latest is not None:
        return latest

    runs = list_run_names(root_path)
    return runs[0] if runs else None


def resolve_checkpoint_from_run(run_dir: str | Path, run_name: str) -> Optional[Path]:
    base = resolve_ui_path(run_dir) / run_name / "checkpoints"
    best = base / "best.pt"
    last = base / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    return None


def infer_run_name_from_scores(scores_path: str | Path) -> str:
    p = Path(scores_path)
    if p.name == "scores.csv":
        return p.parent.name
    return p.stem


def safe_file(path: str | Path) -> Optional[str]:
    p = resolve_ui_path(path)
    return str(p) if p.exists() else None


def list_images(path: str | Path) -> list[str]:
    p = resolve_ui_path(path)
    if not p.exists() or not p.is_dir():
        return []

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    imgs = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts]
    imgs.sort(key=lambda x: x.name)
    return [str(x) for x in imgs]


def empty_hint(msg: str) -> str:
    return f"[No Data] {msg}"
