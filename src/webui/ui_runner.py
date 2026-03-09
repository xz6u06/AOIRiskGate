from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _build_cmd(script_name: str, args: Iterable[str]) -> list[str]:
    script_path = Path(__file__).resolve().parents[1] / "core" / script_name
    return [sys.executable, str(script_path), *args]


def run_script(script_name: str, args: Iterable[str]) -> tuple[bool, str]:
    cmd = _build_cmd(script_name, args)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        return False, f"Failed to run {script_name}: {e}"

    out = []
    out.append(f"$ {' '.join(cmd)}")
    if p.stdout:
        out.append("\n[stdout]\n" + p.stdout.strip())
    if p.stderr:
        out.append("\n[stderr]\n" + p.stderr.strip())

    if p.returncode == 0:
        out.append("\n[status] success")
        return True, "\n".join(out)

    out.append(f"\n[status] failed (exit={p.returncode})")
    return False, "\n".join(out)


def bool_flag(enabled: bool, true_flag: str, false_flag: str | None = None) -> list[str]:
    if enabled:
        return [true_flag]
    if false_flag:
        return [false_flag]
    return []
