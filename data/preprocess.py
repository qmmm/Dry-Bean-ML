"""数据预处理命令封装。"""

from __future__ import annotations

from pathlib import Path

from data import clean as clean_module
from data import select_features as select_features_module


def clean(args: list[str] | None = None) -> int:
    return clean_module.main()


def select_features(args: list[str] | None = None) -> int:
    return select_features_module.main()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
