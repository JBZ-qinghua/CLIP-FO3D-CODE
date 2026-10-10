from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(path.parents[1])
    return cfg


def project_path(cfg: dict[str, Any], value: str | Path) -> Path:
    value = Path(value)
    if value.is_absolute():
        return value
    return Path(cfg["_project_root"]) / value

