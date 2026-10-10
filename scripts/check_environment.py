#!/usr/bin/env python
from __future__ import annotations

import importlib
import json
import platform
import sys

import torch


def main() -> None:
    versions = {}
    for name in ("clip", "MinkowskiEngine", "yaml"):
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "installed")
    result = {
        "python": platform.python_version(),
        "prefix": sys.prefix,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "modules": versions,
    }
    print(json.dumps(result, indent=2))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")


if __name__ == "__main__":
    main()
