#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from clip_fo3d.config import load_config, project_path
from clip_fo3d.projection import verify_target_set
from clip_fo3d.scannet10 import verify_10frame_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the strict CLIP-FO3D ScanNet protocol")
    parser.add_argument("--config", default=str(PROJECT / "configs/scannet.yaml"))
    parser.add_argument("--check-targets", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    point_root = project_path(cfg, cfg["paths"]["point_root"])
    image_root = project_path(cfg, cfg["paths"]["image_root"])
    image_report = verify_10frame_export(point_root, image_root)
    report: dict[str, object] = {"images": image_report}
    success = all(
        split_report["complete_scenes"] == split_report["expected_scenes"]
        for split_report in image_report["splits"].values()
    )
    if args.check_targets:
        target_report = verify_target_set(
            point_root,
            image_root,
            project_path(cfg, cfg["paths"]["target_root"]),
            cfg["train"]["split"],
            cfg["dense_clip"]["scales"],
            cfg["dense_clip"]["stride_ratio"],
        )
        report["targets"] = target_report
        success = success and target_report["complete_scenes"] == target_report["expected_scenes"]
    print(json.dumps(report, indent=2))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
