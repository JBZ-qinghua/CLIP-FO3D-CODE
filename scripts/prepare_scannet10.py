#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from clip_fo3d.scannet10 import prepare_scene_10frames, verify_10frame_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the ScanNet RGB stream every 10 frames for CLIP-FO3D")
    parser.add_argument("--raw-root", required=True, help="ScanNet root containing scans/sceneXXXX_XX")
    parser.add_argument("--point-root", required=True, help="Processed point root containing train/ and val/")
    parser.add_argument("--output-root", default=str(PROJECT / "data/scannet10/images"))
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--scene", action="append", help="Scene id; repeat for a smoke test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--depth-tolerance", type=float, default=0.05)
    parser.add_argument("--export-depth", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def one_task(task):
    return prepare_scene_10frames(*task)


def main() -> None:
    args = parse_args()
    raw_root, point_root, output_root = map(Path, (args.raw_root, args.point_root, args.output_root))
    if args.verify_only:
        report = verify_10frame_export(point_root, output_root, args.splits)
        print(json.dumps(report, indent=2))
        if any(item["complete_scenes"] != item["expected_scenes"] for item in report["splits"].values()):
            raise SystemExit(1)
        return

    tasks, missing = [], []
    for split in args.splits:
        for point_scene in sorted((point_root / split).glob("scene*")):
            if args.scene and point_scene.name not in args.scene:
                continue
            raw_scene = raw_root / "scans" / point_scene.name
            if not (raw_scene / f"{point_scene.name}.sens").is_file():
                missing.append(point_scene.name)
                continue
            tasks.append((raw_scene, point_scene, output_root / split / point_scene.name,
                          10, args.depth_tolerance, args.export_depth))
    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"Missing raw .sens for {len(missing)} scenes (first: {preview})")
    print(f"Preparing {len(tasks)} scenes with frame_gap=10 and {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one_task, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"[{index}/{len(tasks)}] {result['scene']}: {result['exported_frames']} frames")
    report = verify_10frame_export(point_root, output_root, args.splits)
    (output_root / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

