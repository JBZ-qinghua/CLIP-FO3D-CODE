#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from clip_fo3d.config import load_config, project_path
from clip_fo3d.dataset import list_scenes
from clip_fo3d.dense_clip import DenseCLIPExtractor
from clip_fo3d.projection import project_scene, save_targets, scene_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP-FO3D point targets from ScanNet RGB views")
    parser.add_argument("--config", default=str(PROJECT / "configs/scannet.yaml"))
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--scene", action="append", help="Scene id; repeat for multiple scenes")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--max-views", type=int)
    parser.add_argument("--device")
    parser.add_argument("--image-root")
    parser.add_argument("--target-root")
    parser.add_argument("--scales", type=float, nargs="+")
    parser.add_argument("--slic-workers", type=int)
    parser.add_argument("--view-batch-size", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dense_cfg = cfg["dense_clip"]
    point_root = project_path(cfg, cfg["paths"]["point_root"])
    image_root = Path(args.image_root).resolve() if args.image_root else project_path(cfg, cfg["paths"]["image_root"])
    target_root = Path(args.target_root).resolve() if args.target_root else project_path(cfg, cfg["paths"]["target_root"])
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("Require num-shards >= 1 and 0 <= shard-id < num-shards")
    scenes = args.scene or list_scenes(point_root, args.split)
    scenes = scenes[args.shard_id::args.num_shards]
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    extractor = DenseCLIPExtractor(
        project_path(cfg, cfg["paths"]["clip_checkpoint"]),
        device=args.device or cfg["device"],
        n_segments=dense_cfg["n_segments"],
        compactness=dense_cfg["compactness"],
        slic_sigma=dense_cfg["slic_sigma"],
        crop_batch_size=dense_cfg["crop_batch_size"],
        precision=dense_cfg["precision"],
        slic_workers=args.slic_workers or dense_cfg["slic_workers"],
    )
    scales = args.scales or dense_cfg["scales"]
    max_views = args.max_views if args.max_views is not None else dense_cfg["max_views"]
    summary = {}
    for scene in scenes:
        destination = target_root / args.split / f"{scene}.npz"
        if destination.exists() and not args.force:
            total_views = len(scene_views(image_root, args.split, scene))
            requested_views = total_views if max_views is None else min(max_views, total_views)
            complete = False
            try:
                with np.load(destination) as payload:
                    same_scales = "scales" in payload and np.allclose(payload["scales"], scales)
                    same_stride = "stride_ratio" in payload and np.isclose(
                        float(payload["stride_ratio"]), dense_cfg["stride_ratio"]
                    )
                    complete = int(payload["views"]) == requested_views and same_scales and same_stride
            except (KeyError, ValueError, OSError):
                complete = False
            if complete:
                print(f"skip complete target {destination}")
                continue
            print(f"regenerate incomplete or mismatched target {destination}")
        features, valid, stats = project_scene(
            extractor, point_root, image_root, args.split, scene, scales=scales,
            stride_ratio=dense_cfg["stride_ratio"], max_views=max_views,
            view_batch_size=args.view_batch_size or dense_cfg["view_batch_size"],
        )
        save_targets(destination, features, valid, stats, cfg["projection"]["feature_dtype"])
        summary[scene] = stats
        print(json.dumps({scene: stats}, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
